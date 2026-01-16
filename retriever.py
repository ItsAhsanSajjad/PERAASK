from __future__ import annotations

import os
import re
from typing import List, Dict, Any
from collections import defaultdict

from index_store import load_index_and_chunks, embed_texts, _normalize_vectors

# -----------------------------
# Retrieval configuration
# -----------------------------
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "40"))
SIM_THRESHOLD = float(os.getenv("RETRIEVER_SIM_THRESHOLD", "0.20"))
MAX_CHUNKS_PER_DOC = int(os.getenv("RETRIEVER_MAX_CHUNKS_PER_DOC", "6"))

STRONG_SIM_THRESHOLD = float(os.getenv("RETRIEVER_STRONG_SIM_THRESHOLD", "0.28"))

MIN_KEYWORD_MATCHES = int(os.getenv("RETRIEVER_MIN_KEYWORD_MATCHES", "2"))
RELATIVE_DOC_SCORE_KEEP = float(os.getenv("RETRIEVER_RELATIVE_DOC_SCORE_KEEP", "0.80"))
MAX_DOCS_RETURNED = int(os.getenv("RETRIEVER_MAX_DOCS_RETURNED", "4"))

QUERY_VARIANTS_ENABLED = os.getenv("RETRIEVER_QUERY_VARIANTS_ENABLED", "1").strip() != "0"
MAX_QUERY_VARIANTS = int(os.getenv("RETRIEVER_MAX_QUERY_VARIANTS", "3"))

RELAXED_MIN_KEYWORD_MATCHES = int(os.getenv("RETRIEVER_RELAXED_MIN_KEYWORD_MATCHES", "1"))

LEX_FALLBACK_ENABLED = os.getenv("RETRIEVER_LEX_FALLBACK_ENABLED", "1").strip() != "0"
LEX_FALLBACK_MAX = int(os.getenv("RETRIEVER_LEX_FALLBACK_MAX", "80"))
LEX_FALLBACK_PER_DOC = int(os.getenv("RETRIEVER_LEX_FALLBACK_PER_DOC", "3"))

CRITERIA_DOC_PRIORITIZATION = os.getenv("RETRIEVER_CRITERIA_DOC_PRIORITIZATION", "1").strip() != "0"
CRITERIA_MIN_DOCS = int(os.getenv("RETRIEVER_CRITERIA_MIN_DOCS", "2"))

# ✅ FAQ-specific safety thresholds (prevents low-score noise)
FAQ_SIM_THRESHOLD = float(os.getenv("RETRIEVER_FAQ_SIM_THRESHOLD", "0.34"))
FAQ_STRONG_DOCNAME_BONUS = float(os.getenv("RETRIEVER_FAQ_DOCNAME_BONUS", "0.20"))

# -----------------------------
# Keyword extraction
# -----------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "from", "by", "with", "about", "tell", "me",
    "who", "what", "when", "where", "why", "how", "please",
    "pera",
}

_INTENT_STOP = {
    "role", "roles", "duty", "duties", "function", "functions", "responsibility",
    "responsibilities", "tor", "tors", "term", "terms", "reference",
    "criteria", "criterion", "eligibility", "eligible", "qualification", "qualifications",
    "experience", "education", "minimum", "required", "requirement", "requirements",
    "position", "positions", "post", "posts", "job", "jobs", "main", "most", "power", "authority",
}

_KEEP_SHORT = {"ai", "ml", "it", "hr", "ppra", "ipo", "cto", "tor", "tors", "dg"}

_ABBREV_MAP = {
    "cto": "chief technology officer",
    "tor": "terms of reference",
    "tors": "terms of reference",
    "dg": "director general",
    "hr": "human resource",
    "it": "information technology",
    "mgr": "manager",
    "dev": "development",
    "sr": "senior",
    "jr": "junior",
}

# ✅ FAQ intent detection
_FAQ_PAT = re.compile(r"\b(faq|faqs|frequently\s+asked\s+questions?)\b", re.I)

# Tokens that should NEVER be required as entity overlap
_FAQ_STOP_TOKENS = {"faq", "faqs", "frequently", "asked", "question", "questions", "answer", "answers", "q", "a"}

# Docname patterns to prefer for FAQ queries
_FAQ_DOCNAME_PAT = re.compile(r"(faq|faqs|frequently\s+asked\s+questions?)", re.I)


def _expand_abbrev(s: str) -> str:
    t = (s or "").lower()
    for k, v in _ABBREV_MAP.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t)
    return t


def _normalize_text(s: str) -> str:
    s = _expand_abbrev(s or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s


def _stem_token(t: str) -> str:
    t = (t or "").strip().lower()
    if len(t) <= 3:
        return t

    if t.endswith("'s"):
        t = t[:-2]

    if t.endswith("ies") and len(t) > 4:
        return t[:-3] + "y"
    if t.endswith("es") and len(t) > 4:
        return t[:-2]
    if t.endswith("s") and not t.endswith("ss") and len(t) > 4:
        return t[:-1]

    return t


def _tokenize_for_overlap(s: str) -> List[str]:
    q = _normalize_text(s)
    toks: List[str] = []
    for raw in q.split():
        if not raw:
            continue
        if raw in _STOPWORDS:
            continue
        if len(raw) >= 3 or raw in _KEEP_SHORT:
            toks.append(_stem_token(raw))
    return toks


def _extract_keywords(question: str) -> List[str]:
    toks = _tokenize_for_overlap(question)
    seen = set()
    out: List[str] = []
    for t in toks:
        if t in seen:
            continue
        seen.add(t)
        out.append(t)
    return out[:12]


def _entity_keywords(question: str) -> List[str]:
    """
    Entity-focused tokens.
    ✅ FAQ fix: if query is FAQ-intent, do NOT treat faq/question tokens as entities.
    """
    toks = _tokenize_for_overlap(question)
    qn = _normalize_text(question)
    is_faq = _FAQ_PAT.search(qn) is not None

    ent: List[str] = []
    seen = set()
    for t in toks:
        if t in _INTENT_STOP:
            continue
        if is_faq and t in _FAQ_STOP_TOKENS:
            continue
        if t in seen:
            continue
        seen.add(t)
        ent.append(t)
    return ent[:10]


def _keyword_overlap_count(keywords: List[str], text: str) -> int:
    if not keywords:
        return 0
    kw_set = set(keywords)
    text_tokens = set(_tokenize_for_overlap(text))
    return len(kw_set.intersection(text_tokens))


def _rows_by_id(rows: List[Dict[str, Any]]) -> Dict[int, Dict[str, Any]]:
    out: Dict[int, Dict[str, Any]] = {}
    for r in rows:
        try:
            out[int(r.get("id", -1))] = r
        except Exception:
            continue
    return out


def _load_index_rows(index_dir: str):
    try:
        return load_index_and_chunks(index_dir=index_dir)  # type: ignore[arg-type]
    except TypeError:
        return load_index_and_chunks()


# -----------------------------
# Intent helpers
# -----------------------------
_COMPOSITION_PAT = re.compile(r"\b(composition|constitut|constitution|constitute|consist|comprise|members?|authority)\b", re.I)
_COMPOSITION_PHRASES = [
    "authority shall consist of",
    "constitution of the authority",
    "members of the authority",
    "the authority shall consist",
    "shall consist of the following members",
]

_CRITERIA_PAT = re.compile(
    r"\b(criteria|criterion|eligib|qualification|qualify|required|requirement|experience|education|minimum|degree|age|skills?)\b",
    re.I
)
_CRITERIA_PHRASES = [
    "eligibility criteria",
    "minimum qualification",
    "qualification and experience",
    "required qualification",
    "experience required",
    "education",
    "age limit",
    "skills",
]

_ROLE_PAT = re.compile(r"\b(role|roles|tor|tors|terms of reference|duty|duties|responsibil|function|job description)\b", re.I)
_POWER_PAT = re.compile(r"\b(main authority|most power|most powerful|who holds the most power|who is powerful)\b", re.I)


def _intent_extra_keywords(question: str) -> List[str]:
    q = (question or "").lower()
    extras: List[str] = []

    if _COMPOSITION_PAT.search(q):
        extras.extend(["shall", "consist", "comprise", "constitution", "member", "chairperson", "vice", "secretary", "include", "following"])

    if _CRITERIA_PAT.search(q):
        extras.extend(["eligibility", "criteria", "qualification", "experience", "education", "minimum", "required", "requirement", "age", "degree", "competenc", "skill"])

    if _ROLE_PAT.search(q):
        extras.extend(["terms", "reference", "tor", "duties", "responsibilities", "functions", "report", "reports", "wing", "purpose"])

    if _POWER_PAT.search(q):
        extras.extend(["chairperson", "vice", "authority", "director", "general", "member", "secretary"])

    # ✅ FAQ: add Q/A tokens for overlap scoring (NOT as entity)
    if _FAQ_PAT.search(q):
        extras.extend(["question", "questions", "answer", "answers", "q", "a"])

    extras2: List[str] = []
    seen = set()
    for e in extras:
        se = _stem_token(e)
        if se in seen:
            continue
        seen.add(se)
        extras2.append(se)
    return extras2


def _build_query_variants(question: str) -> List[str]:
    q = (question or "").strip()
    if not q:
        return [q]

    variants: List[str] = [q]
    qn = _normalize_text(q)

    # ✅ FAQ query variants (helps semantic search hit Q/A blocks)
    if _FAQ_PAT.search(qn):
        variants.append("frequently asked questions")
        variants.append("questions and answers")
        variants.append("Q: A:")

    # de-dup preserve order
    seen = set()
    out: List[str] = []
    for v in variants:
        v = v.strip()
        if not v:
            continue
        key = v.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(v)

    return out[:MAX_QUERY_VARIANTS]


def _required_overlap(keywords: List[str], strict: bool) -> int:
    n = len(keywords)
    if n == 0:
        return 0
    if strict:
        if n == 1:
            return 1
        return max(2, MIN_KEYWORD_MATCHES)
    return max(1, RELAXED_MIN_KEYWORD_MATCHES)


def _lexical_fallback_hits(rows: List[Dict[str, Any]], all_keywords: List[str], entity_kw: List[str], question: str) -> Dict[int, float]:
    if not LEX_FALLBACK_ENABLED:
        return {}

    qn = _normalize_text(question)
    want_comp = _COMPOSITION_PAT.search(qn) is not None
    want_criteria = _CRITERIA_PAT.search(qn) is not None
    if not want_comp and not want_criteria:
        return {}

    phrases = _COMPOSITION_PHRASES if want_comp else _CRITERIA_PHRASES

    best: Dict[int, float] = {}
    per_doc_counts: Dict[str, int] = defaultdict(int)

    for r in rows:
        if not r.get("active", True):
            continue

        text = r.get("text") or ""
        if not text:
            continue

        tn = _normalize_text(text)

        phrase_hit = any(p in tn for p in phrases)
        overlap_all = _keyword_overlap_count(all_keywords, text)
        overlap_ent = _keyword_overlap_count(entity_kw, text) if entity_kw else 0

        if entity_kw and overlap_ent < 1 and not phrase_hit:
            continue
        if not phrase_hit and overlap_all < 2:
            continue

        doc_name = r.get("doc_name", "Unknown document")
        if per_doc_counts[doc_name] >= LEX_FALLBACK_PER_DOC:
            continue

        try:
            cid = int(r.get("id"))
        except Exception:
            continue

        pseudo = 0.45 + min(0.20, 0.03 * overlap_all)
        if phrase_hit:
            pseudo = max(pseudo, 0.62)

        prev = best.get(cid)
        if prev is None or pseudo > prev:
            best[cid] = pseudo
            per_doc_counts[doc_name] += 1

        if len(best) >= LEX_FALLBACK_MAX:
            break

    return best


def retrieve(question: str, index_dir: str = "assets/index") -> Dict[str, Any]:
    empty = {
        "question": question,
        "has_evidence": False,
        "primary_doc": None,
        "primary_doc_rank": 0,
        "evidence": []
    }

    question = (question or "").strip()
    if not question:
        return empty

    try:
        idx, rows = _load_index_rows(index_dir=index_dir)
        if idx is None or not rows:
            return empty

        id_to_row = _rows_by_id(rows)

        qn = _normalize_text(question)
        is_faq = _FAQ_PAT.search(qn) is not None

        queries = [question]
        if QUERY_VARIANTS_ENABLED:
            queries = _build_query_variants(question)

        entity_kw = _entity_keywords(question)
        base_kw = _extract_keywords(question)
        extras_kw = _intent_extra_keywords(question)
        all_kw = base_kw + [k for k in extras_kw if k not in base_kw]

        q_vecs = embed_texts(queries)
        q_vecs = _normalize_vectors(q_vecs)
        scores_mat, ids_mat = idx.search(q_vecs, TOP_K)

        best_by_id: Dict[int, float] = {}

        for qi in range(len(queries)):
            scores = scores_mat[qi].tolist()
            ids = ids_mat[qi].tolist()
            for score, vid in zip(scores, ids):
                if vid == -1:
                    continue
                score = float(score)

                # ✅ FAQ safety: higher similarity floor to avoid random high-rank docs
                min_score = max(SIM_THRESHOLD, FAQ_SIM_THRESHOLD) if is_faq else SIM_THRESHOLD
                if score < min_score:
                    continue

                vid_i = int(vid)
                prev = best_by_id.get(vid_i)
                if prev is None or score > prev:
                    best_by_id[vid_i] = score

        if not best_by_id:
            return empty

        hits: List[Dict[str, Any]] = []
        for vid_i, score in best_by_id.items():
            r = id_to_row.get(int(vid_i))
            if not r or not r.get("active", True):
                continue

            doc_name = r.get("doc_name", "Unknown document")
            text = (r.get("text") or "")

            overlap_entity = _keyword_overlap_count(entity_kw, text) if entity_kw else 0
            overlap_all = _keyword_overlap_count(all_kw, text)

            # ✅ FAQ: do NOT use entity overlap for gating
            overlap = overlap_all if is_faq else (overlap_entity if entity_kw else overlap_all)

            docname_is_faq = 1 if _FAQ_DOCNAME_PAT.search(doc_name or "") else 0

            # ✅ FAQ docname bonus (only for FAQ queries)
            boosted_score = float(score) + (FAQ_STRONG_DOCNAME_BONUS if (is_faq and docname_is_faq) else 0.0)

            hits.append({
                "id": int(vid_i),
                "score": float(boosted_score),
                "raw_score": float(score),
                "overlap": int(overlap),
                "doc_name": doc_name,
                "doc_rank": int(r.get("doc_rank", 0) or 0),
                "docname_is_faq": int(docname_is_faq),
                "text": text,
                "source_type": r.get("source_type", ""),
                "loc_kind": r.get("loc_kind", ""),
                "loc_start": r.get("loc_start"),
                "loc_end": r.get("loc_end"),
                "path": r.get("path"),
            })

        if not hits:
            return empty

        # Group by document
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        ranks: Dict[str, int] = {}
        doc_is_faq: Dict[str, int] = {}
        for h in hits:
            dn = h["doc_name"]
            grouped[dn].append(h)
            ranks[dn] = max(ranks.get(dn, 0), int(h.get("doc_rank", 0) or 0))
            doc_is_faq[dn] = max(doc_is_faq.get(dn, 0), int(h.get("docname_is_faq", 0) or 0))

        evidence_docs: List[Dict[str, Any]] = []
        for dn, doc_hits in grouped.items():
            doc_hits.sort(
                key=lambda x: (float(x.get("score", 0.0) or 0.0), int(x.get("overlap", 0) or 0)),
                reverse=True
            )
            doc_hits = doc_hits[:MAX_CHUNKS_PER_DOC]
            evidence_docs.append({
                "doc_name": dn,
                "doc_rank": ranks.get(dn, 0),
                "docname_is_faq": doc_is_faq.get(dn, 0),
                "hits": doc_hits
            })

        def best_score(ed: Dict[str, Any]) -> float:
            hs = ed.get("hits", [])
            return float(hs[0]["score"]) if hs else 0.0

        def best_overlap(ed: Dict[str, Any]) -> int:
            hs = ed.get("hits", [])
            return int(hs[0].get("overlap", 0)) if hs else 0

        # ✅ CRITICAL FIX:
        # For FAQ queries: sort by (docname_is_faq, best_score, best_overlap) and IGNORE doc_rank dominance.
        # For non-FAQ: keep original behavior.
        if is_faq:
            evidence_docs.sort(
                key=lambda ed: (int(ed.get("docname_is_faq", 0) or 0), best_score(ed), best_overlap(ed)),
                reverse=True
            )
        else:
            evidence_docs.sort(
                key=lambda ed: (ed.get("doc_rank", 0), best_score(ed), best_overlap(ed)),
                reverse=True
            )

        primary = evidence_docs[0]

        # Prune docs relative to best score
        best = best_score(primary)
        kept_docs: List[Dict[str, Any]] = []
        for ed in evidence_docs:
            if best <= 0:
                continue
            if best_score(ed) >= best * RELATIVE_DOC_SCORE_KEEP:
                kept_docs.append(ed)

        kept_docs = kept_docs[:MAX_DOCS_RETURNED]

        return {
            "question": question,
            "has_evidence": True,
            "primary_doc": primary.get("doc_name"),
            "primary_doc_rank": int(primary.get("doc_rank", 0) or 0),
            "evidence": kept_docs
        }

    except Exception:
        return empty


def reset_retriever_cache() -> None:
    return
