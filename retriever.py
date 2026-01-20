from __future__ import annotations

import os
import re
import json
from typing import List, Dict, Any, Optional
from collections import defaultdict

from dotenv import load_dotenv
load_dotenv()

from openai import OpenAI

from index_store import load_index_and_chunks, embed_texts, _normalize_vectors


# -----------------------------
# Active index pointer
# -----------------------------
class ActiveIndexPointer:
    def __init__(self, pointer_path: str = "assets/indexes/ACTIVE.json"):
        self.pointer_path = (pointer_path or "").replace("\\", "/")

    def read(self) -> Optional[str]:
        if not self.pointer_path:
            return None
        if not os.path.exists(self.pointer_path):
            return None
        try:
            with open(self.pointer_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            p = (data.get("active_index_dir") or "").strip()
            if not p:
                return None
            p = p.replace("\\", "/")
            return p if os.path.isdir(p) else None
        except Exception:
            return None


_ACTIVE_POINTER = ActiveIndexPointer(os.getenv("INDEX_POINTER_PATH", "assets/indexes/ACTIVE.json"))


def _resolve_index_dir(index_dir: Optional[str]) -> str:
    if index_dir and str(index_dir).strip():
        p = str(index_dir).strip().replace("\\", "/")
        return p if os.path.isdir(p) else "assets/index"

    pointer_path = os.getenv("INDEX_POINTER_PATH", "assets/indexes/ACTIVE.json").replace("\\", "/")
    pointer_exists = os.path.exists(pointer_path)

    ptr = _ACTIVE_POINTER.read()
    if ptr:
        return ptr

    if pointer_exists:
        return "assets/indexes/__INVALID_POINTER__"

    env_dir = os.getenv("INDEX_DIR", "").strip()
    if env_dir:
        env_dir = env_dir.replace("\\", "/")
        if os.path.isdir(env_dir):
            return env_dir

    return "assets/index"


# -----------------------------
# Retriever configuration
# -----------------------------
TOP_K = int(os.getenv("RETRIEVER_TOP_K", "40"))
SIM_THRESHOLD = float(os.getenv("RETRIEVER_SIM_THRESHOLD", "0.18"))
MAX_CHUNKS_PER_DOC = int(os.getenv("RETRIEVER_MAX_CHUNKS_PER_DOC", "6"))

STRONG_SIM_THRESHOLD = float(os.getenv("RETRIEVER_STRONG_SIM_THRESHOLD", "0.26"))

MIN_KEYWORD_MATCHES = int(os.getenv("RETRIEVER_MIN_KEYWORD_MATCHES", "2"))
RELAXED_MIN_KEYWORD_MATCHES = int(os.getenv("RETRIEVER_RELAXED_MIN_KEYWORD_MATCHES", "1"))

RELATIVE_DOC_SCORE_KEEP = float(os.getenv("RETRIEVER_RELATIVE_DOC_SCORE_KEEP", "0.80"))
MAX_DOCS_RETURNED = int(os.getenv("RETRIEVER_MAX_DOCS_RETURNED", "4"))

QUERY_VARIANTS_ENABLED = os.getenv("RETRIEVER_QUERY_VARIANTS_ENABLED", "1").strip() != "0"
MAX_QUERY_VARIANTS = int(os.getenv("RETRIEVER_MAX_QUERY_VARIANTS", "3"))

LEX_FALLBACK_ENABLED = os.getenv("RETRIEVER_LEX_FALLBACK_ENABLED", "1").strip() != "0"
LEX_FALLBACK_MAX = int(os.getenv("RETRIEVER_LEX_FALLBACK_MAX", "80"))
LEX_FALLBACK_PER_DOC = int(os.getenv("RETRIEVER_LEX_FALLBACK_PER_DOC", "3"))

CRITERIA_DOC_PRIORITIZATION = os.getenv("RETRIEVER_CRITERIA_DOC_PRIORITIZATION", "1").strip() != "0"
CRITERIA_MIN_DOCS = int(os.getenv("RETRIEVER_CRITERIA_MIN_DOCS", "2"))

# Spell correction
SPELL_CORRECTION_ENABLED = os.getenv("RETRIEVER_SPELL_CORRECTION_ENABLED", "1").strip() != "0"
SPELL_MAX_TOKEN_FIXES = int(os.getenv("RETRIEVER_SPELL_MAX_TOKEN_FIXES", "2"))
SPELL_EDIT_DISTANCE = int(os.getenv("RETRIEVER_SPELL_EDIT_DISTANCE", "2"))
MAX_QUERY_VARIANTS_WITH_SPELL = int(os.getenv("RETRIEVER_MAX_QUERY_VARIANTS_WITH_SPELL", "5"))

# ✅ LLM rewrite controls (ALWAYS)
LLM_REWRITE_ENABLED = os.getenv("RETRIEVER_LLM_QUERY_REWRITE_ENABLED", "1").strip() != "0"
LLM_REWRITE_ALWAYS = os.getenv("RETRIEVER_LLM_QUERY_REWRITE_ALWAYS", "1").strip() != "0"
LLM_REWRITE_MODEL = os.getenv("RETRIEVER_LLM_QUERY_REWRITE_MODEL", "gpt-4.1-mini")
LLM_REWRITE_MAX = int(os.getenv("RETRIEVER_LLM_QUERY_REWRITE_MAX", "3"))

# Optional debug (works same in Streamlit + FastAPI)
DEBUG = os.getenv("RETRIEVER_DEBUG", "0").strip() != "0"


# -----------------------------
# Keyword extraction / normalization
# -----------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "from", "by", "with", "about", "tell", "me",
    "who", "what", "when", "where", "why", "how", "please",
}

_INTENT_STOP = {
    "role", "roles", "duty", "duties", "function", "functions", "responsibility",
    "responsibilities", "tor", "tors", "term", "terms", "reference",
    "criteria", "criterion", "eligibility", "eligible", "qualification", "qualifications",
    "experience", "education", "minimum", "required", "requirement", "requirements",
    "position", "positions", "post", "posts", "job", "jobs", "main", "most", "power",
}

_KEEP_SHORT = {"ai", "ml", "it", "hr", "ppra", "ipo", "cto", "tor", "tors", "dg", "pera"}

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

_COMMON_MISSPELLINGS = {
    "pira": "pera",
    "perra": "pera",
    "peera": "pera",
    "peraa": "pera",
    "peraah": "pera",
}


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
    for k, v in _COMMON_MISSPELLINGS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
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
    toks = _tokenize_for_overlap(question)
    ent: List[str] = []
    seen = set()
    for t in toks:
        if t in _INTENT_STOP:
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
# Spell correction (conservative)
# -----------------------------
_VOCAB_CACHE: Optional[set] = None
_VOCAB_INDEX_DIR: Optional[str] = None


def _build_vocab_from_rows(rows: List[Dict[str, Any]], limit_rows: int = 50000) -> set:
    vocab = set()
    n = 0
    for r in rows:
        if not r.get("active", True):
            continue
        txt = (r.get("text") or "")
        if not txt:
            continue
        for t in _tokenize_for_overlap(txt):
            if len(t) >= 3 or t in _KEEP_SHORT:
                vocab.add(t)
        n += 1
        if n >= limit_rows:
            break

    vocab.update({
        "pera", "authority", "complaint", "grievance", "hearing", "officer",
        "vision", "mission", "purpose", "objectives", "mandate",
        "terms", "reference", "duties", "responsibilities",
        "manager", "monitoring",
    })
    return vocab


def _levenshtein_lte_k(a: str, b: str, k: int) -> Optional[int]:
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if abs(la - lb) > k:
        return None
    if la == 0:
        return lb if lb <= k else None
    if lb == 0:
        return la if la <= k else None

    prev = list(range(lb + 1))
    for i, ca in enumerate(a, start=1):
        cur = [i] + [0] * lb
        row_min = cur[0]
        for j, cb in enumerate(b, start=1):
            cost = 0 if ca == cb else 1
            cur[j] = min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + cost)
            if cur[j] < row_min:
                row_min = cur[j]
        if row_min > k:
            return None
        prev = cur
    return prev[lb] if prev[lb] <= k else None


def _spell_correct_query_variant(question: str, vocab: set) -> Optional[str]:
    if not SPELL_CORRECTION_ENABLED:
        return None

    qn = _normalize_text(question)
    if not qn:
        return None

    toks = qn.split()
    if len(toks) == 0 or len(toks) > 24:
        return None

    corrected: List[str] = []
    changed = 0
    k = max(1, int(SPELL_EDIT_DISTANCE))

    for t in toks:
        if t.isdigit() or (len(t) < 4 and t not in _KEEP_SHORT) or t in _STOPWORDS or t in _KEEP_SHORT:
            corrected.append(t)
            continue

        st = _stem_token(t)
        if t in vocab or st in vocab:
            corrected.append(t)
            continue

        first = t[0]
        lt = len(t)
        best = None
        best_d = k + 1

        for w in vocab:
            if not w:
                continue
            if w[0] != first:
                continue
            lw = len(w)
            if abs(lw - lt) > k:
                continue

            d = _levenshtein_lte_k(t, w, k)
            if d is None:
                continue
            if d < best_d:
                best_d = d
                best = w
                if d == 1:
                    break

        if best and best_d <= k:
            corrected.append(best)
            changed += 1
            if changed >= max(1, int(SPELL_MAX_TOKEN_FIXES)):
                corrected.extend(toks[len(corrected):])
                break
        else:
            corrected.append(t)

    if changed == 0:
        return None

    out = " ".join(corrected).strip()
    return out if out and out != qn else None


# -----------------------------
# Intent patterns + expansions
# -----------------------------
_COMPOSITION_PAT = re.compile(r"\b(composition|constitut|constitution|constitute|consist|comprise|members?|authority)\b", re.I)
_CRITERIA_PAT = re.compile(r"\b(criteria|criterion|eligib|qualification|qualify|required|requirement|experience|education|minimum|degree|age|skills?)\b", re.I)
_COMPLAINT_PAT = re.compile(r"\b(complaint|complain|grievance|petition|hearing|hearing officer|appeal)\b", re.I)
_VISION_PAT = re.compile(r"\b(vision|mission|objective|objectives|purpose|aim|aims|mandate)\b", re.I)
_ROLE_PAT = re.compile(r"\b(role|roles|tor|tors|terms of reference|duty|duties|responsibil|function|job description)\b", re.I)

_COMPOSITION_PHRASES = [
    "authority shall consist of",
    "constitution of the authority",
    "members of the authority",
    "the authority shall consist",
    "shall consist of the following members",
]
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
_COMPLAINT_PHRASES = [
    "file a complaint",
    "how can i file a complaint",
    "public complaint",
    "complaints & hearings",
    "complaint with pera",
    "hearing officer",
    "what happens after i file a complaint",
]
_VISION_PHRASES = [
    "purpose of the authority",
    "purpose of pera",
    "objectives",
    "functions of the authority",
    "mandate",
    "established to",
    "for the purpose of",
]
_ROLE_PHRASES = [
    "terms of reference",
    "tor",
    "job description",
    "duties and responsibilities",
    "responsibilities include",
    "shall be responsible for",
]


def _intent_extra_keywords(question: str) -> List[str]:
    q = _normalize_text(question)
    extras: List[str] = []

    if _COMPOSITION_PAT.search(q):
        extras.extend(["shall", "consist", "comprise", "constitution", "member", "chairperson", "vice", "secretary", "include", "following"])
    if _CRITERIA_PAT.search(q):
        extras.extend(["eligibility", "criteria", "qualification", "experience", "education", "minimum", "required", "requirement", "age", "degree"])
    if _COMPLAINT_PAT.search(q):
        extras.extend(["complaint", "complaints", "hearing", "hearing officer", "file", "submit", "procedure", "process"])
    if _VISION_PAT.search(q):
        extras.extend(["purpose", "objectives", "functions", "mandate", "aim", "established"])
    if _ROLE_PAT.search(q):
        extras.extend(["terms", "reference", "tor", "duties", "responsibilities", "job", "description", "reports", "reporting", "manager"])

    out: List[str] = []
    seen = set()
    for e in extras:
        se = _stem_token(e)
        if se in seen:
            continue
        seen.add(se)
        out.append(se)
    return out


def _swap_two_word_title(ent: List[str]) -> str:
    if len(ent) != 2:
        return ""
    a, b = ent[0], ent[1]
    if not a or not b:
        return ""
    return f"{b} {a}".strip()


def _build_query_variants(question: str) -> List[str]:
    q = (question or "").strip()
    if not q:
        return [q]

    variants: List[str] = [q]
    qn = _normalize_text(q)
    if qn and qn != _normalize_text(variants[0]):
        variants.append(qn)

    # Always add PERA anchor
    if "pera" not in qn.split():
        variants.append((q + " in PERA").strip())

    if _COMPLAINT_PAT.search(qn):
        variants.append("how can i file a complaint with pera procedure")
        variants.append("public complaints and hearings hearing officer process")
    if _VISION_PAT.search(qn):
        variants.append("purpose objectives functions mandate of pera")
        variants.append("objects and purposes for which the authority is established")
    if _ROLE_PAT.search(qn):
        variants.append("terms of reference duties and responsibilities job description in pera")

    ent = _entity_keywords(q)
    ent_phrase = " ".join(ent[:4]).strip()

    if _COMPOSITION_PAT.search(qn):
        variants.append("authority shall consist of the following members chairperson vice chairperson secretary member")
        variants.append("constitution of the authority members of the authority")

    if _CRITERIA_PAT.search(qn):
        variants.append(f"{q} eligibility criteria qualification experience")
        variants.append(f"{q} minimum qualification experience required")

    if _ROLE_PAT.search(qn) and ent_phrase:
        variants.append(f"terms of reference of {ent_phrase} in PERA")
        variants.append(f"duties and responsibilities of {ent_phrase} in PERA")
        variants.append(f"job description of {ent_phrase} in PERA")

    swapped = _swap_two_word_title(ent[:2])
    if swapped:
        variants.append(f"{swapped} job description duties responsibilities in PERA")

    out: List[str] = []
    seen = set()
    for v in variants:
        v = (v or "").strip()
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


def _dedup_and_cap_queries(queries: List[str], cap: int) -> List[str]:
    out: List[str] = []
    seen = set()
    for q in queries:
        q = (q or "").strip()
        if not q:
            continue
        k = q.lower()
        if k in seen:
            continue
        seen.add(k)
        out.append(q)
        if len(out) >= cap:
            break
    return out


# -----------------------------
# Lexical fallback helpers (uses text + search_text)
# -----------------------------
def _row_text_for_matching(r: Dict[str, Any]) -> str:
    a = (r.get("text") or "")
    b = (r.get("search_text") or "")
    if b and b not in a:
        return a + "\n" + b
    return a


def _lexical_fallback_hits(rows: List[Dict[str, Any]], all_keywords: List[str], entity_kw: List[str], question: str) -> Dict[int, float]:
    if not LEX_FALLBACK_ENABLED:
        return {}

    qn = _normalize_text(question)

    want_comp = _COMPOSITION_PAT.search(qn) is not None
    want_criteria = _CRITERIA_PAT.search(qn) is not None
    want_complaint = _COMPLAINT_PAT.search(qn) is not None
    want_vision = _VISION_PAT.search(qn) is not None
    want_role = _ROLE_PAT.search(qn) is not None

    if not want_comp and not want_criteria and not want_complaint and not want_vision and not want_role:
        return {}

    if want_comp:
        phrases = _COMPOSITION_PHRASES
    elif want_criteria:
        phrases = _CRITERIA_PHRASES
    elif want_complaint:
        phrases = _COMPLAINT_PHRASES
    elif want_vision:
        phrases = _VISION_PHRASES
    else:
        phrases = _ROLE_PHRASES

    best: Dict[int, float] = {}
    per_doc_counts: Dict[str, int] = defaultdict(int)

    for r in rows:
        if not r.get("active", True):
            continue

        text = _row_text_for_matching(r)
        if not text:
            continue

        tn = _normalize_text(text)
        phrase_hit = any(p in tn for p in phrases)

        overlap_all = _keyword_overlap_count(all_keywords, text)
        overlap_ent = _keyword_overlap_count(entity_kw, text) if entity_kw else 0

        if want_complaint or want_vision or want_role:
            if not phrase_hit and overlap_all < 1 and overlap_ent < 1:
                continue
        else:
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

        pseudo = 0.40 + min(0.28, 0.05 * overlap_all)
        if phrase_hit:
            pseudo = max(pseudo, 0.62)

        prev = best.get(cid)
        if prev is None or pseudo > prev:
            best[cid] = pseudo
            per_doc_counts[doc_name] += 1

        if len(best) >= LEX_FALLBACK_MAX:
            break

    return best


# -----------------------------
# ✅ LLM rewrite (ALWAYS-first)
# -----------------------------
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is present and loaded.")
    return OpenAI(api_key=key)


def _llm_rewrite_queries(question: str) -> List[str]:
    if not (LLM_REWRITE_ENABLED and LLM_REWRITE_ALWAYS):
        return []

    q = (question or "").strip()
    if not q:
        return []

    system = (
        "You rewrite user questions into search-friendly queries for a PERA (Punjab Enforcement and Regulatory Authority) document chatbot.\n"
        "You must NOT answer.\n"
        "Return ONLY valid JSON with this schema:\n"
        "{\"queries\":[\"...\",\"...\"]}\n"
        "Rules:\n"
        "1) Fix spelling (e.g., pira/peera/perra -> PERA).\n"
        "2) Expand abbreviations: TOR->terms of reference, mgr->manager.\n"
        "3) Add 'PERA' context if missing.\n"
        "4) Produce 1-3 short variants maximum.\n"
        "5) No extra text."
    )

    user = f'User question: "{q}"'

    try:
        resp = _client().chat.completions.create(
            model=LLM_REWRITE_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        ).choices[0].message.content.strip()
    except Exception:
        return []

    try:
        data = json.loads(resp)
        queries = data.get("queries") or []
        if not isinstance(queries, list):
            return []
        cleaned: List[str] = []
        for x in queries:
            if isinstance(x, str) and x.strip():
                cleaned.append(x.strip())
        return _dedup_and_cap_queries(cleaned, cap=max(1, LLM_REWRITE_MAX))
    except Exception:
        return []


# -----------------------------
# Criteria prioritization (unchanged)
# -----------------------------
def _criteria_doc_signal_score(doc: Dict[str, Any]) -> float:
    hits = doc.get("hits", []) or []
    if not hits:
        return 0.0

    h0 = hits[0]
    txt = _normalize_text(h0.get("text") or "")

    phrase_hits = 0
    for p in _CRITERIA_PHRASES:
        if p in txt:
            phrase_hits += 1

    score = float(h0.get("score", 0.0) or 0.0)
    overlap = int(h0.get("overlap", 0) or 0)
    return (phrase_hits * 10.0) + (overlap * 1.5) + (score * 1.0)


def _apply_criteria_doc_prioritization(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    if not CRITERIA_DOC_PRIORITIZATION:
        return evidence_docs

    qn = _normalize_text(question)
    if _CRITERIA_PAT.search(qn) is None:
        return evidence_docs

    scored = [(ed, _criteria_doc_signal_score(ed)) for ed in evidence_docs]
    scored.sort(key=lambda x: x[1], reverse=True)

    kept = [x[0] for x in scored[:max(CRITERIA_MIN_DOCS, 1)]]
    kept_names = {d.get("doc_name") for d in kept}

    for ed in evidence_docs:
        if ed.get("doc_name") in kept_names:
            continue
        kept.append(ed)
        kept_names.add(ed.get("doc_name"))
        if len(kept) >= MAX_DOCS_RETURNED:
            break

    return kept[:MAX_DOCS_RETURNED]


# -----------------------------
# Main retrieval
# -----------------------------
def retrieve(question: str, index_dir: Optional[str] = None) -> Dict[str, Any]:
    index_dir = _resolve_index_dir(index_dir)

    empty = {
        "question": question,
        "has_evidence": False,
        "primary_doc": None,
        "primary_doc_rank": 0,
        "evidence": [],
    }

    question = (question or "").strip()
    if not question:
        return empty

    if index_dir.endswith("__INVALID_POINTER__"):
        return empty

    try:
        idx, rows = _load_index_rows(index_dir=index_dir)
        if idx is None or not rows:
            return empty

        # stale index protection: no docx
        for r in rows:
            dn = str(r.get("doc_name", "")).lower()
            p = str(r.get("path", "")).lower()
            st = str(r.get("source_type", "")).lower()
            if dn.endswith(".docx") or p.endswith(".docx") or st == "docx":
                return empty

        global _VOCAB_CACHE, _VOCAB_INDEX_DIR
        if SPELL_CORRECTION_ENABLED and (_VOCAB_CACHE is None or _VOCAB_INDEX_DIR != index_dir):
            _VOCAB_CACHE = _build_vocab_from_rows(rows)
            _VOCAB_INDEX_DIR = index_dir

        id_to_row = _rows_by_id(rows)

        # -----------------------------
        # ✅ ALWAYS: LLM rewrite FIRST
        # -----------------------------
        llm_queries = _llm_rewrite_queries(question) if (LLM_REWRITE_ENABLED and LLM_REWRITE_ALWAYS) else []

        # -----------------------------
        # Build query set (LLM + rules + spell)
        # -----------------------------
        queries: List[str] = []
        if llm_queries:
            queries.extend(llm_queries)

        # Always include original
        queries.append(question)

        # Rule-based variants
        if QUERY_VARIANTS_ENABLED:
            queries.extend(_build_query_variants(question))

        # normalized
        qn = _normalize_text(question)
        if qn:
            queries.append(qn)

        # spell corrected
        if SPELL_CORRECTION_ENABLED and _VOCAB_CACHE:
            corrected = _spell_correct_query_variant(question, _VOCAB_CACHE)
            if corrected:
                queries.append(corrected)

        # cap + dedup
        cap = max(1, MAX_QUERY_VARIANTS_WITH_SPELL)
        queries = _dedup_and_cap_queries(queries, cap=cap)

        entity_kw = _entity_keywords(question)
        base_kw = _extract_keywords(question)
        extras_kw = _intent_extra_keywords(question)
        all_kw = base_kw + [k for k in extras_kw if k not in base_kw]

        # -----------------------------
        # Semantic search
        # -----------------------------
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
                if score < SIM_THRESHOLD:
                    continue
                vid_i = int(vid)
                prev = best_by_id.get(vid_i)
                if prev is None or score > prev:
                    best_by_id[vid_i] = score

        # lexical fallback (includes role/complaint/vision/etc.)
        lex_best = _lexical_fallback_hits(rows, all_kw, entity_kw, question)
        for cid, pseudo in lex_best.items():
            prev = best_by_id.get(cid)
            if prev is None or pseudo > prev:
                best_by_id[cid] = pseudo

        if not best_by_id:
            if DEBUG:
                empty["debug"] = {"queries_used": queries, "llm_queries": llm_queries}
            return empty

        # -----------------------------
        # Build hits
        # -----------------------------
        hits: List[Dict[str, Any]] = []
        for vid_i, score in best_by_id.items():
            r = id_to_row.get(int(vid_i))
            if not r or not r.get("active", True):
                continue

            text_for_overlap = _row_text_for_matching(r)
            overlap_entity = _keyword_overlap_count(entity_kw, text_for_overlap) if entity_kw else 0
            overlap_all = _keyword_overlap_count(all_kw, text_for_overlap)
            overlap = overlap_entity if entity_kw else overlap_all

            hits.append({
                "id": int(vid_i),
                "score": float(score),
                "overlap": int(overlap),
                "doc_name": r.get("doc_name", "Unknown document"),
                "doc_rank": int(r.get("doc_rank", 0) or 0),
                "text": (r.get("text") or ""),
                "source_type": r.get("source_type", ""),
                "loc_kind": r.get("loc_kind", ""),
                "loc_start": r.get("loc_start"),
                "loc_end": r.get("loc_end"),
                "path": r.get("path"),
                "overlap_all": int(overlap_all),
                "overlap_entity": int(overlap_entity),
            })

        if not hits:
            if DEBUG:
                empty["debug"] = {"queries_used": queries, "llm_queries": llm_queries}
            return empty

        # keep strong semantic hits
        strict_req = _required_overlap(entity_kw if entity_kw else base_kw, strict=True)
        strict_hits = [
            h for h in hits
            if (int(h.get("overlap", 0)) >= strict_req) or (float(h.get("score", 0.0)) >= STRONG_SIM_THRESHOLD)
        ]

        if not strict_hits:
            relaxed_req = _required_overlap(entity_kw if entity_kw else base_kw, strict=False)
            strict_hits = [
                h for h in hits
                if (int(h.get("overlap", 0)) >= relaxed_req) or (float(h.get("score", 0.0)) >= STRONG_SIM_THRESHOLD)
            ]

        final_hits = strict_hits if strict_hits else hits

        # group by doc
        grouped: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        ranks: Dict[str, int] = {}
        for h in final_hits:
            dn = h["doc_name"]
            grouped[dn].append(h)
            ranks[dn] = max(ranks.get(dn, 0), int(h.get("doc_rank", 0) or 0))

        evidence_docs: List[Dict[str, Any]] = []
        for dn, doc_hits in grouped.items():
            doc_hits.sort(
                key=lambda x: (float(x.get("score", 0.0) or 0.0), int(x.get("overlap", 0) or 0)),
                reverse=True
            )
            doc_hits = doc_hits[:MAX_CHUNKS_PER_DOC]
            evidence_docs.append({"doc_name": dn, "doc_rank": ranks.get(dn, 0), "hits": doc_hits})

        def best_score(ed: Dict[str, Any]) -> float:
            hs = ed.get("hits", [])
            return float(hs[0]["score"]) if hs else 0.0

        def best_overlap(ed: Dict[str, Any]) -> int:
            hs = ed.get("hits", [])
            return int(hs[0].get("overlap", 0)) if hs else 0

        evidence_docs.sort(
            key=lambda ed: (ed.get("doc_rank", 0), best_score(ed), best_overlap(ed)),
            reverse=True
        )

        strong = [ed for ed in evidence_docs if ed.get("hits") and float(ed["hits"][0]["score"]) >= STRONG_SIM_THRESHOLD]
        primary = strong[0] if strong else evidence_docs[0]

        best = best_score(primary)
        kept_docs: List[Dict[str, Any]] = []
        for ed in evidence_docs:
            if best <= 0:
                continue
            if best_score(ed) >= best * RELATIVE_DOC_SCORE_KEEP:
                kept_docs.append(ed)

        kept_docs = kept_docs[:MAX_DOCS_RETURNED]
        kept_docs = _apply_criteria_doc_prioritization(question, kept_docs)

        out = {
            "question": question,
            "has_evidence": True,
            "primary_doc": primary.get("doc_name"),
            "primary_doc_rank": int(primary.get("doc_rank", 0) or 0),
            "evidence": kept_docs
        }

        if DEBUG:
            out["debug"] = {"queries_used": queries, "llm_queries": llm_queries}

        return out

    except Exception:
        return empty
