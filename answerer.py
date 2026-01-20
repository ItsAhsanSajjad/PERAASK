from __future__ import annotations

import os
import re
from typing import Dict, Any, List
from urllib.parse import quote

from dotenv import load_dotenv
from openai import OpenAI

from smalltalk_intent import decide_smalltalk

load_dotenv()

# Fixed refusal sentence (exact)
REFUSAL_TEXT = "There is no information available to this question."

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "24000"))

# Gates (typo-resilient while verifier remains strict)
ANSWER_MIN_TOP_SCORE = float(os.getenv("ANSWER_MIN_TOP_SCORE", "0.38"))
HIT_MIN_SCORE = float(os.getenv("HIT_MIN_SCORE", "0.35"))

# if semantic score is strong, allow evidence even with low lexical overlap
HIT_STRONG_SCORE_BYPASS = float(os.getenv("HIT_STRONG_SCORE_BYPASS", "0.55"))

# keep additional supporting chunks from the same doc if doc has strong hit
HIT_MEDIUM_SCORE_KEEP = float(os.getenv("HIT_MEDIUM_SCORE_KEEP", "0.48"))

MAX_HITS_PER_DOC_FOR_PROMPT = int(os.getenv("MAX_HITS_PER_DOC_FOR_PROMPT", "4"))
MAX_DOCS_FOR_PROMPT = int(os.getenv("MAX_DOCS_FOR_PROMPT", "4"))
MAX_REFS_RETURNED = int(os.getenv("MAX_REFS_RETURNED", "8"))

# Reference snippet size
REF_SNIPPET_CHARS = int(os.getenv("REF_SNIPPET_CHARS", "360"))

# Base URL for full links (used by API + mobile clients)
BASE_URL = os.getenv("Base_URL", "https://askpera.infinitysol.agency").rstrip("/")


# -----------------------------
# Client
# -----------------------------
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is present and loaded.")
    return OpenAI(api_key=key)


def _refuse() -> Dict[str, Any]:
    return {"answer": REFUSAL_TEXT, "references": []}


# -----------------------------
# Deterministic cleanup helpers
# -----------------------------
_BRACKET_CIT_RE = re.compile(r"\[[^\]]+\]")


def _strip_inline_citations(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = _BRACKET_CIT_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()


# -----------------------------
# Query normalization for lexical scoring (typo + intent aware)
# -----------------------------
_STOPWORDS = {
    "the", "a", "an", "is", "are", "was", "were", "to", "of", "and", "or",
    "in", "on", "at", "for", "from", "by", "with", "about", "tell", "me",
    "who", "what", "when", "where", "why", "how", "please",
}

_ABBREV_MAP = {
    "cto": "chief technology officer",
    "tor": "terms of reference",
    "tors": "terms of reference",
    "dg": "director general",
    "hr": "human resource",
    "it": "information technology",
    "ppra": "punjab procurement regulatory authority",
    "ipo": "initial public offering",
}

_KEEP_SHORT = {"ai", "ml", "it", "hr", "ppra", "ipo", "cto", "tor", "tors", "dg", "pera"}

# ✅ Domain aliases / common misspellings (mirror retriever.py)
_COMMON_MISSPELLINGS = {
    "pira": "pera",
    "perra": "pera",
    "peera": "pera",
    "peraa": "pera",
    "peraah": "pera",
    "complant": "complaint",
    "complaints": "complaint",
}

# ✅ Intent synonyms (question-side only; helps "vision of PERA" map to purpose/objectives/functions)
_INTENT_SYNONYMS = {
    "vision": ["purpose", "objectives", "functions", "mandate", "aim"],
    "mission": ["purpose", "objectives", "functions", "mandate", "aim"],
    "objective": ["objectives", "purpose", "aim"],
    "objectives": ["purpose", "functions", "mandate", "aim"],
}


def _expand_abbreviations(q: str) -> str:
    s = (q or "").lower()
    for k, v in _ABBREV_MAP.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def _normalize_question_for_scoring(q: str) -> str:
    """
    Normalizes question for overlap checks:
      - abbreviations expansion
      - lowercase + basic cleanup
      - common misspellings fix (pira->pera, complant->complaint)
      - intent synonym expansion (vision/mission -> purpose/objectives/functions/mandate)
    """
    s = _expand_abbreviations(q or "")
    s = s.lower()
    s = re.sub(r"[^a-z0-9\u0600-\u06FF\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()

    # misspellings
    for k, v in _COMMON_MISSPELLINGS.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)

    # intent expansion (append synonyms rather than replace, to keep original intent)
    toks = s.split()
    extras: List[str] = []
    for t in toks:
        if t in _INTENT_SYNONYMS:
            extras.extend(_INTENT_SYNONYMS[t])
    if extras:
        s = (s + " " + " ".join(extras)).strip()

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


def _tokenize(s: str) -> List[str]:
    s = _normalize_question_for_scoring(s or "")
    toks: List[str] = []
    for t in s.split():
        if not t:
            continue
        if t in _STOPWORDS:
            continue
        if len(t) < 3 and t not in _KEEP_SHORT:
            continue
        toks.append(_stem_token(t))
    return toks


def _keyword_overlap(question: str, text: str) -> int:
    q = set(_tokenize(question))
    if not q:
        return 0
    t = set(_tokenize(text))
    return len(q.intersection(t))


# -----------------------------
# Tiny fuzzy lexical matching (edit distance <= 1)
# Used ONLY to decide whether to keep evidence, not to generate content.
# -----------------------------
def _edit_distance_1(a: str, b: str) -> bool:
    if a == b:
        return True
    la, lb = len(a), len(b)
    if abs(la - lb) > 1:
        return False

    # substitution case
    if la == lb:
        diffs = 0
        for x, y in zip(a, b):
            if x != y:
                diffs += 1
                if diffs > 1:
                    return False
        return diffs == 1

    # insertion/deletion: ensure a is shorter
    if la > lb:
        a, b = b, a
        la, lb = lb, la

    i = j = 0
    mismatches = 0
    while i < la and j < lb:
        if a[i] == b[j]:
            i += 1
            j += 1
        else:
            mismatches += 1
            if mismatches > 1:
                return False
            j += 1
    return True


def _keyword_overlap_fuzzy(question: str, text: str) -> int:
    q_tokens = set(_tokenize(question))
    if not q_tokens:
        return 0
    t_tokens = set(_tokenize(text))
    if not t_tokens:
        return 0

    ov = 0
    for qt in q_tokens:
        if qt in t_tokens:
            ov += 1
            continue
        # fuzzy only for longer tokens to reduce false positives
        if len(qt) >= 6:
            for tt in t_tokens:
                if abs(len(qt) - len(tt)) <= 1 and _edit_distance_1(qt, tt):
                    ov += 1
                    break
    return ov


# -----------------------------
# Paths / URLs
# -----------------------------
def _safe_default_url_path(doc_name: str) -> str:
    dn = (doc_name or "").strip()
    if not dn:
        return "/assets/data"
    return f"/assets/data/{dn}".replace("\\", "/")


def _normalize_public_path(path_or_url: str, doc_name: str) -> str:
    p = (path_or_url or "").strip().replace("\\", "/")

    if p.startswith("/assets/data/"):
        return p
    if p.startswith("assets/data/"):
        return "/" + p
    if "/assets/data/" in p:
        tail = p.split("/assets/data/", 1)[1]
        return "/assets/data/" + tail
    if p.lower().endswith(".pdf"):
        filename = p.split("/")[-1]
        return f"/assets/data/{filename}"

    return _safe_default_url_path(doc_name)


def _file_type_from_doc(doc_name: str, public_path: str) -> str:
    dn = (doc_name or "").lower().strip()
    pp = (public_path or "").lower().strip()
    if dn.endswith(".pdf") or pp.endswith(".pdf"):
        return "pdf"
    if dn.endswith(".docx") or pp.endswith(".docx"):
        return "docx"
    return "file"


def _make_snippet(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    t = re.sub(r"\s+", " ", t).strip()
    if len(t) <= REF_SNIPPET_CHARS:
        return t
    return t[: REF_SNIPPET_CHARS].rstrip() + "…"


def _build_open_url(public_path: str, url_hint: str) -> str:
    return f"{BASE_URL}{public_path}{url_hint or ''}"


def _build_download_url(doc_name: str) -> str:
    filename = (doc_name or "").strip()
    return f"{BASE_URL}/download/{quote(filename)}"


# -----------------------------
# Evidence prompt building
# -----------------------------
def _format_loc_for_prompt(hit: Dict[str, Any]) -> str:
    doc = hit.get("doc_name", "Unknown document")
    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    if loc_kind == "page":
        return f"{doc} — p. {loc_start}"
    return f"{doc} — {loc_start}"


def _truncate_evidence_blocks(evidence_docs: List[Dict[str, Any]]) -> str:
    chunks: List[str] = []
    total = 0

    for d in evidence_docs[:MAX_DOCS_FOR_PROMPT]:
        doc_name = d.get("doc_name", "Unknown document")
        doc_rank = d.get("doc_rank", 0)
        hits = d.get("hits", []) or []

        header = f"\n\n=== DOCUMENT: {doc_name} (rank={doc_rank}) ===\n"
        if total + len(header) > MAX_EVIDENCE_CHARS:
            break
        chunks.append(header)
        total += len(header)

        for h in hits[:MAX_HITS_PER_DOC_FOR_PROMPT]:
            loc = _format_loc_for_prompt(h)
            text = (h.get("text") or "").strip()
            if not text:
                continue

            block = f"\n[{loc}]\n{text}\n"
            if total + len(block) > MAX_EVIDENCE_CHARS:
                break
            chunks.append(block)
            total += len(block)

        if total >= MAX_EVIDENCE_CHARS:
            break

    return "".join(chunks).strip()


# -----------------------------
# Evidence filtering (typo-resilient, semantic-first)
# -----------------------------
def _filter_evidence(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []

    # ✅ normalize question for scoring so "pira"/"complant"/"vision" behave correctly
    q_scoring = _normalize_question_for_scoring(question or "")

    for d in evidence_docs:
        hits = d.get("hits", []) or []
        if not hits:
            continue

        doc_has_strong = any(float(h.get("score", 0.0) or 0.0) >= HIT_STRONG_SCORE_BYPASS for h in hits)

        good_hits: List[Dict[str, Any]] = []
        for h in hits:
            score = float(h.get("score", 0.0) or 0.0)
            if score < HIT_MIN_SCORE:
                continue

            txt = (h.get("text") or "").strip()
            if not txt:
                continue

            # 1) strong semantic always wins
            if score >= HIT_STRONG_SCORE_BYPASS:
                good_hits.append(h)
                continue

            # 2) retriever overlap if present
            retr_overlap = h.get("overlap")
            if retr_overlap is not None:
                try:
                    retr_overlap = int(retr_overlap)
                except Exception:
                    retr_overlap = None

            if retr_overlap is not None and retr_overlap >= 1:
                good_hits.append(h)
                continue

            # 3) fuzzy lexical overlap using normalized question
            lex_ov = _keyword_overlap_fuzzy(q_scoring, txt)
            if lex_ov > 0:
                good_hits.append(h)
                continue

            # 4) keep medium semantic supporting hits if doc already has a strong hit
            if doc_has_strong and score >= HIT_MEDIUM_SCORE_KEEP:
                good_hits.append(h)
                continue

        if good_hits:
            good_hits.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            d2 = dict(d)
            d2["hits"] = good_hits[:MAX_HITS_PER_DOC_FOR_PROMPT]
            filtered.append(d2)

    return filtered[:MAX_DOCS_FOR_PROMPT]


# -----------------------------
# References (rich objects)
# -----------------------------
def _make_reference(hit: Dict[str, Any]) -> Dict[str, Any]:
    doc = hit.get("doc_name", "Unknown document")

    raw_path = (hit.get("path") or "").strip()
    public_path = _normalize_public_path(raw_path, doc)

    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    loc_end = hit.get("loc_end")

    snippet = _make_snippet(hit.get("text") or "")

    url_hint = ""
    if loc_kind == "page" and loc_start is not None:
        url_hint = f"#page={loc_start}"

    open_url = _build_open_url(public_path, url_hint)
    download_url = _build_download_url(doc)

    ref: Dict[str, Any] = {
        "document": doc,
        "path": public_path,
        "open_url": open_url,
        "download_url": download_url,
        "file_type": _file_type_from_doc(doc, public_path),
        "loc_kind": loc_kind,
        "loc_start": loc_start,
        "loc_end": loc_end,
        "snippet": snippet,
        "url_hint": url_hint,
    }

    if loc_kind == "page" and loc_start is not None:
        ref["page_start"] = loc_start
        ref["page_end"] = loc_start if loc_end is None else loc_end
    else:
        ref["loc"] = str(loc_start) if loc_start is not None else ""

    return ref


def _build_references_from_filtered(evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    seen = set()
    refs: List[Dict[str, Any]] = []

    for d in evidence_docs[:MAX_DOCS_FOR_PROMPT]:
        for h in (d.get("hits") or [])[:MAX_HITS_PER_DOC_FOR_PROMPT]:
            doc = h.get("doc_name", "Unknown document")
            loc_kind = h.get("loc_kind")
            loc_start = h.get("loc_start")

            key = (doc, loc_kind, str(loc_start))
            if key in seen:
                continue
            seen.add(key)

            refs.append(_make_reference(h))
            if len(refs) >= MAX_REFS_RETURNED:
                return refs

    return refs


# -----------------------------
# Main answering API
# -----------------------------
def answer_question(question: str, retrieval: Dict[str, Any]) -> Dict[str, Any]:
    decision = decide_smalltalk(question or "")
    if decision and decision.is_greeting_only:
        return {"answer": decision.response, "references": []}

    if not retrieval or not retrieval.get("has_evidence"):
        return _refuse()

    evidence_docs = retrieval.get("evidence", []) or []
    if not evidence_docs:
        return _refuse()

    evidence_docs = _filter_evidence(question, evidence_docs)
    if not evidence_docs:
        return _refuse()

    # Hard gate: top hit score AFTER filtering (typo-safe)
    try:
        top_hit = (evidence_docs[0].get("hits") or [{}])[0]
        top_score = float(top_hit.get("score", 0.0) or 0.0)
    except Exception:
        top_score = 0.0

    total_hits = sum(len(d.get("hits") or []) for d in evidence_docs)

    # ✅ refusal only if score low AND evidence is thin
    if top_score < ANSWER_MIN_TOP_SCORE and total_hits < 2:
        return _refuse()

    evidence_text = _truncate_evidence_blocks(evidence_docs)
    if not evidence_text:
        return _refuse()

    # ✅ Persona: Always act as PERA assistant (still evidence-only)
    system = (
        "You are the official AI assistant for PERA (Punjab Enforcement and Regulatory Authority).\n"
        "You must answer as a PERA-focused assistant while remaining strictly document-grounded.\n"
        "RULES:\n"
        "1) Use ONLY the evidence blocks.\n"
        "2) Do NOT guess or add outside knowledge.\n"
        "3) If not explicitly supported, output exactly:\n"
        f"{REFUSAL_TEXT}\n"
        "4) Prefer the latest/highest-ranked document.\n"
        "5) If documents conflict, do NOT merge; describe each version.\n"
        "6) Do NOT include citations, brackets, page numbers, or references in the answer text.\n"
        "7) Keep the answer concise and professional.\n"
        "8) If the user asks for 'vision' or 'mission' and the evidence describes purpose/objectives/functions/mandate,\n"
        "   answer using those statements (without inventing a separate vision statement).\n"
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "TASK:\n"
        "Write the best supported answer.\n"
        "If conflict exists, mention it clearly and concisely.\n"
        f"If unsupported, output exactly: {REFUSAL_TEXT}\n"
    )

    client = _client()
    draft = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    ).choices[0].message.content.strip()

    if not draft or draft.strip() == REFUSAL_TEXT:
        return _refuse()

    draft_clean = _strip_inline_citations(draft)
    if not draft_clean or draft_clean.strip() == REFUSAL_TEXT:
        return _refuse()

    verifier_system = (
        "You are a strict verifier.\n"
        "Check every sentence is directly supported by the evidence.\n"
        "If ANY sentence is not supported, respond with exactly:\n"
        f"{REFUSAL_TEXT}\n"
        "Otherwise return the draft unchanged.\n"
        "Also: the draft must NOT include any bracketed citations like [ ... ].\n"
    )

    verifier_user = (
        f"DRAFT ANSWER:\n{draft_clean}\n\n"
        f"EVIDENCE:\n{evidence_text}\n"
    )

    verified = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": verifier_system},
            {"role": "user", "content": verifier_user},
        ],
    ).choices[0].message.content.strip()

    if not verified or verified.strip() == REFUSAL_TEXT:
        return _refuse()

    verified_clean = _strip_inline_citations(verified)
    if not verified_clean or verified_clean.strip() == REFUSAL_TEXT:
        return _refuse()

    refs = _build_references_from_filtered(evidence_docs)
    return {"answer": verified_clean, "references": refs}
