from __future__ import annotations

import os
import re
from typing import Dict, Any, List

from dotenv import load_dotenv
from openai import OpenAI

from smalltalk_intent import decide_smalltalk

load_dotenv()

# Fixed refusal sentence (exact)
REFUSAL_TEXT = "There is no information available to this question."

ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini")
MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "24000"))

# Gates (keep, but relax + make non-fatal via fallback)
ANSWER_MIN_TOP_SCORE = float(os.getenv("ANSWER_MIN_TOP_SCORE", "0.25"))
HIT_MIN_SCORE = float(os.getenv("HIT_MIN_SCORE", "0.25"))

# If semantic score is strong, allow evidence even with low lexical overlap
HIT_STRONG_SCORE_BYPASS = float(os.getenv("HIT_STRONG_SCORE_BYPASS", "0.62"))

MAX_HITS_PER_DOC_FOR_PROMPT = int(os.getenv("MAX_HITS_PER_DOC_FOR_PROMPT", "4"))
MAX_DOCS_FOR_PROMPT = int(os.getenv("MAX_DOCS_FOR_PROMPT", "4"))
MAX_REFS_RETURNED = int(os.getenv("MAX_REFS_RETURNED", "8"))

REF_SNIPPET_CHARS = int(os.getenv("REF_SNIPPET_CHARS", "360"))

# Prefer-primary behavior
PREFER_PRIMARY_DOC_ONLY_FIRST = os.getenv("PREFER_PRIMARY_DOC_ONLY_FIRST", "1").strip() != "0"

# Fallback behavior (production-safe)
FALLBACK_MIN_SCORE = float(os.getenv("ANSWER_FALLBACK_MIN_SCORE", "0.20"))
FALLBACK_MAX_HITS_PER_DOC = int(os.getenv("ANSWER_FALLBACK_MAX_HITS_PER_DOC", "6"))

# ✅ NEW: Low-information / title-only suppression
MIN_HIT_TEXT_CHARS = int(os.getenv("ANSWER_MIN_HIT_TEXT_CHARS", "180"))
ALLOW_SHORT_IF_QA_PATTERN = os.getenv("ANSWER_ALLOW_SHORT_IF_QA_PATTERN", "1").strip() != "0"


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
# Query normalization for lexical scoring
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


def _expand_abbreviations(q: str) -> str:
    s = (q or "").lower()
    for k, v in _ABBREV_MAP.items():
        s = re.sub(rf"\b{re.escape(k)}\b", v, s)
    return s


def _tokenize(s: str) -> List[str]:
    s = _expand_abbreviations(s or "")
    s = re.sub(r"[^a-z0-9\u0600-\u06FF\s]", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    toks = []
    for t in s.split(" "):
        if len(t) < 3:
            continue
        if t in _STOPWORDS:
            continue
        toks.append(t)
    return toks


def _keyword_overlap(question: str, text: str) -> int:
    q = set(_tokenize(question))
    if not q:
        return 0
    t = set(_tokenize(text))
    return len(q.intersection(t))


def _safe_default_path(doc_name: str) -> str:
    doc_name = (doc_name or "").strip()
    if not doc_name:
        return ""
    return os.path.join("assets", "data", doc_name).replace("\\", "/")


def _file_type_from_path(path: str) -> str:
    p = (path or "").lower()
    if p.endswith(".pdf"):
        return "pdf"
    if p.endswith(".docx"):
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


# -----------------------------
# ✅ NEW: Low-information hit detection
# -----------------------------
_QA_PATTERN = re.compile(r"\b(q:|question:|a:|answer:)\b", re.I)


def _is_low_information_hit(text: str) -> bool:
    """
    Filters out title-only / heading-only chunks that cause refusals.
    Keeps short chunks ONLY if they contain Q/A markers (FAQ style).
    """
    t = (text or "").strip()
    if not t:
        return True

    # normalize for checks
    tn = re.sub(r"\s+", " ", t).strip()

    # allow short if it looks like Q/A content
    if ALLOW_SHORT_IF_QA_PATTERN and _QA_PATTERN.search(tn):
        return False

    # too short to be useful for answering
    if len(tn) < MIN_HIT_TEXT_CHARS:
        return True

    # looks like a title: single line, very few words, no punctuation that indicates content
    one_line = ("\n" not in t) and ("\r" not in t)
    if one_line:
        words = re.findall(r"[A-Za-z\u0600-\u06FF0-9]+", tn)
        if len(words) <= 10 and not re.search(r"[.:;?]", tn):
            return True

    return False


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
# Evidence filtering
# -----------------------------
def _filter_evidence(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Keep ONLY high-confidence hits + remove title-only chunks.
    If this becomes empty, safe fallback will run later.
    """
    filtered: List[Dict[str, Any]] = []

    for d in evidence_docs:
        hits = d.get("hits", []) or []
        good_hits: List[Dict[str, Any]] = []

        for h in hits:
            score = float(h.get("score", 0.0) or 0.0)
            if score < HIT_MIN_SCORE:
                continue

            txt = (h.get("text") or "").strip()
            if not txt:
                continue

            # ✅ critical fix: drop heading/title-only chunks
            if _is_low_information_hit(txt):
                continue

            retr_overlap = h.get("overlap")
            if retr_overlap is not None:
                try:
                    retr_overlap = int(retr_overlap)
                except Exception:
                    retr_overlap = None

            if retr_overlap is not None and retr_overlap >= 1:
                good_hits.append(h)
                continue

            lex_ov = _keyword_overlap(question, txt)
            if lex_ov > 0:
                good_hits.append(h)
                continue

            if score >= HIT_STRONG_SCORE_BYPASS:
                good_hits.append(h)
                continue

        if good_hits:
            good_hits.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            d2 = dict(d)
            d2["hits"] = good_hits[:MAX_HITS_PER_DOC_FOR_PROMPT]
            filtered.append(d2)

    return filtered[:MAX_DOCS_FOR_PROMPT]


def _primary_doc_name(retrieval: Dict[str, Any]) -> str:
    return (retrieval.get("primary_doc") or "").strip()


def _pick_primary_only(evidence_docs: List[Dict[str, Any]], primary_doc: str) -> List[Dict[str, Any]]:
    if not primary_doc:
        return evidence_docs
    only = [d for d in evidence_docs if (d.get("doc_name") or "").strip() == primary_doc]
    return only or evidence_docs


def _fallback_evidence(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Safe fallback when strict filtering returns empty.
    Still grounded: only uses retriever hits.
    Also applies low-information filtering.
    """
    out: List[Dict[str, Any]] = []
    for d in evidence_docs[:MAX_DOCS_FOR_PROMPT]:
        hits = d.get("hits", []) or []
        keep: List[Dict[str, Any]] = []
        for h in hits:
            txt = (h.get("text") or "").strip()
            if not txt:
                continue

            # ✅ still drop low-info in fallback
            if _is_low_information_hit(txt):
                continue

            score = float(h.get("score", 0.0) or 0.0)

            retr_overlap = h.get("overlap")
            try:
                retr_overlap_i = int(retr_overlap) if retr_overlap is not None else 0
            except Exception:
                retr_overlap_i = 0

            if score >= FALLBACK_MIN_SCORE or retr_overlap_i >= 1 or _keyword_overlap(question, txt) > 0:
                keep.append(h)

            if len(keep) >= FALLBACK_MAX_HITS_PER_DOC:
                break

        if keep:
            keep.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)
            d2 = dict(d)
            d2["hits"] = keep[:MAX_HITS_PER_DOC_FOR_PROMPT]
            out.append(d2)

    return out[:MAX_DOCS_FOR_PROMPT]


def _top_score(evidence_docs: List[Dict[str, Any]]) -> float:
    try:
        h0 = (evidence_docs[0].get("hits") or [{}])[0]
        return float(h0.get("score", 0.0) or 0.0)
    except Exception:
        return 0.0


# -----------------------------
# References
# -----------------------------
def _make_reference(hit: Dict[str, Any]) -> Dict[str, Any]:
    doc = hit.get("doc_name", "Unknown document")
    path = (hit.get("path") or "").strip() or _safe_default_path(doc)
    path = path.replace("\\", "/")

    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    loc_end = hit.get("loc_end")

    snippet = _make_snippet(hit.get("text") or "")

    ref: Dict[str, Any] = {
        "document": doc,
        "path": path,
        "file_type": _file_type_from_path(path),
        "loc_kind": loc_kind,
        "loc_start": loc_start,
        "loc_end": loc_end,
        "snippet": snippet,
    }

    if loc_kind == "page" and loc_start is not None:
        ref["page_start"] = loc_start
        ref["page_end"] = loc_start if loc_end is None else loc_end
        ref["url_hint"] = f"#page={loc_start}"
    else:
        ref["loc"] = str(loc_start) if loc_start is not None else ""
        ref["url_hint"] = ""

    return ref


def _build_references(evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
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
# LLM call + verify
# -----------------------------
def _generate_and_verify(client: OpenAI, question: str, evidence_text: str) -> str:
    system = (
        "You are a strict document-grounded assistant.\n"
        "RULES:\n"
        "1) Use ONLY the evidence blocks.\n"
        "2) Do NOT guess.\n"
        "3) If not explicitly supported, output exactly:\n"
        f"{REFUSAL_TEXT}\n"
        "4) Prefer the latest/highest-ranked document.\n"
        "5) If documents conflict, do NOT merge; describe each version.\n"
        "6) IMPORTANT: Do NOT include citations, brackets, page numbers, or references in the answer text.\n"
        "7) Keep the answer concise and professional.\n"
        "8) If the question is broad (e.g., 'What are the FAQs?'), list the FAQ topics/questions explicitly present in the evidence.\n"
    )

    user = (
        f"QUESTION:\n{question}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "TASK:\n"
        "Write the best supported answer.\n"
        "If conflict exists, mention it clearly and concisely.\n"
        f"If unsupported, output exactly: {REFUSAL_TEXT}\n"
    )

    draft = client.chat.completions.create(
        model=ANSWER_MODEL,
        temperature=0.0,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
    ).choices[0].message.content.strip()

    if not draft or draft.strip() == REFUSAL_TEXT:
        return REFUSAL_TEXT

    draft_clean = _strip_inline_citations(draft)
    if not draft_clean or draft_clean.strip() == REFUSAL_TEXT:
        return REFUSAL_TEXT

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
        return REFUSAL_TEXT

    verified_clean = _strip_inline_citations(verified)
    if not verified_clean or verified_clean.strip() == REFUSAL_TEXT:
        return REFUSAL_TEXT

    return verified_clean


# -----------------------------
# Main
# -----------------------------
def answer_question(question: str, retrieval: Dict[str, Any]) -> Dict[str, Any]:
    decision = decide_smalltalk(question or "")
    if decision and decision.is_greeting_only:
        return {"answer": decision.response, "references": []}

    if not retrieval or not retrieval.get("has_evidence"):
        return _refuse()

    evidence_docs_all = retrieval.get("evidence", []) or []
    if not evidence_docs_all:
        return _refuse()

    primary = _primary_doc_name(retrieval)

    # Step 1: strict filter (now removes title-only chunks)
    filtered_all = _filter_evidence(question, evidence_docs_all)

    # Step 2: if strict filter killed everything, safe fallback (still grounded)
    if not filtered_all:
        filtered_all = _fallback_evidence(question, evidence_docs_all)

    if not filtered_all:
        return _refuse()

    client = _client()

    # Prefer primary doc only first
    if PREFER_PRIMARY_DOC_ONLY_FIRST and primary:
        primary_docs = _pick_primary_only(filtered_all, primary)
        ev_text_primary = _truncate_evidence_blocks(primary_docs)

        if ev_text_primary:
            ans1 = _generate_and_verify(client, question, ev_text_primary)
            if ans1 != REFUSAL_TEXT:
                return {"answer": ans1, "references": _build_references(primary_docs)}

    # Full evidence fallback
    ev_text_all = _truncate_evidence_blocks(filtered_all)
    if not ev_text_all:
        return _refuse()

    qn = (question or "").lower()
    is_broad_faq = ("faq" in qn) or ("frequently asked" in qn)

    if (not is_broad_faq) and (_top_score(filtered_all) < ANSWER_MIN_TOP_SCORE):
        return _refuse()

    ans2 = _generate_and_verify(client, question, ev_text_all)
    if ans2 == REFUSAL_TEXT:
        return _refuse()

    return {"answer": ans2, "references": _build_references(filtered_all)}
