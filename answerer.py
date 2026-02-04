from __future__ import annotations

import os
import re
import json
import time
import random
from typing import Dict, Any, List, Optional, Tuple
from urllib.parse import quote

from dotenv import load_dotenv
from openai import OpenAI

# OpenAI error classes (new SDK) — keep flexible
try:
    from openai import (
        APITimeoutError,
        APIConnectionError,
        RateLimitError,
        InternalServerError,
        APIStatusError,
    )
except Exception:  # pragma: no cover
    APITimeoutError = Exception
    APIConnectionError = Exception
    RateLimitError = Exception
    InternalServerError = Exception
    APIStatusError = Exception

from smalltalk_intent import decide_smalltalk

load_dotenv()

# ============================================================
# Hard requirement: exact refusal sentence (ONLY for true empty/unrelated)
# ============================================================
REFUSAL_TEXT = "There is no information available to this question."

# ============================================================
# Supported language refusal message
# ============================================================
SUPPORTED_LANG_REFUSAL: Optional[str] = None
for _mod in ("query_rewrite", "query_preprocessor", "language", "lang", "utils"):
    try:
        _m = __import__(_mod, fromlist=["SUPPORTED_LANG_REFUSAL"])
        if hasattr(_m, "SUPPORTED_LANG_REFUSAL"):
            v = getattr(_m, "SUPPORTED_LANG_REFUSAL")
            if isinstance(v, str) and v.strip():
                SUPPORTED_LANG_REFUSAL = v.strip()
                break
    except Exception:
        continue

if not SUPPORTED_LANG_REFUSAL:
    SUPPORTED_LANG_REFUSAL = os.getenv(
        "SUPPORTED_LANG_REFUSAL",
        "This assistant supports only English, Urdu, and Roman Urdu."
    ).strip()

# ============================================================
# Models / limits
# ============================================================
ANSWER_MODEL = os.getenv("ANSWER_MODEL", "gpt-4.1-mini").strip()
VERIFIER_MODEL = os.getenv("VERIFIER_MODEL", ANSWER_MODEL).strip()

MAX_EVIDENCE_CHARS = int(os.getenv("MAX_EVIDENCE_CHARS", "24000"))
MIN_EVIDENCE_CHARS_TO_ANSWER = int(os.getenv("MIN_EVIDENCE_CHARS_TO_ANSWER", "120"))

HIT_MIN_SCORE = float(os.getenv("HIT_MIN_SCORE", "0.35"))
HIT_STRONG_SCORE_BYPASS = float(os.getenv("HIT_STRONG_SCORE_BYPASS", "0.55"))

MAX_HITS_PER_DOC_FOR_PROMPT = int(os.getenv("MAX_HITS_PER_DOC_FOR_PROMPT", "4"))
MAX_DOCS_FOR_PROMPT = int(os.getenv("MAX_DOCS_FOR_PROMPT", "4"))
MAX_REFS_RETURNED = int(os.getenv("MAX_REFS_RETURNED", "8"))

REF_SNIPPET_CHARS = int(os.getenv("REF_SNIPPET_CHARS", "360"))

BASE_URL = os.getenv("BASE_URL", os.getenv("Base_URL", "")).strip().rstrip("/")

MIN_EVIDENCE_CHARS = int(os.getenv("ANSWER_MIN_EVIDENCE_CHARS", "60"))
MIN_EVIDENCE_WORDS = int(os.getenv("ANSWER_MIN_EVIDENCE_WORDS", "8"))
MIN_EVIDENCE_LETTERS = int(os.getenv("ANSWER_MIN_EVIDENCE_LETTERS", "25"))

DEBUG_ANSWERER = os.getenv("DEBUG_ANSWERER", "0").strip() == "1"

# Prefer JSON mode if available (newer SDK supports response_format={"type":"json_object"})
USE_JSON_MODE = os.getenv("ANSWERER_USE_JSON_MODE", "1").strip() != "0"

# Stronger language enforcement: if model drifts, we re-ask once with stricter lock
MAX_LANGUAGE_REASK = int(os.getenv("ANSWERER_MAX_LANGUAGE_REASK", "1"))

# ============================================================
# Timeout + retry controls
# ============================================================
OPENAI_TIMEOUT_S = float(os.getenv("OPENAI_TIMEOUT_S", "35"))
OPENAI_MAX_RETRIES = int(os.getenv("OPENAI_MAX_RETRIES", "2"))
OPENAI_RETRY_BASE_S = float(os.getenv("OPENAI_RETRY_BASE_S", "0.7"))
OPENAI_RETRY_JITTER_S = float(os.getenv("OPENAI_RETRY_JITTER_S", "0.35"))
OPENAI_RETRY_MAX_SLEEP_S = float(os.getenv("OPENAI_RETRY_MAX_SLEEP_S", "4.0"))

# ============================================================
# Client
# ============================================================
def _client() -> OpenAI:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is present and loaded.")
    # New SDK accepts a float timeout; keep it simple
    return OpenAI(api_key=key, timeout=OPENAI_TIMEOUT_S)

# ============================================================
# Robust OpenAI call wrapper (retry + timeout)
# ============================================================
def _is_transient_status(status_code: Optional[int]) -> bool:
    try:
        if status_code is None:
            return False
        sc = int(status_code)
        return sc == 429 or sc >= 500
    except Exception:
        return False

def _chat_create_with_retry(
    client: OpenAI,
    *,
    model: str,
    messages: List[Dict[str, str]],
    temperature: float = 0.0,
    force_json: bool = False,
) -> str:
    last_err: Optional[Exception] = None

    for attempt in range(OPENAI_MAX_RETRIES + 1):
        try:
            kwargs: Dict[str, Any] = {
                "model": model,
                "temperature": temperature,
                "messages": messages,
                "timeout": OPENAI_TIMEOUT_S,
            }
            if force_json and USE_JSON_MODE:
                # Some SDKs/models support this; if not, we'll catch and retry without it.
                kwargs["response_format"] = {"type": "json_object"}

            resp = client.chat.completions.create(**kwargs)
            return (resp.choices[0].message.content or "").strip()

        except TypeError:
            # response_format not supported by this SDK/model; retry once without it
            if force_json:
                try:
                    resp = client.chat.completions.create(
                        model=model,
                        temperature=temperature,
                        messages=messages,
                        timeout=OPENAI_TIMEOUT_S,
                    )
                    return (resp.choices[0].message.content or "").strip()
                except Exception as e:
                    last_err = e
                    raise
            raise

        except (APITimeoutError, APIConnectionError, InternalServerError, RateLimitError) as e:
            last_err = e
            if attempt >= OPENAI_MAX_RETRIES:
                raise
            sleep_s = (OPENAI_RETRY_BASE_S * (2 ** attempt)) + random.uniform(0.0, OPENAI_RETRY_JITTER_S)
            time.sleep(min(OPENAI_RETRY_MAX_SLEEP_S, sleep_s))

        except APIStatusError as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if _is_transient_status(status) and attempt < OPENAI_MAX_RETRIES:
                sleep_s = (OPENAI_RETRY_BASE_S * (2 ** attempt)) + random.uniform(0.0, OPENAI_RETRY_JITTER_S)
                time.sleep(min(OPENAI_RETRY_MAX_SLEEP_S, sleep_s))
                continue
            raise

        except Exception as e:
            last_err = e
            raise

    raise last_err or RuntimeError("Unknown OpenAI error")

# ============================================================
# Response helpers
# ============================================================
def _refuse(debug: Optional[Dict[str, Any]] = None, *, message: Optional[str] = None) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "decision": "refuse",
        "answer": (message or REFUSAL_TEXT),
        "references": [],
        "used_chunk_ids": [],
    }
    if DEBUG_ANSWERER and debug:
        out["debug"] = debug
    return out

def _clarify(
    message: str,
    references: Optional[List[Dict[str, Any]]] = None,
    used_ids: Optional[List[int]] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "decision": "clarify",
        "answer": (message or "").strip() or "Please clarify your question.",
        "references": references or [],
        "used_chunk_ids": used_ids or [],
    }
    if DEBUG_ANSWERER and debug:
        out["debug"] = debug
    return out

def _answer(
    message: str,
    references: Optional[List[Dict[str, Any]]] = None,
    used_ids: Optional[List[int]] = None,
    debug: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "decision": "answer",
        "answer": (message or "").strip(),
        "references": references or [],
        "used_chunk_ids": used_ids or [],
    }
    if DEBUG_ANSWERER and debug:
        out["debug"] = debug
    return out

# ============================================================
# Deterministic cleanup helpers
# ============================================================
_BRACKET_CIT_RE = re.compile(r"\[[^\]]+\]")

def _strip_inline_citations(text: str) -> str:
    t = (text or "").strip()
    if not t:
        return t
    t = _BRACKET_CIT_RE.sub("", t)
    t = re.sub(r"\s{2,}", " ", t).strip()
    t = re.sub(r"\s+([,.;:!?])", r"\1", t)
    return t.strip()

def _normalize_ws(text: str) -> str:
    t = (text or "").replace("\u00ad", "")
    t = re.sub(r"\s+", " ", t).strip()
    return t

# ============================================================
# Strict language detection + validation
# Supported ONLY: en, ur, roman_ur, unsupported
# ============================================================
_ROMAN_URDU_HINTS = {
    # common particles/pronouns/verbs (roman urdu)
    "kya", "ky", "ka", "ki", "ke", "ko", "se", "par", "aur", "ya",
    "mein", "me", "mai", "hum", "ap", "aap", "tum", "aapka", "apki", "apke",
    "mujhe", "mujhy", "mjy", "mera", "meri", "mere",
    "kaise", "kesy", "kesay", "kis", "kisay", "kisey",
    "batao", "batain", "btao", "btado", "samjhao", "samjhaao",
    "kr", "kro", "kren", "karen", "krna", "krne", "krdo", "karna", "karne",
    "hona", "hai", "hain", "tha", "thi", "thay", "hogaa", "hoga", "hogi",
    "nahi", "nahin", "han", "haan", "jee", "ji",
    "assalam", "salam", "aoa",
    "plz", "please",
    # pakistan-specific roman urdu words you see in queries
    "notification", "qanoon", "qawanin", "qanun", "dafa", "maddah", "shart", "sharaait",
}

_EN_STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "is", "are", "was", "were", "be", "been",
    "for", "with", "on", "as", "by", "at", "from", "that", "this", "it", "you", "your",
    "can", "will", "should", "what", "how", "when", "where", "why", "a", "an", "yes", "no",
    "please", "tell", "explain", "define", "policy", "rule", "procedure",
}

_URDU_LETTERS = set("پچڈڑژگںھٰٖؐؑؒؓؔؕٔۍۓۓی")

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")
_URDU_SPECIFIC_RE = re.compile("[" + re.escape("".join(_URDU_LETTERS)) + "]")

_OTHER_SCRIPT_RE = re.compile(
    r"[\u0900-\u097F"
    r"\u0400-\u04FF"
    r"\u4E00-\u9FFF"
    r"\u3040-\u30FF"
    r"\u0E00-\u0E7F"
    r"\u1100-\u11FF"
    r"\uAC00-\uD7AF"
    r"]"
)

def _tokenize_latin_words(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (s or "").lower())

def _detect_lang_strict(q: str) -> str:
    s = (q or "").strip()
    if not s:
        return "en"

    # Any non-supported scripts => unsupported
    if _OTHER_SCRIPT_RE.search(s):
        return "unsupported"

    # Arabic script => Urdu if strong, else unsupported (Arabic, Persian, etc.)
    if _ARABIC_SCRIPT_RE.search(s):
        if _URDU_SPECIFIC_RE.search(s):
            return "ur"
        return "unsupported"

    # Latin script: choose roman_ur vs en
    tokens = _tokenize_latin_words(s)
    if not tokens:
        return "en"

    roman_hits = sum(1 for t in tokens if t in _ROMAN_URDU_HINTS)
    en_hits = sum(1 for t in tokens if t in _EN_STOPWORDS)

    # Heuristic tuned to avoid “roman urdu -> english” drift:
    # - if roman markers exist and english stopwords are not dominant => roman_ur
    # - short queries with roman marker => roman_ur
    if roman_hits >= 1:
        if len(tokens) <= 6 and roman_hits >= 1:
            return "roman_ur"
        if roman_hits >= en_hits:
            return "roman_ur"
        # if english dominates strongly, keep en
        if en_hits >= roman_hits + 2:
            return "en"
        return "roman_ur"

    return "en"

def _is_urdu_text(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    letters = re.findall(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]", t)
    latin = re.findall(r"[A-Za-z]", t)
    if _URDU_SPECIFIC_RE.search(t) and len(letters) >= 3 and len(letters) >= (len(latin) + 1):
        return True
    return len(letters) >= 8 and len(letters) >= (len(latin) * 2)

def _is_roman_urdu_text(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    if _ARABIC_SCRIPT_RE.search(t):
        return False
    tokens = _tokenize_latin_words(t)
    if not tokens:
        return False

    roman_hits = sum(1 for w in tokens if w in _ROMAN_URDU_HINTS)
    en_hits = sum(1 for w in tokens if w in _EN_STOPWORDS)

    # roman urdu usually has particles/verbs; allow mixed, but avoid pure English
    if roman_hits >= 1 and roman_hits >= en_hits:
        return True
    if roman_hits >= 2:
        return True
    if en_hits / max(1, len(tokens)) > 0.55 and roman_hits == 0:
        return False
    return roman_hits >= 1

def _is_english_text(s: str) -> bool:
    t = (s or "").strip()
    if not t:
        return False
    if _ARABIC_SCRIPT_RE.search(t):
        return False
    tokens = _tokenize_latin_words(t)
    if not tokens:
        return False
    en_hits = sum(1 for w in tokens if w in _EN_STOPWORDS)
    roman_hits = sum(1 for w in tokens if w in _ROMAN_URDU_HINTS)
    # English if English stopwords present and roman markers not stronger
    if en_hits >= 1 and en_hits >= roman_hits:
        return True
    return len(tokens) >= 7 and en_hits >= 1

def _validate_output_language(text: str, target_lang: str) -> bool:
    if target_lang == "ur":
        return _is_urdu_text(text)
    if target_lang == "roman_ur":
        return _is_roman_urdu_text(text)
    return _is_english_text(text)

def _language_lock_for_answer_only(lang: str) -> str:
    if lang == "ur":
        return (
            "ANSWER LANGUAGE LOCK:\n"
            "- The value of 'answer' MUST be Urdu ONLY (Urdu script).\n"
            "- Do NOT include English or Roman Urdu.\n"
        )
    if lang == "roman_ur":
        return (
            "ANSWER LANGUAGE LOCK:\n"
            "- The value of 'answer' MUST be Roman Urdu ONLY.\n"
            "- Use Latin script ONLY (A-Z). Do NOT use Urdu/Arabic script.\n"
            "- Do NOT include English.\n"
        )
    return (
        "ANSWER LANGUAGE LOCK:\n"
        "- The value of 'answer' MUST be English ONLY.\n"
        "- Do NOT include Urdu script or Roman Urdu.\n"
    )

def _rewrite_once_in_language(client: OpenAI, target_lang: str, text: str) -> Optional[str]:
    t = (text or "").strip()
    if not t:
        return None

    system = (
        "Rewrite text with strict language rules.\n"
        + _language_lock_for_answer_only(target_lang) +
        "Rules:\n"
        "- Keep meaning the same.\n"
        "- Do NOT add any new facts.\n"
        "- Do NOT add citations or brackets.\n"
        "- Return plain text only.\n"
    )

    user = f"Rewrite this text:\n\n{t}"

    try:
        out = _chat_create_with_retry(
            client,
            model=ANSWER_MODEL,
            temperature=0.0,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        out = _strip_inline_citations(out).strip()
        return out or None
    except Exception:
        return None

# ============================================================
# Clarify text per language + network hint
# ============================================================
def _clarify_text(lang: str, hint: str = "") -> str:
    hint = (hint or "").strip()
    if lang == "ur":
        base = "براہِ کرم اپنا سوال تھوڑا واضح کریں تاکہ میں دستاویزات کی بنیاد پر درست جواب دے سکوں۔"
        return f"{base}\n\nوضاحت: {hint}" if hint else base
    if lang == "roman_ur":
        base = "Meharbani kar ke apna sawal thora wazeh kar dein taa ke main documents ki bunyaad par sahi jawab de sakun."
        return f"{base}\n\nWazahat chahiye: {hint}" if hint else base
    base = "Please clarify your question so I can answer accurately from the documents."
    return f"{base}\n\nClarification needed: {hint}" if hint else base

def _network_hint(lang: str) -> str:
    if lang == "ur":
        return "انٹرنیٹ کنکشن سست یا غیر مستحکم لگ رہا ہے۔ براہِ کرم دوبارہ کوشش کریں (ممکن ہو تو Wi-Fi استعمال کریں)۔"
    if lang == "roman_ur":
        return "Internet connection slow ya unstable lag raha hai. Meharbani karke dobara try karein (mumkin ho to Wi-Fi use karein)."
    return "Your internet connection seems slow or unstable. Please try again (prefer Wi-Fi)."

# ============================================================
# Evidence quality filters (conservative)
# ============================================================
_PAGE_GARBAGE_RE = re.compile(r"^\s*page\s*\d+\s*(of\s*\d+)?\s*$", re.I)
_ONLY_NUM_PUNCT_RE = re.compile(r"^[\s0-9\-–—_.,:;|/\\()]+$")

def _count_letters(s: str) -> int:
    return len(re.findall(r"[A-Za-z\u0600-\u06FF]", s or ""))

def _count_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z\u0600-\u06FF]{2,}", s or ""))

def _is_low_signal_chunk(txt: str) -> bool:
    t = (txt or "").strip()
    if not t:
        return True
    if _PAGE_GARBAGE_RE.match(t):
        return True
    if len(t) <= 12 and _ONLY_NUM_PUNCT_RE.match(t):
        return True
    if len(t) < MIN_EVIDENCE_CHARS and _count_words(t) < 3:
        return True
    letters = _count_letters(t)
    words = _count_words(t)
    if letters < MIN_EVIDENCE_LETTERS and words < MIN_EVIDENCE_WORDS:
        tl = t.lower()
        # keep identity signals (avoid dropping key short hints)
        if "pera" in tl or "authority" in tl or "پيرا" in tl:
            return False
        return True
    return False

# ============================================================
# Paths / URLs (deterministic + safe encoding)
# ============================================================
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

    if p.startswith("http://") or p.startswith("https://"):
        if "/assets/data/" in p:
            tail = p.split("/assets/data/", 1)[1]
            return "/assets/data/" + tail
        return _safe_default_url_path(doc_name)

    if p.lower().endswith(".pdf") or p.lower().endswith(".docx"):
        filename = p.split("/")[-1]
        return f"/assets/data/{filename}"

    return _safe_default_url_path(doc_name)

def _normalize_public_url_encoded(hit: Dict[str, Any], doc_name: str) -> str:
    pu = (hit.get("public_url") or "").strip().replace("\\", "/")
    if pu.startswith("/assets/data/"):
        return pu

    raw_path = (hit.get("public_path") or hit.get("path") or "").strip()
    p = _normalize_public_path(raw_path, doc_name)

    if p.startswith("/assets/data/"):
        fn = p.split("/assets/data/", 1)[1]
        return "/assets/data/" + quote(fn)
    return "/assets/data/" + quote((doc_name or "").strip()) if (doc_name or "").strip() else "/assets/data"

def _file_type_from_doc(doc_name: str, public_path: str) -> str:
    dn = (doc_name or "").lower().strip()
    pp = (public_path or "").lower().strip()
    if dn.endswith(".pdf") or pp.endswith(".pdf"):
        return "pdf"
    if dn.endswith(".docx") or pp.endswith(".docx"):
        return "docx"
    return "file"

def _make_snippet(text: str) -> str:
    t = _normalize_ws(text or "")
    if len(t) <= REF_SNIPPET_CHARS:
        return t
    return t[:REF_SNIPPET_CHARS].rstrip() + "…"

def _build_open_url(public_url_encoded: str, url_hint: str) -> str:
    if not BASE_URL:
        return f"{public_url_encoded}{url_hint or ''}"
    return f"{BASE_URL}{public_url_encoded}{url_hint or ''}"

def _filename_from_doc_or_path(doc_name: str, public_path: str) -> str:
    dn = (doc_name or "").strip()
    if dn.lower().endswith(".pdf") or dn.lower().endswith(".docx"):
        return dn
    pp = (public_path or "").strip().replace("\\", "/")
    if "/assets/data/" in pp:
        return pp.split("/assets/data/", 1)[1].split("#", 1)[0]
    return dn

def _build_download_url(doc_name: str, public_path: str) -> str:
    filename = _filename_from_doc_or_path(doc_name, public_path)
    filename = (filename or "").strip()
    if not filename:
        return ""
    if not BASE_URL:
        return f"/download/{quote(filename)}"
    return f"{BASE_URL}/download/{quote(filename)}"

def _safe_int(x: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if x is None:
            return default
        return int(str(x).strip())
    except Exception:
        return default

# ============================================================
# Evidence filtering
# ============================================================
def _filter_evidence(question: str, evidence_docs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    filtered: List[Dict[str, Any]] = []
    for d in evidence_docs:
        hits = d.get("hits", []) or []
        if not hits:
            continue

        hits2: List[Dict[str, Any]] = []
        for h in hits:
            txt = (h.get("text") or "").strip()
            stxt = (h.get("search_text") or "").strip()
            combined = (txt + "\n" + stxt).strip() if stxt else txt
            if not _is_low_signal_chunk(combined):
                hits2.append(h)

        if not hits2:
            continue

        hits2.sort(key=lambda x: float(x.get("score", 0.0) or 0.0), reverse=True)

        strong = [h for h in hits2 if float(h.get("score", 0.0) or 0.0) >= HIT_STRONG_SCORE_BYPASS]
        kept = (strong[:MAX_HITS_PER_DOC_FOR_PROMPT] if strong else hits2[:MAX_HITS_PER_DOC_FOR_PROMPT])

        d2 = dict(d)
        d2["hits"] = kept
        filtered.append(d2)

    return filtered[:MAX_DOCS_FOR_PROMPT]

# ============================================================
# Build evidence prompt with deterministic IDs
# ============================================================
def _format_loc_for_prompt(hit: Dict[str, Any]) -> str:
    doc = hit.get("doc_name", "Unknown document")
    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    if loc_kind == "page":
        return f"{doc} — p. {loc_start}" if loc_start is not None else f"{doc}"
    return f"{doc} — {loc_start}" if loc_start is not None else f"{doc}"

def _hit_chunk_id(hit: Dict[str, Any]) -> Optional[int]:
    for k in ("id", "chunk_id"):
        v = hit.get(k)
        try:
            if v is None:
                continue
            return int(v)
        except Exception:
            continue
    return None

def _truncate_evidence_blocks_with_used_hits(
    evidence_docs: List[Dict[str, Any]]
) -> Tuple[str, List[Dict[str, Any]], Dict[int, int]]:
    chunks: List[str] = []
    included_hits: List[Dict[str, Any]] = []
    evidence_id_to_chunk: Dict[int, int] = {}
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

        kept = 0
        for h in hits:
            if kept >= MAX_HITS_PER_DOC_FOR_PROMPT:
                break

            text = (h.get("text") or "").strip()
            if not text:
                continue

            if _is_low_signal_chunk(text):
                continue

            cid = _hit_chunk_id(h)
            loc = _format_loc_for_prompt(h)

            evidence_id = len(included_hits) + 1
            block = (
                f"\n[EVIDENCE_ID: {evidence_id} | CHUNK_ID: {cid if cid is not None else 'NA'} | {loc}]\n"
                f"{text}\n"
            )

            if total + len(block) > MAX_EVIDENCE_CHARS:
                break

            chunks.append(block)
            total += len(block)
            included_hits.append(h)

            if cid is not None:
                evidence_id_to_chunk[evidence_id] = int(cid)

            kept += 1

        if total >= MAX_EVIDENCE_CHARS:
            break

    evidence_text = "".join(chunks).strip()
    return evidence_text, included_hits, evidence_id_to_chunk

# ============================================================
# References from USED hits only
# ============================================================
def _make_reference(hit: Dict[str, Any]) -> Dict[str, Any]:
    doc = hit.get("doc_name", "Unknown document")

    raw_path = (hit.get("public_path") or hit.get("path") or "").strip()
    public_path = _normalize_public_path(raw_path, doc)

    public_url_encoded = _normalize_public_url_encoded(hit, doc)

    loc_kind = hit.get("loc_kind")
    loc_start = hit.get("loc_start")
    loc_end = hit.get("loc_end")

    snippet = _make_snippet(hit.get("text") or "")

    url_hint = ""
    if loc_kind == "page":
        page_i = _safe_int(loc_start, None)
        if page_i is not None:
            url_hint = f"#page={page_i}"

    open_url = _build_open_url(public_url_encoded, url_hint)
    download_url = _build_download_url(doc, public_path)

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

    if loc_kind == "page":
        page_start = _safe_int(loc_start, None)
        page_end = _safe_int(loc_end, page_start)
        ref["page_start"] = page_start if page_start is not None else ""
        ref["page_end"] = page_end if page_end is not None else (page_start if page_start is not None else "")
    else:
        ref["loc"] = str(loc_start) if loc_start is not None else ""

    return ref

def _build_references_from_used_hits(used_hits: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    refs: List[Dict[str, Any]] = []
    seen = set()
    for h in used_hits:
        doc = h.get("doc_name", "Unknown document")
        loc_kind = h.get("loc_kind")
        loc_start = h.get("loc_start")
        key = (doc, loc_kind, str(loc_start))
        if key in seen:
            continue
        seen.add(key)

        refs.append(_make_reference(h))
        if len(refs) >= MAX_REFS_RETURNED:
            break
    return refs

# ============================================================
# Robust JSON extraction
# ============================================================
def _extract_first_valid_json_object(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None
    s = text.strip()

    try:
        obj = json.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass

    start_positions = [m.start() for m in re.finditer(r"\{", s)]
    for start in start_positions:
        depth = 0
        for i in range(start, len(s)):
            ch = s[i]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
                if depth == 0:
                    candidate = s[start:i + 1]
                    try:
                        obj = json.loads(candidate)
                        if isinstance(obj, dict):
                            return obj
                    except Exception:
                        break
    return None

# ============================================================
# Verifier: clause-level verification
# ============================================================
def _verify_supported_clauses(
    client: OpenAI,
    question: str,
    draft: str,
    evidence_text: str,
    lang: str
) -> Tuple[str, bool]:
    d = (draft or "").strip()
    if not d:
        return "", False

    system = (
        "You are a STRICT verifier for a government document-grounded system.\n"
        "Given QUESTION, EVIDENCE, and DRAFT_ANSWER:\n"
        "- Rewrite the answer into short factual clauses.\n"
        "- Keep ONLY clauses explicitly supported by the evidence.\n"
        "- Remove anything not supported.\n"
        "- Do NOT use outside knowledge.\n"
        "Return ONLY valid JSON with ENGLISH keys exactly:\n"
        "  {\"supported_clauses\":[\"...\",\"...\"]}\n\n"
        + _language_lock_for_answer_only(lang) +
        "\nIMPORTANT:\n"
        "- JSON keys MUST remain English.\n"
        "- Only the clause strings must follow the answer-language rule.\n"
    )

    payload = {"question": question, "evidence": evidence_text, "draft_answer": d}

    try:
        raw = _chat_create_with_retry(
            client,
            model=VERIFIER_MODEL,
            temperature=0.0,
            force_json=True,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
            ],
        )
    except Exception:
        return "", False

    data = _extract_first_valid_json_object(raw)
    if not data:
        return "", False

    clauses = data.get("supported_clauses", []) or []
    if not isinstance(clauses, list):
        return "", False

    cleaned: List[str] = []
    for c in clauses:
        if isinstance(c, str):
            cc = _strip_inline_citations(c).strip()
            if cc:
                cleaned.append(cc)

    if not cleaned:
        return "", False

    if lang == "ur":
        out = "۔ ".join(cleaned).strip()
        if not out.endswith("۔"):
            out += "۔"
        return out, True

    return "; ".join(cleaned).strip(), True

# ============================================================
# used ids parsing helpers
# ============================================================
def _parse_int_list(x: Any) -> List[int]:
    out: List[int] = []
    if x is None:
        return out
    if isinstance(x, list):
        items = x
    elif isinstance(x, str):
        items = re.split(r"[,\s]+", x.strip())
    else:
        items = [x]

    for it in items:
        try:
            if it is None:
                continue
            out.append(int(str(it).strip()))
        except Exception:
            continue

    out2: List[int] = []
    seen = set()
    for n in out:
        if n in seen:
            continue
        seen.add(n)
        out2.append(n)
    return out2

# ============================================================
# Smalltalk handling
# ============================================================
def _handle_smalltalk(question: str, lang: str) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    try:
        dec = decide_smalltalk(question or "")
    except Exception:
        return None, None

    if not dec:
        return None, None

    def _ensure_lang(text: str) -> str:
        t = (text or "").strip()
        if not t:
            return t
        if _validate_output_language(t, lang):
            return t
        try:
            client = _client()
            rr = _rewrite_once_in_language(client, lang, t)
            return rr if rr else t
        except Exception:
            return t

    # Dataclass path
    if hasattr(dec, "is_greeting_only"):
        is_only = bool(getattr(dec, "is_greeting_only", False))
        resp = (getattr(dec, "response", "") or "").strip()
        ack = (getattr(dec, "ack", "") or "").strip()
        rem = (getattr(dec, "remaining_question", "") or "").strip()

        if is_only and resp:
            resp = _ensure_lang(resp)
            return None, {"decision": "answer", "answer": resp.strip(), "references": [], "used_chunk_ids": []}

        if (not is_only) and rem:
            return (ack + " ").strip() if ack else None, {"__remaining_question__": rem}

        return None, None

    # Back-compat: dict
    if isinstance(dec, dict):
        if dec.get("is_greeting_only") and dec.get("response"):
            resp = _ensure_lang(str(dec.get("response") or "").strip())
            return None, {"decision": "answer", "answer": resp, "references": [], "used_chunk_ids": []}
        if dec.get("remaining_question"):
            return (str(dec.get("ack") or "").strip() or None), {"__remaining_question__": str(dec.get("remaining_question") or "").strip()}

    # Back-compat: tuple
    if isinstance(dec, tuple) and len(dec) >= 2 and bool(dec[0]):
        resp = _ensure_lang(str(dec[1] or "").strip())
        return None, {"decision": "answer", "answer": resp, "references": [], "used_chunk_ids": []}

    return None, None

# ============================================================
# Model answer JSON prompt builder
# ============================================================
def _answer_system_prompt(lang: str) -> str:
    return (
        "You are the official AI assistant for PERA (Punjab Enforcement and Regulatory Authority).\n\n"
        "OUTPUT FORMAT (STRICT):\n"
        "- Return ONLY valid JSON (no markdown, no extra text).\n"
        "- JSON keys MUST be in ENGLISH and exactly these keys:\n"
        "  decision, answer, used_chunk_ids, used_evidence_ids\n"
        "- decision MUST be one of: answer | clarify | refuse\n\n"
        + _language_lock_for_answer_only(lang) +
        "\nRULES:\n"
        "1) Use ONLY the evidence blocks.\n"
        "2) Do NOT guess or use outside knowledge.\n"
        "3) If evidence does NOT explicitly answer the question, set decision='clarify' and ask ONE precise clarification question in 'answer'.\n"
        "4) Use decision='refuse' ONLY if the evidence is empty or completely unrelated.\n"
        "5) If decision='refuse', 'answer' MUST be exactly:\n"
        f"{REFUSAL_TEXT}\n"
        "6) NEVER include citations, brackets, or page numbers inside 'answer'.\n"
        "7) NEVER output a person's name unless it appears exactly in the evidence.\n"
        "8) NEVER output a number unless it appears exactly in the evidence.\n"
        "9) used_chunk_ids should contain CHUNK_ID integers you relied on.\n"
        "10) used_evidence_ids can contain EVIDENCE_ID integers you relied on.\n"
    )

def _answer_user_prompt(q: str, evidence_text: str) -> str:
    return (
        f"QUESTION:\n{q}\n\n"
        f"EVIDENCE:\n{evidence_text}\n\n"
        "Return JSON now."
    )

# ============================================================
# Main API
# ============================================================
def answer_question(question: str, retrieval: Dict[str, Any]) -> Dict[str, Any]:
    """
    IMPORTANT: This function must NEVER raise in production.
    It should always return a dict with decision/answer/references/used_chunk_ids.
    """
    try:
        q0 = (question or "").strip()
        if not q0:
            return _refuse({"reason": "empty_question"} if DEBUG_ANSWERER else None)

        lang = _detect_lang_strict(q0)

        if lang == "unsupported":
            return _refuse(
                {"reason": "unsupported_language"} if DEBUG_ANSWERER else None,
                message=SUPPORTED_LANG_REFUSAL,
            )

        preface, small_resp = _handle_smalltalk(q0, lang)
        if small_resp:
            if small_resp.get("decision") == "answer":
                return small_resp
            if "__remaining_question__" in small_resp:
                q = str(small_resp["__remaining_question__"] or "").strip()
            else:
                q = q0
        else:
            q = q0

        # Retrieval missing evidence => refuse (ONLY true empty case)
        if not retrieval or not retrieval.get("has_evidence"):
            # If pipeline provided "evidence"/debug but flag false, we should CLARIFY not refuse
            if retrieval and (retrieval.get("evidence") or retrieval.get("debug")):
                msg = _clarify_text(lang, "Mujood material related lagta hai lekin jawab ke liye specific hissa/role batayein." if lang == "roman_ur" else "I found some related material but not enough. Please specify the exact section/role.")
                if preface:
                    msg = f"{preface} {msg}".strip()
                return _clarify(
                    msg,
                    references=[],
                    used_ids=[],
                    debug={"reason": "has_evidence_false_but_payload_present"} if DEBUG_ANSWERER else None,
                )
            return _refuse({"reason": "no_retrieval_or_has_evidence_false"} if DEBUG_ANSWERER else None)

        evidence_docs = retrieval.get("evidence", []) or []
        if not evidence_docs:
            return _refuse({"reason": "no_evidence_docs"} if DEBUG_ANSWERER else None)

        evidence_docs = _filter_evidence(q, evidence_docs)
        if not evidence_docs:
            msg = _clarify_text(lang, "Retrieved snippets were too low-signal. Please specify the exact role/section.")
            if preface:
                msg = f"{preface} {msg}".strip()
            return _clarify(
                msg,
                references=[],
                used_ids=[],
                debug={"reason": "filtered_evidence_empty"} if DEBUG_ANSWERER else None,
            )

        evidence_text, included_hits, evidence_id_to_chunk = _truncate_evidence_blocks_with_used_hits(evidence_docs)

        # Lightweight refs early (so timeouts can still show docs)
        fallback_refs = _build_references_from_used_hits(included_hits[: max(1, min(3, len(included_hits)))]) if included_hits else []
        fallback_used_ids = [cid for cid in (_hit_chunk_id(h) for h in included_hits[:10]) if cid is not None]

        # Allow strong-hit bypass even if evidence is short
        max_score = 0.0
        for h in included_hits:
            try:
                max_score = max(max_score, float(h.get("score", 0.0) or 0.0))
            except Exception:
                pass

        if (not evidence_text) or (len(evidence_text) < MIN_EVIDENCE_CHARS_TO_ANSWER and max_score < HIT_STRONG_SCORE_BYPASS):
            msg = _clarify_text(lang, "The available excerpts are too short/partial. Tell me the exact role/section or page range.")
            if preface:
                msg = f"{preface} {msg}".strip()
            return _clarify(
                msg,
                references=fallback_refs,
                used_ids=fallback_used_ids[:6],
                debug={"reason": "evidence_too_small", "evidence_chars": len(evidence_text), "max_score": max_score} if DEBUG_ANSWERER else None,
            )

        client = _client()

        # ------------------------------------------------------------
        # Generate answer JSON (with strong language lock + JSON mode)
        # Fix: Roman Urdu drift to English => we re-ask once if output language invalid
        # ------------------------------------------------------------
        system = _answer_system_prompt(lang)
        user = _answer_user_prompt(q, evidence_text)

        raw = ""
        last_gen_err: Optional[str] = None
        for attempt in range(MAX_LANGUAGE_REASK + 1):
            try:
                raw = _chat_create_with_retry(
                    client,
                    model=ANSWER_MODEL,
                    temperature=0.0,
                    force_json=True,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": user},
                    ],
                )
                data = _extract_first_valid_json_object(raw)
                if not data:
                    break

                decision_tmp = (data.get("decision") or "").strip().lower()
                ans_tmp = _strip_inline_citations((data.get("answer") or "").strip())

                # If the model is answering/claryfing but violates language, re-ask with stronger warning
                if decision_tmp in {"answer", "clarify"} and ans_tmp and not _validate_output_language(ans_tmp, lang):
                    if attempt < MAX_LANGUAGE_REASK:
                        system = (
                            system
                            + "\n\nHARD ENFORCEMENT:\n"
                            + "If you output any other language, your output will be rejected.\n"
                            + _language_lock_for_answer_only(lang)
                        )
                        continue
                break

            except (APITimeoutError, APIConnectionError, RateLimitError, InternalServerError, APIStatusError) as e:
                last_gen_err = str(e)[:180]
                msg = _network_hint(lang)
                if preface:
                    msg = f"{preface} {msg}".strip()
                return _clarify(
                    msg,
                    references=fallback_refs,
                    used_ids=fallback_used_ids[:6],
                    debug={"reason": "openai_transient_error", "err": last_gen_err} if DEBUG_ANSWERER else None,
                )
            except Exception as e:
                last_gen_err = str(e)[:180]
                msg = _network_hint(lang)
                if preface:
                    msg = f"{preface} {msg}".strip()
                return _clarify(
                    msg,
                    references=fallback_refs,
                    used_ids=fallback_used_ids[:6],
                    debug={"reason": "openai_unknown_error", "err": last_gen_err} if DEBUG_ANSWERER else None,
                )

        data = _extract_first_valid_json_object(raw)
        if not data:
            msg = _clarify_text(lang, "I found relevant documents but could not produce a reliable structured answer. Please rephrase or specify the exact role/section.")
            if preface:
                msg = f"{preface} {msg}".strip()
            return _clarify(
                msg,
                references=fallback_refs,
                used_ids=fallback_used_ids[:6],
                debug={"reason": "model_output_not_json", "raw_head": (raw or "")[:240]} if DEBUG_ANSWERER else None,
            )

        decision = (data.get("decision") or "").strip().lower()
        answer = _strip_inline_citations((data.get("answer") or "").strip())

        if decision not in {"answer", "clarify", "refuse"}:
            decision = "clarify"

        used_chunk_ids = _parse_int_list(data.get("used_chunk_ids"))
        used_evidence_ids = _parse_int_list(data.get("used_evidence_ids"))

        # map evidence ids -> chunk ids
        for eid in used_evidence_ids:
            cid = evidence_id_to_chunk.get(int(eid))
            if cid is not None and cid not in used_chunk_ids:
                used_chunk_ids.append(int(cid))

        # unique used ids
        used_ids: List[int] = []
        seen = set()
        for n in used_chunk_ids:
            if n in seen:
                continue
            seen.add(n)
            used_ids.append(n)

        # map chunk id -> hit
        id_to_hit: Dict[int, Dict[str, Any]] = {}
        for h in included_hits:
            cid = _hit_chunk_id(h)
            if cid is None:
                continue
            id_to_hit[int(cid)] = h

        used_hits: List[Dict[str, Any]] = []
        for cid in used_ids:
            h = id_to_hit.get(int(cid))
            if h:
                used_hits.append(h)

        if not used_hits:
            used_hits = included_hits[: max(1, min(3, len(included_hits)))]
            used_ids = [cid for cid in (_hit_chunk_id(h) for h in used_hits) if cid is not None]

        # Build refs from used hits
        refs = _build_references_from_used_hits(used_hits)

        # ------------------------------------------------------------
        # Decision handling fixes
        # 1) Never refuse if we clearly have related evidence; clarify instead.
        # 2) Enforce language lock on clarify/answer via rewrite fallback.
        # ------------------------------------------------------------
        if decision == "refuse":
            # Only allow refusal if evidence is effectively empty/unrelated.
            # If we have included hits, convert to clarify (your “minimum fallback” rule).
            if included_hits:
                msg = _clarify_text(lang, "Retrieved evidence is related but not explicit. Which exact clause/role/section should I use?")
                if preface:
                    msg = f"{preface} {msg}".strip()
                return _clarify(
                    msg,
                    references=refs[:3],
                    used_ids=used_ids[:6],
                    debug={"reason": "model_refuse_converted_to_clarify"} if DEBUG_ANSWERER else None,
                )
            # True empty case
            return _refuse({"reason": "model_refuse_true_empty"} if DEBUG_ANSWERER else None)

        if decision == "clarify":
            if not answer:
                answer = _clarify_text(lang, "Which exact role/section or page range should I use?")

            if not _validate_output_language(answer, lang):
                rr = _rewrite_once_in_language(client, lang, answer)
                if rr and _validate_output_language(rr, lang):
                    answer = rr
                else:
                    answer = _clarify_text(lang, "Please specify the exact clause/section so I can answer from the documents.")

            if preface:
                answer = f"{preface} {answer}".strip()

            return _clarify(
                answer,
                references=refs[:3],
                used_ids=used_ids[:6],
                debug={"reason": "model_clarify"} if DEBUG_ANSWERER else None,
            )

        # decision == "answer"
        if (not answer) or (answer.strip() == REFUSAL_TEXT):
            msg = _clarify_text(lang, "I found relevant excerpts but they don’t explicitly resolve your question. Please specify the exact role/title or section.")
            if preface:
                msg = f"{preface} {msg}".strip()
            return _clarify(
                msg,
                references=refs[:3],
                used_ids=used_ids[:6],
                debug={"reason": "answer_empty_or_refusal"} if DEBUG_ANSWERER else None,
            )

        # Verify: keep only supported clauses (prevents hallucinations)
        verified, ok = _verify_supported_clauses(client, q, answer, evidence_text, lang)
        verified = _strip_inline_citations(verified).strip()

        if not ok or not verified:
            msg = _clarify_text(lang, "The evidence is related but does not explicitly support a full answer. Please clarify the exact clause/role you mean.")
            if preface:
                msg = f"{preface} {msg}".strip()
            return _clarify(
                msg,
                references=refs[:3],
                used_ids=used_ids[:6],
                debug={"reason": "verifier_could_not_support"} if DEBUG_ANSWERER else None,
            )

        # Enforce output language (best-effort rewrite, then final safety)
        if not _validate_output_language(verified, lang):
            rr = _rewrite_once_in_language(client, lang, verified)
            if rr and _validate_output_language(rr, lang):
                # Verify again (keeps it grounded)
                re_verified, re_ok = _verify_supported_clauses(client, q, rr, evidence_text, lang)
                re_verified = _strip_inline_citations(re_verified).strip()
                if re_ok and re_verified and _validate_output_language(re_verified, lang):
                    verified = re_verified
                else:
                    verified = rr

        verified = _normalize_ws(verified)

        # If still wrong language, do NOT return wrong-language answer; clarify instead (fixes your core issue)
        if not _validate_output_language(verified, lang):
            msg = _clarify_text(lang, "Main same language mein jawab dena chahta hun, lekin abhi output mix ho raha hai. Meharbani karke sawal ko thora aur specific karein (section/page)."
                               if lang == "roman_ur"
                               else "Please specify the exact section/page so I can answer strictly in the same language from the documents.")
            if preface:
                msg = f"{preface} {msg}".strip()
            return _clarify(
                msg,
                references=refs[:3],
                used_ids=used_ids[:6],
                debug={"reason": "final_language_mismatch_blocked"} if DEBUG_ANSWERER else None,
            )

        if preface:
            verified = f"{preface} {verified}".strip()

        out = _answer(
            verified.strip(),
            references=refs,
            used_ids=used_ids,
            debug=({
                "detected_lang": lang,
                "included_hits": len(included_hits),
                "used_hits": len(used_hits),
                "refs": len(refs),
                "evidence_chars": len(evidence_text),
                "mapped_evidence_ids": used_evidence_ids,
                "lang_valid": _validate_output_language(verified, lang),
                "preface_used": bool(preface),
                "timeout_s": OPENAI_TIMEOUT_S,
                "retries": OPENAI_MAX_RETRIES,
                "max_score_included": max_score,
                "json_mode": bool(USE_JSON_MODE),
            } if DEBUG_ANSWERER else None)
        )
        return out

    except Exception as e:
        # Never crash
        msg = "Your internet connection seems slow or unstable. Please try again."
        return _clarify(
            msg,
            references=[],
            used_ids=[],
            debug={"reason": "answerer_unhandled_exception", "err": str(e)[:200]} if DEBUG_ANSWERER else None,
        )
