from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Optional, Tuple, List

# ----------------------------
# Strict language detection (deterministic)
# Supported: english | urdu | roman_urdu | unsupported
# ----------------------------

_OTHER_SCRIPT_RE = re.compile(
    r"[\u0900-\u097F"  # Devanagari
    r"\u0400-\u04FF"   # Cyrillic
    r"\u4E00-\u9FFF"   # CJK
    r"\u3040-\u30FF"   # Japanese
    r"\u0E00-\u0E7F"   # Thai
    r"\u1100-\u11FF"   # Hangul Jamo
    r"\uAC00-\uD7AF"   # Hangul syllables
    r"]"
)

_ARABIC_SCRIPT_RE = re.compile(r"[\u0600-\u06FF\u0750-\u077F\u08A0-\u08FF]")

_URDU_LETTERS = set("پچڈڑژگںھٰٖؐؑؒؓؔؕٔۍۓۓی")
_URDU_SPECIFIC_RE = re.compile("[" + re.escape("".join(_URDU_LETTERS)) + "]")

_ROMAN_URDU_HINTS = {
    "aoa", "a.o.a", "assalam", "assalamu", "asalam", "salam", "salaam", "slm",
    "ap", "aap", "kya", "ky", "ka", "ki", "ke", "ko", "se", "par", "aur", "ya",
    "kaise", "kaisay", "kesy", "kese",
    "hain", "hai", "theek", "thik",
    "shukriya", "jazakallah",
    "haal", "hal",
    "meharbani", "plz", "please",
    "ji", "haan", "han", "nahi",
}

_EN_STOPWORDS = {
    "the", "and", "or", "of", "to", "in", "is", "are", "was", "were", "be", "been",
    "for", "with", "on", "as", "by", "at", "from", "that", "this", "it", "you", "your",
    "can", "will", "should", "what", "how", "when", "where", "why", "a", "an"
}

def _tokenize_latin_words(s: str) -> List[str]:
    return re.findall(r"[a-zA-Z']+", (s or "").lower())

def detect_language(text: str) -> str:
    """
    Returns: "english" | "urdu" | "roman_urdu" | "unsupported"
    """
    s = (text or "").strip()
    if not s:
        return "english"

    if _OTHER_SCRIPT_RE.search(s):
        return "unsupported"

    if _ARABIC_SCRIPT_RE.search(s):
        if _URDU_SPECIFIC_RE.search(s):
            return "urdu"
        return "unsupported"

    tokens = _tokenize_latin_words(s)
    if not tokens:
        return "english"

    roman_hits = sum(1 for t in tokens if t in _ROMAN_URDU_HINTS)
    en_hits = sum(1 for t in tokens if t in _EN_STOPWORDS)

    if roman_hits >= 1 and (roman_hits >= en_hits or roman_hits >= 2):
        return "roman_urdu"

    return "english"

# ----------------------------
# Normalize for latin matching
# ----------------------------
def _norm_latin(s: str) -> str:
    s = (s or "").lower()
    s = re.sub(r"[^a-z0-9\s\.\-']", " ", s)
    s = re.sub(r"\s+", " ", s).strip()
    return s

# ----------------------------
# Greeting / smalltalk patterns
# ----------------------------

# Anchored patterns (start-of-string)
_GREETING_PATTERNS_LATIN = [
    r"^(hi|hello|hey|hy)\b",
    r"^(aoa|a\.o\.a|assalam|assalamu|asalam|salam|salaam|slm)\b",
    r"^(good\s+morning|good\s+afternoon|good\s+evening)\b",
]

_SMALLTALK_PATTERNS_LATIN = [
    r"^(how\s+are\s+you|how\s+r\s+you|how\s+ru)\b",
    r"^(kya\s+haal|kya\s+hal|haal\s+chaal|hal\s+chaal)\b",
    r"^(ap\s+kaise|aap\s+kaise|kaise\s+hain|kaisay\s+hain|kesy\s+hain|kese\s+hain)\b",
    r"^(thanks|thank\s+you|thx|shukriya|jazakallah)\b",
]

_GREETING_PATTERNS_URDU = [
    r"^(السلام\s*علیکم|اسلام\s*علیکم|سلام)\b",
]

_SMALLTALK_PATTERNS_URDU = [
    r"^(آپ\s*کیسے\s*ہیں|آپ\s*کیسی\s*ہیں|کیا\s*حال\s*ہے|کیا\s*حال)\b",
    r"^(کیسے\s*ہو|کیسا\s*ہے|کیسی\s*ہو)\b",
    r"^(آپ\s*ٹھیک\s*ہیں|آپ\s*خیریت\s*سے\s*ہیں)\b",
    r"^(شکریہ|جزاک\s*اللہ)\b",
]

def _starts_with_any(patterns: List[str], text: str) -> bool:
    for p in patterns:
        if re.search(p, text):
            return True
    return False

def is_smalltalk_or_greeting(text: str) -> bool:
    """
    True if the message looks like greeting/smalltalk ONLY.
    IMPORTANT: This should NOT swallow greeting+question.
    """
    raw = (text or "").strip()
    if not raw:
        return False

    # Urdu
    if _ARABIC_SCRIPT_RE.search(raw) and _URDU_SPECIFIC_RE.search(raw):
        return _starts_with_any(_GREETING_PATTERNS_URDU + _SMALLTALK_PATTERNS_URDU, raw.strip())

    # Latin
    t = _norm_latin(raw)
    return _starts_with_any(_GREETING_PATTERNS_LATIN + _SMALLTALK_PATTERNS_LATIN, t)

def is_greeting_prefix(text: str) -> bool:
    raw = (text or "").strip()
    if not raw:
        return False

    if _ARABIC_SCRIPT_RE.search(raw) and _URDU_SPECIFIC_RE.search(raw):
        return _starts_with_any(_GREETING_PATTERNS_URDU, raw.strip())

    t = _norm_latin(raw)
    return _starts_with_any(_GREETING_PATTERNS_LATIN, t)

# ----------------------------
# Greeting + question split (RAW, deterministic)
# ----------------------------

@dataclass
class SmalltalkDecision:
    is_greeting_only: bool
    ack: str
    response: str
    remaining_question: str
    language: str

_GREET_PREFIX_RAW_RE = re.compile(
    r"^\s*(hi|hello|hey|hy|aoa|a\.o\.a|slm|salam|salaam|assalam|assalamu|asalam|good\s+morning|good\s+afternoon|good\s+evening)\b",
    re.IGNORECASE,
)

_URDU_GREET_PREFIX_RAW_RE = re.compile(r"^\s*(السلام\s*علیکم|اسلام\s*علیکم|سلام)\b")

def _language_from_greeting_token(token: str) -> Optional[str]:
    """
    Infer language from the greeting token itself:
    - Islamic greetings in latin -> roman_urdu
    - hi/hello/good morning -> english
    """
    t = (token or "").lower()
    if any(x in t for x in ["aoa", "a.o.a", "slm", "salam", "salaam", "assalam", "assalamu", "asalam"]):
        return "roman_urdu"
    if any(x in t for x in ["hi", "hello", "hey", "hy", "good morning", "good afternoon", "good evening"]):
        return "english"
    return None

def split_greeting_and_question(text: str) -> Tuple[bool, str, str, str]:
    """
    Returns: (has_greeting_prefix, ack, remaining_question, lang)

    lang selection priority:
      1) If Urdu script greeting -> urdu
      2) Else infer from greeting token (roman_urdu vs english)
      3) Else fallback detect_language(full text)
    """
    raw = (text or "").strip()
    if not raw:
        return False, "", "", "english"

    # Urdu greeting prefix
    m_ur = _URDU_GREET_PREFIX_RAW_RE.search(raw)
    if m_ur:
        remaining = raw[m_ur.end():].lstrip(" ,:-—–\n\t")
        ack = "وعلیکم السلام! "
        return True, ack, remaining, "urdu"

    # Latin greeting prefix
    m = _GREET_PREFIX_RAW_RE.search(raw)
    if not m:
        return False, "", raw, detect_language(raw)

    token = (m.group(0) or "").strip()
    remaining = raw[m.end():].lstrip(" ,:-—–\n\t")

    token_lang = _language_from_greeting_token(token)
    lang = token_lang or detect_language(raw)

    if token_lang == "roman_urdu":
        ack = "Walikum Assalam! "
    else:
        # English-style greeting ack
        ack = "Hello! "

    return True, ack, remaining, lang

# ----------------------------
# Deterministic responses (no retrieval)
# ----------------------------
SUPPORTED_LANG_REFUSAL = os.getenv(
    "SUPPORTED_LANG_REFUSAL",
    "This assistant supports only English, Urdu, and Roman Urdu."
).strip()

_RESPONSES = {
    "english": "Hello! I am the PERA AI Assistant. How can I help you with PERA-related questions?",
    "roman_urdu": "Assalam-o-Alaikum! Main PERA AI Assistant hoon. Aap PERA se related sawal poochain, main madad kar doon ga.",
    "urdu": "السلام علیکم! میں PERA AI Assistant ہوں۔ آپ PERA سے متعلق سوالات پوچھیں، میں آپ کی رہنمائی کر دوں گا۔",
    "unsupported": SUPPORTED_LANG_REFUSAL,
}

def decide_smalltalk(text: str) -> Optional[SmalltalkDecision]:
    """
    - If greeting-only / smalltalk-only -> returns response (is_greeting_only=True)
    - If greeting + real question -> returns ack + remaining_question (is_greeting_only=False)
    - Else -> None
    """
    raw = (text or "").strip()
    if not raw:
        return SmalltalkDecision(
            is_greeting_only=True,
            ack="",
            response=_RESPONSES["english"],
            remaining_question="",
            language="english",
        )

    # Hard refuse unsupported scripts
    lang_full = detect_language(raw)
    if lang_full == "unsupported":
        return SmalltalkDecision(
            is_greeting_only=True,
            ack="",
            response=_RESPONSES["unsupported"],
            remaining_question="",
            language="unsupported",
        )

    has_greet, ack, remaining, lang = split_greeting_and_question(raw)

    # If we have greeting prefix and a meaningful remaining part, send remaining to RAG
    if has_greet and remaining and remaining.strip():
        # If remaining is NOT smalltalk, it's a real question
        if not is_smalltalk_or_greeting(remaining.strip()):
            return SmalltalkDecision(
                is_greeting_only=False,
                ack=ack,
                response="",
                remaining_question=remaining.strip(),
                language=lang,
            )

    # Greeting/smalltalk-only -> deterministic response
    if is_smalltalk_or_greeting(raw):
        return SmalltalkDecision(
            is_greeting_only=True,
            ack="",
            response=_RESPONSES.get(lang, _RESPONSES["english"]),
            remaining_question="",
            language=lang,
        )

    return None
