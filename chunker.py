from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Any, Optional

from extractors import ExtractedUnit


# -----------------------------
# Chunk structure
# -----------------------------
@dataclass
class Chunk:
    doc_name: str
    doc_rank: int
    source_type: str         # "pdf" | "docx"
    loc_kind: str            # "page" | "section" | "paragraphs"
    loc_start: Any
    loc_end: Any
    chunk_text: str
    path: Optional[str] = None


# -----------------------------
# Utilities
# -----------------------------
_WS_RE = re.compile(r"[ \t]+")
_NUL_RE = re.compile(r"\x00+")

# Sentence-ish boundaries (English + Urdu punctuation)
_SENT_BOUNDARY_RE = re.compile(r"([.!?]|[\u06D4\u061F])\s")  # ۔  ؟
# Numeric heading like "1.", "1.1", "(a)", "2)" etc
_NUM_HEADING_RE = re.compile(r"^\s*(\(?[0-9]+(\.[0-9]+){0,4}\)?[\)\.]|[\(\[]?[a-zA-Z][\)\.]|\([ivxIVX]+\))\s+")
# Urdu heading-ish keywords
_URDU_HEADING_RE = re.compile(r"^\s*(باب|حصہ|شق|دفعہ|ضمیمہ|شیڈول|فہرست)\b")

# Keep unicode (Urdu) and punctuation. Only normalize whitespace.
def _clean_text(s: str) -> str:
    s = s or ""
    s = _NUL_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    # collapse horizontal whitespace, preserve newlines
    s = _WS_RE.sub(" ", s)
    s = re.sub(r"\n{4,}", "\n\n\n", s)
    s = "\n".join([ln.strip() for ln in s.split("\n")])
    return s.strip()


def _parse_book_rank(filename: str) -> int:
    """
    book1, book2, ... bookN => higher number = newer/higher priority.
    If no match => rank 0.
    """
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"\bbook\s*([0-9]+)\b", base, flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


# --- structural heuristics: detect tables/lists/headings ---
_BULLET_RE = re.compile(r"^\s*([•\-\u2022]|\d+[\)\.]|[a-zA-Z][\)\.])\s+")
_PIPE_TABLE_RE = re.compile(r"\s\|\s")  # " | " delimiter
_TAB_TABLE_RE = re.compile(r"\t+")
_MULTI_SPACE_COL_RE = re.compile(r"\s{2,}")  # raw column-like spacing

_HEADING_RE = re.compile(r"^\s*(schedule|annex|annexure|appendix|chapter|section|rule|article|clause)\b", re.I)


def _looks_like_table_line(line: str) -> bool:
    """
    Improved table detection:
    - pipe tables (" | ")
    - tabs
    - multi-space aligned columns
    """
    if not line:
        return False
    s = line.strip()
    if len(s) < 12:
        return False

    # Pipe table: allow smaller rows, but require at least 1 delimiter + multiple tokens
    if _PIPE_TABLE_RE.search(s):
        if s.count("|") >= 1:
            toks = re.findall(r"[A-Za-z\u0600-\u06FF0-9]{2,}", s)
            return len(toks) >= 3

    # Tabs strongly indicate a table
    if _TAB_TABLE_RE.search(s):
        return True

    # Multi-space columns: require multiple gaps and multiple tokens
    if _MULTI_SPACE_COL_RE.search(s):
        # Avoid treating normal sentences as tables: must have multiple big gaps
        gaps = len(re.findall(r"\s{3,}", s))
        if gaps >= 2:
            toks = re.findall(r"[A-Za-z\u0600-\u06FF0-9]{2,}", s)
            return len(toks) >= 4

    return False


def _looks_like_list_line(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if _BULLET_RE.search(s) is not None:
        return True
    # also treat "•" anywhere at start, and Urdu/Arabic bullet-like dash
    if s.startswith(("•", "-", "–", "—")) and len(s) > 6:
        return True
    # numeric list: "1)" "2." "a)" etc
    if _NUM_HEADING_RE.search(s) is not None:
        return True
    return False


def _looks_like_table_or_list(line: str) -> bool:
    return _looks_like_list_line(line) or _looks_like_table_line(line)


def _is_heading(line: str) -> bool:
    if not line:
        return False
    s = line.strip()
    if not s:
        return False

    # common English heading keywords
    if _HEADING_RE.search(s):
        return True

    # Urdu heading keywords
    if _URDU_HEADING_RE.search(s):
        return True

    # numbered headings like "1. Definitions" / "(a) Scope" etc
    if _NUM_HEADING_RE.search(s) and len(s) <= 120:
        return True

    # short ALL-CAPS headings (English only; don't break Urdu)
    letters = re.sub(r"[^A-Za-z]+", "", s)
    if 4 <= len(letters) <= 60 and letters.isupper() and len(s) <= 90:
        return True

    return False


def _split_into_blocks(text: str) -> List[str]:
    """
    Production chunking blocks:
    - Split on blank lines
    - Start new block on headings
    - Keep tables/lists as their own blocks; do not mix with narrative text
    - Prevent mode mixing: narrative vs list/table
    """
    t = _clean_text(text)
    if not t:
        return []

    lines = t.split("\n")
    blocks: List[str] = []
    buf: List[str] = []

    def flush() -> None:
        nonlocal buf
        if not buf:
            return
        b = _clean_text("\n".join(buf))
        if b:
            blocks.append(b)
        buf = []

    def last_is_structured() -> bool:
        if not buf:
            return False
        return _looks_like_table_or_list(buf[-1])

    for ln in lines:
        raw = (ln or "").strip()

        # blank line = boundary
        if not raw:
            flush()
            continue

        # headings start a new block
        if _is_heading(raw):
            flush()
            buf.append(raw)
            continue

        structured = _looks_like_table_or_list(raw)

        # If switching between narrative <-> structured, flush to avoid mixing
        if buf:
            prev_structured = last_is_structured()
            if structured != prev_structured:
                flush()

        buf.append(raw)

    flush()
    return blocks if blocks else [t]


def _is_mid_word_boundary(s: str) -> bool:
    """
    Detect if a string likely starts mid-word (bad overlap stitch).
    Conservative: only for latin/urdu letters/digits at start and no boundary early.
    """
    if not s:
        return False
    s = s.lstrip()
    if not s:
        return False

    # if starts with punctuation or newline, it's safe
    if re.match(r"^[\s\.\,\;\:\!\?\)\]\}\"\']+", s):
        return False

    # if starts with a letter/digit and the first 12 chars contain no whitespace/punct, likely mid-token
    head = s[:12]
    if re.match(r"^[A-Za-z\u0600-\u06FF0-9]+$", head):
        return True
    return False


def _safe_overlap_tail(prev: str, cap: int) -> str:
    """
    Compute a safe overlap tail:
    - avoid structured tails
    - avoid mid-word starts
    - try to begin at a sentence boundary when possible
    """
    if not prev:
        return ""
    prev = prev.strip()
    if not prev:
        return ""

    # If previous chunk is mostly structured, skip overlap entirely (it harms tables/lists)
    lines = prev.split("\n")
    structured_lines = sum(1 for ln in lines if _looks_like_table_or_list(ln.strip()))
    if lines and structured_lines / max(1, len(lines)) >= 0.60:
        return ""

    cap = max(60, min(int(cap or 0), 700))
    tail = prev[-cap:].strip()
    if not tail:
        return ""

    # Prefer sentence boundary: find last boundary inside tail and start there
    m = None
    for m in _SENT_BOUNDARY_RE.finditer(tail):
        pass
    if m is not None:
        cut = m.end()
        candidate = tail[cut:].strip()
        if candidate and len(candidate) >= 40:
            tail = candidate

    # If tail starts mid-word, drop until a boundary
    if _is_mid_word_boundary(tail):
        b = re.search(r"[\s\.,;:\)\]\}!\?\u06D4\u061F]", tail)
        if b and b.start() < 25:
            tail = tail[b.start():].strip()

    # keep only if still meaningful
    if len(tail) < 40:
        return ""
    return tail


def _split_huge_block_soft(text: str, max_chars: int, overlap_chars: int) -> List[str]:
    """
    Split a huge block into parts, preferring boundaries (newline/sentence),
    rather than raw character slicing.
    """
    t = _clean_text(text)
    if not t:
        return []

    parts: List[str] = []
    step = max(200, max_chars - max(0, overlap_chars))

    start = 0
    while start < len(t):
        end = min(len(t), start + max_chars)
        window = t[start:end]

        # try to break on newline near the end
        if end < len(t):
            nl = window.rfind("\n")
            if nl >= int(0.65 * len(window)):
                end = start + nl

        # try to break on sentence boundary near the end (if still large)
        if end < len(t):
            win2 = t[start:end]
            last = None
            for m in _SENT_BOUNDARY_RE.finditer(win2):
                last = m
            if last is not None and last.end() >= int(0.70 * len(win2)):
                end = start + last.end()

        part = _clean_text(t[start:end])
        if part:
            parts.append(part)

        if end >= len(t):
            break
        start = start + step

    return parts


def _chunk_by_char_budget(blocks: List[str], max_chars: int, overlap_chars: int) -> List[str]:
    """
    Creates chunks up to max_chars.
    Overlap is applied as a tail snippet of previous chunk.

    Hardening:
    - Never allow overlap >= max_chars
    - Safe splitting for huge blocks using soft boundaries
    - Cap total chunks globally
    """
    if not blocks:
        return []

    max_chars = max(500, int(max_chars or 0))
    overlap_chars = max(0, int(overlap_chars or 0))
    if overlap_chars >= max_chars:
        overlap_chars = max(0, max_chars // 5)

    chunks: List[str] = []
    buf: List[str] = []
    size = 0

    def flush() -> None:
        nonlocal buf, size
        if not buf:
            return
        chunk = _clean_text("\n\n".join(buf))
        if chunk:
            chunks.append(chunk)
        buf = []
        size = 0

    GLOBAL_MAX_CHUNKS = 20000

    for b in blocks:
        b = (b or "").strip()
        if not b:
            continue

        # Huge block safeguard (improved)
        if len(b) > max_chars:
            flush()
            parts = _split_huge_block_soft(b, max_chars=max_chars, overlap_chars=overlap_chars)
            for part in parts:
                if part:
                    chunks.append(part)
                    if len(chunks) >= GLOBAL_MAX_CHUNKS:
                        return chunks
            continue

        if size + len(b) + 2 > max_chars and buf:
            flush()

        buf.append(b)
        size += len(b) + 2

        if len(chunks) >= GLOBAL_MAX_CHUNKS:
            flush()
            return chunks

    flush()

    # Overlap: add tail of previous chunk (safe boundary)
    if overlap_chars > 0 and len(chunks) > 1:
        out: List[str] = []
        cap = max(80, min(overlap_chars, 600))

        for i, c in enumerate(chunks):
            if i == 0:
                out.append(c)
                continue

            prev = chunks[i - 1]
            tail = _safe_overlap_tail(prev, cap=cap)

            if tail:
                stitched = _clean_text(tail + "\n\n" + c)
            else:
                stitched = _clean_text(c)

            out.append(stitched)

        return out

    return chunks


def _force_keep_chunk(ctext: str) -> bool:
    """
    Some chunks must be kept even if short, because they answer common questions.
    """
    t = (ctext or "").lower()

    # Schedules/Annexures
    if "schedule" in t or "annex" in t or "annexure" in t or "appendix" in t:
        return True

    # PERA identity / short definition-like
    if "punjab enforcement and regulatory authority" in t or re.search(r"\bpera\b", t):
        if len(t) < 700:
            return True

    # Roles / org
    if "chief technology officer" in t or re.search(r"\bcto\b", t):
        return True

    # TOR
    if "terms of reference" in t or re.search(r"\btor\b", t):
        return True

    # definition patterns
    if re.search(r"\bmeans\b", t) and len(t) < 900:
        return True
    if re.search(r"^\s*definition(s)?\b", t) and len(t) < 1200:
        return True

    # Urdu definition-ish: "مراد" / "سے مراد"
    if ("سے مراد" in ctext) or ("مراد" in ctext and len(ctext) < 900):
        return True

    return False


def _count_real_words(s: str) -> int:
    return len(re.findall(r"[A-Za-z\u0600-\u06FF]{3,}", s or ""))


# -----------------------------
# Main chunking API
# -----------------------------
def chunk_units(
    units: List[ExtractedUnit],
    max_chars: int = 4500,
    overlap_chars: int = 350,
    min_chunk_chars: int = 200
) -> List[Chunk]:
    """
    Converts extracted units into chunks while preserving traceability.

    Guarantees:
      - PDF units are page-scoped (never mix pages)
      - DOCX units are unit-scoped (never mix sections/ranges)

    Safety:
      - Keep short but high-value chunks (Schedule/Annex/definitions/role titles)
      - Drop short tails only when clearly low-signal
    """
    out: List[Chunk] = []

    for u in units:
        txt = _clean_text(getattr(u, "text", "") or "")
        if not txt:
            continue

        rank = getattr(u, "doc_rank", 0) or _parse_book_rank(getattr(u, "doc_name", ""))

        # If unit itself is short, keep it as one chunk
        if len(txt) < min_chunk_chars:
            out.append(
                Chunk(
                    doc_name=u.doc_name,
                    doc_rank=rank,
                    source_type=u.source_type,
                    loc_kind=u.loc_kind,
                    loc_start=u.loc_start,
                    loc_end=u.loc_end,
                    chunk_text=txt,
                    path=getattr(u, "path", None),
                )
            )
            continue

        blocks = _split_into_blocks(txt)
        chunk_texts = _chunk_by_char_budget(blocks, max_chars=max_chars, overlap_chars=overlap_chars)
        if not chunk_texts:
            chunk_texts = [txt]

        for i, ctext in enumerate(chunk_texts):
            ctext = _clean_text(ctext)
            if not ctext:
                continue

            if len(ctext) < min_chunk_chars:
                if len(chunk_texts) == 1:
                    pass
                elif _force_keep_chunk(ctext):
                    pass
                elif i == len(chunk_texts) - 1:
                    # allow last tail only if it has enough real words
                    if _count_real_words(ctext) >= 10:
                        pass
                    else:
                        continue
                else:
                    continue

            out.append(
                Chunk(
                    doc_name=u.doc_name,
                    doc_rank=rank,
                    source_type=u.source_type,
                    loc_kind=u.loc_kind,
                    loc_start=u.loc_start,
                    loc_end=u.loc_end,
                    chunk_text=ctext,
                    path=getattr(u, "path", None),
                )
            )

    return out
