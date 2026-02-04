from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any


# -----------------------------
# Data structures
# -----------------------------
@dataclass
class ExtractedUnit:
    """
    A traceable extraction unit that can later be chunked.
    - PDF  => one unit per page
    - DOCX => one unit per section (heading) or paragraph-range block
    """
    doc_name: str
    source_type: str              # "pdf" | "docx"
    loc_kind: str                 # "page" | "section" | "paragraphs"
    loc_start: Any                # int page number, or str anchor
    loc_end: Any                  # int page number, or str anchor
    text: str

    # optional metadata
    path: Optional[str] = None
    doc_rank: int = 0


# -----------------------------
# Helpers
# -----------------------------
SUPPORTED_EXTS = (".pdf", ".docx")

_NUL_RE = re.compile(r"\x00+")
_PAGE_NUM_RE = re.compile(r"^\s*(page\s*)?\d+\s*(of\s*\d+)?\s*$", re.I)
_MULTI_NEWLINES_RE = re.compile(r"\n{4,}")

_BULLET_RE = re.compile(r"^\s*([•\-\u2022]|\d+[\)\.]|[A-Za-z][\)\.])\s+")
_TABLE_LIKE_RE = re.compile(r"(\t+|\s{2,})")
_HYPHEN_END_RE = re.compile(r".*[\w\u0600-\u06FF]-$")

# Useful heading-ish keywords (English + common legal doc markers)
_HEADING_KEYWORDS_RE = re.compile(
    r"^\s*(schedule|annex|annexure|appendix|chapter|section|part|rule|rules|procedure|definitions?)\b",
    re.I
)

_ALLCAPS_WORDS_RE = re.compile(r"^[A-Z0-9\s\-\–—_:;,/\\\.\(\)\[\]]{4,}$")


def _clean_text_general(s: str) -> str:
    """
    General cleaning after structure decisions are already made.
    IMPORTANT: we do NOT collapse multiple spaces here globally
    because tables may have been converted into " | " already.
    """
    s = s or ""
    s = _NUL_RE.sub(" ", s)
    s = s.replace("\r\n", "\n").replace("\r", "\n")
    s = _MULTI_NEWLINES_RE.sub("\n\n\n", s)
    s = "\n".join([ln.strip() for ln in s.split("\n")])
    return s.strip()


def _is_heading_style(style_name: str) -> bool:
    """
    Robust heading style detection:
    - Handles "Heading 1", "heading 2", "Title", and some localized variations
    """
    if not style_name:
        return False
    sn = style_name.strip().lower()
    if sn.startswith("heading"):
        return True
    if "heading" in sn:
        return True
    if sn in ("title", "subtitle"):
        return True
    return False


def _looks_like_heading_text(txt: str) -> bool:
    """
    Fallback heading heuristic when style is not reliable.
    Conservative (avoid false positives).
    """
    t = (txt or "").strip()
    if not t:
        return False

    # Strong legal-structure markers
    if _HEADING_KEYWORDS_RE.search(t):
        return True

    # Short all-caps headings
    if len(t) <= 90 and _ALLCAPS_WORDS_RE.match(t):
        # require some letters to avoid "----"
        letters = len(re.findall(r"[A-Z]", t))
        if letters >= 4:
            return True

    # "X:" style headings (short)
    if len(t) <= 80 and t.endswith(":"):
        # require at least one real word
        if len(re.findall(r"[A-Za-z\u0600-\u06FF]{2,}", t)) >= 1:
            return True

    return False


def discover_documents(data_dir: str = "assets/data") -> List[str]:
    data_dir = data_dir.replace("\\", "/")
    if not os.path.isdir(data_dir):
        return []
    out: List[str] = []
    for name in os.listdir(data_dir):
        p = os.path.join(data_dir, name).replace("\\", "/")
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if low.endswith(SUPPORTED_EXTS):
            out.append(p)
    return sorted(out)


# -----------------------------
# PDF extraction quality helpers
# -----------------------------
def _pdf_lines_raw(text: str) -> List[str]:
    """
    Split PDF raw text into lines WITHOUT collapsing multiple spaces.
    This is critical for table detection.
    """
    t = text or ""
    t = _NUL_RE.sub(" ", t)
    t = t.replace("\r\n", "\n").replace("\r", "\n")
    lines = [ln.strip("\n") for ln in t.split("\n")]
    lines = [ln.strip() for ln in lines if ln and ln.strip()]
    return lines


def _normalize_line_for_header_footer(line: str) -> str:
    l = (line or "").strip().lower()
    l = re.sub(r"\d+", "0", l)
    l = re.sub(r"\s+", " ", l).strip()
    return l


def _is_header_footer_candidate(line: str) -> bool:
    s = (line or "").strip()
    if not s:
        return False
    if len(s) > 120:
        return False
    if _PAGE_NUM_RE.match(s):
        return True

    letters = len(re.findall(r"[A-Za-z\u0600-\u06FF]", s))
    if letters <= 6:
        return True

    sl = s.lower()
    if "punjab" in sl and ("authority" in sl or "regulatory" in sl or "enforcement" in sl):
        return True
    return False


def _detect_repeated_header_footer(page_lines: List[List[str]], min_pages: int = 3) -> Dict[str, set]:
    if len(page_lines) < min_pages:
        return {"header": set(), "footer": set()}

    first_counts: Dict[str, int] = {}
    last_counts: Dict[str, int] = {}

    eligible_pages = 0
    for lines in page_lines:
        if not lines:
            continue
        eligible_pages += 1

        for ln in lines[:2]:
            if not _is_header_footer_candidate(ln):
                continue
            k = _normalize_line_for_header_footer(ln)
            if k:
                first_counts[k] = first_counts.get(k, 0) + 1

        for ln in lines[-2:]:
            if not _is_header_footer_candidate(ln):
                continue
            k = _normalize_line_for_header_footer(ln)
            if k:
                last_counts[k] = last_counts.get(k, 0) + 1

    if eligible_pages < min_pages:
        return {"header": set(), "footer": set()}

    threshold = max(2, int(0.60 * eligible_pages))
    header = {k for k, c in first_counts.items() if c >= threshold}
    footer = {k for k, c in last_counts.items() if c >= threshold}
    return {"header": header, "footer": footer}


def _strip_headers_footers(lines: List[str], hf: Dict[str, set]) -> List[str]:
    if not lines:
        return lines
    out: List[str] = []
    for ln in lines:
        s = (ln or "").strip()
        if not s:
            continue
        norm = _normalize_line_for_header_footer(s)
        if norm in hf.get("header", set()) or norm in hf.get("footer", set()):
            continue
        if _PAGE_NUM_RE.match(s):
            continue
        out.append(s)
    return out


def _looks_like_table_row(raw_line: str) -> bool:
    s = (raw_line or "").rstrip()
    if len(s) < 20:
        return False
    if _TABLE_LIKE_RE.search(s) is None:
        return False
    if s.count("  ") >= 1 or "\t" in s:
        tokens = re.findall(r"[A-Za-z\u0600-\u06FF0-9]{2,}", s)
        return len(tokens) >= 3
    return False


def _normalize_table_row(raw_line: str) -> str:
    s = (raw_line or "").strip()
    s = re.sub(r"\t+", "  ", s)
    s = re.sub(r"\s{2,}", " | ", s).strip()
    return s


def _join_pdf_lines(lines: List[str]) -> str:
    """
    Join PDF-extracted lines into cleaner text:
    - keep bullets / table-like rows as new lines
    - merge narrative lines into paragraphs
    - fix hyphenated line breaks
    """
    if not lines:
        return ""

    merged: List[str] = []
    buf: List[str] = []

    def flush_buf() -> None:
        nonlocal buf
        if not buf:
            return
        merged.append(" ".join(buf).strip())
        buf = []

    for ln in lines:
        raw = (ln or "").strip()
        if not raw:
            flush_buf()
            continue

        if _BULLET_RE.search(raw) is not None:
            flush_buf()
            merged.append(raw)
            continue

        if _looks_like_table_row(raw):
            flush_buf()
            merged.append(_normalize_table_row(raw))
            continue

        if buf:
            prev = buf[-1]
            if _HYPHEN_END_RE.match(prev):
                buf[-1] = prev[:-1] + raw
                continue

        if buf and buf[-1].endswith(":"):
            flush_buf()
            buf.append(raw)
            continue

        buf.append(raw)

    flush_buf()
    return _clean_text_general("\n".join(merged))


# -----------------------------
# PDF Extraction
# -----------------------------
def extract_pdf_units(pdf_path: str) -> List[ExtractedUnit]:
    """
    Extract PDF page-by-page.
    - conservative repeated header/footer removal
    - table preservation
    - hyphenation repair
    """
    units: List[ExtractedUnit] = []
    pdf_path = (pdf_path or "").replace("\\", "/")
    doc_name = os.path.basename(pdf_path)

    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = reader.pages
    except Exception:
        return units

    raw_lines_by_page: List[List[str]] = []
    for page in pages:
        try:
            raw = page.extract_text() or ""
        except Exception:
            raw = ""
        raw_lines_by_page.append(_pdf_lines_raw(raw))

    hf = _detect_repeated_header_footer(raw_lines_by_page)

    for i, lines in enumerate(raw_lines_by_page):
        page_no = i + 1

        lines2 = _strip_headers_footers(lines, hf)
        if len(lines2) < max(3, int(0.25 * len(lines))):
            lines2 = lines

        text = _join_pdf_lines(lines2)
        text = _clean_text_general(text)
        if not text:
            continue

        units.append(
            ExtractedUnit(
                doc_name=doc_name,
                source_type="pdf",
                loc_kind="page",
                loc_start=page_no,
                loc_end=page_no,
                text=text,
                path=pdf_path,
            )
        )

    return units


# -----------------------------
# DOCX Extraction
# -----------------------------
def _para_is_heading(p) -> bool:
    """
    Decide if a docx paragraph is a heading.
    Uses style first, then a conservative text heuristic.
    """
    txt = (getattr(p, "text", "") or "").strip()
    if not txt:
        return False

    style_name = ""
    try:
        style_name = p.style.name if p.style else ""
    except Exception:
        style_name = ""

    if _is_heading_style(style_name):
        return True

    # fallback heuristic
    return _looks_like_heading_text(txt)


def _format_docx_unit_text(heading: str, body: str) -> str:
    """
    IMPORTANT FIX: include heading in the unit text so embeddings capture it.
    """
    h = (heading or "").strip()
    b = (body or "").strip()
    if h and b:
        return _clean_text_general(f"{h}\n\n{b}")
    if h:
        return _clean_text_general(h)
    return _clean_text_general(b)


def extract_docx_units(
    docx_path: str,
    min_chars_per_unit: int = 800,
    max_chars_per_unit: int = 6000
) -> List[ExtractedUnit]:
    """
    Extract DOCX into stable units using headings.
    FIX: headings are included into text payload so retrieval doesn't miss section titles.
    """
    units: List[ExtractedUnit] = []
    docx_path = (docx_path or "").replace("\\", "/")
    doc_name = os.path.basename(docx_path)

    try:
        from docx import Document
        doc = Document(docx_path)
    except Exception:
        return units

    # Build an ordered stream of paragraphs with heading context
    stream: List[Dict[str, Any]] = []
    para_idx = 0
    current_heading = ""

    # Track heading-only situations
    last_was_heading = False

    for p in doc.paragraphs:
        txt = (p.text or "").strip()
        if not txt:
            continue

        if _para_is_heading(p):
            current_heading = txt
            last_was_heading = True
            continue

        para_idx += 1
        stream.append({"i": para_idx, "heading": current_heading, "text": txt})
        last_was_heading = False

    # If DOCX had headings but no body paragraphs, keep at least the headings as units (rare but possible)
    if not stream:
        # best-effort: add document title-ish headings if present
        # (we cannot reliably access headings now because we skipped them above)
        return units

    has_any_heading = any((x.get("heading") or "").strip() for x in stream)

    if has_any_heading:
        _emit_docx_stream_grouped_by_heading(
            units=units,
            doc_name=doc_name,
            docx_path=docx_path,
            stream=stream,
            min_chars=min_chars_per_unit,
            max_chars=max_chars_per_unit,
        )
        return units

    # No headings detected: fall back to paragraph blocks
    _emit_docx_paragraph_blocks(
        units=units,
        doc_name=doc_name,
        docx_path=docx_path,
        items=stream,
        min_chars=min_chars_per_unit,
        max_chars=max_chars_per_unit
    )
    return units


def _emit_docx_stream_grouped_by_heading(
    units: List[ExtractedUnit],
    doc_name: str,
    docx_path: str,
    stream: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int,
) -> None:
    """
    Ordered grouping by heading (do NOT use dict grouping; it destroys order).
    Each emitted unit includes its heading in the text.
    """
    current_heading = ""
    buffer: List[str] = []
    start_i: Optional[int] = None
    end_i: Optional[int] = None
    char_count = 0

    def flush() -> None:
        nonlocal buffer, start_i, end_i, char_count, current_heading
        if not buffer and not current_heading:
            return

        body = _clean_text_general("\n".join(buffer)) if buffer else ""
        text = _format_docx_unit_text(current_heading, body)

        if text:
            # Anchor includes paragraph range (when available)
            if start_i is not None and end_i is not None:
                anchor = f'Section: "{current_heading or "Untitled"}" (Paragraphs {start_i}–{end_i})'
            else:
                anchor = f'Section: "{current_heading or "Untitled"}"'

            units.append(
                ExtractedUnit(
                    doc_name=doc_name,
                    source_type="docx",
                    loc_kind="section",
                    loc_start=anchor,
                    loc_end=anchor,
                    text=text,
                    path=docx_path,
                )
            )

        buffer = []
        start_i = None
        end_i = None
        char_count = 0

    for item in stream:
        h = (item.get("heading") or "").strip()
        txt = (item.get("text") or "").strip()
        i = int(item.get("i") or 0)

        # Heading changed => flush previous section
        if h != current_heading:
            flush()
            current_heading = h

        if start_i is None:
            start_i = i
        end_i = i

        buffer.append(txt)
        char_count += len(txt) + 1

        if char_count >= max_chars:
            flush()

    flush()


def _emit_docx_paragraph_blocks(
    units: List[ExtractedUnit],
    doc_name: str,
    docx_path: str,
    items: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int
) -> None:
    buffer: List[str] = []
    start_i: Optional[int] = None
    end_i: Optional[int] = None
    char_count = 0

    for p in items:
        txt = (p.get("text") or "").strip()
        i = int(p.get("i") or 0)
        if not txt:
            continue

        if start_i is None:
            start_i = i
        end_i = i

        buffer.append(txt)
        char_count += len(txt) + 1

        if char_count >= max_chars:
            text = _clean_text_general("\n".join(buffer))
            if text:
                anchor = f"Paragraphs {start_i}–{end_i}"
                units.append(
                    ExtractedUnit(
                        doc_name=doc_name,
                        source_type="docx",
                        loc_kind="paragraphs",
                        loc_start=anchor,
                        loc_end=anchor,
                        text=text,
                        path=docx_path,
                    )
                )
            buffer = []
            start_i = None
            end_i = None
            char_count = 0

    if buffer:
        text = _clean_text_general("\n".join(buffer))
        if text:
            anchor = f"Paragraphs {start_i}–{end_i}"
            units.append(
                ExtractedUnit(
                    doc_name=doc_name,
                    source_type="docx",
                    loc_kind="paragraphs",
                    loc_start=anchor,
                    loc_end=anchor,
                    text=text,
                    path=docx_path,
                )
            )


# -----------------------------
# Unified interface (PDF + DOCX)
# -----------------------------
def extract_units_from_file(path: str) -> List[ExtractedUnit]:
    p = (path or "").replace("\\", "/")
    low = p.lower()
    if low.endswith(".pdf"):
        return extract_pdf_units(p)
    if low.endswith(".docx"):
        return extract_docx_units(p)
    return []


def extract_units_from_files(paths: List[str]) -> List[ExtractedUnit]:
    all_units: List[ExtractedUnit] = []
    for p in paths:
        all_units.extend(extract_units_from_file(p))
    return all_units
