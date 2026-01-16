from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import List, Optional, Dict, Any, Iterable, Union


# -----------------------------
# Data structures
# -----------------------------

@dataclass
class ExtractedUnit:
    """
    A traceable extraction unit that can later be chunked.
    - PDF => one unit per page
    - DOCX => one unit per section (heading) or paragraph-range block
    """
    doc_name: str
    source_type: str              # "pdf" | "docx"
    loc_kind: str                 # "page" | "section" | "paragraphs" | "xml"
    loc_start: Any
    loc_end: Any
    text: str

    # optional metadata
    path: Optional[str] = None
    doc_rank: int = 0             # optional; can be filled by registry later


# -----------------------------
# Helpers
# -----------------------------

SUPPORTED_EXTS = (".pdf", ".docx")


def _clean_text(s: str) -> str:
    s = s or ""
    s = s.replace("\x00", " ")
    s = re.sub(r"[ \t]+", " ", s)
    s = re.sub(r"\n{3,}", "\n\n", s)
    return s.strip()


def _is_heading_style(style_name: str) -> bool:
    if not style_name:
        return False
    sn = style_name.strip().lower()
    return sn.startswith("heading")  # "Heading 1", "Heading 2", etc.


def discover_documents(data_dir: str = "assets/data") -> List[str]:
    """
    Utility: list PDF/DOCX files in assets/data. (Used in testing)
    Excludes MS Word temp lock files (~$*.docx).
    """
    data_dir = data_dir.replace("\\", "/")
    if not os.path.isdir(data_dir):
        return []
    out: List[str] = []
    for name in os.listdir(data_dir):
        if name.startswith("~$"):
            continue
        p = os.path.join(data_dir, name).replace("\\", "/")
        if not os.path.isfile(p):
            continue
        low = name.lower()
        if low.endswith(SUPPORTED_EXTS):
            out.append(p)
    return sorted(out)


# -----------------------------
# PDF Extraction
# -----------------------------

def extract_pdf_units(pdf_path: str) -> List[ExtractedUnit]:
    """
    Extract PDF page-by-page.
    Each unit includes:
      doc_name, type="pdf", page=1..N, text
    """
    units: List[ExtractedUnit] = []
    doc_name = os.path.basename(pdf_path)

    try:
        from pypdf import PdfReader
        reader = PdfReader(pdf_path)
        pages = reader.pages
    except Exception:
        return units

    for i, page in enumerate(pages):
        page_no = i + 1  # 1-indexed page numbers
        try:
            txt = page.extract_text() or ""
        except Exception:
            txt = ""

        txt = _clean_text(txt)
        if not txt:
            continue

        units.append(
            ExtractedUnit(
                doc_name=doc_name,
                source_type="pdf",
                loc_kind="page",
                loc_start=page_no,
                loc_end=page_no,
                text=txt,
                path=pdf_path.replace("\\", "/"),
            )
        )

    return units


# -----------------------------
# DOCX Extraction
# -----------------------------
# Key fix:
# - python-docx's doc.paragraphs does NOT include table text OR textbox/shape text.
# - Many government/job docs store core content in tables and/or textboxes/shapes.
# This extractor reads:
#   ✅ paragraphs + tables in document order
#   ✅ headers/footers
#   ✅ XML fallback for textboxes / content controls (w:txbxContent, w:sdt, etc.)

def _iter_block_items(doc) -> Iterable[Union["Paragraph", "Table"]]:
    """
    Yield paragraphs and tables in document order.
    Standard python-docx approach using the underlying XML.
    """
    from docx.table import Table
    from docx.text.paragraph import Paragraph

    body_elm = doc.element.body
    for child in body_elm.iterchildren():
        if child.tag.endswith("}p"):
            yield Paragraph(child, doc)
        elif child.tag.endswith("}tbl"):
            yield Table(child, doc)


def _table_to_text(tbl) -> str:
    """
    Convert a docx table to readable text deterministically.
    We join cell texts with " | " and rows with newlines.
    """
    rows_out: List[str] = []
    try:
        for row in tbl.rows:
            cells = []
            for cell in row.cells:
                t = _clean_text(cell.text or "")
                if t:
                    cells.append(t)
            if cells:
                rows_out.append(" | ".join(cells))
    except Exception:
        return ""
    return _clean_text("\n".join(rows_out))


def _extract_header_footer_text(doc) -> str:
    """
    Extract paragraphs from headers/footers (common for titles, references).
    """
    out: List[str] = []
    try:
        for sec in getattr(doc, "sections", []):
            for part in [getattr(sec, "header", None), getattr(sec, "footer", None)]:
                if not part:
                    continue
                for p in getattr(part, "paragraphs", []):
                    t = _clean_text(getattr(p, "text", "") or "")
                    if t:
                        out.append(t)
                for tbl in getattr(part, "tables", []):
                    t = _table_to_text(tbl)
                    if t:
                        out.append(t)
    except Exception:
        pass
    return _clean_text("\n".join(out))


def _extract_docx_xml_text(doc) -> str:
    """
    Robust fallback: extract visible text from the DOCX XML.
    This captures many textboxes/shapes (w:txbxContent), content controls (w:sdt),
    and other text runs not exposed via python-docx high-level API.

    We keep this as a fallback (or supplement) to avoid duplicates.
    """
    try:
        root = doc.part._element  # lxml element
    except Exception:
        return ""

    # Namespaces are embedded; xpath with local-name() is more robust here.
    # Collect w:t nodes from:
    # - regular paragraphs/tables: //w:t
    # - textboxes: //w:txbxContent//w:t
    # - content controls: //w:sdt//w:t
    try:
        texts: List[str] = []
        # Focus on textbox + sdt first (least likely to be captured elsewhere)
        for node in root.xpath(".//*[local-name()='txbxContent']//*[local-name()='t']"):
            t = node.text or ""
            if t:
                texts.append(t)
        for node in root.xpath(".//*[local-name()='sdt']//*[local-name()='t']"):
            t = node.text or ""
            if t:
                texts.append(t)

        # If still empty, fall back to all runs
        if not texts:
            for node in root.xpath(".//*[local-name()='t']"):
                t = node.text or ""
                if t:
                    texts.append(t)

        return _clean_text("\n".join(texts))
    except Exception:
        return ""


def extract_docx_units(
    docx_path: str,
    min_chars_per_unit: int = 800,
    max_chars_per_unit: int = 6000
) -> List[ExtractedUnit]:
    """
    Extract DOCX into stable "section/page equivalent" units.

    Strategy:
      - Use Heading 1/2/3 as section boundaries.
      - Include TABLE text (critical).
      - Include header/footer text (often titles/refs).
      - XML fallback for textboxes/shapes/content controls when body yields little/no text.
      - If no headings, create blocks by paragraph ranges.

    Output units:
      source_type="docx"
      loc_kind="section" or "paragraphs" or "xml"
      loc_start/loc_end: anchor strings
    """
    units: List[ExtractedUnit] = []
    docx_path = (docx_path or "").replace("\\", "/")
    doc_name = os.path.basename(docx_path)

    # ✅ Skip Word temp/lock files
    if doc_name.startswith("~$"):
        return units

    try:
        from docx import Document
        doc = Document(docx_path)
    except Exception:
        return units

    # Extract header/footer (optional enrichment)
    hf_text = _extract_header_footer_text(doc)

    # Gather content blocks (paragraphs + tables) in order
    paras: List[Dict[str, Any]] = []
    para_idx = 0
    current_heading = ""

    # Iterate in document order
    try:
        blocks = list(_iter_block_items(doc))
    except Exception:
        blocks = list(getattr(doc, "paragraphs", []))

    for b in blocks:
        # Paragraph
        if hasattr(b, "text") and not hasattr(b, "rows"):
            txt = (getattr(b, "text", "") or "").strip()
            if not txt:
                continue

            style_name = ""
            try:
                st = getattr(b, "style", None)
                style_name = st.name if st else ""
            except Exception:
                style_name = ""

            if _is_heading_style(style_name):
                current_heading = txt
                para_idx += 1
                paras.append({"i": para_idx, "heading": current_heading or "Untitled", "text": txt})
                continue

            para_idx += 1
            paras.append({"i": para_idx, "heading": current_heading, "text": txt})
            continue

        # Table
        if hasattr(b, "rows"):
            ttxt = _table_to_text(b)
            if not ttxt:
                continue

            para_idx += 1
            paras.append({"i": para_idx, "heading": current_heading, "text": ttxt})
            continue

    body_text_len = sum(len(p.get("text", "") or "") for p in paras)

    # ✅ If body extraction is empty OR suspiciously small, use XML fallback (textboxes/shapes)
    xml_text = ""
    if body_text_len < 200:  # small threshold to catch "all in textbox" docs
        xml_text = _extract_docx_xml_text(doc)

    # If we still have nothing, try XML anyway (some docs are purely in shapes)
    if not paras and not xml_text:
        xml_text = _extract_docx_xml_text(doc)

    # If header/footer exists, prepend it once as context (lightweight)
    if hf_text and (paras or xml_text):
        # Add as a small synthetic paragraph block so it participates in section grouping
        para_idx += 1
        paras.insert(0, {"i": 1, "heading": "Header/Footer", "text": hf_text})

    # If body blocks exist, proceed with normal grouping
    if paras:
        has_any_heading = any(p.get("heading") for p in paras)
        if has_any_heading:
            order: List[str] = []
            groups: Dict[str, List[Dict[str, Any]]] = {}

            for p in paras:
                h = p.get("heading") or "Untitled"
                if h not in groups:
                    groups[h] = []
                    order.append(h)
                groups[h].append(p)

            for heading in order:
                _emit_docx_group_as_units(
                    units=units,
                    doc_name=doc_name,
                    docx_path=docx_path,
                    heading=heading,
                    items=groups[heading],
                    min_chars=min_chars_per_unit,
                    max_chars=max_chars_per_unit
                )
        else:
            _emit_docx_paragraph_blocks(
                units=units,
                doc_name=doc_name,
                docx_path=docx_path,
                items=paras,
                min_chars=min_chars_per_unit,
                max_chars=max_chars_per_unit
            )

    # If normal extraction produced nothing useful but XML has text, emit XML unit(s)
    if not units and xml_text:
        # Chunk XML text into units ~ max_chars_per_unit
        txt = xml_text
        if len(txt) <= max_chars_per_unit:
            units.append(
                ExtractedUnit(
                    doc_name=doc_name,
                    source_type="docx",
                    loc_kind="xml",
                    loc_start="XML extracted text",
                    loc_end="XML extracted text",
                    text=txt,
                    path=docx_path,
                )
            )
        else:
            start = 0
            part = 1
            while start < len(txt):
                chunk = _clean_text(txt[start:start + max_chars_per_unit])
                if chunk:
                    anchor = f"XML extracted text (part {part})"
                    units.append(
                        ExtractedUnit(
                            doc_name=doc_name,
                            source_type="docx",
                            loc_kind="xml",
                            loc_start=anchor,
                            loc_end=anchor,
                            text=chunk,
                            path=docx_path,
                        )
                    )
                start += max_chars_per_unit
                part += 1

    return units


def _emit_docx_group_as_units(
    units: List[ExtractedUnit],
    doc_name: str,
    docx_path: str,
    heading: str,
    items: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int
) -> None:
    buffer: List[str] = []
    start_i = None
    end_i = None
    char_count = 0

    for p in items:
        txt = p["text"]
        i = p["i"]

        if start_i is None:
            start_i = i
        end_i = i

        buffer.append(txt)
        char_count += len(txt) + 1

        if char_count >= max_chars:
            text = _clean_text("\n".join(buffer))
            if text:
                anchor = f'Section: "{heading}" (Paragraphs {start_i}–{end_i})'
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

    if buffer:
        text = _clean_text("\n".join(buffer))
        if text:
            anchor = f'Section: "{heading}" (Paragraphs {start_i}–{end_i})'
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


def _emit_docx_paragraph_blocks(
    units: List[ExtractedUnit],
    doc_name: str,
    docx_path: str,
    items: List[Dict[str, Any]],
    min_chars: int,
    max_chars: int
) -> None:
    buffer: List[str] = []
    start_i = None
    end_i = None
    char_count = 0

    for p in items:
        txt = p["text"]
        i = p["i"]

        if start_i is None:
            start_i = i
        end_i = i

        buffer.append(txt)
        char_count += len(txt) + 1

        if char_count >= max_chars:
            text = _clean_text("\n".join(buffer))
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
        text = _clean_text("\n".join(buffer))
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
    low = (path or "").lower()
    base = os.path.basename(path or "")
    if base.startswith("~$"):
        return []

    if low.endswith(".pdf"):
        return extract_pdf_units(path)
    if low.endswith(".docx"):
        return extract_docx_units(path)
    return []


def extract_units_from_files(paths: List[str]) -> List[ExtractedUnit]:
    all_units: List[ExtractedUnit] = []
    for p in paths:
        all_units.extend(extract_units_from_file(p))
    return all_units
