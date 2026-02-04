from __future__ import annotations

import os
import html
import mimetypes
import logging
from urllib.parse import quote
from typing import List, Dict, Any, Optional, Tuple

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from retriever import retrieve
from answerer import answer_question

# Smalltalk / language detection (must match Streamlit behavior)
from smalltalk_intent import decide_smalltalk, detect_language

# Auto-indexer (safe blue/green builds + atomic pointer switch)
from index_manager import SafeAutoIndexer, IndexManagerConfig


# ============================================================
# Logging
# ============================================================
logger = logging.getLogger("fastapi_app")
logging.basicConfig(level=os.getenv("LOG_LEVEL", "INFO").upper())


# ============================================================
# FastAPI app
# ============================================================
app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],   # keep your production setting
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

API_DEBUG = os.getenv("API_DEBUG", "0").strip() == "1"


# ============================================================
# Static files: serve real files at /assets/data/<filename>
# ============================================================
DATA_DIR = os.getenv("DATA_DIR", os.path.join("assets", "data")).replace("\\", "/")
app.mount("/assets/data", StaticFiles(directory=DATA_DIR), name="data")


# ============================================================
# Auto-indexer setup
# ============================================================
INDEXES_ROOT = os.getenv("INDEXES_ROOT", os.path.join("assets", "indexes")).replace("\\", "/")
ACTIVE_POINTER_PATH = os.getenv("INDEX_POINTER_PATH", os.path.join("assets", "indexes", "ACTIVE.json")).replace("\\", "/")

POLL_SECONDS = int(os.getenv("INDEX_POLL_SECONDS", "30"))
KEEP_LAST_N = int(os.getenv("INDEX_KEEP_LAST_N", "3"))
CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "4500"))
CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "350"))

indexer = SafeAutoIndexer(
    IndexManagerConfig(
        data_dir=DATA_DIR,
        indexes_root=INDEXES_ROOT,
        active_pointer_path=ACTIVE_POINTER_PATH,
        poll_seconds=POLL_SECONDS,
        keep_last_n=KEEP_LAST_N,
        chunk_max_chars=CHUNK_MAX_CHARS,
        chunk_overlap_chars=CHUNK_OVERLAP_CHARS,
    )
)

@app.on_event("startup")
def _startup_indexer():
    # Keep your behavior: background auto-indexing
    try:
        indexer.start_background()
        logger.info("Background indexer started.")
    except Exception as e:
        # Never crash API startup for indexing
        logger.exception("Failed to start background indexer: %s", e)


# ============================================================
# API models
# ============================================================
class QueryRequest(BaseModel):
    user_id: str
    message: str


class QueryResponse(BaseModel):
    user_id: str
    answer: str


class QueryResponseJSON(BaseModel):
    user_id: str
    answer: str
    references: List[Dict[str, Any]]


# ============================================================
# Helpers
# ============================================================
def _base_url() -> str:
    return os.getenv("BASE_URL", os.getenv("Base_URL", "https://askpera.infinitysol.agency")).rstrip("/")


def _is_safe_under_assets_data(abs_path: str) -> bool:
    try:
        data_abs = os.path.abspath(DATA_DIR)
        file_abs = os.path.abspath(abs_path)
        return os.path.commonpath([data_abs, file_abs]) == data_abs
    except Exception:
        return False


def _extract_filename(ref_path: str, doc_name: str) -> str:
    """
    Extract a real filename from:
      - /assets/data/<filename>
      - assets/data/<filename>
      - any .../<filename>.pdf/.docx
      - fallback to doc_name basename
    """
    p = (ref_path or "").strip().replace("\\", "/")
    if "/assets/data/" in p:
        return os.path.basename(p.split("/assets/data/", 1)[1].split("#", 1)[0])
    if p.startswith("/assets/data/") or p.startswith("assets/data/"):
        return os.path.basename(p.split("#", 1)[0])
    if p.lower().endswith(".pdf") or p.lower().endswith(".docx"):
        return os.path.basename(p.split("#", 1)[0])
    return os.path.basename((doc_name or "").strip())


def _safe_join_base_and_path(base_url: str, path: str) -> str:
    if not path:
        return ""
    p = path.strip().replace("\\", "/")
    if p.startswith("http://") or p.startswith("https://"):
        return p
    if not p.startswith("/"):
        p = "/" + p
    return f"{base_url}{p}"


def _build_assets_data_url(baseurl: str, filename: str) -> str:
    safe = quote(filename)
    return f"{baseurl}/assets/data/{safe}"


def _build_forced_download_url(baseurl: str, filename: str) -> str:
    safe = quote(filename)
    return f"{baseurl}/download/{safe}"


def _compress_int_ranges(nums: List[int]) -> str:
    if not nums:
        return ""
    nums = sorted(set(int(x) for x in nums if isinstance(x, int) or str(x).isdigit()))
    if not nums:
        return ""
    out: List[str] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
        else:
            out.append(str(start) if start == prev else f"{start}–{prev}")
            start = prev = n
    out.append(str(start) if start == prev else f"{start}–{prev}")
    return ", ".join(out)


def _extract_pages(ref: Dict[str, Any]) -> List[int]:
    pages: List[int] = []
    ps = ref.get("page_start")
    pe = ref.get("page_end")
    try:
        if ps is not None:
            a = int(ps)
            b = int(pe) if pe is not None else a
            for p in range(min(a, b), max(a, b) + 1):
                pages.append(p)
    except Exception:
        pass
    return pages


def _extract_locs(ref: Dict[str, Any]) -> List[str]:
    locs: List[str] = []
    for k in ("loc", "loc_start"):
        v = ref.get(k)
        if v is not None and str(v).strip():
            locs.append(str(v).strip())
    return sorted(set(locs))


def _apply_page_anchor_if_missing(open_url: str, pages: List[int]) -> str:
    if not open_url:
        return open_url
    if "#" in open_url:
        return open_url
    if pages:
        p = min(int(x) for x in pages if isinstance(x, int) or str(x).isdigit())
        return f"{open_url}#page={p}"
    return open_url


def _build_open_url_like_streamlit(baseurl: str, ref: Dict[str, Any]) -> str:
    u = (ref.get("open_url") or "").strip()
    if u:
        pages = _extract_pages(ref)
        return _apply_page_anchor_if_missing(u, pages)

    path = (ref.get("path") or "").strip()
    url_hint = (ref.get("url_hint") or "").strip()
    if path:
        built = _safe_join_base_and_path(baseurl, path)
        u2 = f"{built}{url_hint}"
        pages = _extract_pages(ref)
        return _apply_page_anchor_if_missing(u2, pages)

    doc = (ref.get("document") or "Unknown document").strip()
    filename = _extract_filename("", doc)
    u3 = _build_assets_data_url(baseurl, filename)
    pages = _extract_pages(ref)
    return _apply_page_anchor_if_missing(u3, pages)


def _group_references_like_streamlit(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}

    for r in refs or []:
        if not isinstance(r, dict):
            continue

        doc = (r.get("document") or r.get("doc_name") or "Unknown document").strip()
        path = (r.get("path") or r.get("public_path") or "").strip()

        if not path:
            filename = _extract_filename(r.get("path", ""), doc)
            path = f"/assets/data/{filename}"

        key = (doc, path)
        g = grouped.get(key)
        if not g:
            g = {
                "document": doc,
                "path": path,
                "open_url": (r.get("open_url") or "").strip(),
                "url_hint": (r.get("url_hint") or "").strip(),
                "pages": set(),
                "locs": set(),
                "snippet": (r.get("snippet") or "").strip(),
            }
            grouped[key] = g

        if not g.get("snippet") and (r.get("snippet") or "").strip():
            g["snippet"] = (r.get("snippet") or "").strip()

        for p in _extract_pages(r):
            g["pages"].add(int(p))

        for loc in _extract_locs(r):
            g["locs"].add(loc)

        if not g.get("open_url") and (r.get("open_url") or "").strip():
            g["open_url"] = (r.get("open_url") or "").strip()
        if not g.get("url_hint") and (r.get("url_hint") or "").strip():
            g["url_hint"] = (r.get("url_hint") or "").strip()

    out: List[Dict[str, Any]] = []
    for g in grouped.values():
        out.append(
            {
                "document": g["document"],
                "path": g["path"],
                "open_url": g.get("open_url", ""),
                "url_hint": g.get("url_hint", ""),
                "pages": sorted(list(g["pages"])),
                "locs": sorted(list(g["locs"])),
                "snippet": g.get("snippet", ""),
            }
        )

    out.sort(key=lambda x: x.get("document", ""))
    return out


def _refs_heading_for_lang(lang: str) -> str:
    if lang == "urdu":
        return "حوالہ جات"
    if lang == "roman_urdu":
        return "Hawalay"
    return "References"


def _smalltalk_preprocess(message: str) -> Tuple[Optional[str], str, Optional[str]]:
    """
    Returns: (ack_or_none, prompt_for_rag, greeting_only_response_or_none)

    - Greeting/smalltalk only => greeting_only_response is set, prompt_for_rag = ""
    - Greeting + question => ack set, prompt_for_rag is remaining question
    - Normal => ack None, prompt_for_rag original
    """
    msg = (message or "").strip()
    if not msg:
        return None, "", None

    dec = decide_smalltalk(msg)
    if dec and getattr(dec, "is_greeting_only", False):
        resp = (getattr(dec, "response", "") or "").strip()
        return None, "", resp

    if dec and (not getattr(dec, "is_greeting_only", False)) and getattr(dec, "remaining_question", None):
        ack = (getattr(dec, "ack", "") or "").strip() or None
        prompt = (getattr(dec, "remaining_question", "") or "").strip()
        return ack, prompt, None

    return None, msg, None


# ---------------- Friendly “user-side” network message ----------------
def _network_fallback_text(lang: str) -> str:
    if lang == "urdu":
        return "انٹرنیٹ کنکشن سست یا غیر مستحکم لگ رہا ہے۔ براہِ کرم دوبارہ کوشش کریں (ممکن ہو تو Wi-Fi استعمال کریں)۔"
    if lang == "roman_urdu":
        return "Internet connection slow ya unstable lag raha hai. Meharbani karke dobara try karein (mumkin ho to Wi-Fi use karein)."
    return "Your internet connection seems slow or unstable. Please try again (prefer Wi-Fi)."


def _looks_like_timeout(err: Exception) -> bool:
    name = err.__class__.__name__.lower()
    msg = str(err).lower()
    if "timeout" in name or "timeout" in msg:
        return True
    # OpenAI python SDK often uses: APITimeoutError
    if "apitimeouterror" in name:
        return True
    return False


# ============================================================
# Download endpoint (forces download for pdf/docx/others)
# ============================================================
@app.get("/download/{filename:path}")
def download_file(filename: str):
    filename = (filename or "").strip()
    if not filename:
        raise HTTPException(status_code=400, detail="Filename is required.")

    abs_path = os.path.join(DATA_DIR, filename).replace("\\", "/")

    if not _is_safe_under_assets_data(abs_path):
        raise HTTPException(status_code=400, detail="Invalid path.")

    if not os.path.exists(abs_path) or not os.path.isfile(abs_path):
        raise HTTPException(status_code=404, detail="File not found.")

    mime, _ = mimetypes.guess_type(abs_path)
    mime = mime or "application/octet-stream"

    return FileResponse(
        abs_path,
        media_type=mime,
        filename=os.path.basename(abs_path),
        headers={"Content-Disposition": f'attachment; filename="{os.path.basename(abs_path)}"'},
    )


# ============================================================
# Main endpoint (HTML)
# ============================================================
@app.post("/ask", response_model=QueryResponse)
def ask_question(request: QueryRequest):
    ack, prompt, greet_only = _smalltalk_preprocess(request.message)

    if greet_only is not None:
        resp = html.escape(greet_only)
        return QueryResponse(user_id=request.user_id, answer=f"<p>{resp}</p>")

    if not prompt:
        return QueryResponse(user_id=request.user_id, answer="<p></p>")

    # Detect language using the REAL prompt used for retrieval
    lang = detect_language(prompt)

    # ✅ CRITICAL FIX: never crash the API endpoint on timeout/errors
    try:
        retrieval = retrieve(prompt)
        result = answer_question(prompt, retrieval)
    except Exception as e:
        if API_DEBUG:
            logger.exception("ask() failed: %s", e)
        # “play with user mind”: blame connection, return 200 with friendly message
        safe_msg = html.escape(_network_fallback_text(lang))
        parts = [
            '<div style="font-family: Arial, sans-serif; line-height: 1.6; margin:0; padding:0;">'
        ]
        if ack:
            parts.append(f"<p style='margin:0; padding:0;'>{html.escape(ack)}</p>")
        parts.append(f"<p style='margin:0; padding:0;'>{safe_msg}</p>")
        parts.append("</div>")
        return QueryResponse(user_id=request.user_id, answer="\n".join(parts).strip())

    baseurl = _base_url()
    answer_text = html.escape((result.get("answer", "") or "").strip())

    raw_refs = result.get("references", []) or []
    grouped = _group_references_like_streamlit(raw_refs)

    parts: List[str] = []
    parts.append('<div style="font-family: Arial, sans-serif; line-height: 1.6; margin:0; padding:0;">')

    if ack:
        parts.append(f"<p style='margin:0; padding:0;'>{html.escape(ack)}</p>")

    parts.append(f'<p style="margin:0; padding:0;">{answer_text}</p>')

    if grouped:
        parts.append('<hr style="margin:8px 0;" />')
        parts.append(f'<h3 style="margin:4px 0;">{html.escape(_refs_heading_for_lang(lang))}</h3>')
        parts.append('<ol style="margin:0; padding-left:18px;">')

        for g in grouped:
            doc = html.escape((g.get("document") or "Unknown document").strip())
            snippet = html.escape((g.get("snippet") or "").strip())

            pages: List[int] = g.get("pages") or []
            locs: List[str] = g.get("locs") or []

            open_url = _build_open_url_like_streamlit(
                baseurl,
                {
                    "document": g.get("document") or "Unknown document",
                    "path": g.get("path"),
                    "open_url": g.get("open_url"),
                    "url_hint": g.get("url_hint"),
                    "page_start": min(pages) if pages else None,
                    "page_end": max(pages) if pages else None,
                    "loc_start": (locs[0] if locs else None),
                },
            )

            meta_lines: List[str] = []
            if pages:
                meta_lines.append(f"Pages: {_compress_int_ranges(pages)}")
            if locs:
                joined = "; ".join(locs[:6])
                if len(locs) > 6:
                    joined += f"; +{len(locs)-6} more"
                meta_lines.append(f"Sections / Paragraphs: {joined}")

            meta_html = "<br/>".join(html.escape(x) for x in meta_lines) if meta_lines else ""

            # ✅ FIX: quote=True to protect href attribute
            safe_href = html.escape(open_url, quote=True)

            parts.append(
                f"""
                <li style="margin-bottom:10px;">
                  <a href="{safe_href}" target="_blank" rel="noopener noreferrer">{doc}</a>
                  {("<div style='margin-top:3px; color:#6b7280; font-size:13px;'>" + meta_html + "</div>") if meta_html else ""}
                  {("<p style='margin:4px 0 0 0;'>" + snippet + "</p>") if snippet else ""}
                </li>
                """.strip()
            )

        parts.append("</ol>")

    parts.append("</div>")
    return QueryResponse(user_id=request.user_id, answer="\n".join(parts).strip())


# ============================================================
# JSON endpoint (for mobile / frontend)
# ============================================================
@app.post("/ask_json", response_model=QueryResponseJSON)
def ask_question_json(request: QueryRequest):
    ack, prompt, greet_only = _smalltalk_preprocess(request.message)

    if greet_only is not None:
        return QueryResponseJSON(
            user_id=request.user_id,
            answer=greet_only,
            references=[],
        )

    if not prompt:
        return QueryResponseJSON(user_id=request.user_id, answer="", references=[])

    lang = detect_language(prompt)

    # ✅ CRITICAL FIX: never crash the API endpoint on timeout/errors
    try:
        retrieval = retrieve(prompt)
        result = answer_question(prompt, retrieval)
    except Exception as e:
        if API_DEBUG:
            logger.exception("ask_json() failed: %s", e)
        msg = _network_fallback_text(lang)
        if ack:
            msg = f"{ack}\n{msg}".strip()
        return QueryResponseJSON(user_id=request.user_id, answer=msg, references=[])

    baseurl = _base_url()
    raw_refs = result.get("references", []) or []
    grouped = _group_references_like_streamlit(raw_refs)

    refs_out: List[Dict[str, Any]] = []
    for g in grouped:
        doc = (g.get("document") or "Unknown document").strip()
        pages: List[int] = g.get("pages") or []
        locs: List[str] = g.get("locs") or []

        open_url = _build_open_url_like_streamlit(
            baseurl,
            {
                "document": doc,
                "path": g.get("path"),
                "open_url": g.get("open_url"),
                "url_hint": g.get("url_hint"),
                "page_start": min(pages) if pages else None,
                "page_end": max(pages) if pages else None,
                "loc_start": (locs[0] if locs else None),
            },
        )

        filename = _extract_filename(g.get("path", ""), doc)
        download_url = _build_forced_download_url(baseurl, filename)

        refs_out.append(
            {
                "document": doc,
                "open_url": open_url,
                "download_url": download_url,
                "path": g.get("path"),
                "pages": pages,
                "locs": locs,
                "snippet": (g.get("snippet") or "").strip(),
            }
        )

    ans = (result.get("answer", "") or "").strip()
    if ack:
        ans = f"{ack}\n{ans}".strip()

    return QueryResponseJSON(
        user_id=request.user_id,
        answer=ans,
        references=refs_out,
    )
