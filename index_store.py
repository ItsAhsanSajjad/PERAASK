from __future__ import annotations

import os
import json
import time
import hashlib
import re
from typing import List, Dict, Any, Optional, Tuple

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import faiss  # type: ignore

from doc_registry import scan_assets_data, compare_with_manifest
from extractors import extract_units_from_files
from chunker import chunk_units, Chunk


# -----------------------------
# Config (env-tunable)
# -----------------------------
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")

# ✅ Bump defaults so improvements force a systematic rebuild unless env pins older versions.
# If you already set these in .env, update them there too.
EMBED_TEXT_VERSION = int(os.getenv("EMBED_TEXT_VERSION", "3"))
SEARCH_TEXT_VERSION = int(os.getenv("SEARCH_TEXT_VERSION", "2"))

MAX_EMBED_CHARS_PER_TEXT = int(os.getenv("MAX_EMBED_CHARS_PER_TEXT", "7000"))
MAX_EMBED_CHARS_PER_BATCH = int(os.getenv("MAX_EMBED_CHARS_PER_BATCH", "120000"))

DEFAULT_CHUNK_MAX_CHARS = int(os.getenv("CHUNK_MAX_CHARS", "4500"))
DEFAULT_CHUNK_OVERLAP_CHARS = int(os.getenv("CHUNK_OVERLAP_CHARS", "350"))

# ✅ Critical: if index files are missing/empty, force full ingest even if manifest says "unchanged"
FORCE_REBUILD_IF_INDEX_MISSING = os.getenv("FORCE_REBUILD_IF_INDEX_MISSING", "1").strip() != "0"


# -----------------------------
# Helpers: filesystem
# -----------------------------
def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)

def _p(index_dir: str, name: str) -> str:
    return os.path.join(index_dir, name).replace("\\", "/")

def _read_jsonl(path: str) -> List[Dict[str, Any]]:
    if not os.path.exists(path):
        return []
    out: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except Exception:
                continue
    return out

def _rewrite_jsonl(path: str, rows: List[Dict[str, Any]]) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    os.replace(tmp, path)

def _sha256_text(s: str) -> str:
    return hashlib.sha256((s or "").encode("utf-8", errors="ignore")).hexdigest()

def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()

def _safe_default_path(doc_name: str) -> str:
    doc_name = (doc_name or "").strip()
    if not doc_name:
        return ""
    return os.path.join("assets", "data", doc_name).replace("\\", "/")


# -----------------------------
# FAISS helpers
# -----------------------------
def _load_or_create_faiss(faiss_path: str, dim: int) -> faiss.Index:
    if os.path.exists(faiss_path):
        return faiss.read_index(faiss_path)
    base = faiss.IndexFlatIP(dim)
    return faiss.IndexIDMap(base)

def _save_faiss(index: faiss.Index, faiss_path: str) -> None:
    faiss.write_index(index, faiss_path)

def _normalize_vectors(v: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(v, axis=1, keepdims=True) + 1e-12
    return v / norms


# -----------------------------
# OpenAI embeddings (safe batching)
# -----------------------------
def _require_api_key() -> str:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise RuntimeError("OPENAI_API_KEY is missing. Ensure .env is present and loaded.")
    return key

def _truncate_text_for_embedding(t: str) -> str:
    t = (t or "").strip()
    if len(t) <= MAX_EMBED_CHARS_PER_TEXT:
        return t
    return t[:MAX_EMBED_CHARS_PER_TEXT]

def embed_texts(texts: List[str]) -> np.ndarray:
    if not texts:
        return np.zeros((0, 1), dtype=np.float32)

    from openai import OpenAI
    client = OpenAI(api_key=_require_api_key())

    safe_texts = [_truncate_text_for_embedding(t) for t in texts]

    all_vecs: List[List[float]] = []
    batch: List[str] = []
    batch_chars = 0

    def flush():
        nonlocal batch, batch_chars, all_vecs
        if not batch:
            return
        resp = client.embeddings.create(model=EMBEDDING_MODEL, input=batch)
        all_vecs.extend([d.embedding for d in resp.data])
        batch = []
        batch_chars = 0

    for t in safe_texts:
        tlen = len(t)
        if tlen >= MAX_EMBED_CHARS_PER_BATCH:
            flush()
            resp = client.embeddings.create(model=EMBEDDING_MODEL, input=[t])
            all_vecs.extend([d.embedding for d in resp.data])
            continue

        if batch_chars + tlen > MAX_EMBED_CHARS_PER_BATCH and batch:
            flush()

        batch.append(t)
        batch_chars += tlen

    flush()
    vectors = np.array(all_vecs, dtype=np.float32)
    return vectors


# -----------------------------
# Embedding/search text builders
# -----------------------------
def _loc_label(loc_kind: Any, loc_start: Any, loc_end: Any) -> str:
    lk = (loc_kind or "").strip()
    if lk == "page" and loc_start is not None:
        try:
            ps = int(loc_start)
            pe = int(loc_end) if loc_end is not None else ps
            return f"Page {ps}" if ps == pe else f"Pages {ps}-{pe}"
        except Exception:
            return f"Page {loc_start}"
    if loc_start is not None:
        return str(loc_start)
    return ""


# ✅ PERA identity + aliases for embeddings ONLY
_PERA_IDENTITY_LINE = (
    "ENTITY: PERA (Punjab Enforcement and Regulatory Authority, Punjab). "
    "ALIASES: pira, perra, peera, peraa. "
    "TOPICS: enforcement, regulation, scheduled laws, complaints, hearings, recruitment, HR, discipline, contracts."
)

# ✅ Richer tag patterns: improves retrieval for “complaint procedure”, “vision”, etc.
_TAG_PATTERNS: List[Tuple[str, str]] = [
    # composition / authority
    (r"\bshall\s+consist\b", "composition"),
    (r"\bconsist\s+of\b", "composition"),
    (r"\bcomposition\b", "composition"),
    (r"\bconstitution\b", "constitution"),
    (r"\bmember(s)?\b", "members"),
    (r"\bchairperson\b", "chairperson"),
    (r"\bvice\s+chairperson\b", "vice chairperson"),
    (r"\bsecretary\b", "secretary"),
    (r"\bdirector\s+general\b", "director general"),

    # complaint / hearings
    (r"\bcomplaint(s)?\b", "complaint"),
    (r"\bpublic\s+complaint(s)?\b", "public complaint"),
    (r"\bhearing(s)?\b", "hearing"),
    (r"\bhearing\s+officer\b", "hearing officer"),
    (r"\bgrievance\b", "grievance"),
    (r"\bappeal\b", "appeal"),
    (r"\bprocedure\b", "procedure"),
    (r"\bprocess\b", "process"),
    (r"\bhow\s+to\b", "how to"),

    # purpose / functions / mandate (vision-like queries map here)
    (r"\bpurpose\b", "purpose"),
    (r"\bobjective(s)?\b", "objectives"),
    (r"\bfunction(s)?\b", "functions"),
    (r"\bmandate\b", "mandate"),
    (r"\bestablished\s+to\b", "established to"),
    (r"\bvision\b", "vision"),
    (r"\bmission\b", "mission"),

    # HR / recruitment / criteria
    (r"\brecruitment\b", "recruitment"),
    (r"\beligibil", "eligibility"),
    (r"\bqualification(s)?\b", "qualification"),
    (r"\bexperience\b", "experience"),
    (r"\bcontract(ual)?\b", "contract"),
    (r"\bprobation\b", "probation"),
    (r"\btermination\b", "termination"),
    (r"\bdisciplin", "discipline"),
    (r"\bmisconduct\b", "misconduct"),

    # FAQ signals (important for complaint procedure in FAQ PDFs)
    (r"\bfaq\b", "faq"),
    (r"\bfrequently\s+asked\b", "faq"),
    (r"\bquestion(s)?\b", "questions"),
    (r"\banswer(s)?\b", "answers"),
]

def _derive_search_tags(raw_text: str) -> List[str]:
    t = (raw_text or "")
    tl = t.lower()
    tags: List[str] = []

    # Always include PERA tag if PERA appears or doc looks like PERA content
    if "pera" in tl or "punjab enforcement" in tl or "enforcement and regulatory" in tl:
        tags.append("pera")

    for pat, tag in _TAG_PATTERNS:
        if re.search(pat, t, flags=re.IGNORECASE):
            tags.append(tag)

    # De-dup preserving order
    seen = set()
    out: List[str] = []
    for x in tags:
        if x in seen:
            continue
        seen.add(x)
        out.append(x)
    return out


def _build_embed_text_from_parts(
    doc_name: str,
    doc_rank: Any,
    source_type: str,
    loc_kind: Any,
    loc_start: Any,
    loc_end: Any,
    raw_text: str,
) -> str:
    """
    ✅ Key improvement:
    Add PERA identity line + derived tags into embedding input (NOT into the raw evidence text).
    This makes semantic retrieval robust for typos & indirect wording like:
      - "complant procedure" -> complaint/hearing/FAQ
      - "vision of pera" -> purpose/objectives/functions/mandate
    """
    dn = (doc_name or "Unknown document").strip()
    stype = (source_type or "").strip()
    loc = _loc_label(loc_kind, loc_start, loc_end)
    rank = str(doc_rank) if doc_rank is not None else "0"

    body = (raw_text or "").strip()
    tags = _derive_search_tags(body)
    tags_line = f"TAGS: {', '.join(tags)}" if tags else "TAGS:"

    header = (
        f"DOCUMENT: {dn}\n"
        f"RANK: {rank}\n"
        f"TYPE: {stype}\n"
        f"LOCATION: {loc}\n"
        f"{tags_line}\n"
        f"{_PERA_IDENTITY_LINE}\n"
    )

    return (header + "\n" + body).strip()


def _build_embed_text_for_chunk(ch: Chunk) -> str:
    return _build_embed_text_from_parts(
        getattr(ch, "doc_name", "Unknown document"),
        getattr(ch, "doc_rank", 0),
        getattr(ch, "source_type", ""),
        getattr(ch, "loc_kind", ""),
        getattr(ch, "loc_start", None),
        getattr(ch, "loc_end", None),
        getattr(ch, "chunk_text", "") or "",
    )

def _build_embed_text_for_row(r: Dict[str, Any]) -> str:
    return _build_embed_text_from_parts(
        r.get("doc_name", "Unknown document"),
        r.get("doc_rank", 0),
        r.get("source_type", ""),
        r.get("loc_kind", ""),
        r.get("loc_start"),
        r.get("loc_end"),
        r.get("text", "") or "",
    )


def _build_search_text_from_parts(
    doc_name: str,
    source_type: str,
    loc_kind: Any,
    loc_start: Any,
    loc_end: Any,
    raw_text: str,
) -> str:
    dn = (doc_name or "Unknown document").strip()
    stype = (source_type or "").strip()
    loc = _loc_label(loc_kind, loc_start, loc_end)
    body = (raw_text or "").strip()

    tags = _derive_search_tags(body)
    tags_line = f"TAGS: {', '.join(tags)}" if tags else "TAGS:"

    # Keep search_text separate; retriever may choose to use it later
    header = f"DOC: {dn}\nTYPE: {stype}\nLOC: {loc}\n{tags_line}\n"
    return (header + "\n" + body).strip()

def _build_search_text_for_chunk(ch: Chunk) -> str:
    return _build_search_text_from_parts(
        getattr(ch, "doc_name", "Unknown document"),
        getattr(ch, "source_type", ""),
        getattr(ch, "loc_kind", ""),
        getattr(ch, "loc_start", None),
        getattr(ch, "loc_end", None),
        getattr(ch, "chunk_text", "") or "",
    )

def _build_search_text_for_row(r: Dict[str, Any]) -> str:
    return _build_search_text_from_parts(
        r.get("doc_name", "Unknown document"),
        r.get("source_type", ""),
        r.get("loc_kind", ""),
        r.get("loc_start"),
        r.get("loc_end"),
        r.get("text", "") or "",
    )


def _needs_embed_version_rebuild(rows: List[Dict[str, Any]]) -> bool:
    for r in rows:
        if not r.get("active", True):
            continue
        v = r.get("embed_text_version")
        if v is None or int(v) != int(EMBED_TEXT_VERSION):
            return True
    return False

def _needs_search_version_rebuild(rows: List[Dict[str, Any]]) -> bool:
    for r in rows:
        if not r.get("active", True):
            continue
        v = r.get("search_text_version")
        if v is None or int(v) != int(SEARCH_TEXT_VERSION):
            return True
    return False


# -----------------------------
# Chunk store / id assignment
# -----------------------------
def _next_chunk_id(existing_rows: List[Dict[str, Any]]) -> int:
    if not existing_rows:
        return 1
    return max(int(r.get("id", 0)) for r in existing_rows) + 1

def _mark_inactive_for_doc(rows: List[Dict[str, Any]], doc_name: str) -> int:
    n = 0
    now = int(time.time())
    for r in rows:
        if r.get("doc_name") == doc_name and r.get("active", True):
            r["active"] = False
            r["deactivated_at"] = now
            n += 1
    return n


# -----------------------------
# Manifest builder for cold-start rebuild
# -----------------------------
def _build_manifest_from_scanned(scanned: List[Dict[str, Any]]) -> Dict[str, Any]:
    files_map: Dict[str, Dict[str, Any]] = {}
    for e in scanned:
        try:
            sha = _sha256_file(e["path"])
        except Exception:
            sha = ""
        files_map[e["filename"]] = {
            "filename": e["filename"],
            "path": e["path"],
            "ext": e.get("ext", ""),
            "mtime": int(e.get("mtime", 0) or 0),
            "size": int(e.get("size", 0) or 0),
            "rank": int(e.get("rank", 0) or 0),
            "sha256": sha,
        }
    return {"version": 1, "files": files_map}


# -----------------------------
# Main: scan + incremental ingest
# -----------------------------
def scan_and_ingest_if_needed(
    data_dir: str = "assets/data",
    index_dir: str = "assets/index",
    manifest_name: str = "manifest.json",
    chunk_max_chars: int = DEFAULT_CHUNK_MAX_CHARS,
    chunk_overlap_chars: int = DEFAULT_CHUNK_OVERLAP_CHARS,
) -> Dict[str, Any]:
    _safe_mkdir(index_dir)

    faiss_path = _p(index_dir, "faiss.index")
    chunks_path = _p(index_dir, "chunks.jsonl")
    manifest_path = _p(index_dir, "manifest.json")

    scanned = scan_assets_data(data_dir=data_dir)

    rows = _read_jsonl(chunks_path)

    chunks_missing_or_empty = (not os.path.exists(chunks_path)) or (os.path.getsize(chunks_path) == 0)
    faiss_missing = not os.path.exists(faiss_path)
    cold_start = FORCE_REBUILD_IF_INDEX_MISSING and (faiss_missing or chunks_missing_or_empty)

    if cold_start:
        new_or_changed = list(scanned)
        unchanged: List[Dict[str, Any]] = []
        removed: List[Dict[str, Any]] = []
        updated_manifest = _build_manifest_from_scanned(scanned)

        rows = []
        start_id = 1
        deactivated = 0
    else:
        new_or_changed, unchanged, removed, updated_manifest = compare_with_manifest(
            scanned=scanned,
            index_dir=index_dir,
            manifest_name=manifest_name,
            compute_hash=True
        )

        if os.path.exists(faiss_path) and rows and (_needs_embed_version_rebuild(rows) or _needs_search_version_rebuild(rows)):
            _ = rebuild_index_from_chunks(index_dir=index_dir)
            rows = _read_jsonl(chunks_path)

        start_id = _next_chunk_id(rows)

        deactivated = 0
        for r in removed:
            docname = r.get("filename") or r.get("name")
            if docname:
                deactivated += _mark_inactive_for_doc(rows, docname)

        if not new_or_changed and os.path.exists(faiss_path):
            _rewrite_jsonl(chunks_path, rows)
            with open(manifest_path, "w", encoding="utf-8") as f:
                json.dump(updated_manifest, f, ensure_ascii=False, indent=2)
            return {
                "found": len(scanned),
                "new_or_changed": 0,
                "unchanged": len(unchanged),
                "removed": len(removed),
                "chunks_added": 0,
                "chunks_deactivated": deactivated,
                "faiss_vectors_added": 0,
                "cold_start_rebuild": False,
            }

    changed_files = [e["path"] for e in new_or_changed]
    changed_names = [e["filename"] for e in new_or_changed]

    for doc in changed_names:
        _ = _mark_inactive_for_doc(rows, doc)

    units = extract_units_from_files(changed_files)

    # Apply rank
    rank_map = {e["filename"]: int(e.get("rank", 0) or 0) for e in new_or_changed}
    for u in units:
        try:
            u.doc_rank = rank_map.get(u.doc_name, 0)
        except Exception:
            pass

    chunks: List[Chunk] = chunk_units(
        units,
        max_chars=chunk_max_chars,
        overlap_chars=chunk_overlap_chars
    )

    embed_text_list: List[str] = []
    kept_chunks: List[Chunk] = []
    for c in chunks:
        raw = (c.chunk_text or "").strip()
        if not raw:
            continue
        kept_chunks.append(c)
        embed_text_list.append(_build_embed_text_for_chunk(c))

    if not embed_text_list:
        _rewrite_jsonl(chunks_path, rows)
        with open(manifest_path, "w", encoding="utf-8") as f:
            json.dump(updated_manifest, f, ensure_ascii=False, indent=2)
        return {
            "found": len(scanned),
            "new_or_changed": len(new_or_changed),
            "unchanged": 0 if cold_start else len(unchanged),
            "removed": 0 if cold_start else len(removed),
            "chunks_added": 0,
            "chunks_deactivated": 0,
            "faiss_vectors_added": 0,
            "cold_start_rebuild": bool(cold_start),
            "note": "No chunks extracted from PDFs (check extractors/chunker).",
        }

    vectors = embed_texts(embed_text_list)
    vectors = _normalize_vectors(vectors)
    dim = vectors.shape[1]

    idx = _load_or_create_faiss(faiss_path, dim)

    if idx.d != dim:
        rebuilt = rebuild_index_from_chunks(index_dir=index_dir)
        idx = rebuilt["index"]
        rows = _read_jsonl(chunks_path)
        start_id = _next_chunk_id(rows)

    ids = np.arange(start_id, start_id + len(kept_chunks), dtype=np.int64)
    idx.add_with_ids(vectors, ids)

    now = int(time.time())
    new_rows: List[Dict[str, Any]] = []

    for cid, ch in zip(ids.tolist(), kept_chunks):
        t = (ch.chunk_text or "").strip()
        if not t:
            continue

        doc_name = getattr(ch, "doc_name", "Unknown document")
        path = (getattr(ch, "path", "") or "").strip() or _safe_default_path(doc_name)
        path = path.replace("\\", "/")

        embed_text = _build_embed_text_for_chunk(ch)
        search_text = _build_search_text_for_chunk(ch)

        new_rows.append({
            "id": int(cid),
            "active": True,
            "created_at": now,
            "doc_name": doc_name,
            "doc_rank": int(getattr(ch, "doc_rank", 0) or 0),
            "source_type": getattr(ch, "source_type", ""),
            "loc_kind": getattr(ch, "loc_kind", ""),
            "loc_start": getattr(ch, "loc_start", None),
            "loc_end": getattr(ch, "loc_end", None),
            "path": path,

            # ✅ RAW evidence text stays clean
            "text": t,
            "text_sha256": _sha256_text(t),

            # ✅ versions + embedding/search payload hashes
            "embed_text_version": int(EMBED_TEXT_VERSION),
            "embed_text_sha256": _sha256_text(embed_text),

            "search_text_version": int(SEARCH_TEXT_VERSION),
            "search_text_sha256": _sha256_text(search_text),
            "search_text": search_text,
        })

    rows.extend(new_rows)
    _rewrite_jsonl(chunks_path, rows)
    _save_faiss(idx, faiss_path)

    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(updated_manifest, f, ensure_ascii=False, indent=2)

    return {
        "found": len(scanned),
        "new_or_changed": len(new_or_changed),
        "unchanged": 0 if cold_start else len(unchanged),
        "removed": 0 if cold_start else len(removed),
        "chunks_added": len(new_rows),
        "chunks_deactivated": 0,
        "faiss_vectors_added": len(new_rows),
        "cold_start_rebuild": bool(cold_start),
    }


def rebuild_index_from_chunks(index_dir: str = "assets/index") -> Dict[str, Any]:
    faiss_path = _p(index_dir, "faiss.index")
    chunks_path = _p(index_dir, "chunks.jsonl")

    rows = _read_jsonl(chunks_path)
    active = [r for r in rows if r.get("active", True)]

    if not active:
        if os.path.exists(faiss_path):
            os.remove(faiss_path)
        empty_idx = faiss.IndexIDMap(faiss.IndexFlatIP(1))
        _save_faiss(empty_idx, faiss_path)
        return {"index": empty_idx, "rebuilt": True, "count": 0}

    embed_texts_list = [_build_embed_text_for_row(r) for r in active]
    vectors = embed_texts(embed_texts_list)
    vectors = _normalize_vectors(vectors)
    dim = vectors.shape[1]

    idx = faiss.IndexIDMap(faiss.IndexFlatIP(dim))
    ids = np.array([int(r["id"]) for r in active], dtype=np.int64)
    idx.add_with_ids(vectors, ids)
    _save_faiss(idx, faiss_path)

    changed = False
    for r, et in zip(active, embed_texts_list):
        if int(r.get("embed_text_version", -1)) != int(EMBED_TEXT_VERSION):
            r["embed_text_version"] = int(EMBED_TEXT_VERSION)
            r["embed_text_sha256"] = _sha256_text(et)
            changed = True

        stxt = _build_search_text_for_row(r)
        if int(r.get("search_text_version", -1)) != int(SEARCH_TEXT_VERSION) or not r.get("search_text"):
            r["search_text_version"] = int(SEARCH_TEXT_VERSION)
            r["search_text_sha256"] = _sha256_text(stxt)
            r["search_text"] = stxt
            changed = True

    if changed:
        _rewrite_jsonl(chunks_path, rows)

    return {"index": idx, "rebuilt": True, "count": len(active)}


def load_index_and_chunks(index_dir: str = "assets/index") -> Tuple[Optional[faiss.Index], List[Dict[str, Any]]]:
    faiss_path = _p(index_dir, "faiss.index")
    chunks_path = _p(index_dir, "chunks.jsonl")

    idx = None
    if os.path.exists(faiss_path):
        idx = faiss.read_index(faiss_path)

    rows = _read_jsonl(chunks_path)
    return idx, rows
