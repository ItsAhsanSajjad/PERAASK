from __future__ import annotations

import os
import json
import hashlib
import re
from typing import List, Dict, Tuple, Optional


SUPPORTED_EXTS = (".pdf", ".docx")


def _safe_mkdir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_book_rank(filename: str) -> int:
    base = os.path.splitext(os.path.basename(filename))[0]
    m = re.search(r"\bbook\s*([0-9]+)\b", base, flags=re.IGNORECASE)
    if not m:
        return 0
    try:
        return int(m.group(1))
    except Exception:
        return 0


def scan_assets_data(data_dir: str = "assets/data") -> List[Dict]:
    """
    Scans assets/data for .pdf and .docx (top-level only).
    Excludes MS Word temp lock files (~$*.docx).
    """
    data_dir = data_dir.replace("\\", "/")
    if not os.path.isdir(data_dir):
        return []

    out: List[Dict] = []
    for name in os.listdir(data_dir):
        if name.startswith("~$"):
            continue

        full = os.path.join(data_dir, name).replace("\\", "/")
        if not os.path.isfile(full):
            continue

        low = name.lower()
        if not low.endswith(SUPPORTED_EXTS):
            continue

        out.append({
            "path": full,
            "filename": name,
            "ext": os.path.splitext(name)[1].lower(),
            "mtime": int(os.path.getmtime(full)),
            "size": int(os.path.getsize(full)),
            "rank": _parse_book_rank(name),
        })

    out.sort(key=lambda x: (-x["rank"], x["filename"].lower()))
    return out


def _load_manifest(manifest_path: str) -> Dict:
    if not os.path.exists(manifest_path):
        return {"version": 1, "files": {}}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"version": 1, "files": {}}


def _save_manifest(manifest_path: str, manifest: Dict) -> None:
    tmp = manifest_path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    os.replace(tmp, manifest_path)


def compare_with_manifest(
    scanned: List[Dict],
    index_dir: str = "assets/index",
    manifest_name: str = "manifest.json",
    compute_hash: bool = True
) -> Tuple[List[Dict], List[Dict], List[Dict], Dict]:
    """
    Returns (new_or_changed, unchanged, removed, updated_manifest)
    """
    index_dir = index_dir.replace("\\", "/")
    _safe_mkdir(index_dir)

    manifest_path = os.path.join(index_dir, manifest_name).replace("\\", "/")
    manifest = _load_manifest(manifest_path)
    mf_files: Dict[str, Dict] = manifest.get("files", {})

    current_keys = set()
    new_or_changed: List[Dict] = []
    unchanged: List[Dict] = []

    sha_cache: Dict[str, str] = {}

    def get_sha(p: str) -> Optional[str]:
        if not compute_hash:
            return None
        if p in sha_cache:
            return sha_cache[p]
        sha_cache[p] = _sha256_file(p)
        return sha_cache[p]

    for entry in scanned:
        key = entry["filename"]
        current_keys.add(key)

        prev = mf_files.get(key)
        entry_hash = get_sha(entry["path"])

        entry_out = dict(entry)
        if entry_hash:
            entry_out["sha256"] = entry_hash

        if prev is None:
            new_or_changed.append(entry_out)
            continue

        changed = False
        if compute_hash and prev.get("sha256") != entry_hash:
            changed = True
        elif prev.get("size") != entry["size"]:
            changed = True
        elif prev.get("mtime") != entry["mtime"]:
            changed = True

        if changed:
            new_or_changed.append(entry_out)
        else:
            unchanged.append(entry_out)

    removed: List[Dict] = []
    for key, prev in mf_files.items():
        if key not in current_keys:
            removed.append(prev)

    new_files_map: Dict[str, Dict] = {}
    for e in scanned:
        sha = get_sha(e["path"]) if compute_hash else mf_files.get(e["filename"], {}).get("sha256")
        new_files_map[e["filename"]] = {
            "filename": e["filename"],
            "path": e["path"],
            "ext": e["ext"],
            "mtime": e["mtime"],
            "size": e["size"],
            "rank": e["rank"],
            "sha256": sha,
        }

    updated_manifest = {"version": 1, "files": new_files_map}
    return new_or_changed, unchanged, removed, updated_manifest


def scan_status_only(
    data_dir: str = "assets/data",
    index_dir: str = "assets/index",
    manifest_name: str = "manifest.json",
) -> Dict:
    """
    UI-safe scan: does NOT write manifest.
    """
    scanned = scan_assets_data(data_dir=data_dir)
    new_or_changed, unchanged, removed, _updated_manifest = compare_with_manifest(
        scanned=scanned,
        index_dir=index_dir,
        manifest_name=manifest_name,
        compute_hash=True
    )

    manifest_path = os.path.join(index_dir, manifest_name).replace("\\", "/")
    return {
        "found": len(scanned),
        "new_or_changed": len(new_or_changed),
        "unchanged": len(unchanged),
        "removed": len(removed),
        "new_or_changed_files": new_or_changed,
        "removed_files": removed,
        "data_dir": data_dir,
        "manifest_path": manifest_path,
    }


def scan_and_update_manifest(
    data_dir: str = "assets/data",
    index_dir: str = "assets/index",
    manifest_name: str = "manifest.json",
) -> Dict:
    """
    This writes manifest. Use ONLY during ingestion/index build.
    """
    scanned = scan_assets_data(data_dir=data_dir)
    new_or_changed, unchanged, removed, updated_manifest = compare_with_manifest(
        scanned=scanned,
        index_dir=index_dir,
        manifest_name=manifest_name,
        compute_hash=True
    )

    manifest_path = os.path.join(index_dir, manifest_name).replace("\\", "/")
    _save_manifest(manifest_path, updated_manifest)

    return {
        "found": len(scanned),
        "new_or_changed": len(new_or_changed),
        "unchanged": len(unchanged),
        "removed": len(removed),
        "new_or_changed_files": new_or_changed,
        "removed_files": removed,
        "data_dir": data_dir,
        "manifest_path": manifest_path,
    }
