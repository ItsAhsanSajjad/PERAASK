from __future__ import annotations

import os
import re
import json
import uuid
import time
import html
from datetime import datetime
from typing import List, Dict, Any, Tuple, Optional

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from speech import transcribe_audio

# Step 3: auto-discovery + manifest status
from doc_registry import scan_and_update_manifest

# ✅ Blue/Green auto-indexer
from index_manager import SafeAutoIndexer, IndexManagerConfig

from retriever import retrieve
from answerer import answer_question

# ✅ Smalltalk / greeting intent layer
from smalltalk_intent import decide_smalltalk, detect_language


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="PERA AI Dashboard",
    layout="wide",
    page_icon="assets/pera_logo.png",
)

# accept both spellings
BASE_URL = os.getenv("BASE_URL", os.getenv("Base_URL", "https://askpera.infinitysol.agency")).rstrip("/")
DATA_DIR = os.getenv("DATA_DIR", "assets/data")
INDEXES_ROOT = os.getenv("INDEXES_ROOT", "assets/indexes")
INDEX_POINTER_PATH = os.getenv("INDEX_POINTER_PATH", "assets/indexes/ACTIVE.json")

REFUSAL_TEXT = "There is no information available to this question."
APP_DEBUG = os.getenv("APP_DEBUG", "0").strip() != "0"

# ---------------- Chat store (ChatGPT-style history) ----------------
CHATS_STORE_PATH = os.getenv("CHATS_STORE_PATH", "assets/chats/chat_store.json").replace("\\", "/")
os.makedirs(os.path.dirname(CHATS_STORE_PATH), exist_ok=True)

# ✅ Windows-safe lock file for cross-process / rerun safety
CHAT_LOCK_PATH = (CHATS_STORE_PATH + ".lock").replace("\\", "/")


def _now_iso() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


# ---------------- File lock (cross-platform, no extra libs) ----------------
class _FileLock:
    """
    Simple lock via exclusive file creation.
    Works cross-process on Windows/Linux.
    """
    def __init__(self, lock_path: str, timeout_s: float = 3.0, poll_s: float = 0.05, stale_s: float = 45.0):
        self.lock_path = lock_path
        self.timeout_s = timeout_s
        self.poll_s = poll_s
        self.stale_s = stale_s
        self._fd: Optional[int] = None

    def __enter__(self):
        start = time.time()
        while True:
            try:
                self._fd = os.open(self.lock_path, os.O_CREAT | os.O_EXCL | os.O_RDWR)
                os.write(self._fd, str(os.getpid()).encode("utf-8", errors="ignore"))
                return self
            except FileExistsError:
                # stale lock breaker
                try:
                    age = time.time() - os.path.getmtime(self.lock_path)
                    if age > self.stale_s:
                        try:
                            os.remove(self.lock_path)
                            continue
                        except Exception:
                            pass
                except Exception:
                    pass

                if (time.time() - start) >= self.timeout_s:
                    raise TimeoutError(f"Timed out acquiring lock: {self.lock_path}")
                time.sleep(self.poll_s)

    def __exit__(self, exc_type, exc, tb):
        try:
            if self._fd is not None:
                os.close(self._fd)
        except Exception:
            pass
        try:
            if os.path.exists(self.lock_path):
                os.remove(self.lock_path)
        except Exception:
            pass


def _load_chat_store() -> Dict[str, Any]:
    if not os.path.exists(CHATS_STORE_PATH):
        return {"chats": {}, "order": []}
    try:
        with open(CHATS_STORE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, dict):
            return {"chats": {}, "order": []}
        data.setdefault("chats", {})
        data.setdefault("order", [])
        return data
    except Exception:
        return {"chats": {}, "order": []}


def _atomic_write_json(path: str, obj: Dict[str, Any]) -> None:
    """
    ✅ Windows-safe atomic write:
    - write to unique tmp file in same dir
    - fsync to flush
    - replace with retries (handles WinError 5)
    - fallback to non-atomic write (prevents app crash)
    """
    path = path.replace("\\", "/")
    dirp = os.path.dirname(path) or "."
    os.makedirs(dirp, exist_ok=True)

    tmp = f"{path}.{uuid.uuid4().hex}.tmp"
    last_err: Optional[Exception] = None

    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=2)
        f.flush()
        try:
            os.fsync(f.fileno())
        except Exception:
            pass

    for attempt in range(10):
        try:
            os.replace(tmp, path)
            return
        except PermissionError as e:
            last_err = e
            time.sleep(0.015 * (2 ** attempt))
        except OSError as e:
            last_err = e
            time.sleep(0.015 * (2 ** attempt))

    # fallback (avoid crash)
    try:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
            f.flush()
            try:
                os.fsync(f.fileno())
            except Exception:
                pass
    finally:
        try:
            if os.path.exists(tmp):
                os.remove(tmp)
        except Exception:
            pass

    if last_err:
        raise last_err


def _save_chat_store(store: Dict[str, Any]) -> None:
    """
    ✅ Best-effort persistence:
    - lock to prevent concurrent writes across reruns/sessions
    - atomic write with retry/backoff
    - NEVER crash the app if saving fails (storage is not mission critical)
    """
    try:
        with _FileLock(CHAT_LOCK_PATH, timeout_s=3.0, poll_s=0.05, stale_s=45.0):
            _atomic_write_json(CHATS_STORE_PATH, store)
    except Exception as e:
        if APP_DEBUG:
            st.sidebar.warning(f"Chat store save failed (non-fatal): {e}")


# ---------------- Conversation context state (multi-turn robustness) ----------------
_TOKEN_RE = re.compile(r"[A-Za-z\u0600-\u06FF]{2,}")

_ABBREV = {
    "cto": "chief technology officer",
    "tor": "terms of reference",
    "dg": "director general",
    "sso": "senior staff officer",
    "eo": "enforcement officer",
    "io": "investigation officer",
    # add more org-specific short forms here
}

def _norm_for_ctx(text: str) -> str:
    t = (text or "").lower()
    t = re.sub(r"[^a-z0-9\u0600-\u06FF\s]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t

def _expand_abbrev(text: str) -> str:
    t = _norm_for_ctx(text)
    for k, v in _ABBREV.items():
        t = re.sub(rf"\b{re.escape(k)}\b", v, t)
    return t

def _extract_terms(text: str, cap: int = 14) -> List[str]:
    t = _norm_for_ctx(text)
    toks = _TOKEN_RE.findall(t)
    seen = set()
    out: List[str] = []
    for tok in toks:
        if tok in seen:
            continue
        seen.add(tok)
        out.append(tok)
        if len(out) >= cap:
            break
    return out

def _init_context_state() -> Dict[str, Any]:
    return {"topic_terms": [], "recent_user_turns": []}

def _update_context_state(ctx: Dict[str, Any], user_text: str) -> Dict[str, Any]:
    ctx = ctx or _init_context_state()
    expanded = _expand_abbrev(user_text)
    terms = _extract_terms(expanded, cap=14)

    topic_terms: List[str] = ctx.get("topic_terms") or []
    for t in terms:
        if t not in topic_terms:
            topic_terms.append(t)
    ctx["topic_terms"] = topic_terms[-25:]  # last 25 unique terms

    turns: List[str] = ctx.get("recent_user_turns") or []
    turns.append(user_text.strip())
    ctx["recent_user_turns"] = turns[-5:]  # last 5 turns
    return ctx

def _lang_labels(lang: str) -> Dict[str, str]:
    # smalltalk_intent.detect_language returns: english | urdu | roman_urdu | unsupported
    if lang == "urdu":
        return {
            "ctx_terms": "سیاقی الفاظ",
            "prev_q": "پچھلا سوال",
            "follow": "ضمنی سوال",
            "regarding": "حوالہ",
        }
    if lang == "roman_urdu":
        return {
            "ctx_terms": "Mozooi alfaaz",
            "prev_q": "Pichla sawal",
            "follow": "Follow-up",
            "regarding": "Hawala",
        }
    return {
        "ctx_terms": "Context terms",
        "prev_q": "Previous question",
        "follow": "Follow-up",
        "regarding": "Regarding",
    }

def _build_retrieval_query(prompt: str, ctx: Dict[str, Any], is_follow: bool, last_question: str) -> str:
    """
    Deterministic: always add light topic-term hint.
    Cap hint hard to avoid embedding drift.
    """
    expanded = _expand_abbrev(prompt)

    lang = detect_language(expanded)
    labels = _lang_labels(lang)

    topic_terms = (ctx.get("topic_terms") or [])[-10:]
    hint = " ".join(topic_terms).strip()
    if len(hint) > 180:
        hint = hint[:180].rstrip()

    if is_follow and last_question:
        base = _rewrite_followup_to_standalone(expanded, last_question, lang)
    else:
        base = expanded

    if hint:
        return f"{base}\n{labels['ctx_terms']}: {hint}"
    return base


def _title_from_prompt(prompt: str) -> str:
    q = (prompt or "").strip()
    if not q:
        return "New chat"
    q = re.sub(r"\s+", " ", q)
    q_low = q.lower()
    for pref in [
        "please", "kindly", "tell me", "can you", "could you", "would you",
        "what is", "what are", "who is", "who are", "how to", "how can i",
        "explain", "define"
    ]:
        if q_low.startswith(pref + " "):
            q = q[len(pref):].strip()
            break
    words = q.split()
    title = " ".join(words[:7])
    if len(words) > 7:
        title += "…"
    return title[:60] if title else "New chat"


def _create_new_chat(store: Dict[str, Any]) -> str:
    cid = str(uuid.uuid4())
    store["chats"][cid] = {
        "id": cid,
        "title": "New chat",
        "created_at": _now_iso(),
        "messages": [],
        "last_question": None,
        "last_retrieval": None,
        "last_answer": None,
        "voice_text": "",
        "context_state": _init_context_state(),
    }
    store["order"].insert(0, cid)
    _save_chat_store(store)
    return cid


def _clear_all_chats_and_start_fresh() -> None:
    """
    ✅ Clears ALL chat history (file + session) and starts a fresh chat.
    """
    store = {"chats": {}, "order": []}
    new_id = _create_new_chat(store)
    st.session_state.chat_store = store
    st.session_state.active_chat_id = new_id


def _init_chat_state():
    if "chat_store" not in st.session_state:
        st.session_state.chat_store = _load_chat_store()

    store = st.session_state.chat_store

    if "active_chat_id" not in st.session_state:
        if store.get("order"):
            st.session_state.active_chat_id = store["order"][0]
        else:
            st.session_state.active_chat_id = _create_new_chat(store)

    if st.session_state.active_chat_id not in store.get("chats", {}):
        st.session_state.active_chat_id = _create_new_chat(store)

    # ensure each chat has context_state (migration)
    changed = False
    for cid, chat in (store.get("chats") or {}).items():
        if "context_state" not in chat or not isinstance(chat.get("context_state"), dict):
            chat["context_state"] = _init_context_state()
            store["chats"][cid] = chat
            changed = True
    if changed:
        _save_chat_store(store)


def _get_active_chat() -> Dict[str, Any]:
    store = st.session_state.chat_store
    return store["chats"][st.session_state.active_chat_id]


# ---------------- In-memory append (NO save here) ----------------
def _append_message_mem(chat: Dict[str, Any], role: str, content: str, references: Optional[List[Dict[str, Any]]] = None) -> None:
    chat.setdefault("messages", [])
    chat["messages"].append({"role": role, "content": content, "references": references or []})


# ---------------- UI CSS ----------------
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700;900&display=swap');

.stApp { background: linear-gradient(180deg, #f3f7f4, #edf3ef); font-family: 'Inter', sans-serif; }
.block-container { max-width: 100% !important; width: 100% !important; padding: 2rem !important; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, #0b3b2a, #0e2a1f); }
[data-testid="stSidebar"] * { color: #ffffff; }
[data-testid="stSidebar"] h2 { font-weight: 900; letter-spacing: 0.5px; }

/* Header */
.dashboard-subtitle { margin-top:23px; font-size: 17px; text-align: center; margin-bottom: 10px; color: #065f46; font-weight: 500; }
.doc-status { text-align: center; margin-bottom: 28px; color: #064e3b; font-size: 13px; opacity: 0.9; }

/* Cards */
.card {
  background: rgba(255,255,255,0.95); backdrop-filter: blur(10px);
  border-radius: 18px; padding: 24px;
  box-shadow: 0 6px 18px rgba(0,0,0,0.08);
  transition: all 0.3s ease; text-align: center;
  border: 1px solid rgba(255,255,255,0.6);
  height: 100%; display: flex; flex-direction: column; justify-content: center;
}
.card:hover { transform: translateY(-6px); box-shadow: 0 12px 28px rgba(0,0,0,0.12); }
.card-icon { font-size: 32px; margin-bottom: 10px; color: #1f7a4d; }
.card-title { font-size: 18px; font-weight: 700; color: #064e3b; }
.card-desc { font-size: 13px; color: #6b7280; margin-top: 4px; }

/* Chat */
[data-testid="stChatMessage"] { border-radius: 18px; padding: 16px; margin-bottom: 14px; box-shadow: 0 6px 18px rgba(0,0,0,0.08); }
[data-testid="stChatMessage"][aria-label="assistant message"] { background: #ffffff; }
[data-testid="stChatMessage"][aria-label="user message"] { background: linear-gradient(135deg, #1f7a4d, #2ea96f); color: white; }

/* Inputs */
[data-testid="stChatInput"] textarea { border-radius: 30px; padding: 15px 18px; font-size: 15px; border: 1px solid #d1d5db; }
textarea[aria-label="Transcribed text"] {
  border-radius: 30px !important; padding: 15px 18px !important; font-size: 15px !important;
  border: 1px solid #d1d5db !important; background-color: #ffffff !important; color: #111827 !important;
}

/* Buttons */
.stButton button {
  background: linear-gradient(135deg, #1f7a4d, #2ea96f);
  color: white; border-radius: 30px; padding: 12px 26px; border: none;
  font-weight: 700; transition: all 0.3s ease;
}
.stButton button:hover { transform: scale(1.04); box-shadow: 0 10px 22px rgba(0,0,0,0.15); }

/* ---- Chat list spacing + ChatGPT-like compact list ---- */
.chat-list { margin-top: 6px; }
.chat-item-btn { margin: 6px 0 !important; }
.chat-item-btn .stButton { margin: 0 !important; padding: 0 !important; }
.chat-item-btn button {
  width: 100% !important;
  text-align: left !important;
  border-radius: 12px !important;
  padding: 8px 12px !important;
  background: rgba(255,255,255,0.12) !important;
  border: 1px solid rgba(255,255,255,0.10) !important;
  font-size: 14px !important;
  margin: 0 !important;
}
.chat-item-btn button:hover { background: rgba(255,255,255,0.22) !important; }

/* Footer */
.footer {
  margin-top: 50px; padding: 18px; text-align: center; font-size: 13px; color: #374151;
  background: rgba(255,255,255,0.7); backdrop-filter: blur(8px);
  border-radius: 16px; box-shadow: 0 6px 18px rgba(0,0,0,0.06);
}
.footer a { color: #1f7a4d; font-weight: 700; text-decoration: none; }
.footer a:hover { text-decoration: underline; }

/* Responsive cards grid */
@media (min-width: 1200px) { .stColumns { display: grid !important; grid-template-columns: repeat(4, 1fr) !important; gap: 20px !important; } }
@media (max-width: 992px) { .stColumns { display: grid !important; grid-template-columns: repeat(2, 1fr) !important; gap: 16px !important; } }
@media (max-width: 600px) {
  .stColumns { display: grid !important; grid-template-columns: 1fr !important; gap: 14px !important; }
  .card { padding: 20px; }
  .card-title { font-size: 16px; }
  .card-icon { font-size: 28px; }
}

/* -------- References pills (click to open) -------- */
.ref-row { display:flex; align-items:center; gap:10px; margin: 6px 0; flex-wrap: wrap; }
.ref-pill {
  background:#f3f4f6; color:#111827; border:1px solid #e5e7eb;
  border-radius:999px; padding:6px 12px; font-size:13px; font-weight:500;
  display:inline-block; text-decoration:none; cursor:pointer;
}
.ref-pill:hover { background:#eef2ff; border-color:#dbeafe; }
.ref-meta { margin-left: 8px; color: #6b7280; font-size: 13px; line-height: 1.45; }
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
def _normalize_assistant_output(raw: Any) -> Tuple[str, List[Dict[str, Any]]]:
    if isinstance(raw, dict):
        return (raw.get("answer", "") or ""), (raw.get("references", []) or [])
    return str(raw), []


def _strip_legacy_refs_from_answer(answer: str) -> str:
    if not answer:
        return answer
    lines = (answer or "").splitlines()
    out = []
    in_refs = False
    for line in lines:
        s = (line or "").strip()
        if re.match(r"^references\s*:?\s*$", s, flags=re.IGNORECASE):
            in_refs = True
            continue
        if in_refs:
            continue
        if re.match(r"^(open|download)\s*$", s, flags=re.IGNORECASE):
            continue
        out.append(line)
    return "\n".join(out).strip()


def _compress_int_ranges(nums: List[int]) -> str:
    if not nums:
        return ""
    nums = sorted(set(int(n) for n in nums if isinstance(n, (int, float)) or str(n).isdigit()))
    if not nums:
        return ""
    ranges: List[Tuple[int, int]] = []
    start = prev = nums[0]
    for n in nums[1:]:
        if n == prev + 1:
            prev = n
            continue
        ranges.append((start, prev))
        start = prev = n
    ranges.append((start, prev))
    parts = []
    for a, b in ranges:
        parts.append(str(a) if a == b else f"{a}–{b}")
    return ", ".join(parts)


def _safe_join_base_and_path(base_url: str, path: str) -> str:
    if not path:
        return ""
    p = path.strip().replace("\\", "/")
    if p.startswith("http://") or p.startswith("https://"):
        return p
    if not p.startswith("/"):
        p = "/" + p
    return f"{base_url}{p}"


def _group_references(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in refs:
        if not isinstance(r, dict):
            continue
        doc = (r.get("document") or r.get("doc_name") or "Unknown document").strip()
        path = (r.get("public_path") or r.get("path") or "").strip()

        # ✅ FIX: fallback path when missing
        if not path:
            path = f"/assets/data/{doc}"

        key = (doc, path)
        g = grouped.get(key)
        if not g:
            g = {"document": doc, "path": path, "pages": set(), "locs": set(), "open_url": "", "url_hint": ""}
            grouped[key] = g

        if not g["open_url"]:
            g["open_url"] = (r.get("open_url") or "").strip()
        if not g["url_hint"]:
            g["url_hint"] = (r.get("url_hint") or "").strip()

        ps = r.get("page_start")
        pe = r.get("page_end")
        if ps is not None:
            try:
                ps_i = int(ps)
                pe_i = int(pe) if pe is not None else ps_i
                for p in range(min(ps_i, pe_i), max(ps_i, pe_i) + 1):
                    g["pages"].add(p)
            except Exception:
                pass

        loc = r.get("loc") or r.get("loc_start")
        if loc:
            g["locs"].add(str(loc).strip())

    out: List[Dict[str, Any]] = []
    for g in grouped.values():
        out.append(
            {
                "document": g["document"],
                "path": g["path"],
                "open_url": g["open_url"],
                "url_hint": g["url_hint"],
                "pages": sorted(list(g["pages"])),
                "locs": sorted(list(g["locs"])),
            }
        )
    out.sort(key=lambda x: x.get("document", ""))
    return out


def _apply_page_anchor_if_missing(open_url: str, pages: List[int]) -> str:
    if not open_url:
        return open_url
    if "#" in open_url:
        return open_url
    if pages:
        p = min(int(x) for x in pages if isinstance(x, int) or str(x).isdigit())
        return f"{open_url}#page={p}"
    return open_url


def _render_references_click_to_open(references: List[Dict[str, Any]]) -> None:
    if not references:
        return

    grouped = _group_references(references)
    if not grouped:
        return

    st.markdown("---")
    st.markdown("**References:**")

    for g in grouped:
        doc = g.get("document", "Unknown document")
        path = (g.get("path") or "").strip()
        pages: List[int] = g.get("pages") or []
        locs: List[str] = g.get("locs") or []
        url_hint = (g.get("url_hint") or "").strip()

        open_url = (g.get("open_url") or "").strip()
        if not open_url:
            built = _safe_join_base_and_path(BASE_URL, path)
            open_url = f"{built}{url_hint}"

        open_url = _apply_page_anchor_if_missing(open_url, pages)

        # ✅ escape values used in HTML
        safe_href = html.escape(open_url, quote=True)
        safe_doc = html.escape(doc, quote=False)

        st.markdown(
            f"""
            <div class="ref-row">
              <a class="ref-pill" href="{safe_href}" target="_blank" rel="noopener noreferrer">{safe_doc}</a>
            </div>
            """,
            unsafe_allow_html=True,
        )

        meta_lines: List[str] = []
        if pages:
            meta_lines.append(f"Pages: {_compress_int_ranges(pages)}")
        if locs:
            joined = "; ".join(locs[:6])
            if len(locs) > 6:
                joined += f"; +{len(locs)-6} more"
            meta_lines.append(f"Sections / Paragraphs: {joined}")

        if meta_lines:
            safe_meta = "<br/>".join(html.escape(x) for x in meta_lines)
            st.markdown(
                f"<div class='ref-meta'>{safe_meta}</div>",
                unsafe_allow_html=True,
            )


def _render_assistant_message(answer: str, references: List[Dict[str, Any]]) -> None:
    st.markdown(answer)
    _render_references_click_to_open(references)


# ---------------- Follow-up detection ----------------
_FOLLOWUP_PATTERNS = [
    r"\b(explain|simplify|summari[sz]e|elaborate|clarify|rephrase|detail|more)\b",
    r"\b(in simpler terms|in simple words|simple words|asaan|asan|سادہ|آسان)\b",
    r"^(why\??|how\??|and\??|then\??|ok\??|okay\??|yes\??|no\??)$",
    r"\b(it|this|that|above|previous|earlier)\b",
]

def _is_followup(text: str) -> bool:
    t = (text or "").strip().lower()
    if not t:
        return False
    short_bias = len(t.split()) <= 6
    for p in _FOLLOWUP_PATTERNS:
        if re.search(p, t, flags=re.IGNORECASE):
            return True
    return short_bias and any(w in t for w in ["it", "this", "that", "explain", "simpler", "simple", "clarify"])


def _rewrite_followup_to_standalone(followup: str, last_question: str, lang: str) -> str:
    f = (followup or "").strip()
    lq = (last_question or "").strip()
    if not lq:
        return f

    if len(f.split()) >= 10 or re.search(r"\b(pera|authority|regulation|policy|rule|notification|composition)\b", f, re.I):
        return f

    labels = _lang_labels(lang)
    return f"{labels['regarding']}: {lq}\n{labels['follow']}: {f}"


# ---------------- Cached index (Blue/Green) ----------------
@st.cache_resource(show_spinner=False)
def _ensure_index_ready() -> Dict[str, Any]:
    cfg = IndexManagerConfig(
        data_dir=DATA_DIR,
        indexes_root=INDEXES_ROOT,
        active_pointer_path=INDEX_POINTER_PATH,
        poll_seconds=int(os.getenv("INDEX_POLL_SECONDS", "30")),
        keep_last_n=int(os.getenv("INDEX_KEEP_LAST_N", "3")),
        chunk_max_chars=int(os.getenv("CHUNK_MAX_CHARS", "4500")),
        chunk_overlap_chars=int(os.getenv("CHUNK_OVERLAP_CHARS", "350")),
    )
    ix = SafeAutoIndexer(cfg)
    return ix.ensure_index_once()


# ---------------- Init chat history ----------------
_init_chat_state()


# ---------------- User-facing network fallback (blame connection) ----------------
def _network_fallback_text(lang: str) -> str:
    if lang == "urdu":
        return "انٹرنیٹ کنکشن سست یا غیر مستحکم لگ رہا ہے۔ براہِ کرم دوبارہ کوشش کریں (ممکن ہو تو Wi-Fi استعمال کریں)۔"
    if lang == "roman_urdu":
        return "Internet connection slow ya unstable lag raha hai. Meharbani karke dobara try karein (mumkin ho to Wi-Fi use karein)."
    return "Your internet connection seems slow or unstable. Please try again (prefer Wi-Fi)."


# ---------------- Core handler (ONE SAVE at end) ----------------
def _handle_user_message(user_raw: str):
    user_raw = (user_raw or "").strip()
    if not user_raw:
        return

    cid = st.session_state.active_chat_id
    store = st.session_state.chat_store
    chat_obj = _get_active_chat()

    # Title on first meaningful prompt (keep production behavior)
    if (chat_obj.get("title") or "New chat") == "New chat":
        chat_obj["title"] = _title_from_prompt(user_raw)

    # Append user message exactly as typed/spoken
    _append_message_mem(chat_obj, "user", user_raw, [])

    # Smalltalk gate
    decision = decide_smalltalk(user_raw)

    # Greeting-only
    if decision and getattr(decision, "is_greeting_only", False):
        _append_message_mem(chat_obj, "assistant", (decision.response or "").strip(), [])
        store["chats"][cid] = chat_obj
        _save_chat_store(store)
        return

    # Greeting + question
    prompt = user_raw
    if decision and (not getattr(decision, "is_greeting_only", False)) and getattr(decision, "remaining_question", None):
        prompt = (decision.remaining_question or "").strip()
        ack = (getattr(decision, "ack", "") or "").strip()
        if ack:
            _append_message_mem(chat_obj, "assistant", ack, [])

    prompt = (prompt or "").strip()
    if not prompt:
        store["chats"][cid] = chat_obj
        _save_chat_store(store)
        return

    # ✅ context state update uses the REAL prompt
    ctx = chat_obj.get("context_state") or _init_context_state()
    ctx = _update_context_state(ctx, prompt)
    chat_obj["context_state"] = ctx

    lang = detect_language(prompt)
    labels = _lang_labels(lang)

    last_ret = chat_obj.get("last_retrieval")
    last_q = chat_obj.get("last_question")
    is_follow = bool(last_q and _is_followup(prompt))

    retrieval_query = _build_retrieval_query(prompt, ctx, is_follow=is_follow, last_question=last_q or "")

    retrieval_used: Dict[str, Any] = {}
    raw: Any = {"answer": _network_fallback_text(lang), "references": []}

    # ✅ CRITICAL FIX: never allow exceptions to crash Streamlit UI
    with st.spinner("PERA AI is thinking..."):
        try:
            retrieval_used = retrieve(retrieval_query)

            composed_q_for_answerer = prompt
            if is_follow and last_q:
                composed_q_for_answerer = f"{prompt}\n\n{labels['prev_q']}: {last_q}"

            raw = answer_question(composed_q_for_answerer, retrieval_used)

            # Follow-up fallback: if refused but last retrieval had evidence, retry once
            if (
                isinstance(raw, dict)
                and (raw.get("answer") or "").strip() == REFUSAL_TEXT
                and is_follow
                and last_ret
                and isinstance(last_ret, dict)
                and last_ret.get("has_evidence")
            ):
                raw2 = answer_question(composed_q_for_answerer, last_ret)
                if isinstance(raw2, dict) and (raw2.get("answer") or "").strip() != REFUSAL_TEXT:
                    raw = raw2
                    retrieval_used = last_ret

        except Exception as e:
            # show friendly message (user-side network) and optionally log debug
            raw = {"answer": _network_fallback_text(lang), "references": []}
            if APP_DEBUG:
                with st.sidebar:
                    st.markdown("---")
                    st.markdown("### 🛠 Debug (Exception)")
                    st.write({"error": str(e)[:500], "prompt": prompt, "retrieval_query": retrieval_query})

    answer, refs = _normalize_assistant_output(raw)
    final_answer = _strip_legacy_refs_from_answer(answer)

    _append_message_mem(chat_obj, "assistant", final_answer, refs)

    # Update follow-up context ONLY when retrieval has evidence
    if isinstance(retrieval_used, dict) and retrieval_used.get("has_evidence"):
        chat_obj["last_question"] = prompt
        chat_obj["last_retrieval"] = retrieval_used
        chat_obj["last_answer"] = answer

    # Persist once
    store["chats"][cid] = chat_obj
    st.session_state.chat_store = store
    _save_chat_store(store)

    if APP_DEBUG:
        with st.sidebar:
            st.markdown("---")
            st.markdown("### 🛠 Debug")
            st.write({"prompt": prompt, "detected_lang": lang, "is_follow": is_follow})
            st.write({"retrieval_query": retrieval_query})
            if isinstance(retrieval_used, dict):
                st.write({
                    "has_evidence": retrieval_used.get("has_evidence"),
                    "primary_doc": retrieval_used.get("primary_doc"),
                    "docs": [d.get("doc_name") for d in (retrieval_used.get("evidence") or [])],
                })


# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("assets/pera_logo.png", width=150)
    st.markdown("## 🤖 PERA AI")

    if st.button("➕ New Chat", use_container_width=True):
        store = st.session_state.chat_store
        st.session_state.active_chat_id = _create_new_chat(store)
        st.rerun()

    if st.button("🧹 Clear Chat History", use_container_width=True):
        _clear_all_chats_and_start_fresh()
        st.rerun()

    st.markdown("---")
    st.markdown("### 💬 Chats")

    store = st.session_state.chat_store

    st.markdown("<div class='chat-list'>", unsafe_allow_html=True)

    for chat_id in store.get("order", []):
        chat = store["chats"].get(chat_id)
        if not chat:
            continue
        title = chat.get("title") or "New chat"
        prefix = "✅ " if chat_id == st.session_state.active_chat_id else ""
        st.markdown("<div class='chat-item-btn'>", unsafe_allow_html=True)
        if st.button(prefix + title, key=f"open_chat_{chat_id}", use_container_width=True):
            st.session_state.active_chat_id = chat_id
            st.rerun()
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("👤 User")
    st.markdown("🚪 Logout")


# ---------------- Header ----------------
st.markdown(
    "<div class='dashboard-subtitle'>Ask questions strictly from PERA policies & notifications</div>",
    unsafe_allow_html=True,
)

# Step 3: status line (manifest only for UI status)
try:
    status = scan_and_update_manifest(data_dir=DATA_DIR, index_dir=INDEXES_ROOT)
    st.markdown(
        f"<div class='doc-status'>📚 Documents: {status['found']} found | "
        f"{status['new_or_changed']} new/changed | "
        f"{status['unchanged']} unchanged | "
        f"{status['removed']} removed</div>",
        unsafe_allow_html=True,
    )
except Exception as e:
    st.markdown(f"<div class='doc-status'>📚 Document scan failed: {e}</div>", unsafe_allow_html=True)

# Step 6: ensure index ready (blue/green)
with st.spinner("Indexing documents (auto)..."):
    ingest_status = _ensure_index_ready()

st.markdown(
    f"<div class='doc-status'>🧠 Active Index: {ingest_status.get('active_index_dir','')} "
    f"| Changed: {ingest_status.get('changed', False)}</div>",
    unsafe_allow_html=True,
)

# ---------------- Dashboard cards ----------------
c1, c2, c3, c4 = st.columns(4)
cards = [
    ("🤖", "Ask AI", "Answers from PERA policies"),
    ("🎙️", "Voice Query", "Speak & get accurate transcription"),
    ("📄", "Policy Help", "Official PERA notifications"),
    ("⭐", "Saved Answers", "Bookmark important replies"),
]
for col, (icon, title, desc) in zip([c1, c2, c3, c4], cards):
    with col:
        st.markdown(
            f"""
        <div class='card'>
            <div class='card-icon'>{icon}</div>
            <div class='card-title'>{title}</div>
            <div class='card-desc'>{desc}</div>
        </div>
        """,
            unsafe_allow_html=True,
        )

# ---------------- Voice query ----------------
st.markdown("### 🎙️ Voice Query")

audio_dict = mic_recorder(
    start_prompt="🎤 Start Recording",
    stop_prompt="⏹ Stop Recording",
    just_once=True,
    key="mic_rec",
)

if audio_dict and audio_dict.get("bytes"):
    with st.spinner("Transcribing..."):
        try:
            store = st.session_state.chat_store
            active = _get_active_chat()
            active["voice_text"] = transcribe_audio(audio_dict["bytes"])
            store["chats"][st.session_state.active_chat_id] = active
            _save_chat_store(store)
        except Exception as e:
            st.error(f"⚠️ Voice transcription failed: {e}")

active_chat = _get_active_chat()
voice_val = active_chat.get("voice_text") or ""

voice_text = st.text_area(
    "Transcribed text",
    value=voice_val,
    height=90,
    placeholder="Ask a PERA policy question...",
)

send_voice = st.button("Send Voice Query")

# ---------------- Chat display ----------------
active_chat = _get_active_chat()
for item in active_chat.get("messages", []):
    role = item.get("role")
    content = item.get("content", "")
    references = item.get("references", [])
    with st.chat_message(role):
        if role == "assistant":
            _render_assistant_message(content, references)
        else:
            st.markdown(content)

# ---------------- Send handlers ----------------
if send_voice and (voice_text or "").strip():
    store = st.session_state.chat_store
    active = _get_active_chat()
    active["voice_text"] = ""
    store["chats"][st.session_state.active_chat_id] = active
    _save_chat_store(store)

    _handle_user_message(voice_text.strip())
    st.rerun()

user_input = st.chat_input("Ask a PERA policy question...")
if user_input:
    _handle_user_message(user_input)
    st.rerun()

# ---------------- Footer ----------------
st.markdown(
    """
<div class='footer'>
    Powered by: PERA AI TEAM · 
    <a href='#'>Support</a> .
    <a href='#'>Feedback</a>
</div>
""",
    unsafe_allow_html=True,
)
