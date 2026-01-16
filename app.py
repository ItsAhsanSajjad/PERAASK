from __future__ import annotations

import re
import os
import mimetypes
import hashlib
import json
from typing import List, Dict, Any, Tuple

import streamlit as st
from streamlit_mic_recorder import mic_recorder

from speech import transcribe_audio

# ✅ UI-safe scan (does NOT write manifest)
from doc_registry import scan_status_only

# Pipeline
from index_store import scan_and_ingest_if_needed
from retriever import retrieve, reset_retriever_cache
from answerer import answer_question

from smalltalk_intent import decide_smalltalk


# ---------------- CONFIG ----------------
st.set_page_config(
    page_title="PERA AI Dashboard",
    layout="wide",
    page_icon="assets/pera_logo.png",
)

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

/* Chat history */
.chat-item { padding: 10px 14px; border-radius: 12px; background: rgba(255,255,255,0.12); margin-bottom: 8px; cursor: pointer; font-size: 14px; }
.chat-item:hover { background: rgba(255,255,255,0.22); }

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

/* ---------------- ChatGPT-style references (ONLY affects download buttons) ---------------- */
div.stDownloadButton > button {
  background: #f3f4f6 !important;
  color: #111827 !important;
  border: 1px solid #e5e7eb !important;
  border-radius: 999px !important;
  padding: 6px 12px !important;
  font-size: 13px !important;
  font-weight: 500 !important;
  box-shadow: none !important;
  width: auto !important;
}
div.stDownloadButton > button:hover {
  background: #eef2ff !important;
  border-color: #dbeafe !important;
  transform: none !important;
  box-shadow: none !important;
}

/* Reference metadata text */
.ref-meta {
  margin-left: 8px;
  color: #6b7280;
  font-size: 13px;
  line-height: 1.45;
}
</style>
""",
    unsafe_allow_html=True,
)

# ---------------- Helpers ----------------
def _normalize_assistant_output(raw: Any) -> Tuple[str, List[Dict[str, Any]]]:
    if isinstance(raw, dict):
        return (raw.get("answer", "") or ""), (raw.get("references", []) or [])
    return str(raw), []


@st.cache_data(show_spinner=False)
def _read_file_bytes(path: str) -> bytes:
    with open(path, "rb") as f:
        return f.read()


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


def _group_references(refs: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for r in refs:
        if not isinstance(r, dict):
            continue
        doc = (r.get("document") or r.get("doc_name") or "Unknown document").strip()
        path = (r.get("path") or "").strip()
        key = (doc, path)

        g = grouped.get(key)
        if not g:
            g = {"document": doc, "path": path, "count": 0, "pages": set(), "locs": set()}
            grouped[key] = g

        g["count"] += 1

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
                "count": g["count"],
                "pages": sorted(list(g["pages"])),
                "locs": sorted(list(g["locs"])),
            }
        )

    out.sort(key=lambda x: x.get("document", ""))
    return out


def _render_references_chatgpt_style(references: List[Dict[str, Any]]) -> None:
    if not references:
        return

    grouped = _group_references(references)
    if not grouped:
        return

    st.markdown("---")
    st.markdown("**References:**")

    for i, g in enumerate(grouped, start=1):
        doc = g.get("document", "Unknown document")
        path = g.get("path", "")
        count = int(g.get("count", 1) or 1)
        label = f"{doc} ({count})"

        if path and os.path.exists(path):
            try:
                data = _read_file_bytes(path)
                mime = mimetypes.guess_type(path)[0] or "application/octet-stream"
                st.download_button(
                    label=label,
                    data=data,
                    file_name=os.path.basename(path),
                    mime=mime,
                    key=f"refpill_{hash(path)}_{i}",
                )
            except Exception:
                st.markdown(f"- {label}")
        else:
            st.markdown(f"- {label}")
            st.caption("Source file not found on server.")

        pages: List[int] = g.get("pages") or []
        locs: List[str] = g.get("locs") or []

        meta_lines: List[str] = []
        if pages:
            meta_lines.append(f"Pages: {_compress_int_ranges(pages)}")
        if locs:
            joined = "; ".join(locs[:6])
            if len(locs) > 6:
                joined += f"; +{len(locs)-6} more"
            meta_lines.append(f"Sections / Paragraphs: {joined}")

        if meta_lines:
            st.markdown(
                f"<div class='ref-meta'>{'<br/>'.join(meta_lines)}</div>",
                unsafe_allow_html=True,
            )


def _render_assistant_message(answer: str, references: List[Dict[str, Any]]) -> None:
    st.markdown(answer)
    _render_references_chatgpt_style(references)


# ---------------- Follow-up detection (deterministic) ----------------
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


def _rewrite_followup_to_standalone(followup: str, last_question: str) -> str:
    f = (followup or "").strip()
    lq = (last_question or "").strip()
    if not lq:
        return f

    if len(f.split()) >= 10 or re.search(r"\b(pera|authority|regulation|policy|rule|notification|composition)\b", f, re.I):
        return f

    return f"Regarding: {lq}\nFollow-up: {f}"


# ---------------- Index signature to defeat caching ----------------
def _compute_data_signature(data_dir: str = "assets/data") -> str:
    items = []
    try:
        for name in os.listdir(data_dir):
            if name.startswith("~$"):
                continue
            if not name.lower().endswith((".pdf", ".docx")):
                continue
            p = os.path.join(data_dir, name)
            try:
                st_ = os.stat(p)
                items.append((name, int(st_.st_mtime), int(st_.st_size)))
            except Exception:
                items.append((name, 0, 0))
    except Exception:
        pass

    items.sort()
    raw = json.dumps(items, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()


@st.cache_resource(show_spinner=False)
def _ensure_index_ready(_sig: str, _version: int):
    return scan_and_ingest_if_needed(data_dir="assets/data", index_dir="assets/index")


# ---------------- Session state ----------------
if "chat" not in st.session_state:
    st.session_state.chat = []
if "voice_text" not in st.session_state:
    st.session_state.voice_text = ""

if "last_question" not in st.session_state:
    st.session_state.last_question = None
if "last_retrieval" not in st.session_state:
    st.session_state.last_retrieval = None
if "last_answer" not in st.session_state:
    st.session_state.last_answer = None

if "index_version" not in st.session_state:
    st.session_state.index_version = 0


# ---------------- Core handler ----------------
def _handle_user_message(user_raw: str):
    user_raw = (user_raw or "").strip()
    if not user_raw:
        return

    st.session_state.chat.append({"role": "user", "content": user_raw, "references": []})

    decision = decide_smalltalk(user_raw)
    if decision and decision.is_greeting_only:
        st.session_state.chat.append({"role": "assistant", "content": decision.response, "references": []})
        return

    prompt = user_raw
    ack = ""
    if decision and (not decision.is_greeting_only) and decision.remaining_question:
        prompt = decision.remaining_question.strip()
        ack = (decision.ack or "").strip()

    last_ret = st.session_state.last_retrieval
    last_q = st.session_state.last_question

    is_follow = bool(last_q and _is_followup(prompt))

    retrieval_query = prompt
    if is_follow:
        retrieval_query = _rewrite_followup_to_standalone(prompt, last_q)

    with st.spinner("PERA AI is thinking..."):
        retrieval_used = retrieve(retrieval_query)
        composed_q_for_answerer = prompt

        if is_follow and last_q:
            composed_q_for_answerer = f"{prompt}\n\nContext (previous question): {last_q}"

        raw = answerer = answer_question(composed_q_for_answerer, retrieval_used)

        if (
            isinstance(raw, dict)
            and (raw.get("answer") or "").strip() == "There is no information available to this question."
            and is_follow
            and last_ret
            and isinstance(last_ret, dict)
            and last_ret.get("has_evidence")
        ):
            raw = answer_question(composed_q_for_answerer, last_ret)
            if isinstance(raw, dict) and (raw.get("answer") or "").strip() != "There is no information available to this question.":
                retrieval_used = last_ret

    answer, refs = _normalize_assistant_output(raw)
    final_answer = f"{ack} {answer}".strip() if ack else answer
    st.session_state.chat.append({"role": "assistant", "content": final_answer, "references": refs})

    if isinstance(retrieval_used, dict) and retrieval_used.get("has_evidence"):
        if not is_follow:
            st.session_state.last_question = prompt
        st.session_state.last_retrieval = retrieval_used
        st.session_state.last_answer = answer


# ---------------- Sidebar ----------------
with st.sidebar:
    st.image("assets/pera_logo.png", width=150)
    st.markdown("## 🤖 PERA AI")

    if st.button("➕ New Chat"):
        st.session_state.chat = []
        st.session_state.last_question = None
        st.session_state.last_retrieval = None
        st.session_state.last_answer = None
        st.rerun()

    if st.button("🔄 Refresh Index"):
        st.session_state.index_version += 1
        st.cache_resource.clear()
        st.cache_data.clear()
        try:
            reset_retriever_cache()
        except Exception:
            pass
        st.success("Index refresh triggered. Rebuilding now…")
        st.rerun()

    st.markdown("### 💬 Chat History")
    if st.session_state.chat:
        for item in st.session_state.chat[-8:]:
            role = item.get("role", "")
            preview = (item.get("content", "") or "").replace("\n", " ")[:38]
            st.markdown(f"<div class='chat-item'>{role}: {preview}...</div>", unsafe_allow_html=True)
    else:
        st.markdown("<div class='chat-item'>No chats yet</div>", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("👤 User")
    st.markdown("🚪 Logout")


# ---------------- Header ----------------
st.markdown(
    "<div class='dashboard-subtitle'>Ask questions strictly from PERA policies & notifications</div>",
    unsafe_allow_html=True,
)

# ✅ Ensure index first (so ingestion updates index/manifest correctly)
sig = _compute_data_signature("assets/data")
with st.spinner("Indexing documents (auto)..."):
    ingest_status = _ensure_index_ready(sig, st.session_state.index_version)

st.markdown(
    f"<div class='doc-status'>🧠 Index: {ingest_status.get('chunks_added', 0)} chunks added "
    f"(new/changed: {ingest_status.get('new_or_changed', 0)})</div>",
    unsafe_allow_html=True,
)

# ✅ UI status scan AFTER ingest (read-only scan; does not write manifest)
try:
    status = scan_status_only(data_dir="assets/data", index_dir="assets/index")
    st.markdown(
        f"<div class='doc-status'>📚 Documents: {status['found']} found | "
        f"{status['new_or_changed']} new/changed | "
        f"{status['unchanged']} unchanged | "
        f"{status['removed']} removed</div>",
        unsafe_allow_html=True,
    )
except Exception as e:
    st.markdown(f"<div class='doc-status'>📚 Document scan failed: {e}</div>", unsafe_allow_html=True)


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
            st.session_state.voice_text = transcribe_audio(audio_dict["bytes"])
        except Exception as e:
            st.session_state.voice_text = ""
            st.error(f"⚠️ Voice transcription failed: {e}")

st.session_state.voice_text = st.text_area(
    "Transcribed text",
    value=st.session_state.voice_text,
    height=90,
    placeholder="Ask a PERA policy question...",
)

send_voice = st.button("Send Voice Query")

# ---------------- Chat display ----------------
for item in st.session_state.chat:
    role = item.get("role")
    content = item.get("content", "")
    references = item.get("references", [])
    with st.chat_message(role):
        if role == "assistant":
            _render_assistant_message(content, references)
        else:
            st.markdown(content)

# ---------------- Send handlers ----------------
if send_voice and st.session_state.voice_text.strip():
    _handle_user_message(st.session_state.voice_text.strip())
    st.session_state.voice_text = ""
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
    <a href='#'> Feedback</a>
</div>
""",
    unsafe_allow_html=True,
)
