"""
app.py  ·  AI Document Comparison Agent
Two modes: Simple (non-technical) and Expert (technical)
"""

import os
import streamlit as st
from pdf_processor import extract_langchain_docs, extraction_summary
from vector_store import VectorStoreManager
from agent import DocumentComparisonAgent

LIVE: dict = {}


def _get_store_and_agent():
    store = LIVE.get("store")
    agent = LIVE.get("agent")
    if store and agent:
        return store, agent
    api_key = st.session_state.get("mistral_api_key") or os.environ.get("MISTRAL_API_KEY")
    model   = st.session_state.get("model", "mistral-large-latest")
    if api_key:
        store = VectorStoreManager(api_key)
        agent = DocumentComparisonAgent(api_key, model)
        LIVE["store"] = store
        LIVE["agent"] = agent
    return store, agent


# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="DocCompare AI", page_icon="🤖", layout="wide",
                   initial_sidebar_state="expanded")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;600;700&family=DM+Sans:wght@300;400;500;600&display=swap');

/* ── Base ── */
html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    background: #07090f;
    color: #c8d4e8;
}
.main, [data-testid="stAppViewContainer"] { background: #07090f; }
[data-testid="stSidebar"] {
    background: #0b0e18;
    border-right: 1px solid #151a28;
}
h1,h2,h3,code { font-family: 'JetBrains Mono', monospace !important; }

/* ── Mode toggle pills ── */
.mode-bar {
    display: flex;
    gap: 8px;
    background: #0f1220;
    border: 1px solid #1a2035;
    border-radius: 12px;
    padding: 5px;
    width: fit-content;
    margin-bottom: 1.4rem;
}
.mode-pill {
    padding: 6px 20px;
    border-radius: 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 12px;
    font-weight: 600;
    cursor: pointer;
    border: none;
    transition: all .2s;
    letter-spacing: .04em;
}
.mode-pill.active-simple {
    background: linear-gradient(135deg,#10b981,#059669);
    color: #fff;
    box-shadow: 0 0 14px rgba(16,185,129,.35);
}
.mode-pill.active-expert {
    background: linear-gradient(135deg,#3b82f6,#6366f1);
    color: #fff;
    box-shadow: 0 0 14px rgba(99,102,241,.35);
}
.mode-pill.inactive {
    background: transparent;
    color: #4a5568;
}

/* ── Doc cards ── */
.doc-card {
    background: #0f1220;
    border: 1px solid #1a2035;
    border-radius: 12px;
    padding: 1rem 1.2rem;
    margin-top: .5rem;
    transition: border-color .2s;
}
.doc-card.ready-a { border-color: #3b82f6; }
.doc-card.ready-b { border-color: #8b5cf6; }

/* ── Simple mode answer ── */
.simple-card {
    background: #0f1220;
    border: 1px solid #1a3050;
    border-radius: 16px;
    padding: 2rem 2.2rem;
    margin-top: 1rem;
    line-height: 2;
    font-size: 15px;
    color: #d0dff0;
}
.simple-section-title {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    letter-spacing: 2px;
    text-transform: uppercase;
    color: #4a5568;
    margin: 1.4rem 0 .5rem;
    padding-bottom: .3rem;
    border-bottom: 1px solid #1a2035;
}
.simple-progress {
    background: #0f1220;
    border: 1px solid #1a2035;
    border-radius: 12px;
    padding: 1.2rem 1.5rem;
    margin-bottom: .6rem;
}
.progress-label {
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4a5568;
    margin-bottom: .4rem;
    letter-spacing: 1px;
}
.progress-bar-wrap {
    background: #151a28;
    border-radius: 6px;
    height: 4px;
    overflow: hidden;
}
.progress-bar-fill {
    height: 4px;
    border-radius: 6px;
    background: linear-gradient(90deg, #3b82f6, #8b5cf6);
    animation: fill-bar 1.5s ease forwards;
}
@keyframes fill-bar { from{width:0} to{width:var(--w)} }
.step-dot {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    padding: .5rem 1rem;
    background: #0f1220;
    border: 1px solid #1a2035;
    border-radius: 8px;
    margin: .25rem 0;
    font-size: 13px;
    color: #6b7a99;
    width: 100%;
}
.step-dot.done { border-color: #10b981; color: #34d399; }
.step-dot .dot { width:8px;height:8px;border-radius:50%;background:#10b981;flex-shrink:0; }
.step-dot .spin {
    width:12px;height:12px;border-radius:50%;
    border:2px solid #3b82f6;border-top-color:transparent;
    animation:spin .8s linear infinite;flex-shrink:0;
}
@keyframes spin { to{transform:rotate(360deg)} }

/* ── Expert mode ── */
.exp-thought {
    background: #0d1117;
    border-left: 3px solid #f59e0b;
    border-radius: 0 8px 8px 0;
    padding: .7rem 1rem;
    margin: .35rem 0;
    font-size: 13px;
    color: #d0a060;
    font-style: italic;
}
.exp-tool {
    background: #090f1f;
    border-left: 3px solid #3b82f6;
    border-radius: 0 8px 8px 0;
    padding: .65rem 1rem;
    margin: .25rem 0;
    font-size: 12.5px;
    color: #7aabf0;
    font-family: 'JetBrains Mono', monospace;
}
.exp-obs {
    background: #090f0d;
    border-left: 3px solid #10b981;
    border-radius: 0 8px 8px 0;
    padding: .65rem 1rem;
    margin: .25rem 0;
    font-size: 11.5px;
    color: #5aab88;
    font-family: 'JetBrains Mono', monospace;
    white-space: pre-wrap;
    max-height: 160px;
    overflow-y: auto;
}
.exp-error {
    background: #1a0808;
    border-left: 3px solid #ef4444;
    border-radius: 0 8px 8px 0;
    padding: .7rem 1rem;
    font-size: 13px;
    color: #f87171;
}
.final-card {
    background: #0b1020;
    border: 1px solid #1e3a6e;
    border-top: 3px solid #3b82f6;
    border-radius: 0 0 14px 14px;
    padding: 1.8rem 2rem;
    margin-top: .4rem;
    color: #cdd8f0;
    line-height: 1.95;
    font-size: 14.5px;
}

/* ── Shared badges ── */
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: .05em;
    text-transform: uppercase;
}
.badge-a     { background:#0e2040; color:#60a0ff; }
.badge-b     { background:#1a1040; color:#a080ff; }
.badge-ok    { background:#0a2018; color:#34d399; }
.badge-agent { background:#1a1008; color:#fbbf24; }
.badge-simple{ background:#0a2018; color:#34d399; }

/* ── Buttons ── */
.stButton > button {
    background: linear-gradient(135deg,#3b82f6,#6366f1) !important;
    color:#fff !important;
    font-family:'JetBrains Mono',monospace !important;
    font-weight:700 !important;
    border:none !important;
    border-radius:10px !important;
    letter-spacing:.04em !important;
}
.stTextArea textarea {
    background:#0f1220 !important;
    border:1px solid #1a2035 !important;
    color:#c8d4e8 !important;
    border-radius:10px !important;
    font-family:'DM Sans',sans-serif !important;
}
.stTextInput > div > div > input {
    background:#0f1220 !important;
    border:1px solid #1a2035 !important;
    color:#c8d4e8 !important;
    border-radius:8px !important;
}
.tool-chip {
    display:inline-block;
    background:#0a1628;
    border:1px solid #1a3050;
    border-radius:6px;
    padding:2px 8px;
    font-family:'JetBrains Mono',monospace;
    font-size:11px;
    color:#5080c0;
    margin:2px 3px;
}
#MainMenu, footer { visibility:hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state init ────────────────────────────────────────────────────────
for k, v in {
    "doc_a_ready": False, "doc_b_ready": False,
    "doc_a_name": "",     "doc_b_name": "",
    "doc_a_meta": {},     "doc_b_meta": {},
    "history": [],        "current_query": "",
    "mistral_api_key": "", "model": "mistral-large-latest",
    "ui_mode": "simple",
}.items():
    if k not in st.session_state:
        st.session_state[k] = v


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 DocCompare AI")
    st.markdown(
        "<p style='color:#2a4070;font-size:12px;font-family:JetBrains Mono,monospace;'>"
        "LangGraph · Mistral AI · ChromaDB</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    api_key = st.text_input("Mistral API Key", type="password",
                             placeholder="Enter your key…",
                             value=st.session_state.get("mistral_api_key", ""))
    model   = st.selectbox("Model", ["mistral-large-latest", "mistral-small-latest"])
    top_k   = st.slider("Chunks per retrieval", 3, 10, 5)

    if api_key:
        os.environ["MISTRAL_API_KEY"] = api_key
        st.session_state["mistral_api_key"] = api_key
        st.session_state["model"] = model
        _get_store_and_agent()
        st.success("✓ Connected to Mistral", icon="🔑")

    st.divider()

    # ── Mode selector in sidebar ──────────────────────────────────────────────
    st.markdown("**View Mode**")
    mode_col1, mode_col2 = st.columns(2)
    with mode_col1:
        if st.button("🟢 Simple", use_container_width=True,
                     type="primary" if st.session_state["ui_mode"] == "simple" else "secondary"):
            st.session_state["ui_mode"] = "simple"
            st.rerun()
    with mode_col2:
        if st.button("🔵 Expert", use_container_width=True,
                     type="primary" if st.session_state["ui_mode"] == "expert" else "secondary"):
            st.session_state["ui_mode"] = "expert"
            st.rerun()

    mode = st.session_state["ui_mode"]
    if mode == "simple":
        st.markdown(
            "<p style='color:#10b981;font-size:11px;font-family:JetBrains Mono,monospace;"
            "margin-top:.3rem;'>✓ Clean answers, no jargon</p>",
            unsafe_allow_html=True,
        )
    else:
        st.markdown(
            "<p style='color:#3b82f6;font-size:11px;font-family:JetBrains Mono,monospace;"
            "margin-top:.3rem;'>✓ Live tool calls, observations, reasoning</p>",
            unsafe_allow_html=True,
        )

    st.divider()
    if mode == "expert":
        st.markdown("**Agent Tools**")
        for t in ["search_document_a","search_document_b","compare_topic",
                  "find_conflicts","find_common_ground","get_document_overview"]:
            st.markdown(f"<span class='tool-chip'>{t}</span>", unsafe_allow_html=True)
        st.divider()

    st.markdown(
        "<p style='font-size:11px;color:#1a2540;'>Built by Maaz Ali · AI Eng Portfolio</p>",
        unsafe_allow_html=True,
    )


# ── Header ────────────────────────────────────────────────────────────────────
mode = st.session_state["ui_mode"]

if mode == "simple":
    st.markdown("""
    <h1 style='color:#e2ecff;font-size:1.8rem;margin-bottom:2px;'>📄 Document Compare AI</h1>
    <p style='color:#3a5080;font-size:13px;margin-top:0;'>
    Upload two documents — get a clear, plain-English comparison
    </p>
    """, unsafe_allow_html=True)
else:
    st.markdown("""
    <h1 style='color:#c0d4f8;font-size:1.8rem;margin-bottom:2px;'>🤖 DocCompare Agent</h1>
    <p style='color:#3a5080;font-size:13px;margin-top:0;'>
    LangGraph ReAct · Live tool calls · Streaming reasoning
    </p>
    """, unsafe_allow_html=True)

st.divider()


# ── Upload helpers ────────────────────────────────────────────────────────────
def ingest_pdf(file, doc_key: str):
    store, agent = _get_store_and_agent()
    api_key  = st.session_state.get("mistral_api_key") or os.environ.get("MISTRAL_API_KEY")
    pdf_bytes = file.read()

    with st.spinner(f"📖 Reading {file.name}…"):
        docs    = extract_langchain_docs(pdf_bytes, doc_key, api_key=api_key)
        summary = extraction_summary(docs)

    methods      = summary.get("methods", {})
    vision_pages = methods.get("vision", 0)
    text_pages   = methods.get("text", 0)

    if summary["total_chunks"] == 0:
        st.error("❌ Could not read this PDF. Check your API key is set.")
        return 0

    mode = st.session_state["ui_mode"]
    if vision_pages > 0 and text_pages == 0:
        if mode == "simple":
            st.success(f"✓ Document read successfully! Found {summary['total_chunks']} sections.")
        else:
            st.info(f"🤖 Image-based PDF → AI Vision extracted {summary['total_chunks']} sections.", icon="🔍")
    elif vision_pages > 0:
        if mode == "simple":
            st.success(f"✓ Document read successfully! Found {summary['total_chunks']} sections.")
        else:
            st.info(f"🤖 Mixed PDF: {text_pages} text + {vision_pages} vision pages → {summary['total_chunks']} sections.", icon="🔍")

    with st.spinner("⚡ Indexing…"):
        n    = store.ingest(doc_key, docs, file.name)
        meta = store.meta(doc_key)
        meta["vision_pages"] = vision_pages
        meta["text_pages"]   = text_pages

    st.session_state[f"{doc_key}_ready"] = True
    st.session_state[f"{doc_key}_name"]  = file.name
    st.session_state[f"{doc_key}_meta"]  = meta

    agent.configure(store, st.session_state["doc_a_name"],
                    st.session_state["doc_b_name"], top_k)
    return n


def _render_doc_card(doc_key: str, card_class: str):
    m    = st.session_state[f"{doc_key}_meta"]
    name = st.session_state[f"{doc_key}_name"]
    mode = st.session_state["ui_mode"]

    if mode == "simple":
        pages  = m.get("page_count", "?")
        st.markdown(f"""<div class='doc-card {card_class}'>
            <p style='margin:0;font-weight:600;color:#c8d4e8;'>📄 {name}</p>
            <p style='margin:4px 0 6px;font-size:13px;color:#4a5a7a;'>{pages} page(s) · Ready to compare</p>
            <span class='badge badge-ok'>✓ Ready</span>
        </div>""", unsafe_allow_html=True)
    else:
        chunks = m.get("chunk_count", "?")
        pages  = m.get("page_count", "?")
        vp     = m.get("vision_pages", 0)
        method = ("🤖 AI Vision" if vp > 0 else "📝 Text")
        method_style = ("color:#fbbf24" if vp > 0 else "color:#34d399")
        st.markdown(f"""<div class='doc-card {card_class}'>
            <p style='margin:0;font-weight:600;color:#c8d4e8;'>📄 {name}</p>
            <p style='margin:4px 0 2px;font-size:12px;color:#2a4060;'>
              {pages} pages · {chunks} chunks · <span style='{method_style}'>{method}</span>
            </p>
            <span class='badge badge-ok'>✓ Indexed</span>
        </div>""", unsafe_allow_html=True)


# ── Upload columns ────────────────────────────────────────────────────────────
mode = st.session_state["ui_mode"]

if mode == "simple":
    st.markdown("### Step 1 — Upload your two documents")
    up_label = "Drag and drop your PDF here (any type — CV, report, contract, scan)"
else:
    st.markdown("#### Documents")
    up_label = "Upload any PDF — text, scanned, or designed CV"

col_a, _, col_b = st.columns([1, 0.05, 1])

with col_a:
    badge_a = "<span class='badge badge-a'>Document A</span>" if mode == "expert" else "<span style='font-size:13px;color:#3b82f6;font-family:JetBrains Mono,monospace;font-weight:600;'>📘 First Document</span>"
    st.markdown(badge_a, unsafe_allow_html=True)
    if st.session_state["doc_a_ready"]:
        _render_doc_card("doc_a", "ready-a")
    fa = st.file_uploader(up_label, type=["pdf"], key="up_a", label_visibility="collapsed")
    store, _ = _get_store_and_agent()
    if fa and store and not st.session_state["doc_a_ready"]:
        ingest_pdf(fa, "doc_a")
        st.rerun()
    elif fa and not store:
        st.warning("⚠️ Enter your Mistral API key in the sidebar first.")

with col_b:
    badge_b = "<span class='badge badge-b'>Document B</span>" if mode == "expert" else "<span style='font-size:13px;color:#8b5cf6;font-family:JetBrains Mono,monospace;font-weight:600;'>📙 Second Document</span>"
    st.markdown(badge_b, unsafe_allow_html=True)
    if st.session_state["doc_b_ready"]:
        _render_doc_card("doc_b", "ready-b")
    fb = st.file_uploader(up_label, type=["pdf"], key="up_b", label_visibility="collapsed")
    store, _ = _get_store_and_agent()
    if fb and store and not st.session_state["doc_b_ready"]:
        ingest_pdf(fb, "doc_b")
        st.rerun()
    elif fb and not store:
        st.warning("⚠️ Enter your Mistral API key in the sidebar first.")

st.divider()
both_ready = st.session_state["doc_a_ready"] and st.session_state["doc_b_ready"]


# ── Query section ─────────────────────────────────────────────────────────────
if mode == "simple":
    st.markdown("### Step 2 — Ask your question")
    SUGGESTIONS_SIMPLE = [
        ("📋", "What are the main differences?"),
        ("🤝", "What do both documents agree on?"),
        ("⚠️", "Are there any contradictions?"),
        ("📌", "What does only one document mention?"),
        ("📊", "Give me a full comparison summary"),
        ("🔍", "Which is more detailed?"),
    ]
    if both_ready:
        st.markdown(
            "<p style='color:#4a5a7a;font-size:13px;margin-bottom:.8rem;'>Choose a question or type your own:</p>",
            unsafe_allow_html=True,
        )
        cols = st.columns(3)
        for i, (icon, label) in enumerate(SUGGESTIONS_SIMPLE):
            with cols[i % 3]:
                if st.button(f"{icon} {label}", key=f"sq_{i}", use_container_width=True):
                    st.session_state["current_query"] = label
                    st.rerun()

    query = st.text_area(
        "Your question",
        value=st.session_state.get("current_query", ""),
        placeholder="e.g. What are the main differences between these two documents?",
        height=80,
        label_visibility="visible",
    )
    _, agent = _get_store_and_agent()
    run = st.button("🔍 Compare Now", disabled=not (both_ready and query and agent),
                    use_container_width=True)
    if not both_ready:
        st.info("📂 Upload both documents above to get started.", icon="👆")

else:
    # Expert mode
    st.markdown("#### Ask the Agent")
    SUGGESTIONS_EXPERT = [
        "What are the key differences between these two documents?",
        "Do both documents agree on their main conclusions?",
        "Are there any contradictions or conflicts between them?",
        "What topics does only one document cover?",
        "Give me a detailed comparison of the methodologies.",
        "What are the most important similarities?",
    ]
    with st.expander("💡 Suggested queries", expanded=False):
        for s in SUGGESTIONS_EXPERT:
            if st.button(s, key=f"eq_{s}", use_container_width=True):
                st.session_state["current_query"] = s
                st.rerun()

    query = st.text_area(
        "Query",
        value=st.session_state.get("current_query", ""),
        placeholder="e.g. What are the key differences?",
        height=85,
        label_visibility="collapsed",
    )
    _, agent = _get_store_and_agent()
    run = st.button("🚀 Run Agent", disabled=not (both_ready and query and agent),
                    use_container_width=True)
    if not both_ready:
        st.info("Upload both PDFs above to activate the agent.", icon="⬆️")


# ── Agent execution ───────────────────────────────────────────────────────────
if run and query and both_ready:
    store, agent = _get_store_and_agent()
    agent.configure(store, st.session_state["doc_a_name"],
                    st.session_state["doc_b_name"], top_k)

    st.markdown("---")
    mode = st.session_state["ui_mode"]

    collected_steps  = []
    final_answer     = ""
    tool_call_count  = 0

    # ── SIMPLE MODE: friendly progress ────────────────────────────────────── #
    if mode == "simple":
        st.markdown(
            "<p style='color:#4a5a7a;font-size:13px;'>🔍 Analysing your documents…</p>",
            unsafe_allow_html=True,
        )
        step_names = {
            "get_document_overview": "📖 Reading document structure",
            "compare_topic":         "⚖️  Comparing key topics",
            "search_document_a":     "🔎 Searching first document",
            "search_document_b":     "🔎 Searching second document",
            "find_conflicts":        "⚠️  Checking for contradictions",
            "find_common_ground":    "🤝 Finding common ground",
        }
        progress_container = st.container()
        steps_done = []

        with progress_container:
            for step in agent.stream_steps(query):
                collected_steps.append(step)
                t = step["type"]

                if t == "tool_call":
                    tool_call_count += 1
                    friendly = step_names.get(step["tool"], f"🔧 Checking details ({step['tool']})")
                    steps_done.append(friendly)
                    # Rerender all steps done so far
                    progress_container.empty()
                    with progress_container:
                        for s in steps_done[:-1]:
                            st.markdown(
                                f"<div class='step-dot done'><span class='dot'></span>{s}</div>",
                                unsafe_allow_html=True,
                            )
                        st.markdown(
                            f"<div class='step-dot'><span class='spin'></span>{steps_done[-1]}</div>",
                            unsafe_allow_html=True,
                        )
                elif t == "final":
                    final_answer = step["content"]
                elif t == "error":
                    st.error(f"Something went wrong: {step['content']}")

        # Mark all done
        if steps_done:
            progress_container.empty()
            for s in steps_done:
                st.markdown(
                    f"<div class='step-dot done'><span class='dot'></span>{s}</div>",
                    unsafe_allow_html=True,
                )

        # Render final answer in simple readable card
        if final_answer:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("<span class='badge badge-simple'>✅ Here's what I found</span>",
                        unsafe_allow_html=True)
            # Strip any markdown headers and reformat for plain reading
            clean = final_answer.replace("###", "").replace("##", "").replace("**", "")
            formatted = clean.replace("\n", "<br>")
            st.markdown(f"<div class='simple-card'>{formatted}</div>",
                        unsafe_allow_html=True)

    # ── EXPERT MODE: raw technical stream ─────────────────────────────────── #
    else:
        st.markdown("<span class='badge badge-agent'>🤖 Agent Reasoning</span>",
                    unsafe_allow_html=True)
        steps_container = st.container()

        with steps_container:
            for step in agent.stream_steps(query):
                collected_steps.append(step)
                t = step["type"]

                if t == "thought" and step["content"]:
                    st.markdown(
                        f"<div class='exp-thought'>🤔 <b>Thinking:</b> {step['content']}</div>",
                        unsafe_allow_html=True,
                    )
                elif t == "tool_call":
                    tool_call_count += 1
                    st.markdown(
                        f"<div class='exp-tool'>"
                        f"🔧 <b>Tool #{tool_call_count}:</b> <code>{step['tool']}</code><br>"
                        f"<span style='color:#4060a0;'>↳ {step['input']}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                elif t == "observation":
                    preview = step["content"][:450] + ("…" if len(step["content"]) > 450 else "")
                    st.markdown(
                        f"<div class='exp-obs'>👁 {preview}</div>",
                        unsafe_allow_html=True,
                    )
                elif t == "final":
                    final_answer = step["content"]
                elif t == "error":
                    st.markdown(
                        f"<div class='exp-error'>❌ {step['content']}</div>",
                        unsafe_allow_html=True,
                    )

        if final_answer:
            st.markdown("<span class='badge badge-ok'>✅ Final Answer</span>",
                        unsafe_allow_html=True)
            formatted = final_answer.replace("\n", "<br>")
            st.markdown(f"<div class='final-card'>{formatted}</div>",
                        unsafe_allow_html=True)

    # Save to history
    if final_answer:
        st.session_state["history"].insert(0, {
            "query":      query,
            "answer":     final_answer,
            "steps":      collected_steps,
            "tool_calls": tool_call_count,
            "mode":       mode,
        })
        st.session_state["current_query"] = ""


# ── History ───────────────────────────────────────────────────────────────────
if st.session_state["history"]:
    st.markdown("---")
    mode = st.session_state["ui_mode"]
    label = "📜 Previous Questions" if mode == "simple" else f"📜 Query History ({len(st.session_state['history'])})"

    with st.expander(label, expanded=False):
        for i, item in enumerate(st.session_state["history"]):
            tag = "Latest" if i == 0 else f"#{i+1}"
            item_mode = item.get("mode", "expert")

            if item_mode == "simple":
                st.markdown(
                    f"<p style='color:#60a0d0;font-weight:600;font-size:13px;'>❓ {item['query']}</p>",
                    unsafe_allow_html=True,
                )
                clean = item["answer"].replace("###","").replace("##","").replace("**","")
                st.markdown(
                    f"<div class='simple-card' style='font-size:13px;'>"
                    f"{clean[:500].replace(chr(10),'<br>')}{'…' if len(clean)>500 else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.markdown(
                    f"<p style='color:#4488ff;font-size:13px;font-weight:600;'>"
                    f"[{tag}] {item['query']} "
                    f"<span style='color:#1a3050;font-size:11px;'>({item['tool_calls']} tool calls)</span>"
                    f"</p>",
                    unsafe_allow_html=True,
                )
                st.markdown(
                    f"<div class='final-card' style='font-size:13px;'>"
                    f"{item['answer'][:500].replace(chr(10),'<br>')}{'…' if len(item['answer'])>500 else ''}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            st.markdown("")
