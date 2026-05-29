"""
app.py  ·  AI Document Comparison Agent (LangGraph + Mistral + ChromaDB)
Run:  streamlit run app.py
"""

import os
import streamlit as st
from pdf_processor import extract_langchain_docs, extraction_summary
from vector_store import VectorStoreManager
from agent import DocumentComparisonAgent

# Live in-memory cache for non-serializable objects
LIVE: dict = {}


def _get_store_and_agent():
    store = LIVE.get("store")
    agent = LIVE.get("agent")
    if store and agent:
        return store, agent

    api_key = st.session_state.get("mistral_api_key") or os.environ.get("MISTRAL_API_KEY")
    model = st.session_state.get("model", "mistral-large-latest")
    if api_key:
        store = VectorStoreManager(api_key)
        agent = DocumentComparisonAgent(api_key, model)
        LIVE["store"] = store
        LIVE["agent"] = agent
    return store, agent


# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocAgent AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ─────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.main, [data-testid="stAppViewContainer"] { background: #080b14; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #161b27; }

h1, h2, h3, code { font-family: 'JetBrains Mono', monospace !important; }

.step-thought {
    background: #0d1117;
    border-left: 3px solid #f0c040;
    border-radius: 0 8px 8px 0;
    padding: .7rem 1rem;
    margin: .4rem 0;
    font-size: 13px;
    color: #d0c090;
    font-style: italic;
}
.step-tool-call {
    background: #0a1628;
    border-left: 3px solid #3b8ef3;
    border-radius: 0 8px 8px 0;
    padding: .7rem 1rem;
    margin: .4rem 0;
    font-size: 13px;
    color: #90b8f8;
}
.step-obs {
    background: #0d1f0d;
    border-left: 3px solid #30a050;
    border-radius: 0 8px 8px 0;
    padding: .7rem 1rem;
    margin: .4rem 0;
    font-size: 12px;
    color: #80c090;
    font-family: 'JetBrains Mono', monospace;
    white-space: pre-wrap;
    max-height: 180px;
    overflow-y: auto;
}
.step-error {
    background: #1f0d0d;
    border-left: 3px solid #e05050;
    border-radius: 0 8px 8px 0;
    padding: .7rem 1rem;
    margin: .4rem 0;
    font-size: 13px;
    color: #e09090;
}
.final-answer {
    background: #0d1117;
    border: 1px solid #1e3a5f;
    border-top: 3px solid #3b8ef3;
    border-radius: 0 0 12px 12px;
    padding: 1.6rem;
    margin-top: .5rem;
    color: #d0dff0;
    line-height: 1.9;
    font-size: 14.5px;
}
.badge {
    display: inline-block;
    padding: 3px 12px;
    border-radius: 20px;
    font-size: 11px;
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    letter-spacing: .06em;
    text-transform: uppercase;
}
.badge-a    { background: #112244; color: #4488ff; }
.badge-b    { background: #331144; color: #bb66ff; }
.badge-ok   { background: #0d2d1a; color: #33cc77; }
.badge-agent{ background: #1a1200; color: #ffcc33; }
.doc-card {
    background: #0d1117;
    border: 1px solid #161b27;
    border-radius: 10px;
    padding: 1rem 1.2rem;
    margin-top: .5rem;
}
.doc-card.ready { border-color: #33cc77; }
.tool-chip {
    display: inline-block;
    background: #0a1628;
    border: 1px solid #1e3a5f;
    border-radius: 6px;
    padding: 2px 8px;
    font-family: 'JetBrains Mono', monospace;
    font-size: 11px;
    color: #4488ff;
    margin: 2px 3px;
}
.stButton > button {
    background: linear-gradient(135deg, #3b8ef3, #7c3af3) !important;
    color: #fff !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-weight: 700 !important;
    border: none !important;
    border-radius: 8px !important;
}
.stTextArea textarea {
    background: #0d1117 !important;
    border: 1px solid #1e2d42 !important;
    color: #c0d0e8 !important;
    border-radius: 8px !important;
}
#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ── Session state ─────────────────────────────────────────────────────────────
def _init():
    defs = {
        "doc_a_ready": False, "doc_b_ready": False,
        "doc_a_name": "", "doc_b_name": "",
        "doc_a_meta": {}, "doc_b_meta": {},
        "history": [],
        "current_query": "",
        "mistral_api_key": "",
        "model": "mistral-large-latest",
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 DocAgent AI")
    st.markdown(
        "<p style='color:#4060a0;font-size:12px;'>Agentic RAG · LangGraph ReAct<br>"
        "Mistral AI · ChromaDB · LangChain</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    api_key = st.text_input("Mistral API Key", type="password", placeholder="your key here")
    model = st.selectbox("Model", ["mistral-large-latest", "mistral-small-latest"], index=0)
    top_k = st.slider("Chunks per retrieval", 3, 10, 5)

    if api_key:
        os.environ["MISTRAL_API_KEY"] = api_key
        st.session_state["mistral_api_key"] = api_key
        st.session_state["model"] = model
        _get_store_and_agent()
        st.success("✓ Connected to Mistral", icon="🔑")

    st.divider()
    st.markdown("**Agent Tools Available**")
    for t in ["search_document_a", "search_document_b", "compare_topic",
              "find_conflicts", "find_common_ground", "get_document_overview"]:
        st.markdown(f"<span class='tool-chip'>{t}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-size:11px;color:#2a3050;'>Built by Maaz Ali · AI Eng Portfolio</p>",
                unsafe_allow_html=True)


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#c0d4f8;margin-bottom:2px;font-size:1.9rem;'>🤖 AI Document Comparison Agent</h1>
<p style='color:#3a5080;margin-top:0;font-size:13px;'>
Agentic RAG · The AI autonomously decides what to search, how many times, and how to reason
</p>
""", unsafe_allow_html=True)
st.divider()


# ── Upload ────────────────────────────────────────────────────────────────────
def ingest_pdf(file, doc_key: str):
    """
    Smart ingestion with automatic vision fallback.
    Works on: normal text PDFs, designed CVs, scanned docs, handwritten notes.
    """
    store, agent = _get_store_and_agent()
    api_key = st.session_state.get("mistral_api_key") or os.environ.get("MISTRAL_API_KEY")
    pdf_bytes = file.read()

    with st.spinner(f"📖 Reading {file.name}…"):
        docs = extract_langchain_docs(pdf_bytes, doc_key, api_key=api_key)
        summary = extraction_summary(docs)

    methods = summary.get("methods", {})
    vision_pages = methods.get("vision", 0)
    text_pages   = methods.get("text", 0)

    if summary["total_chunks"] == 0:
        st.error("❌ Could not read this PDF. Make sure your Mistral API key is entered (needed for image-based PDFs).")
        return 0
    elif vision_pages > 0 and text_pages == 0:
        st.info(f"🤖 Image-based PDF detected — AI Vision read it automatically. Extracted {summary['total_chunks']} sections.", icon="🔍")
    elif vision_pages > 0:
        st.info(f"🤖 Mixed PDF: {text_pages} text pages + {vision_pages} image pages read via AI Vision. Total: {summary['total_chunks']} sections.", icon="🔍")

    with st.spinner(f"⚡ Indexing {summary['total_chunks']} sections…"):
        n = store.ingest(doc_key, docs, file.name)
        meta = store.meta(doc_key)
        meta["vision_pages"] = vision_pages
        meta["text_pages"]   = text_pages

    st.session_state[f"{doc_key}_ready"] = True
    st.session_state[f"{doc_key}_name"]  = file.name
    st.session_state[f"{doc_key}_meta"]  = meta

    agent.configure(
        store,
        st.session_state["doc_a_name"],
        st.session_state["doc_b_name"],
        top_k,
    )
    return n


def _render_doc_card(doc_key: str, color: str):
    m    = st.session_state[f"{doc_key}_meta"]
    name = st.session_state[f"{doc_key}_name"]
    vision_pages = m.get("vision_pages", 0)
    if vision_pages > 0:
        method_badge = "<span style='background:#1a1200;color:#ffcc33;border-radius:4px;padding:1px 7px;font-size:11px;'>🤖 AI Vision</span>"
    else:
        method_badge = "<span style='background:#0d2d1a;color:#33cc77;border-radius:4px;padding:1px 7px;font-size:11px;'>📝 Text</span>"
    st.markdown(f"""<div class='doc-card ready'>
        <p style='color:{color};margin:0;font-weight:600;'>📄 {name}</p>
        <p style='color:#2a4060;font-size:12px;margin:4px 0 0;'>
          {m.get('page_count','?')} pages · {m.get('chunk_count','?')} sections indexed
        </p>
        <span class='badge badge-ok'>✓ Ready</span>&nbsp;{method_badge}
    </div>""", unsafe_allow_html=True)


col_a, _, col_b = st.columns([1, 0.06, 1])

with col_a:
    st.markdown("<span class='badge badge-a'>Document A</span>", unsafe_allow_html=True)
    if st.session_state["doc_a_ready"]:
        _render_doc_card("doc_a", "#4488ff")
    fa = st.file_uploader(
        "Upload any PDF — text, scanned, or designed CV",
        type=["pdf"], key="up_a", label_visibility="visible"
    )
    store, _ = _get_store_and_agent()
    if fa and store and not st.session_state["doc_a_ready"]:
        ingest_pdf(fa, "doc_a")
        st.rerun()
    elif fa and not store:
        st.warning("⚠️ Enter your Mistral API key in the sidebar first.")

with col_b:
    st.markdown("<span class='badge badge-b'>Document B</span>", unsafe_allow_html=True)
    if st.session_state["doc_b_ready"]:
        _render_doc_card("doc_b", "#bb66ff")
    fb = st.file_uploader(
        "Upload any PDF — text, scanned, or designed CV",
        type=["pdf"], key="up_b", label_visibility="visible"
    )
    store, _ = _get_store_and_agent()
    if fb and store and not st.session_state["doc_b_ready"]:
        ingest_pdf(fb, "doc_b")
        st.rerun()
    elif fb and not store:
        st.warning("⚠️ Enter your Mistral API key in the sidebar first.")

st.divider()


# ── Query ─────────────────────────────────────────────────────────────────────
st.markdown("### 💬 Ask the Agent")

SUGGESTIONS = [
    "What are the key differences between these two documents?",
    "Do both documents agree on their main conclusions?",
    "Are there any contradictions or conflicts between them?",
    "What topics does only one document cover?",
    "Give me a detailed comparison of the methodologies described.",
    "What are the most important similarities between these documents?",
]

with st.expander("💡 Suggested questions", expanded=False):
    for s in SUGGESTIONS:
        if st.button(s, key=f"sug_{s}", use_container_width=True):
            st.session_state["current_query"] = s
            st.rerun()

query = st.text_area(
    "Question",
    value=st.session_state.get("current_query", ""),
    placeholder="e.g. What are the key differences between these two documents?",
    height=85,
    label_visibility="collapsed",
)

both_ready = st.session_state["doc_a_ready"] and st.session_state["doc_b_ready"]
_, agent = _get_store_and_agent()
run = st.button(
    "🚀 Run Agent",
    disabled=not (both_ready and query and agent),
    use_container_width=True,
)

if not both_ready:
    st.info("Upload both PDFs above to activate the agent.", icon="⬆️")


# ── Agent execution ───────────────────────────────────────────────────────────
if run and query and both_ready:
    store, agent = _get_store_and_agent()
    agent.configure(
        store,
        st.session_state["doc_a_name"],
        st.session_state["doc_b_name"],
        top_k,
    )

    st.markdown("---")
    st.markdown("<span class='badge badge-agent'>🤖 Agent Reasoning</span>", unsafe_allow_html=True)

    steps_container = st.container()
    collected_steps = []
    final_answer = ""
    tool_call_count = 0

    with steps_container:
        for step in agent.stream_steps(query):
            collected_steps.append(step)
            t = step["type"]

            if t == "thought" and step["content"]:
                st.markdown(
                    f"<div class='step-thought'>🤔 <b>Thinking:</b> {step['content']}</div>",
                    unsafe_allow_html=True,
                )
            elif t == "tool_call":
                tool_call_count += 1
                st.markdown(
                    f"<div class='step-tool-call'>🔧 <b>Tool #{tool_call_count}:</b> "
                    f"<code>{step['tool']}</code><br>"
                    f"<span style='color:#5580c0;font-size:12px;'>↳ query: \"{step['input']}\"</span>"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            elif t == "observation":
                preview = step["content"][:400] + ("…" if len(step["content"]) > 400 else "")
                st.markdown(
                    f"<div class='step-obs'>👁 {preview}</div>",
                    unsafe_allow_html=True,
                )
            elif t == "final":
                final_answer = step["content"]
            elif t == "error":
                st.markdown(
                    f"<div class='step-error'>❌ Error: {step['content']}</div>",
                    unsafe_allow_html=True,
                )

    if final_answer:
        st.markdown("<span class='badge badge-ok'>✅ Final Answer</span>", unsafe_allow_html=True)
        formatted = final_answer.replace("\n", "<br>")
        st.markdown(f"<div class='final-answer'>{formatted}</div>", unsafe_allow_html=True)

        st.session_state["history"].insert(0, {
            "query": query,
            "answer": final_answer,
            "steps": collected_steps,
            "tool_calls": tool_call_count,
        })
        st.session_state["current_query"] = ""


# ── History ───────────────────────────────────────────────────────────────────
if st.session_state["history"]:
    st.markdown("---")
    with st.expander(
        f"📜 Query history ({len(st.session_state['history'])} session{'s' if len(st.session_state['history'])>1 else ''})",
        expanded=False,
    ):
        for i, item in enumerate(st.session_state["history"]):
            label = "Latest" if i == 0 else f"#{i+1}"
            st.markdown(
                f"<p style='color:#4488ff;font-size:13px;font-weight:600;'>"
                f"[{label}] {item['query']} "
                f"<span style='color:#2a4060;font-size:11px;'>({item['tool_calls']} tool calls)</span>"
                f"</p>",
                unsafe_allow_html=True,
            )
            st.markdown(
                f"<div class='final-answer' style='font-size:13px;'>"
                f"{item['answer'][:600]}{'…' if len(item['answer']) > 600 else ''}"
                f"</div>",
                unsafe_allow_html=True,
            )
            st.markdown("")
