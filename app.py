"""
app.py  ·  AI Document Comparison Agent (LangGraph + Mistral + ChromaDB)
Run:  streamlit run app.py
"""

import os
import time
import streamlit as st
from pdf_processor import extract_langchain_docs
from vector_store import VectorStoreManager
from agent import DocumentComparisonAgent

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DocAgent AI",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── CSS ────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@400;700&family=Sora:wght@300;400;600;700&display=swap');

html, body, [class*="css"] { font-family: 'Sora', sans-serif; }
.main, [data-testid="stAppViewContainer"] { background: #080b14; }
[data-testid="stSidebar"] { background: #0d1117; border-right: 1px solid #161b27; }

h1, h2, h3, code { font-family: 'JetBrains Mono', monospace !important; }

/* Step cards */
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
.badge-a   { background: #112244; color: #4488ff; }
.badge-b   { background: #331144; color: #bb66ff; }
.badge-ok  { background: #0d2d1a; color: #33cc77; }
.badge-agent{ background: #1a1200; color: #ffcc33; }
.doc-card  {
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


# ── Session state ──────────────────────────────────────────────────────────────
def _init():
    defs = {
        "store": None, "agent": None,
        "doc_a_ready": False, "doc_b_ready": False,
        "doc_a_name": "", "doc_b_name": "",
        "doc_a_meta": {}, "doc_b_meta": {},
        "history": [],
        "current_query": "",
    }
    for k, v in defs.items():
        if k not in st.session_state:
            st.session_state[k] = v

_init()


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🤖 DocAgent AI")
    st.markdown(
        "<p style='color:#4060a0;font-size:12px;'>Agentic RAG · LangGraph ReAct<br>"
        "Mistral AI · ChromaDB · LangChain</p>",
        unsafe_allow_html=True,
    )
    st.divider()

    api_key = st.text_input("Mistral API Key", type="password", placeholder="your key here")
    model = st.selectbox(
        "Model",
        ["mistral-large-latest", "mistral-small-latest"],
        index=0,
    )
    top_k = st.slider("Chunks per retrieval", 3, 10, 5)

    if api_key:
        os.environ["MISTRAL_API_KEY"] = api_key
        if st.session_state["store"] is None:
            st.session_state["store"] = VectorStoreManager(api_key)
            st.session_state["agent"] = DocumentComparisonAgent(api_key, model)
        st.success("✓ Connected to Mistral", icon="🔑")

    st.divider()
    st.markdown("**Agent Tools Available**")
    TOOL_LABELS = [
        "search_document_a", "search_document_b",
        "compare_topic", "find_conflicts",
        "find_common_ground", "get_document_overview",
    ]
    for t in TOOL_LABELS:
        st.markdown(f"<span class='tool-chip'>{t}</span>", unsafe_allow_html=True)

    st.divider()
    st.markdown("<p style='font-size:11px;color:#2a3050;'>Built by Maaz · AI Eng Portfolio</p>",
                unsafe_allow_html=True)


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown("""
<h1 style='color:#c0d4f8;margin-bottom:2px;font-size:1.9rem;'>🤖 AI Document Comparison Agent</h1>
<p style='color:#3a5080;margin-top:0;font-size:13px;'>
Agentic RAG · The AI autonomously decides what to search, how many times, and how to reason
</p>
""", unsafe_allow_html=True)
st.divider()


# ── Upload columns ─────────────────────────────────────────────────────────────
def ingest_pdf(file, doc_key: str):
    store: VectorStoreManager = st.session_state["store"]
    agent: DocumentComparisonAgent = st.session_state["agent"]
    pdf_bytes = file.read()
    with st.spinner(f"Embedding {file.name}…"):
        docs = extract_langchain_docs(pdf_bytes, doc_key)
        n = store.ingest(doc_key, docs, file.name)
        meta = store.meta(doc_key)

    st.session_state[f"{doc_key}_ready"] = True
    st.session_state[f"{doc_key}_name"] = file.name
    st.session_state[f"{doc_key}_meta"] = meta

    # Re-configure tools with latest store + names
    agent.configure(
        store,
        st.session_state["doc_a_name"],
        st.session_state["doc_b_name"],
        top_k,
    )
    return n


col_a, _, col_b = st.columns([1, 0.06, 1])

with col_a:
    st.markdown("<span class='badge badge-a'>Document A</span>", unsafe_allow_html=True)
    if st.session_state["doc_a_ready"]:
        m = st.session_state["doc_a_meta"]
        st.markdown(f"""<div class='doc-card ready'>
            <p style='color:#4488ff;margin:0;font-weight:600;'>📄 {st.session_state['doc_a_name']}</p>
            <p style='color:#2a4060;font-size:12px;margin:4px 0 0;'>
              {m.get('page_count','?')} pages · {m.get('chunk_count','?')} vectors
            </p>
            <span class='badge badge-ok'>✓ Indexed</span>
        </div>""", unsafe_allow_html=True)

    fa = st.file_uploader("Upload Document A", type=["pdf"], key="up_a", label_visibility="collapsed")
    if fa and st.session_state["store"]:
        ingest_pdf(fa, "doc_a")
        st.rerun()
    elif fa:
        st.warning("Set Mistral API key first.")

with col_b:
    st.markdown("<span class='badge badge-b'>Document B</span>", unsafe_allow_html=True)
    if st.session_state["doc_b_ready"]:
        m = st.session_state["doc_b_meta"]
        st.markdown(f"""<div class='doc-card ready'>
            <p style='color:#bb66ff;margin:0;font-weight:600;'>📄 {st.session_state['doc_b_name']}</p>
            <p style='color:#3a2060;font-size:12px;margin:4px 0 0;'>
              {m.get('page_count','?')} pages · {m.get('chunk_count','?')} vectors
            </p>
            <span class='badge badge-ok'>✓ Indexed</span>
        </div>""", unsafe_allow_html=True)

    fb = st.file_uploader("Upload Document B", type=["pdf"], key="up_b", label_visibility="collapsed")
    if fb and st.session_state["store"]:
        ingest_pdf(fb, "doc_b")
        st.rerun()
    elif fb:
        st.warning("Set Mistral API key first.")

st.divider()


# ── Query ──────────────────────────────────────────────────────────────────────
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
run = st.button(
    "🚀 Run Agent",
    disabled=not (both_ready and query and st.session_state["agent"]),
    use_container_width=True,
)

if not both_ready:
    st.info("Upload both PDFs above to activate the agent.", icon="⬆️")


# ── Agent execution with live step rendering ───────────────────────────────────
if run and query and both_ready:
    agent: DocumentComparisonAgent = st.session_state["agent"]
    # Re-configure in case top_k changed
    agent.configure(
        st.session_state["store"],
        st.session_state["doc_a_name"],
        st.session_state["doc_b_name"],
        top_k,
    )

    st.markdown("---")
    st.markdown(
        "<span class='badge badge-agent'>🤖 Agent Reasoning</span>",
        unsafe_allow_html=True,
    )

    steps_container = st.container()
    final_placeholder = st.empty()
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

    # Display final answer
    if final_answer:
        st.markdown(
            "<span class='badge badge-ok'>✅ Final Answer</span>",
            unsafe_allow_html=True,
        )
        formatted = final_answer.replace("\n", "<br>")
        st.markdown(
            f"<div class='final-answer'>{formatted}</div>",
            unsafe_allow_html=True,
        )

        # Save to history
        st.session_state["history"].insert(0, {
            "query": query,
            "answer": final_answer,
            "steps": collected_steps,
            "tool_calls": tool_call_count,
        })
        st.session_state["current_query"] = ""


# ── History ────────────────────────────────────────────────────────────────────
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
