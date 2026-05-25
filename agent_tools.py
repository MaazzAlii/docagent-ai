"""
agent_tools.py  ·  6 LangChain tools for intelligent document comparison.

The agent autonomously decides which tools to call, in what order, and
how many times — enabling multi-step reasoning far beyond simple RAG.

Tools:
  1. search_document_a      → semantic search in Document A
  2. search_document_b      → semantic search in Document B
  3. compare_topic          → parallel search + structural comparison
  4. find_conflicts         → contradiction detection between docs
  5. find_common_ground     → agreement / consensus detection
  6. get_document_overview  → stats + broad topic sweep of a doc
"""

from __future__ import annotations
from typing import Annotated
from langchain_core.tools import tool
from vector_store import VectorStoreManager

# ── Registry ───────────────────────────────────────────────────────────────── #
# Populated at runtime after PDFs are ingested.
_registry: dict = {
    "store": None,       # VectorStoreManager instance
    "name_a": "Doc A",
    "name_b": "Doc B",
    "top_k": 5,
}


def configure_tools(store: VectorStoreManager, name_a: str, name_b: str, top_k: int = 5):
    """Call this after ingesting both PDFs to wire tools to the live store."""
    _registry["store"] = store
    _registry["name_a"] = name_a
    _registry["name_b"] = name_b
    _registry["top_k"] = top_k


def _store() -> VectorStoreManager:
    if _registry["store"] is None:
        raise RuntimeError("Tools not configured — call configure_tools() first.")
    return _registry["store"]


# ── Tool 1: Search Document A ──────────────────────────────────────────────── #

@tool
def search_document_a(
    query: Annotated[str, "Specific search query to find relevant passages in Document A"]
) -> str:
    """
    Perform a semantic search inside Document A and return the most relevant excerpts.
    Use this when you need to find what Document A specifically says about a topic,
    claim, section, or concept. You can call this multiple times with different queries
    to gather more evidence.
    """
    vs = _store()
    name = _registry["name_a"]
    docs = vs.search("doc_a", query, k=_registry["top_k"])
    if not docs:
        return f"[{name}]: No relevant content found for query: '{query}'"

    lines = [f"Results from [{name}] for query: '{query}'\n"]
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        lines.append(f"  [{i}] Page {page}: {doc.page_content}")
    return "\n".join(lines)


# ── Tool 2: Search Document B ──────────────────────────────────────────────── #

@tool
def search_document_b(
    query: Annotated[str, "Specific search query to find relevant passages in Document B"]
) -> str:
    """
    Perform a semantic search inside Document B and return the most relevant excerpts.
    Use this when you need to find what Document B specifically says about a topic,
    claim, section, or concept. You can call this multiple times with different queries
    to gather more evidence.
    """
    vs = _store()
    name = _registry["name_b"]
    docs = vs.search("doc_b", query, k=_registry["top_k"])
    if not docs:
        return f"[{name}]: No relevant content found for query: '{query}'"

    lines = [f"Results from [{name}] for query: '{query}'\n"]
    for i, doc in enumerate(docs, 1):
        page = doc.metadata.get("page", "?")
        lines.append(f"  [{i}] Page {page}: {doc.page_content}")
    return "\n".join(lines)


# ── Tool 3: Compare Topic ──────────────────────────────────────────────────── #

@tool
def compare_topic(
    topic: Annotated[str, "The topic, concept, or claim to compare across both documents simultaneously"]
) -> str:
    """
    Search BOTH documents simultaneously for a given topic and return a structured
    side-by-side view of what each document says. Use this as your primary
    comparison tool when the user asks how two documents differ on a subject.
    More efficient than calling search_document_a + search_document_b separately.
    """
    vs = _store()
    name_a, name_b = _registry["name_a"], _registry["name_b"]
    docs_a, docs_b = vs.search_both(topic, k=_registry["top_k"])

    result = [f"=== TOPIC COMPARISON: '{topic}' ===\n"]

    result.append(f"--- [{name_a}] says: ---")
    if docs_a:
        for i, d in enumerate(docs_a, 1):
            result.append(f"  [{i}] Page {d.metadata.get('page','?')}: {d.page_content}")
    else:
        result.append(f"  (No relevant content found in {name_a})")

    result.append(f"\n--- [{name_b}] says: ---")
    if docs_b:
        for i, d in enumerate(docs_b, 1):
            result.append(f"  [{i}] Page {d.metadata.get('page','?')}: {d.page_content}")
    else:
        result.append(f"  (No relevant content found in {name_b})")

    return "\n".join(result)


# ── Tool 4: Find Conflicts ─────────────────────────────────────────────────── #

@tool
def find_conflicts(
    topic: Annotated[str, "The topic or claim to check for contradictions or conflicts between the two documents"]
) -> str:
    """
    Specifically designed to detect contradictions, conflicts, and disagreements
    between the two documents. Searches both with conflict-oriented sub-queries
    (negations, opposites, alternative framings) and returns findings side by side.
    Use this when asked about disagreements, contradictions, or inconsistencies.
    """
    vs = _store()
    name_a, name_b = _registry["name_a"], _registry["name_b"]
    k = _registry["top_k"]

    # Use multiple angle queries to surface contradictions
    conflict_queries = [
        topic,
        f"NOT {topic}",
        f"contrary {topic}",
        f"limitations of {topic}",
        f"against {topic}",
    ]

    all_a, all_b = [], []
    seen_a, seen_b = set(), set()

    for q in conflict_queries[:3]:  # top 3 angles
        for d in vs.search("doc_a", q, k=3):
            if d.page_content not in seen_a:
                all_a.append(d)
                seen_a.add(d.page_content)
        for d in vs.search("doc_b", q, k=3):
            if d.page_content not in seen_b:
                all_b.append(d)
                seen_b.add(d.page_content)

    result = [f"=== CONFLICT ANALYSIS: '{topic}' ===\n"]
    result.append(
        "NOTE: The following excerpts have been retrieved with conflict-detection queries.\n"
        "Look for statements that directly oppose each other.\n"
    )

    result.append(f"--- [{name_a}] positions: ---")
    for i, d in enumerate(all_a[:k], 1):
        result.append(f"  [{i}] Page {d.metadata.get('page','?')}: {d.page_content}")

    result.append(f"\n--- [{name_b}] positions: ---")
    for i, d in enumerate(all_b[:k], 1):
        result.append(f"  [{i}] Page {d.metadata.get('page','?')}: {d.page_content}")

    if not all_a and not all_b:
        result.append("No content found in either document for conflict analysis.")

    return "\n".join(result)


# ── Tool 5: Find Common Ground ─────────────────────────────────────────────── #

@tool
def find_common_ground(
    topic: Annotated[str, "The topic or concept to find agreements or shared positions on across both documents"]
) -> str:
    """
    Find areas where both documents AGREE, share similar conclusions, or cover
    the same ground. Use this when the user asks about similarities, consensus,
    or shared perspectives between the two documents.
    """
    vs = _store()
    name_a, name_b = _registry["name_a"], _registry["name_b"]
    k = _registry["top_k"]

    agreement_queries = [
        topic,
        f"importance of {topic}",
        f"benefits of {topic}",
        f"summary {topic}",
    ]

    all_a, all_b = [], []
    seen_a, seen_b = set(), set()

    for q in agreement_queries[:2]:
        for d in vs.search("doc_a", q, k=k):
            if d.page_content not in seen_a:
                all_a.append(d)
                seen_a.add(d.page_content)
        for d in vs.search("doc_b", q, k=k):
            if d.page_content not in seen_b:
                all_b.append(d)
                seen_b.add(d.page_content)

    result = [f"=== COMMON GROUND ANALYSIS: '{topic}' ===\n"]

    result.append(f"--- [{name_a}] on this topic: ---")
    for i, d in enumerate(all_a[:k], 1):
        result.append(f"  [{i}] Page {d.metadata.get('page','?')}: {d.page_content}")

    result.append(f"\n--- [{name_b}] on this topic: ---")
    for i, d in enumerate(all_b[:k], 1):
        result.append(f"  [{i}] Page {d.metadata.get('page','?')}: {d.page_content}")

    return "\n".join(result)


# ── Tool 6: Document Overview ──────────────────────────────────────────────── #

@tool
def get_document_overview(
    doc_key: Annotated[str, "Which document to summarise: must be exactly 'doc_a' or 'doc_b'"]
) -> str:
    """
    Get a broad overview of a document by sampling key sections.
    Use this at the START of an analysis to understand what each document
    is about before diving into specific comparisons. Also useful when the user
    asks 'what does document X cover?' or 'what is the scope of document Y?'
    doc_key must be exactly 'doc_a' or 'doc_b'.
    """
    vs = _store()
    if doc_key not in ("doc_a", "doc_b"):
        return "Invalid doc_key. Must be 'doc_a' or 'doc_b'."

    name = _registry["name_a"] if doc_key == "doc_a" else _registry["name_b"]
    meta = vs.meta(doc_key)

    # Broad sweep queries to surface the doc's main topics
    overview_queries = [
        "main topic introduction overview",
        "key findings conclusions results",
        "methodology approach methods",
        "recommendations implications",
    ]

    result = [
        f"=== DOCUMENT OVERVIEW: [{name}] ===",
        f"Pages: {meta.get('page_count', '?')} | Chunks indexed: {meta.get('chunk_count', '?')}\n",
    ]

    seen = set()
    for q in overview_queries:
        docs = vs.search(doc_key, q, k=2)
        for d in docs:
            if d.page_content not in seen:
                result.append(f"[Page {d.metadata.get('page','?')}] {d.page_content[:300]}...")
                seen.add(d.page_content)

    return "\n".join(result)


# ── Exported tool list ─────────────────────────────────────────────────────── #

ALL_TOOLS = [
    search_document_a,
    search_document_b,
    compare_topic,
    find_conflicts,
    find_common_ground,
    get_document_overview,
]
