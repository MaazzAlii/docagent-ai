"""
vector_store.py  ·  LangChain-Chroma vector store manager
Two isolated collections: one per document.
"""

from __future__ import annotations
import chromadb
from chromadb.config import Settings
from typing import List, Tuple
try:
    from langchain_chroma import Chroma
except Exception:
    # Fallback: newer langchain exposes Chroma under langchain.vectorstores
    try:
        from langchain.vectorstores import Chroma
    except Exception:
        Chroma = None
from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document


class VectorStoreManager:
    """
    Manages two Chroma vector stores (doc_a, doc_b) with shared Mistral embeddings.
    Provides retrieval helpers consumed by agent tools.
    """

    def __init__(self, api_key: str):
        self._embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=api_key,
        )
        # Single Chroma client shared across both collections.
        # Use an explicit Settings instance to prefer the duckdb+parquet
        # implementation and avoid issues with Rust bindings in some environments.
        # Try the default client first (works for many chromadb versions).
        try:
            self._chroma_client = chromadb.Client()
        except Exception as e:
            # Provide a clear, actionable error if chromadb's Rust bindings or
            # new configuration model cause a failure. A common fix is to
            # install a chromadb version compatible with this codebase:
            #   pip install chromadb==0.3.25
            raise RuntimeError(
                "Failed to initialize chromadb.Client().\n"
                "If you see Rust-binding or deprecated-configuration errors, try: `pip install chromadb==0.3.25`\n"
                "Then restart Streamlit and try again."
            ) from e
        self._stores: dict[str, Chroma] = {}
        self._meta: dict[str, dict] = {}  # filename, page_count, chunk_count per doc

    # ── Ingestion ──────────────────────────────────────────────────────────── #

    def ingest(self, doc_key: str, docs: List[Document], filename: str) -> int:
        """
        Embed and store LangChain Documents into a named Chroma collection.
        doc_key: 'doc_a' or 'doc_b'
        Returns number of chunks stored.
        """
        collection_name = f"agent_{doc_key}"

        # Wipe existing collection for this key (handles re-uploads)
        try:
            self._chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            client=self._chroma_client,
        )
        store.add_documents(docs)
        self._stores[doc_key] = store

        pages = {d.metadata["page"] for d in docs}
        self._meta[doc_key] = {
            "filename": filename,
            "page_count": len(pages),
            "chunk_count": len(docs),
        }
        return len(docs)

    # ── Retrieval ──────────────────────────────────────────────────────────── #

    def search(self, doc_key: str, query: str, k: int = 5) -> List[Document]:
        """Semantic search within a single document collection."""
        if doc_key not in self._stores:
            return []
        return self._stores[doc_key].similarity_search(query, k=k)

    def search_with_score(
        self, doc_key: str, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        """Semantic search with cosine similarity scores."""
        if doc_key not in self._stores:
            return []
        return self._stores[doc_key].similarity_search_with_relevance_scores(query, k=k)

    def search_both(
        self, query: str, k: int = 5
    ) -> Tuple[List[Document], List[Document]]:
        """Search both collections simultaneously."""
        return self.search("doc_a", query, k), self.search("doc_b", query, k)

    # ── Helpers ────────────────────────────────────────────────────────────── #

    def is_ready(self, doc_key: str) -> bool:
        return doc_key in self._stores

    def both_ready(self) -> bool:
        return self.is_ready("doc_a") and self.is_ready("doc_b")

    def meta(self, doc_key: str) -> dict:
        return self._meta.get(doc_key, {})

    def format_docs(
        self, docs: List[Document], label: str, show_scores: bool = False
    ) -> str:
        """Format retrieved docs into readable context string."""
        if not docs:
            return f"[{label}]: No relevant content found."
        parts = [f"[{label}]"]
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            parts.append(f"  Excerpt {i} (Page {page}):\n  {doc.page_content}")
        return "\n".join(parts)
