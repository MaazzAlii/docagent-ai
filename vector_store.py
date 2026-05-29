"""
vector_store.py  ·  LangChain-Chroma vector store manager

Two isolated collections: agent_doc_a and agent_doc_b.
Uses PersistentClient so data survives Streamlit hot-reloads.
"""

from __future__ import annotations
import os
import chromadb
from typing import List, Tuple

try:
    from langchain_chroma import Chroma
except ImportError:
    from langchain.vectorstores import Chroma  # type: ignore

from langchain_mistralai import MistralAIEmbeddings
from langchain_core.documents import Document

# Persist to local disk so reruns don't lose indexed data
_CHROMA_PATH = "./chroma_store"


class VectorStoreManager:
    """
    Manages two Chroma vector stores (doc_a, doc_b) with Mistral embeddings.
    """

    def __init__(self, api_key: str):
        self._embeddings = MistralAIEmbeddings(
            model="mistral-embed",
            mistral_api_key=api_key,
        )
        os.makedirs(_CHROMA_PATH, exist_ok=True)
        self._chroma_client = chromadb.PersistentClient(path=_CHROMA_PATH)
        self._stores: dict[str, Chroma] = {}
        self._meta:   dict[str, dict]   = {}

    # ── Ingestion ────────────────────────────────────────────────────────────── #

    def ingest(self, doc_key: str, docs: List[Document], filename: str) -> int:
        """
        Embed + store Documents into a named Chroma collection.
        Always wipes the old collection first (fresh upload = fresh index).
        """
        collection_name = f"agent_{doc_key}"

        # Delete stale collection
        try:
            self._chroma_client.delete_collection(collection_name)
        except Exception:
            pass

        store = Chroma(
            collection_name=collection_name,
            embedding_function=self._embeddings,
            client=self._chroma_client,
        )

        # Add in batches to avoid hitting embed rate limits
        batch_size = 50
        for i in range(0, len(docs), batch_size):
            store.add_documents(docs[i: i + batch_size])

        self._stores[doc_key] = store

        pages = {d.metadata.get("page", 1) for d in docs}
        self._meta[doc_key] = {
            "filename":    filename,
            "page_count":  len(pages),
            "chunk_count": len(docs),
        }
        return len(docs)

    # ── Retrieval ────────────────────────────────────────────────────────────── #

    def search(self, doc_key: str, query: str, k: int = 5) -> List[Document]:
        """Semantic search within one collection."""
        store = self._get_store(doc_key)
        if not store:
            return []
        try:
            return store.similarity_search(query, k=k)
        except Exception:
            return []

    def search_with_score(
        self, doc_key: str, query: str, k: int = 5
    ) -> List[Tuple[Document, float]]:
        store = self._get_store(doc_key)
        if not store:
            return []
        try:
            return store.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            return []

    def search_both(
        self, query: str, k: int = 5
    ) -> Tuple[List[Document], List[Document]]:
        return self.search("doc_a", query, k), self.search("doc_b", query, k)

    # ── Helpers ──────────────────────────────────────────────────────────────── #

    def is_ready(self, doc_key: str) -> bool:
        return doc_key in self._stores

    def both_ready(self) -> bool:
        return self.is_ready("doc_a") and self.is_ready("doc_b")

    def meta(self, doc_key: str) -> dict:
        return self._meta.get(doc_key, {})

    def _get_store(self, doc_key: str):
        """Return in-memory store, or try to reload from persistent client."""
        if doc_key in self._stores:
            return self._stores[doc_key]
        # Try to reconnect to persisted collection
        collection_name = f"agent_{doc_key}"
        try:
            store = Chroma(
                collection_name=collection_name,
                embedding_function=self._embeddings,
                client=self._chroma_client,
            )
            # Verify it has data
            if store._collection.count() > 0:
                self._stores[doc_key] = store
                return store
        except Exception:
            pass
        return None

    def format_docs(self, docs: List[Document], label: str) -> str:
        if not docs:
            return f"[{label}]: No relevant content found."
        parts = [f"[{label}]"]
        for i, doc in enumerate(docs, 1):
            page = doc.metadata.get("page", "?")
            parts.append(f"  Excerpt {i} (Page {page}):\n  {doc.page_content}")
        return "\n".join(parts)
