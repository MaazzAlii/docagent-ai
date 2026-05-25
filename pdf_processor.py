"""
pdf_processor.py  ·  Extract + chunk PDFs with page-level metadata
"""

import io
import re
from typing import List, Dict
import pdfplumber
from langchain_core.documents import Document


def extract_langchain_docs(pdf_bytes: bytes, source_label: str) -> List[Document]:
    """
    Extract PDF text as LangChain Documents with rich metadata.
    Each page → one or more Document objects with:
      - source: doc label ('doc_a' / 'doc_b')
      - page: page number (1-indexed)
      - chunk_index: position within page
    """
    documents = []
    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        total_pages = len(pdf.pages)
        for page_num, page in enumerate(pdf.pages, start=1):
            raw = page.extract_text() or ""
            raw = _clean(raw)
            if not raw:
                continue
            chunks = _chunk_text(raw, chunk_size=450, overlap=60)
            for idx, chunk in enumerate(chunks):
                documents.append(
                    Document(
                        page_content=chunk,
                        metadata={
                            "source": source_label,
                            "page": page_num,
                            "total_pages": total_pages,
                            "chunk_index": idx,
                        },
                    )
                )
    return documents


def _chunk_text(text: str, chunk_size: int, overlap: int) -> List[str]:
    words = text.split()
    chunks, start = [], 0
    while start < len(words):
        piece = " ".join(words[start : start + chunk_size])
        if len(piece) > 40:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()
