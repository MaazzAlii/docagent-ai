"""
pdf_processor.py  ·  Smart PDF extraction with Vision LLM fallback

Strategy (fully automatic, no user input needed):
  1. Try pdfplumber  → fast, free, works on text-based PDFs
  2. If page yields < 50 chars  → page is image-based (scanned / designed CV / handwritten)
  3. Render that page as image → send to Mistral Vision → extract full text via OCR
  4. Chunk extracted text → return LangChain Documents with page metadata

Supports:
  ✓ Normal text PDFs (reports, papers, contracts)
  ✓ Designed/styled CVs (Canva, Adobe, Word-exported)
  ✓ Scanned documents
  ✓ Handwritten notes
  ✓ Mixed documents (some pages text, some image)
"""

from __future__ import annotations
import io
import re
import base64
from typing import List

import pdfplumber
import fitz  # PyMuPDF — for rendering pages as images
from langchain_core.documents import Document


# ── Public API ─────────────────────────────────────────────────────────────── #

def extract_langchain_docs(
    pdf_bytes: bytes,
    source_label: str,
    api_key: str | None = None,
) -> List[Document]:
    """
    Extract PDF → LangChain Documents with smart text/vision fallback.

    Args:
        pdf_bytes:    Raw PDF file bytes
        source_label: 'doc_a' or 'doc_b'
        api_key:      Mistral API key (required for vision fallback on image PDFs)

    Returns:
        List of LangChain Documents, one or more per page, with metadata:
          page, total_pages, source, extraction_method ('text' or 'vision')
    """
    documents: List[Document] = []
    total_pages = _count_pages(pdf_bytes)

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):
            raw_text = _safe_extract_text(page)

            if _is_text_sufficient(raw_text):
                # ── Path A: normal text PDF ──────────────────────────────── #
                chunks = _chunk_text(_clean(raw_text))
                for idx, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source": source_label,
                            "page": page_num,
                            "total_pages": total_pages,
                            "chunk_index": idx,
                            "extraction_method": "text",
                        },
                    ))
            else:
                # ── Path B: image-based page → Vision LLM ────────────────── #
                vision_text = _extract_via_vision(pdf_bytes, page_num, api_key)
                if vision_text:
                    chunks = _chunk_text(_clean(vision_text))
                    for idx, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source": source_label,
                                "page": page_num,
                                "total_pages": total_pages,
                                "chunk_index": idx,
                                "extraction_method": "vision",
                            },
                        ))
                else:
                    # Vision also failed — store a placeholder so page isn't lost
                    documents.append(Document(
                        page_content=f"[Page {page_num}: Could not extract content]",
                        metadata={
                            "source": source_label,
                            "page": page_num,
                            "total_pages": total_pages,
                            "chunk_index": 0,
                            "extraction_method": "failed",
                        },
                    ))

    return documents


def extraction_summary(docs: List[Document]) -> dict:
    """Return a human-readable summary of how pages were extracted."""
    methods: dict[str, int] = {}
    for d in docs:
        m = d.metadata.get("extraction_method", "unknown")
        methods[m] = methods.get(m, 0) + 1

    pages = {d.metadata.get("page") for d in docs}
    return {
        "total_chunks": len(docs),
        "total_pages": len(pages),
        "methods": methods,
    }


# ── Internal helpers ────────────────────────────────────────────────────────── #

def _count_pages(pdf_bytes: bytes) -> int:
    try:
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            return len(pdf.pages)
    except Exception:
        return 0


def _safe_extract_text(page) -> str:
    try:
        return page.extract_text() or ""
    except Exception:
        return ""


def _is_text_sufficient(text: str, min_chars: int = 80) -> bool:
    """Returns True if the page has enough real text to skip vision."""
    cleaned = re.sub(r"\s+", " ", text).strip()
    return len(cleaned) >= min_chars


def _extract_via_vision(pdf_bytes: bytes, page_num: int, api_key: str | None) -> str:
    """
    Render a single PDF page as PNG image and send to Mistral Vision for OCR.
    page_num is 1-indexed.
    """
    if not api_key:
        return ""

    try:
        # Render page to image with PyMuPDF
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        page = doc[page_num - 1]  # 0-indexed
        mat = fitz.Matrix(2.0, 2.0)  # 2x zoom = better OCR quality
        pix = page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        doc.close()

        # Encode to base64
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        # Call Mistral Vision (pixtral-12b)
        from mistralai import Mistral
        client = Mistral(api_key=api_key)

        response = client.chat.complete(
            model="pixtral-12b-2409",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image_url",
                            "image_url": f"data:image/png;base64,{img_b64}",
                        },
                        {
                            "type": "text",
                            "text": (
                                "Extract ALL text from this document image exactly as it appears. "
                                "Preserve section headings, bullet points, dates, names, contact info, "
                                "skills, job titles, and all other content. "
                                "Output plain text only — no markdown, no commentary."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=2000,
        )

        return response.choices[0].message.content or ""

    except Exception as e:
        print(f"[Vision fallback failed for page {page_num}]: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int = 400, overlap: int = 60) -> List[str]:
    """Split text into overlapping word-based chunks."""
    words = text.split()
    if not words:
        return []
    chunks, start = [], 0
    while start < len(words):
        piece = " ".join(words[start: start + chunk_size])
        if len(piece) > 40:
            chunks.append(piece)
        start += chunk_size - overlap
    return chunks or [text]  # always return at least one chunk


def _clean(text: str) -> str:
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^\x00-\x7F]+", " ", text)
    return text.strip()
