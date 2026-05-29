"""
pdf_processor.py  ·  Smart PDF extraction with Vision LLM fallback

Strategy (fully automatic):
  1. Try pdfplumber for text extraction
  2. Check QUALITY not just length:
       - Enough real words? (not just symbols/coordinates)
       - Reasonable word-to-char ratio? (not garbled layout data)
       - Enough unique words? (not repetitive junk)
  3. If quality is poor → use Mistral Vision (pixtral-12b) regardless
  4. Vision always wins for designed CVs, scanned docs, handwritten notes

Supports:
  ✓ Normal text PDFs (reports, papers, contracts)
  ✓ Designed/styled CVs (Canva, Adobe, InDesign, Word-exported)
  ✓ Scanned documents
  ✓ Handwritten notes
  ✓ Mixed documents (some pages text, some image)
  ✓ Password-free PDFs of any layout complexity
"""

from __future__ import annotations
import io
import re
import base64
from typing import List

import pdfplumber
import fitz  # PyMuPDF — renders pages as images for vision
from langchain_core.documents import Document


# ── Tunable thresholds ─────────────────────────────────────────────────────── #
MIN_REAL_WORDS      = 30    # minimum real English/Urdu words on a page
MIN_WORD_LENGTH_AVG = 2.5   # avg word length — garbled text has very short "words"
MIN_UNIQUE_RATIO    = 0.4   # at least 40% of words must be unique (no repetitive junk)
VISION_ZOOM         = 2.5   # render scale — higher = better OCR, larger image
VISION_MAX_TOKENS   = 3000  # plenty for a full CV page


# ── Public API ─────────────────────────────────────────────────────────────── #

def extract_langchain_docs(
    pdf_bytes: bytes,
    source_label: str,
    api_key: str | None = None,
) -> List[Document]:
    """
    Extract PDF → LangChain Documents with smart text/vision pipeline.

    Args:
        pdf_bytes:    Raw PDF bytes
        source_label: 'doc_a' or 'doc_b'
        api_key:      Mistral API key (needed for vision on image/designed PDFs)

    Returns:
        List of LangChain Documents with metadata:
          page, total_pages, source, extraction_method, word_count
    """
    documents: List[Document] = []
    total_pages = _count_pages(pdf_bytes)

    with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
        for page_num, page in enumerate(pdf.pages, start=1):

            raw_text  = _safe_extract_text(page)
            quality   = _text_quality(raw_text)

            if quality["is_good"]:
                # ── Path A: Clean readable text ──────────────────────────── #
                chunks = _chunk_text(_clean(raw_text))
                for idx, chunk in enumerate(chunks):
                    documents.append(Document(
                        page_content=chunk,
                        metadata={
                            "source":             source_label,
                            "page":               page_num,
                            "total_pages":        total_pages,
                            "chunk_index":        idx,
                            "extraction_method":  "text",
                            "word_count":         quality["word_count"],
                        },
                    ))
            else:
                # ── Path B: Poor/no text → Vision LLM ───────────────────── #
                # Log why text was rejected (visible in terminal for debugging)
                print(
                    f"[pdf_processor] Page {page_num}: text quality too low "
                    f"(words={quality['word_count']}, avg_len={quality['avg_word_len']:.1f}, "
                    f"unique_ratio={quality['unique_ratio']:.2f}) → using Vision"
                )

                vision_text = _extract_via_vision(pdf_bytes, page_num, api_key)

                if vision_text and len(vision_text.strip()) > 30:
                    chunks = _chunk_text(_clean(vision_text))
                    for idx, chunk in enumerate(chunks):
                        documents.append(Document(
                            page_content=chunk,
                            metadata={
                                "source":             source_label,
                                "page":               page_num,
                                "total_pages":        total_pages,
                                "chunk_index":        idx,
                                "extraction_method":  "vision",
                                "word_count":         len(vision_text.split()),
                            },
                        ))
                elif not api_key:
                    # No API key — can't use vision, store warning chunk
                    documents.append(Document(
                        page_content=(
                            f"[Page {page_num}: This appears to be an image-based PDF. "
                            f"Add your Mistral API key to enable AI Vision reading.]"
                        ),
                        metadata={
                            "source":             source_label,
                            "page":               page_num,
                            "total_pages":        total_pages,
                            "chunk_index":        0,
                            "extraction_method":  "no_key",
                            "word_count":         0,
                        },
                    ))
                else:
                    # Vision failed for another reason
                    documents.append(Document(
                        page_content=f"[Page {page_num}: Content could not be extracted]",
                        metadata={
                            "source":             source_label,
                            "page":               page_num,
                            "total_pages":        total_pages,
                            "chunk_index":        0,
                            "extraction_method":  "failed",
                            "word_count":         0,
                        },
                    ))

    return documents


def extraction_summary(docs: List[Document]) -> dict:
    """Return a summary dict for the UI to display."""
    methods: dict[str, int] = {}
    total_words = 0
    for d in docs:
        m = d.metadata.get("extraction_method", "unknown")
        methods[m] = methods.get(m, 0) + 1
        total_words += d.metadata.get("word_count", 0)

    pages = {d.metadata.get("page") for d in docs}
    return {
        "total_chunks": len(docs),
        "total_pages":  len(pages),
        "total_words":  total_words,
        "methods":      methods,
    }


# ── Text quality checker ────────────────────────────────────────────────────── #

def _text_quality(text: str) -> dict:
    """
    Assess whether extracted text is genuinely readable content.

    Designed CVs often produce garbled pdfplumber output like:
      "Maaz 2024 A l i E n g i n e e r"  ← spaced-out characters
      "▪ ▪ ▪ Python ▪ ▪"                  ← mostly symbols
      "MaazAliResume.indd"                 ← InDesign artefacts

    Returns a dict with quality metrics and a boolean `is_good`.
    """
    if not text or not text.strip():
        return {"is_good": False, "word_count": 0, "avg_word_len": 0, "unique_ratio": 0}

    # Tokenise: only keep tokens that look like real words (≥2 alphanumeric chars)
    tokens    = re.findall(r"[A-Za-z0-9\u0600-\u06FF]{2,}", text)  # includes Urdu/Arabic
    wc        = len(tokens)
    avg_len   = (sum(len(t) for t in tokens) / wc) if wc > 0 else 0
    unique_r  = (len(set(t.lower() for t in tokens)) / wc) if wc > 0 else 0

    is_good = (
        wc       >= MIN_REAL_WORDS
        and avg_len   >= MIN_WORD_LENGTH_AVG
        and unique_r  >= MIN_UNIQUE_RATIO
    )

    return {
        "is_good":       is_good,
        "word_count":    wc,
        "avg_word_len":  avg_len,
        "unique_ratio":  unique_r,
    }


# ── Internal helpers ─────────────────────────────────────────────────────────── #

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


def _extract_via_vision(pdf_bytes: bytes, page_num: int, api_key: str | None) -> str:
    """
    Render PDF page as PNG → Mistral pixtral-12b Vision → extracted text.
    page_num is 1-indexed.
    """
    if not api_key:
        return ""

    try:
        # 1. Render page to high-res PNG via PyMuPDF
        fitz_doc = fitz.open(stream=pdf_bytes, filetype="pdf")
        fitz_page = fitz_doc[page_num - 1]  # 0-indexed
        mat  = fitz.Matrix(VISION_ZOOM, VISION_ZOOM)
        pix  = fitz_page.get_pixmap(matrix=mat)
        img_bytes = pix.tobytes("png")
        fitz_doc.close()

        # 2. Base64 encode
        img_b64 = base64.standard_b64encode(img_bytes).decode("utf-8")

        # 3. Send to Mistral Vision
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
                                "This is a page from a document. Extract ALL text exactly as it appears.\n"
                                "Include: names, contact info, dates, job titles, company names, "
                                "education details, skills, bullet points, section headings, and any other text.\n"
                                "Format: output plain readable text. Preserve line breaks between sections.\n"
                                "Do NOT add any commentary, headers, or markdown formatting.\n"
                                "Just the raw extracted text."
                            ),
                        },
                    ],
                }
            ],
            max_tokens=VISION_MAX_TOKENS,
        )

        extracted = response.choices[0].message.content or ""
        print(f"[pdf_processor] Vision extracted {len(extracted.split())} words from page {page_num}")
        return extracted

    except Exception as e:
        print(f"[pdf_processor] Vision failed for page {page_num}: {e}")
        return ""


def _chunk_text(text: str, chunk_size: int = 350, overlap: int = 50) -> List[str]:
    """
    Split text into overlapping chunks by word count.
    Smaller chunk_size = more precise retrieval for CVs.
    """
    words = text.split()
    if not words:
        return []

    chunks = []
    start  = 0
    while start < len(words):
        piece = " ".join(words[start: start + chunk_size])
        if len(piece.strip()) > 30:
            chunks.append(piece)
        start += chunk_size - overlap

    return chunks if chunks else [text]


def _clean(text: str) -> str:
    # Collapse whitespace
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Remove non-printable / non-ASCII junk (but keep Urdu/Arabic range)
    text = re.sub(r"[^\x20-\x7E\u0600-\u06FF\n]", " ", text)
    return text.strip()
