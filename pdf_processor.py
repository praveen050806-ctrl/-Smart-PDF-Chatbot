"""
pdf_processor.py
------------------
Extracts text from a PDF and splits it into overlapping passages
("chunks") that the QA engine can search over. Each chunk remembers
which page(s) it came from so answers can cite a source.
"""

import re
from PyPDF2 import PdfReader


class PDFProcessError(Exception):
    """Raised when a PDF cannot be read or contains no extractable text."""
    pass


def extract_pages(file_path):
    """
    Extract raw text per page from a PDF.

    Returns:
        list[dict]: [{"page": 1, "text": "..."}, ...]
    """
    try:
        reader = PdfReader(file_path)
    except Exception as e:
        raise PDFProcessError(f"Could not open PDF: {e}")

    pages = []
    for i, page in enumerate(reader.pages, start=1):
        text = page.extract_text() or ""
        pages.append({"page": i, "text": _clean_text(text)})

    if not any(p["text"].strip() for p in pages):
        raise PDFProcessError(
            "No readable text could be extracted from this PDF. "
            "It may be a scanned/image-only document."
        )

    return pages


def _clean_text(text):
    text = text.replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{2,}", "\n", text)
    return text.strip()


def chunk_pages(pages, target_words=150, overlap_words=30):
    """
    Split each page's text into ~target_words-sized chunks with a small
    overlap between consecutive chunks (so an answer near a chunk
    boundary isn't cut off from its context).

    Returns:
        list[dict]: [{"id": 0, "page": 1, "text": "..."}, ...]
    """
    chunks = []
    chunk_id = 0

    for page in pages:
        words = page["text"].split()
        if not words:
            continue

        start = 0
        while start < len(words):
            end = min(start + target_words, len(words))
            chunk_text = " ".join(words[start:end])
            if chunk_text.strip():
                chunks.append({
                    "id": chunk_id,
                    "page": page["page"],
                    "text": chunk_text,
                })
                chunk_id += 1
            if end == len(words):
                break
            start = end - overlap_words

    return chunks


def get_document_stats(pages, chunks):
    total_words = sum(len(p["text"].split()) for p in pages)
    return {
        "num_pages": len(pages),
        "num_chunks": len(chunks),
        "total_words": total_words,
    }
