"""PDF service compatibility wrapper."""

from app.pdf_extractor import (
    extract_title,
    extract_abstract,
    extract_keywords,
    extract_chunks,
    extract_pdf_info,
)

# Alias for backward compatibility
extract_paragraphs = extract_chunks

__all__ = [
    "extract_title",
    "extract_abstract",
    "extract_keywords",
    "extract_chunks",
    "extract_paragraphs",
    "extract_pdf_info",
]
