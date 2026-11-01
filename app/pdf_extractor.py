"""PDF extraction service for ResearchMate."""

import re
from typing import Dict, Any, List
import pymupdf  # PyMuPDF


def extract_title(doc) -> str:
    """Extract PDF title from metadata, fallback to first meaningful heading."""
    metadata = doc.metadata
    if metadata and metadata.get("title"):
        title = metadata["title"].strip()
        if title:
            return title

    # Fallback: find first meaningful heading on first page
    for page_num in range(min(2, len(doc))):
        page = doc[page_num]
        for b in page.get_text("blocks"):
            if b[6] == 0:  # text block
                text = b[4].strip()
                lines = [line.strip() for line in text.split("\n") if line.strip()]
                for line in lines:
                    if 15 < len(line) < 200:
                        if line.isupper() or re.match(r"^[A-Z][a-zA-Z0-9\s:,-]+$", line):
                            # Avoid generic headings like 'ABSTRACT'
                            if line.lower() not in ["abstract", "introduction", "contents", "index"]:
                                return line
    return "Untitled Document"


def extract_abstract(doc) -> str:
    """Extract abstract text from document."""
    full_text = ""
    for page in doc[:3]:  # abstract is usually within the first 3 pages
        full_text += page.get_text() + "\n"

    lines = full_text.split("\n")
    in_abstract = False
    abstract_lines = []
    for line in lines:
        stripped = line.strip()
        if stripped.lower() == "abstract" or stripped.lower().startswith("abstract:"):
            in_abstract = True
            content_after = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if content_after:
                abstract_lines.append(content_after)
            continue
        if in_abstract:
            # Stop at next major section marker
            if re.match(r"^[IVX]+\.|^[0-9]+\.\s+[A-Z]|^Keywords|^Index Terms|^1\.\s+Introduction", stripped, re.I):
                break
            if stripped:
                abstract_lines.append(stripped)

    if abstract_lines:
        abstract = " ".join(abstract_lines)
        abstract = re.sub(r"\s+", " ", abstract).strip()
        if abstract:
            return abstract

    return ""


def extract_keywords(doc) -> List[str]:
    """Extract keywords from document."""
    full_text = ""
    for page in doc[:3]:
        full_text += page.get_text() + "\n"

    lines = full_text.split("\n")
    for idx, line in enumerate(lines):
        stripped = line.strip()
        if stripped.lower().startswith("keywords") or stripped.lower().startswith("index terms"):
            content = stripped.split(":", 1)[1].strip() if ":" in stripped else ""
            if not content and idx + 1 < len(lines):
                content = lines[idx + 1].strip()
            if content:
                keywords = [k.strip() for k in re.split(r"[,;•]+", content) if k.strip()]
                if keywords:
                    return keywords
    return []


def extract_chunks(doc, paper_id: str) -> List[Dict[str, Any]]:
    """Extract meaningful text chunks with page numbers, removing headers, footers, and references."""
    chunks = []
    in_references = False

    for page_num in range(len(doc)):
        page = doc[page_num]
        page_height = page.rect.height
        blocks = page.get_text("blocks")

        for b in blocks:
            if b[6] != 0:  # Skip non-text
                continue

            # Skip header (top 5% of page) and footer (bottom 5% of page) if single small line
            y0, y1 = b[1], b[3]
            raw_text = b[4].strip()
            if not raw_text:
                continue

            lines = [l.strip() for l in raw_text.split("\n") if l.strip()]
            if len(lines) == 1 and len(lines[0]) < 50:
                if y0 < page_height * 0.06 or y1 > page_height * 0.94:
                    continue  # Likely running header or footer page number

            # Check for References section marker
            first_line = lines[0].lower()
            if first_line in ["references", "bibliography", "works cited"] or re.match(r"^[0-9IVX]+\.\s*references", first_line):
                in_references = True
                continue

            if in_references:
                continue

            clean_text = " ".join(lines)
            clean_text = re.sub(r"\s+", " ", clean_text).strip()

            if len(clean_text) >= 20:
                chunk = {
                    "paper_id": paper_id,
                    "document_id": paper_id,
                    "page_number": page_num + 1,
                    "page": page_num + 1,
                    "text": clean_text,
                }
                chunks.append(chunk)

    return chunks


def extract_pdf_info(file_data: bytes, filename: str) -> Dict[str, Any]:
    """Extract all metadata, text, and chunks from PDF."""
    doc = pymupdf.open(stream=file_data, filetype="pdf")

    title = extract_title(doc)
    abstract = extract_abstract(doc)
    keywords = extract_keywords(doc)
    chunks = extract_chunks(doc, filename)

    # Full text reconstruction
    main_text = "\n\n".join([c["text"] for c in chunks])

    info = {
        "document_id": filename,
        "paper_id": filename,
        "filename": filename,
        "title": title,
        "abstract": abstract,
        "keywords": keywords,
        "page_count": len(doc),
        "main_text": main_text,
        "extracted_text": main_text,
        "chunks": chunks,
        "paragraphs": chunks,
    }
    doc.close()
    return info

