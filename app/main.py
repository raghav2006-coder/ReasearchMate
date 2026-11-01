import logging
import re
from typing import Dict, Any, List, Optional
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Request
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from rank_bm25 import BM25Okapi
import pymupdf

from app.pdf_extractor import extract_pdf_info
from app.llm import answer_question_llm, summarize_paper_llm

logger = logging.getLogger(__name__)

app = FastAPI(title="ResearchMate API")

# Enable CORS for seamless frontend-backend connection
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory storage for uploaded documents: document_id -> {"data": bytes, "info": dict}
documents: Dict[str, Dict[str, Any]] = {}


@app.get("/")
def serve_frontend():
    return FileResponse("static/index.html")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Upload and process a research paper PDF."""
    filename = file.filename or "unknown.pdf"
    content_type = file.content_type or ""

    # Validate file type
    if not (content_type.startswith("application/pdf") or filename.lower().endswith(".pdf")):
        raise HTTPException(status_code=400, detail="Invalid file type. Only PDFs are allowed.")

    file_data = await file.read()
    if not file_data:
        raise HTTPException(status_code=400, detail="Empty PDF file.")

    try:
        doc = pymupdf.open(stream=file_data, filetype="pdf")
        if len(doc) == 0:
            doc.close()
            raise HTTPException(status_code=400, detail="Invalid PDF: no pages found.")
        doc.close()
    except Exception:
        raise HTTPException(status_code=400, detail="Invalid or corrupted PDF file.")

    info = extract_pdf_info(file_data, filename=filename)

    document_id = filename
    documents[document_id] = {
        "data": file_data,
        "info": info,
    }

    return info


async def _get_doc_and_query(request: Request, document_id: Optional[str], query: Optional[str]):
    """Helper to extract document_id and query from either Form data or JSON body."""
    if not document_id or not query:
        try:
            body = await request.json()
            if isinstance(body, dict):
                document_id = document_id or body.get("document_id") or body.get("paper_id")
                query = query or body.get("query") or body.get("question")
        except Exception:
            pass
    return document_id, query


@app.post("/search")
async def search_pdf(
    request: Request,
    document_id: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
):
    """Search an uploaded PDF's passages using BM25."""
    doc_id, q = await _get_doc_and_query(request, document_id, query)

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not doc_id or doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    info = documents[doc_id]["info"]
    chunks = info.get("chunks", info.get("paragraphs", []))

    if not chunks:
        return {"results": [], "message": "No text passages found in document."}

    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = q.lower().split()
    scores = bm25.get_scores(tokenized_query)

    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:5]
    results = [
        {
            "text": chunks[idx]["text"],
            "page": chunks[idx].get("page", chunks[idx].get("page_number", 1)),
            "score": float(scores[idx]),
        }
        for idx in top_indices if scores[idx] > 0
    ]

    return {"results": results}


@app.post("/ask")
async def ask_question(
    request: Request,
    document_id: Optional[str] = Form(None),
    query: Optional[str] = Form(None),
):
    """Answer questions about an uploaded paper, grounded in content with page citations."""
    doc_id, q = await _get_doc_and_query(request, document_id, query)

    if not q or not q.strip():
        raise HTTPException(status_code=400, detail="Query cannot be empty.")

    if not doc_id or doc_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    info = documents[doc_id]["info"]
    chunks = info.get("chunks", info.get("paragraphs", []))

    if not chunks:
        return {"answer": "Not found in this paper.", "sources": []}

    tokenized_chunks = [c["text"].lower().split() for c in chunks]
    bm25 = BM25Okapi(tokenized_chunks)
    tokenized_query = q.lower().split()
    scores = bm25.get_scores(tokenized_query)

    # Rank passages by BM25
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)
    # Filter passages that scored > 0, up to top 5
    scored_indices = [idx for idx in top_indices if scores[idx] > 0][:5]

    # If no passage scored > 0, check if we have any passages
    if not scored_indices:
        # Check if query is a general high-level query about the paper
        general_terms = {"about", "paper", "overview", "summary", "topic", "work", "findings", "method"}
        if any(term in query.lower() for term in general_terms) and len(chunks) > 0:
            scored_indices = list(range(min(3, len(chunks))))
        else:
            return {"answer": "Not found in this paper.", "sources": []}

    top_passages = [
        {
            "text": chunks[idx]["text"],
            "page": chunks[idx].get("page", chunks[idx].get("page_number", 1)),
        }
        for idx in scored_indices
    ]

    # 1. Try LLM first
    answer = answer_question_llm(q, top_passages)

    # 2. Extractive fallback if LLM is unavailable or fails
    if not answer or not answer.strip():
        answer = _extractive_qa(q, top_passages)

    return {"answer": answer, "sources": top_passages}


def _extractive_qa(query: str, passages: List[Dict[str, Any]]) -> str:
    """Extractive QA fallback grounded strictly in passages, with page citations."""
    if not passages:
        return "Not found in this paper."

    stopwords = {
        "what", "is", "are", "was", "were", "the", "a", "an", "in", "on", "at",
        "by", "for", "with", "about", "against", "between", "into", "through",
        "during", "before", "after", "above", "below", "to", "from", "up",
        "down", "of", "off", "over", "under", "again", "further", "then",
        "once", "here", "there", "when", "where", "why", "how", "all", "any",
        "both", "each", "few", "more", "most", "other", "some", "such", "no",
        "nor", "not", "only", "own", "same", "so", "than", "too", "very",
        "can", "will", "just", "should", "now", "does", "did", "do"
    }

    query_tokens = [w.lower() for w in re.findall(r"\b\w+\b", query)]
    meaningful_terms = [t for t in query_tokens if t not in stopwords and len(t) > 2]
    if not meaningful_terms:
        meaningful_terms = [t for t in query_tokens if len(t) > 2]

    best_sentences = []
    for p in passages:
        text = p["text"]
        page = p.get("page", p.get("page_number", 1))
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 15]
        for s in sentences:
            s_lower = s.lower()
            matches = sum(1 for term in meaningful_terms if term in s_lower)
            if matches > 0:
                best_sentences.append((matches, s, page))

    if not best_sentences:
        # General question fallback to first passage sentence
        top_p = passages[0]
        sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", top_p["text"]) if len(s.strip()) > 20]
        if sentences:
            page = top_p.get("page", top_p.get("page_number", 1))
            return f"{sentences[0].rstrip('. ')}. (Page {page})"
        return "Not found in this paper."

    best_sentences.sort(key=lambda x: x[0], reverse=True)
    top_matches = best_sentences[:2]
    answer_parts = []
    for _, sent, page in top_matches:
        clean_sent = sent.rstrip(". ")
        answer_parts.append(f"{clean_sent}. (Page {page})")

    return " ".join(answer_parts)


@app.post("/summarize")
@app.post("/summarize/{paper_id}")
async def summarize_pdf(
    request: Request,
    paper_id: Optional[str] = None,
    document_id: Optional[str] = Form(None),
    level: str = Form("beginner"),
):
    """Summarize an uploaded paper with structured sections: Problem, Method, Key Findings, Limitations, TL;DR."""
    target_id = paper_id or document_id
    if not target_id:
        try:
            body = await request.json()
            if isinstance(body, dict):
                target_id = body.get("paper_id") or body.get("document_id")
                level = body.get("level", level)
        except Exception:
            pass

    if not target_id or target_id not in documents:
        raise HTTPException(status_code=404, detail="Document not found.")

    info = documents[target_id]["info"]
    full_text = info.get("main_text", info.get("extracted_text", ""))
    chunks = info.get("chunks", info.get("paragraphs", []))

    if not full_text.strip():
        return {
            "Problem": "No text content available.",
            "Method": "N/A",
            "Key Findings": "N/A",
            "Limitations": "N/A",
            "TL;DR": "No content available.",
            "summary": "No text content available for summarization."
        }

    # 1. Try LLM summarization
    llm_summary = summarize_paper_llm(full_text, level=level)
    if llm_summary and llm_summary.strip():
        parsed = _parse_structured_summary(llm_summary)
        parsed["summary"] = llm_summary.strip()
        return parsed

    # 2. Extractive fallback
    return _extractive_summary(info, chunks)


def _parse_structured_summary(summary_text: str) -> Dict[str, str]:
    """Parse sections from LLM summary text."""
    sections = {
        "Problem": "",
        "Method": "",
        "Key Findings": "",
        "Limitations": "",
        "TL;DR": "",
    }
    pattern = r"(Problem|Method|Key Findings|Limitations|TL;DR)\s*:\s*(.*?)(?=(Problem|Method|Key Findings|Limitations|TL;DR)\s*:|$)"
    matches = re.findall(pattern, summary_text, re.DOTALL | re.IGNORECASE)

    for match in matches:
        key_raw = match[0].strip().title()
        content = match[1].strip()
        for k in sections.keys():
            if k.lower() == key_raw.lower() or (k == "TL;DR" and "tl" in key_raw.lower()):
                sections[k] = content
                break

    # If parsing didn't match structured keys, provide the full text as TL;DR
    if not any(sections.values()):
        sections["TL;DR"] = summary_text.strip()
        sections["Problem"] = "See TL;DR"
        sections["Method"] = "See TL;DR"
        sections["Key Findings"] = "See TL;DR"
        sections["Limitations"] = "See TL;DR"

    return sections


def _extractive_summary(info: Dict[str, Any], chunks: List[Dict[str, Any]]) -> Dict[str, str]:
    """Generate structured summary by extracting abstract and BM25 top sentences."""
    abstract = info.get("abstract", "")
    tldr = abstract if abstract else (chunks[0]["text"] if chunks else "No text found.")

    def find_best_sentence(query_words: List[str]) -> str:
        best_sent = "Not explicitly stated."
        max_matches = 0
        for c in chunks:
            text = c["text"]
            sentences = [s.strip() for s in re.split(r"(?<=[.!?])\s+", text) if len(s.strip()) > 20]
            for s in sentences:
                s_lower = s.lower()
                matches = sum(1 for w in query_words if w in s_lower)
                if matches > max_matches:
                    max_matches = matches
                    page = c.get("page", c.get("page_number", 1))
                    best_sent = f"{s.rstrip('. ')}. (Page {page})"
        return best_sent

    problem = find_best_sentence(["problem", "challenge", "motivation", "address", "issue"])
    method = find_best_sentence(["method", "methodology", "approach", "framework", "architecture", "model"])
    findings = find_best_sentence(["results", "findings", "demonstrate", "show", "performance", "achieve"])
    limitations = find_best_sentence(["limitation", "limitations", "future work", "constraint", "drawback"])

    return {
        "Problem": problem,
        "Method": method,
        "Key Findings": findings,
        "Limitations": limitations,
        "TL;DR": tldr[:500],
        "summary": f"Problem: {problem}\nMethod: {method}\nKey Findings: {findings}\nLimitations: {limitations}\nTL;DR: {tldr[:500]}"
    }
