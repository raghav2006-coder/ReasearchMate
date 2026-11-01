"""Unit tests for ResearchMate API and core logic."""

import unittest
from unittest.mock import patch, MagicMock
import pymupdf
from fastapi.testclient import TestClient

from app.main import app, documents
from app.llm import answer_question_llm, summarize_paper_llm, response_cache


def create_sample_pdf_bytes() -> bytes:
    """Helper to generate an in-memory sample research paper PDF."""
    doc = pymupdf.open()
    
    # Page 1: Metadata, Abstract, Introduction
    p1 = doc.new_page()
    p1.insert_text(
        (50, 50),
        "Title: Deep Neural Architectures in Healthcare Diagnostics\n\n"
        "Abstract\n"
        "This paper investigates deep neural network architectures for automated medical image analysis.\n"
        "Our methods improve diagnostic reliability across multiple medical conditions.\n\n"
        "Keywords: deep learning, diagnostics, medical imaging, artificial intelligence\n\n"
        "1. Introduction\n"
        "Medical diagnosis requires reliable and interpretable algorithmic solutions.\n"
        "The problem addressed in this work is the high false positive rate in legacy diagnostic tools.\n"
    )

    # Page 2: Methodology, Findings, Limitations
    p2 = doc.new_page()
    p2.insert_text(
        (50, 50),
        "2. Methodology\n"
        "We propose a novel hybrid attention mechanism combined with convolutional feature extractors.\n"
        "The method was evaluated using stratified five-fold cross-validation on hospital cohorts.\n\n"
        "3. Key Findings\n"
        "Our model achieved a 96.4 percent sensitivity and 94.8 percent specificity.\n"
        "These results significantly outperform established baseline architectures.\n\n"
        "4. Limitations\n"
        "A major limitation is the dependency on high-resolution image inputs and substantial computational resources.\n\n"
        "References\n"
        "1. Smith et al. Medical AI Foundations. 2024.\n"
    )

    pdf_bytes = doc.tobytes()
    doc.close()
    return pdf_bytes


class TestResearchMateAPI(unittest.TestCase):
    def setUp(self):
        self.client = TestClient(app)
        self.pdf_bytes = create_sample_pdf_bytes()
        self.filename = "sample_test_paper.pdf"
        documents.clear()
        response_cache.clear()

    def test_health_endpoint(self):
        """GET /health must return status ok."""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response.json(), {"status": "ok"})

    def test_frontend_served(self):
        """GET / must serve the static frontend HTML."""
        response = self.client.get("/")
        self.assertEqual(response.status_code, 200)
        self.assertIn("ResearchMate", response.text)
        self.assertIn("submitQuestion", response.text)

    def test_upload_pdf(self):
        """POST /upload must extract and store metadata and chunks."""
        response = self.client.post(
            "/upload",
            files={"file": (self.filename, self.pdf_bytes, "application/pdf")},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["filename"], self.filename)
        self.assertEqual(data["document_id"], self.filename)
        self.assertTrue(len(data["chunks"]) > 0)
        self.assertIn("deep learning", " ".join(data.get("keywords", [])).lower())
        self.assertIn(self.filename, documents)

    def test_upload_invalid_file(self):
        """POST /upload rejects non-PDF files."""
        response = self.client.post(
            "/upload",
            files={"file": ("notes.txt", b"plain text content", "text/plain")},
        )
        self.assertEqual(response.status_code, 400)

    def test_upload_empty_file(self):
        """POST /upload rejects empty files."""
        response = self.client.post(
            "/upload",
            files={"file": ("empty.pdf", b"", "application/pdf")},
        )
        self.assertEqual(response.status_code, 400)

    def test_ask_extractive_fallback_with_citation(self):
        """POST /ask returns answers grounded in paper with page citations."""
        # 1. Upload first
        self.client.post(
            "/upload",
            files={"file": (self.filename, self.pdf_bytes, "application/pdf")},
        )

        # 2. Ask question about methodology
        response = self.client.post(
            "/ask",
            data={"document_id": self.filename, "query": "What methodology was used?"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertIn("answer", data)
        self.assertIn("sources", data)
        self.assertTrue(len(data["sources"]) > 0)
        # Must cite page number
        self.assertIn("Page", data["answer"])

    def test_ask_unrelated_query(self):
        """POST /ask returns 'Not found in this paper.' when not found."""
        self.client.post(
            "/upload",
            files={"file": (self.filename, self.pdf_bytes, "application/pdf")},
        )

        response = self.client.post(
            "/ask",
            data={"document_id": self.filename, "query": "What is the capital of Mars?"},
        )
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["answer"], "Not found in this paper.")

    def test_search_endpoint(self):
        """POST /search returns ranked passages with scores."""
        self.client.post(
            "/upload",
            files={"file": (self.filename, self.pdf_bytes, "application/pdf")},
        )

        response = self.client.post(
            "/search",
            data={"document_id": self.filename, "query": "sensitivity specificity findings"},
        )
        self.assertEqual(response.status_code, 200)
        results = response.json()["results"]
        self.assertTrue(len(results) > 0)
        self.assertIn("score", results[0])
        self.assertIn("page", results[0])

    def test_summarize_endpoint(self):
        """POST /summarize/{paper_id} returns structured summary with required sections."""
        self.client.post(
            "/upload",
            files={"file": (self.filename, self.pdf_bytes, "application/pdf")},
        )

        response = self.client.post(f"/summarize/{self.filename}")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        for field in ["Problem", "Method", "Key Findings", "Limitations", "TL;DR"]:
            self.assertIn(field, data)
            self.assertTrue(len(data[field]) > 0)

    @patch("app.llm.get_client")
    def test_llm_qa_mock_and_cache(self, mock_get_client):
        """Test LLM answering and caching logic."""
        mock_openai = MagicMock()
        mock_get_client.return_value = mock_openai
        mock_choice = MagicMock()
        mock_choice.message.content = "The study uses a hybrid attention mechanism. (Page 2)"
        mock_openai.chat.completions.create.return_value = MagicMock(choices=[mock_choice])

        passages = [{"page": 2, "text": "We propose a novel hybrid attention mechanism."}]
        query = "What mechanism is proposed?"

        # First call: calls LLM
        ans1 = answer_question_llm(query, passages)
        self.assertEqual(ans1, "The study uses a hybrid attention mechanism. (Page 2)")
        self.assertEqual(mock_openai.chat.completions.create.call_count, 1)

        # Second call: hits in-memory cache
        ans2 = answer_question_llm(query, passages)
        self.assertEqual(ans2, ans1)
        self.assertEqual(mock_openai.chat.completions.create.call_count, 1)


if __name__ == "__main__":
    unittest.main()

