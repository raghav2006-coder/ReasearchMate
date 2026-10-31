# AGENTS.md

## Guidelines for adding features to ResearchMate

1. **LLM Configuration** – Use environment variables `LLM_BASE_URL`, `LLM_API_KEY`, and `LLM_MODEL` (not `OPENAI_API_KEY`). The `llm.py` module (under `app/`) wraps the `openai` package with a 30‑second timeout and exponential‑backoff retry on rate‑limit errors. An in‑memory cache stores recent responses.

2. **Prompts** – All prompt templates live in `app/prompts.py`. Keep them readable and version‑controlled; change only through that file.

3. **PDF Extraction** – `pdf_extractor.py` must return a dict with keys: `title`, `abstract`, `keywords`, `main_text`, and `chunks`. Each chunk carries `paper_id` and `page_number`. Remove headers, footers, and References sections.

4. **Search & QA** – `/ask` returns answers grounded in the paper. Every answer must cite page numbers. If retrieved chunks contain no answer, respond with "Not found in this paper."

5. **Summarization** – `POST /summarize/{paper_id}` returns a structured summary with fields: `Problem`, `Method`, `Key Findings`, `Limitations`, `TL;DR`. Offer a `beginner` or `expert` level. If the LLM fails or no key exists, fall back to extracting the abstract plus the top‑scoring BM25 sentences.

6. **Caching** – In‑memory dict cache keyed by (endpoint, hash of input). Optional persistence later.

7. **Testing** – Add unit tests under `tests/`. Mock the LLM calls.

8. **Git** – `.env` is git‑ignored (see `.gitignore`). Keep API keys out of the repository.

9. **Run the server** – `uvicorn app.main:app --reload` from the project root. The health endpoint is `GET /health`.