"""LLM client and cache for ResearchMate."""

import hashlib
import json
import logging
import time
from typing import Optional, List, Dict, Any
from openai import OpenAI, RateLimitError

from app.config import LLM_BASE_URL, LLM_API_KEY, LLM_MODEL
from app.prompts import (
    QA_SYSTEM_PROMPT,
    QA_USER_PROMPT,
    SUMMARIZE_SYSTEM_PROMPT,
    SUMMARIZE_USER_PROMPT,
)

logger = logging.getLogger(__name__)

# In-memory response cache: (endpoint, hash_of_input) -> response_text
response_cache: Dict[tuple, str] = {}


def _get_cache_key(endpoint: str, payload: Any) -> tuple:
    payload_str = json.dumps(payload, sort_keys=True)
    hash_val = hashlib.sha256(payload_str.encode("utf-8")).hexdigest()
    return (endpoint, hash_val)


def get_client() -> Optional[OpenAI]:
    """Return configured OpenAI client or None if credentials not present."""
    if not LLM_API_KEY or not LLM_BASE_URL or not LLM_MODEL:
        return None
    try:
        return OpenAI(
            base_url=LLM_BASE_URL,
            api_key=LLM_API_KEY,
            timeout=30.0,
        )
    except Exception as e:
        logger.warning(f"Could not initialize OpenAI client: {e}")
        return None


def call_chat_completion(
    messages: List[Dict[str, str]],
    temperature: float = 0.1,
    max_retries: int = 3,
) -> Optional[str]:
    """Call chat completion with 30s timeout and exponential backoff retry on rate limit."""
    client = get_client()
    if not client:
        return None

    delay = 1.0
    for attempt in range(max_retries):
        try:
            response = client.chat.completions.create(
                model=LLM_MODEL,
                messages=messages,
                temperature=temperature,
            )
            if response.choices and len(response.choices) > 0:
                content = response.choices[0].message.content
                if content and content.strip():
                    return content.strip()
            return None
        except RateLimitError as e:
            logger.warning(f"Rate limited (attempt {attempt + 1}/{max_retries}): {e}")
            if attempt < max_retries - 1:
                time.sleep(delay)
                delay *= 2.0
            else:
                return None
        except Exception as e:
            logger.warning(f"LLM API error: {e}")
            return None
    return None


def answer_question_llm(query: str, passages: List[Dict[str, Any]]) -> Optional[str]:
    """Answer question with LLM using cached responses when available."""
    cache_key = _get_cache_key("ask", {"query": query, "passages": passages})
    if cache_key in response_cache:
        return response_cache[cache_key]

    passage_texts = "\n\n".join([
        f"[Page {p.get('page', p.get('page_number', '?'))}] {p['text']}"
        for p in passages
    ])

    messages = [
        {"role": "system", "content": QA_SYSTEM_PROMPT},
        {"role": "user", "content": QA_USER_PROMPT.format(passages=passage_texts, query=query)},
    ]

    answer = call_chat_completion(messages, temperature=0.1)
    if answer:
        response_cache[cache_key] = answer
    return answer


def summarize_paper_llm(text: str, level: str = "beginner") -> Optional[str]:
    """Summarize paper excerpt with LLM using cached responses when available."""
    cache_key = _get_cache_key("summarize", {"text": text[:8000], "level": level})
    if cache_key in response_cache:
        return response_cache[cache_key]

    messages = [
        {"role": "system", "content": SUMMARIZE_SYSTEM_PROMPT},
        {"role": "user", "content": SUMMARIZE_USER_PROMPT.format(text=text[:8000], level=level)},
    ]

    summary = call_chat_completion(messages, temperature=0.3)
    if summary:
        response_cache[cache_key] = summary
    return summary

