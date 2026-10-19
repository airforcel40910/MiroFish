"""
OpenAI Chat Completions compatibility helpers.

This module keeps existing behavior for legacy models/providers while
gracefully adapting request parameters for GPT-5 family models and for
Anthropic's OpenAI-compatible endpoint (https://api.anthropic.com/v1/).

Every backend Chat Completions request goes through create_chat_completion,
which also applies the shared LLM concurrency cap (LLM_MAX_CONCURRENCY) and
retries rate-limit (429) and overloaded (529) errors with backoff.
"""

from __future__ import annotations

import logging
import os
import random
import threading
import time
from typing import Any, Dict, List, Optional
from urllib.parse import urlparse


logger = logging.getLogger(__name__)

# Rate-limit / overload protection
LLM_RETRY_MAX_ATTEMPTS = 6
LLM_RETRY_INITIAL_DELAY = 2.0
LLM_RETRY_MAX_DELAY = 60.0
_RETRYABLE_STATUS_CODES = {429, 529}
_DEFAULT_MAX_CONCURRENCY = 4

_semaphore: Optional[threading.BoundedSemaphore] = None
_semaphore_lock = threading.Lock()


def get_max_concurrency() -> int:
    """Read LLM_MAX_CONCURRENCY from the environment (default 4, minimum 1)."""
    raw = (os.environ.get("LLM_MAX_CONCURRENCY") or "").strip()
    try:
        value = int(raw) if raw else _DEFAULT_MAX_CONCURRENCY
    except ValueError:
        logger.warning("Invalid LLM_MAX_CONCURRENCY=%r; using %d", raw, _DEFAULT_MAX_CONCURRENCY)
        value = _DEFAULT_MAX_CONCURRENCY
    return max(1, value)


def _get_semaphore() -> threading.BoundedSemaphore:
    global _semaphore
    if _semaphore is None:
        with _semaphore_lock:
            if _semaphore is None:
                _semaphore = threading.BoundedSemaphore(get_max_concurrency())
    return _semaphore


def is_anthropic_base_url(base_url: Any) -> bool:
    """Return True when base_url points at Anthropic's API (api.anthropic.com)."""
    if not base_url:
        return False
    host = (urlparse(str(base_url)).hostname or "").lower()
    return host == "anthropic.com" or host.endswith(".anthropic.com")


def is_anthropic_client(client: Any) -> bool:
    return is_anthropic_base_url(getattr(client, "base_url", None))


def is_retryable_llm_error(error: Exception) -> bool:
    """True for rate-limit (429) and overloaded (529 / overloaded_error) errors."""
    if getattr(error, "status_code", None) in _RETRYABLE_STATUS_CODES:
        return True
    body = getattr(error, "body", None)
    if isinstance(body, dict):
        details = body.get("error", body)
        if isinstance(details, dict) and details.get("type") == "overloaded_error":
            return True
    return False


def _retry_after_seconds(error: Exception) -> Optional[float]:
    response = getattr(error, "response", None)
    headers = getattr(response, "headers", None)
    if not headers:
        return None
    try:
        value = headers.get("retry-after")
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


def call_with_llm_retry(func, *args, **kwargs):
    """
    Call func under the shared concurrency cap, retrying 429/529 errors.

    Up to LLM_RETRY_MAX_ATTEMPTS attempts with exponential backoff plus
    jitter (honouring a larger retry-after header). The semaphore is released
    while sleeping so a backing-off call does not hold a slot.
    """
    delay = LLM_RETRY_INITIAL_DELAY
    for attempt in range(1, LLM_RETRY_MAX_ATTEMPTS + 1):
        try:
            with _get_semaphore():
                return func(*args, **kwargs)
        except Exception as error:
            if attempt >= LLM_RETRY_MAX_ATTEMPTS or not is_retryable_llm_error(error):
                raise
            wait = min(delay, LLM_RETRY_MAX_DELAY) * (0.5 + random.random())
            retry_after = _retry_after_seconds(error)
            if retry_after is not None:
                wait = max(wait, min(retry_after, LLM_RETRY_MAX_DELAY))
            logger.warning(
                "LLM rate-limited/overloaded (status=%s), attempt %d/%d; retrying in %.1fs",
                getattr(error, "status_code", None), attempt, LLM_RETRY_MAX_ATTEMPTS, wait,
            )
            time.sleep(wait)
            delay *= 2


def is_gpt5_family(model: Optional[str]) -> bool:
    """Return True when model belongs to GPT-5 family aliases/snapshots."""
    if not model:
        return False
    return model.strip().lower().startswith("gpt-5")


def create_chat_completion(
    client: Any,
    *,
    model: str,
    messages: List[Dict[str, Any]],
    temperature: Optional[float] = None,
    max_tokens: Optional[int] = None,
    response_format: Optional[Dict[str, Any]] = None,
) -> Any:
    """
    Create a chat completion with model-specific request parameters.

    Compatibility strategy:
    - For GPT-5 family, avoid sending temperature by default.
    - For token limit, use `max_completion_tokens` on GPT-5, `max_tokens` otherwise.
    - For Anthropic's OpenAI-compatible endpoint, drop response_format
      (ignored there; JSON is requested in the prompt and parsed robustly) and
      temperature (Claude 4.7+ models reject any value other than 1).
    - Preserve the legacy request shape for every other model/provider.
    - Retry only 429/529 overload errors; propagate all other provider errors
      unchanged instead of guessing from message text.
    """
    kwargs: Dict[str, Any] = {
        "model": model,
        "messages": messages,
    }

    anthropic = is_anthropic_client(client)

    if response_format is not None and not anthropic:
        kwargs["response_format"] = response_format

    gpt5_family = is_gpt5_family(model)

    if temperature is not None and not gpt5_family and not anthropic:
        kwargs["temperature"] = temperature

    if max_tokens is not None:
        if gpt5_family:
            kwargs["max_completion_tokens"] = max_tokens
        else:
            kwargs["max_tokens"] = max_tokens

    return call_with_llm_retry(client.chat.completions.create, **kwargs)


def extract_chat_completion_text(response: Any) -> str:
    """Extract plain text from chat completion response across SDK content shapes."""
    choices = getattr(response, "choices", None) or []
    if not choices:
        return ""

    message = getattr(choices[0], "message", None)
    if message is None:
        return ""

    content = getattr(message, "content", "")

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: List[str] = []
        for item in content:
            if isinstance(item, dict):
                text_obj = item.get("text")
                if isinstance(text_obj, dict):
                    text_obj = text_obj.get("value")
                if isinstance(text_obj, str):
                    chunks.append(text_obj)
                elif isinstance(item.get("content"), str):
                    chunks.append(item["content"])
                continue

            text_obj = getattr(item, "text", None)
            if isinstance(text_obj, dict):
                text_obj = text_obj.get("value")
            if isinstance(text_obj, str):
                chunks.append(text_obj)
                continue

            content_obj = getattr(item, "content", None)
            if isinstance(content_obj, str):
                chunks.append(content_obj)

        return "".join(chunks).strip()

    return str(content or "")
