import asyncio
import sys
import threading
from pathlib import Path
from types import SimpleNamespace

import pytest

from app.utils import openai_chat_compat
from app.utils.llm_client import LLMClient
from app.utils.openai_chat_compat import (
    call_with_llm_retry,
    create_chat_completion,
    get_max_concurrency,
    is_anthropic_base_url,
    is_retryable_llm_error,
)

ANTHROPIC_URL = "https://api.anthropic.com/v1/"


class ProviderError(RuntimeError):
    def __init__(self, status_code, body=None):
        super().__init__(f"status {status_code}")
        self.status_code = status_code
        self.body = body or {}


class Sequence:
    def __init__(self, *results):
        self.results = list(results)
        self.calls = []

    def create(self, **kwargs):
        self.calls.append(kwargs)
        result = self.results.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


def _client(sequence, base_url=None):
    return SimpleNamespace(base_url=base_url, chat=SimpleNamespace(completions=sequence))


def _response(content):
    return SimpleNamespace(
        choices=[SimpleNamespace(finish_reason="stop", message=SimpleNamespace(content=content))]
    )


@pytest.fixture(autouse=True)
def no_sleep(monkeypatch):
    sleeps = []
    monkeypatch.setattr(openai_chat_compat.time, "sleep", sleeps.append)
    return sleeps


@pytest.mark.parametrize(
    ("url", "expected"),
    [
        (ANTHROPIC_URL, True),
        ("https://api.anthropic.com/v1", True),
        ("https://api.openai.com/v1", False),
        ("https://anthropic.com.evil.example/v1", False),
        ("", False),
        (None, False),
    ],
)
def test_anthropic_base_url_detection(url, expected):
    assert is_anthropic_base_url(url) is expected


def test_anthropic_drops_temperature_and_response_format():
    sequence = Sequence(object())
    messages = [{"role": "user", "content": "hi"}]

    create_chat_completion(
        _client(sequence, ANTHROPIC_URL),
        model="claude-sonnet-5-5",
        messages=messages,
        temperature=0.3,
        max_tokens=100,
        response_format={"type": "json_object"},
    )

    assert sequence.calls == [
        {"model": "claude-sonnet-5-5", "messages": messages, "max_tokens": 100}
    ]


def test_openai_request_shape_is_unchanged():
    sequence = Sequence(object())

    create_chat_completion(
        _client(sequence, "https://api.openai.com/v1"),
        model="gpt-4o-mini",
        messages=[],
        temperature=0.3,
        response_format={"type": "json_object"},
    )

    assert sequence.calls[0]["temperature"] == 0.3
    assert sequence.calls[0]["response_format"] == {"type": "json_object"}


@pytest.mark.parametrize(
    "error",
    [
        ProviderError(429),
        ProviderError(529),
        ProviderError(500, {"error": {"type": "overloaded_error"}}),
    ],
)
def test_rate_limit_and_overload_errors_are_retried(error, no_sleep):
    result = object()
    sequence = Sequence(error, error, result)

    assert create_chat_completion(_client(sequence), model="m", messages=[]) is result
    assert len(sequence.calls) == 3
    assert len(no_sleep) == 2
    assert no_sleep[1] > 0


def test_retry_gives_up_after_six_attempts(no_sleep):
    sequence = Sequence(*[ProviderError(429) for _ in range(7)])

    with pytest.raises(ProviderError):
        create_chat_completion(_client(sequence), model="m", messages=[])

    assert len(sequence.calls) == 6
    assert len(no_sleep) == 5


@pytest.mark.parametrize("status_code", [400, 401, 500])
def test_other_errors_are_not_retried(status_code):
    assert not is_retryable_llm_error(ProviderError(status_code))


def test_concurrency_cap_defaults_to_four(monkeypatch):
    monkeypatch.delenv("LLM_MAX_CONCURRENCY", raising=False)
    assert get_max_concurrency() == 4
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "")
    assert get_max_concurrency() == 4
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "2")
    assert get_max_concurrency() == 2
    monkeypatch.setenv("LLM_MAX_CONCURRENCY", "nope")
    assert get_max_concurrency() == 4


def test_concurrency_cap_limits_parallel_calls(monkeypatch):
    monkeypatch.setattr(openai_chat_compat, "_semaphore", threading.BoundedSemaphore(2))
    active = 0
    peak = 0
    lock = threading.Lock()
    release = threading.Event()

    def call():
        nonlocal active, peak
        with lock:
            active += 1
            peak = max(peak, active)
        release.wait(timeout=2)
        with lock:
            active -= 1

    threads = [threading.Thread(target=call_with_llm_retry, args=(call,)) for _ in range(5)]
    for thread in threads:
        thread.start()
    release.set()
    for thread in threads:
        thread.join(timeout=5)

    assert peak <= 2


def _llm_client(sequence, base_url):
    client = object.__new__(LLMClient)
    client.model = "claude-sonnet-5-5"
    client.base_url = base_url
    client.client = _client(sequence, base_url)
    return client


def test_chat_json_strips_fences_and_retries_once_on_anthropic():
    sequence = Sequence(_response("not json"), _response('```json\n{"ok": true}\n```'))

    result = _llm_client(sequence, ANTHROPIC_URL).chat_json(
        messages=[{"role": "user", "content": "Return JSON"}]
    )

    assert result == {"ok": True}
    assert len(sequence.calls) == 2
    assert "response_format" not in sequence.calls[0]


# --- simulation subprocess model routing (scripts/camel_llm.py) ---

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
camel_llm = pytest.importorskip("camel_llm")

LLM_VARS = [
    "LLM_API_KEY", "LLM_BASE_URL", "LLM_MODEL_NAME",
    "LLM_BOOST_API_KEY", "LLM_BOOST_BASE_URL", "LLM_BOOST_MODEL_NAME",
]


@pytest.fixture
def llm_env(monkeypatch):
    for name in LLM_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("LLM_API_KEY", "main-key")
    monkeypatch.setenv("LLM_BASE_URL", ANTHROPIC_URL)
    monkeypatch.setenv("LLM_MODEL_NAME", "claude-sonnet-5-5")
    return monkeypatch


def test_boost_absent_falls_back_to_main(llm_env):
    settings = camel_llm.resolve_llm_settings(use_boost=True)
    assert settings["model"] == "claude-sonnet-5-5"
    assert settings["is_boost"] is False


def test_boost_blank_falls_back_to_main(llm_env):
    for name in LLM_VARS[3:]:
        llm_env.setenv(name, "  ")
    settings = camel_llm.resolve_llm_settings(use_boost=True)
    assert settings["model"] == "claude-sonnet-5-5"
    assert settings["api_key"] == "main-key"


def test_boost_model_only_reuses_main_key_and_url(llm_env):
    llm_env.setenv("LLM_BOOST_MODEL_NAME", "claude-haiku-5-5")
    settings = camel_llm.resolve_llm_settings(use_boost=True)
    assert settings == {
        "api_key": "main-key",
        "base_url": ANTHROPIC_URL,
        "model": "claude-haiku-5-5",
        "is_boost": True,
    }


def test_camel_model_gets_explicit_credentials_and_retry(llm_env):
    llm_env.setenv("LLM_BOOST_MODEL_NAME", "claude-haiku-5-5")
    model = camel_llm.create_camel_model({}, use_boost=True)

    assert str(model.model_type) == "claude-haiku-5-5"
    assert model._api_key == "main-key"
    assert model._url == ANTHROPIC_URL

    calls = []

    async def flaky(*args, **kwargs):
        calls.append(1)
        if len(calls) == 1:
            raise ProviderError(529)
        return "ok"

    real_sleep = asyncio.sleep
    llm_env.setattr(camel_llm.asyncio, "sleep", lambda _s: real_sleep(0))
    model.arun = flaky
    camel_llm._wrap_arun(model)
    assert asyncio.run(model.arun([])) == "ok"
    assert len(calls) == 2
