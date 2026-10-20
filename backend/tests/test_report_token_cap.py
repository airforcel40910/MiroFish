from types import SimpleNamespace

import pytest

from app.services import report_agent as report_agent_module
from app.services.report_agent import (
    ReportAgent,
    ReportOutline,
    ReportSection,
)
from app.utils import llm_client as llm_client_module
from app.utils.llm_client import LLMClient


class RecordingLogger:
    def __init__(self):
        self.warnings = []

    def warning(self, message, *args, **_kwargs):
        self.warnings.append(message % args if args else message)

    def __getattr__(self, _name):
        return lambda *a, **k: None


def _response(text, finish_reason):
    return SimpleNamespace(
        choices=[
            SimpleNamespace(
                finish_reason=finish_reason,
                message=SimpleNamespace(content=text),
            )
        ]
    )


def _client(monkeypatch, finish_reason, sent=None):
    client = LLMClient.__new__(LLMClient)
    client.model = "test-model"
    client.base_url = "https://api.anthropic.com/v1/"
    client.last_finish_reason = None

    def create(**kwargs):
        if sent is not None:
            sent.append(kwargs)
        return _response("Final Answer: Section text that stops", finish_reason)

    monkeypatch.setattr(client, "_create_completion", create)
    return client


@pytest.mark.parametrize("finish_reason", ["length", "max_tokens"])
def test_chat_records_and_warns_on_truncation(monkeypatch, finish_reason):
    log = RecordingLogger()
    monkeypatch.setattr(llm_client_module, "logger", log)
    client = _client(monkeypatch, finish_reason)

    client.chat([{"role": "user", "content": "hi"}], max_tokens=10)

    assert client.last_finish_reason == finish_reason
    assert len(log.warnings) == 1
    assert "truncated" in log.warnings[0]


def test_chat_does_not_warn_on_normal_stop(monkeypatch):
    log = RecordingLogger()
    monkeypatch.setattr(llm_client_module, "logger", log)
    client = _client(monkeypatch, "stop")

    client.chat([{"role": "user", "content": "hi"}])

    assert client.last_finish_reason == "stop"
    assert log.warnings == []


def _agent(llm):
    agent = ReportAgent.__new__(ReportAgent)
    agent.llm = llm
    agent.report_logger = None
    agent.graph_id = "graph"
    agent.simulation_id = "sim"
    agent.simulation_requirement = "requirement"
    agent.zep_tools = None
    agent.tools = {}
    return agent


def test_report_agent_cap_is_raised():
    assert ReportAgent.MAX_OUTPUT_TOKENS >= 16000


def test_section_generation_uses_raised_cap_and_warns_when_truncated(monkeypatch):
    sent = []
    log = RecordingLogger()
    monkeypatch.setattr(report_agent_module, "logger", log)
    agent = _agent(_client(monkeypatch, "length", sent))
    section = ReportSection(title="Funnel")
    outline = ReportOutline(title="T", summary="S", sections=[section])

    try:
        agent._generate_section_react(section, outline, [])
    except Exception:
        # Only the first LLM call matters here; later bookkeeping may need
        # collaborators this unit test does not provide.
        pass

    assert sent, "section generation made no LLM call"
    assert all(call["max_tokens"] == ReportAgent.MAX_OUTPUT_TOKENS for call in sent)
    assert any("Funnel" in w for w in log.warnings)


def test_no_warning_for_complete_section(monkeypatch):
    log = RecordingLogger()
    monkeypatch.setattr(report_agent_module, "logger", log)
    agent = _agent(SimpleNamespace(last_finish_reason="stop"))

    agent._warn_if_truncated("Funnel")

    assert log.warnings == []


def test_chat_calls_use_raised_cap(monkeypatch):
    sent = []
    agent = _agent(_client(monkeypatch, "stop", sent))
    monkeypatch.setattr(agent, "_get_tools_description", lambda: "", raising=False)

    try:
        agent.chat("question?", chat_history=[])
    except Exception:
        pass

    assert sent, "chat made no LLM call"
    assert all(call["max_tokens"] == ReportAgent.MAX_OUTPUT_TOKENS for call in sent)
