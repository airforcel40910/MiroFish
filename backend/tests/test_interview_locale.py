import re
from types import SimpleNamespace

import pytest
from flask import Flask

from app.api import simulation as simulation_api
from app.services import simulation_runner as runner_module
from app.services.zep_tools import (
    AgentInterview,
    InterviewResult,
    ZepToolsService,
)
from app.utils.locale import get_all_translations

CJK = re.compile(r"[一-鿿]")
app = Flask(__name__)


def _locale(lang):
    return app.test_request_context(headers={"Accept-Language": lang})


class FakeLLM:
    def __init__(self):
        self.calls = []

    def chat_json(self, messages, **_kwargs):
        self.calls.append(messages)
        return {"selected_indices": [0], "reasoning": "r", "questions": ["Q1?"]}

    def chat(self, messages, **_kwargs):
        self.calls.append(messages)
        return "summary"


def _service():
    service = ZepToolsService.__new__(ZepToolsService)
    service._llm_client = FakeLLM()
    return service


def _text(messages):
    return "\n".join(m["content"] for m in messages)


def test_chat_prefix_follows_locale():
    with _locale("en"):
        english = simulation_api.optimize_interview_prompt("What do you think?")
    with _locale("zh"):
        chinese = simulation_api.optimize_interview_prompt("What do you think?")

    assert english.endswith("What do you think?")
    assert not CJK.search(english)
    assert CJK.search(chinese)
    assert english != chinese


def test_prefix_check_recognises_both_languages():
    prefixes = get_all_translations("interview.chatPrefix")
    assert len(prefixes) >= 2
    for active in ("en", "zh"):
        for prefix in prefixes:
            prompt = f"{prefix}already prefixed"
            with _locale(active):
                assert simulation_api.optimize_interview_prompt(prompt) == prompt


def test_empty_prompt_is_left_alone():
    with _locale("en"):
        assert simulation_api.optimize_interview_prompt("") == ""


@pytest.mark.parametrize("lang,has_cjk", [("en", False), ("zh", True)])
def test_interview_llm_prompts_follow_locale(lang, has_cjk):
    service = _service()
    profiles = [{"realname": "Ann", "bio": "b", "interested_topics": []}]
    agents = [{"profession": "Barista"}]
    with _locale(lang):
        service._select_agents_for_interview(profiles, "req", "", 1)
        service._generate_interview_questions("req", "", agents)
        service._generate_interview_summary(
            [AgentInterview("Ann", "Barista", "bio", "Q", "A")], "req"
        )

    assert len(service._llm_client.calls) == 3
    for messages in service._llm_client.calls:
        assert bool(CJK.search(_text(messages))) is has_cjk


def test_report_interview_prefix_follows_locale(monkeypatch):
    sent = {}

    def interview_agents_batch(_cls, simulation_id, interviews, **_kwargs):
        sent["prompt"] = interviews[0]["prompt"]
        return {"success": True, "result": {"results": {}}}

    monkeypatch.setattr(
        runner_module.SimulationRunner,
        "interview_agents_batch",
        classmethod(interview_agents_batch),
    )

    for lang, has_cjk in (("en", False), ("zh", True)):
        service = _service()
        service._load_agent_profiles = lambda _id: [
            {"realname": "Ann", "profession": "Barista", "bio": "b"}
        ]
        with _locale(lang):
            service.interview_agents("sim", "req", "", 1, custom_questions=["How?"])
        assert bool(CJK.search(sent["prompt"])) is has_cjk
        assert "1. How?" in sent["prompt"]
    # English asks for the label the response parser strips.
    with _locale("en"):
        service = _service()
        service._load_agent_profiles = lambda _id: [{"realname": "Ann", "bio": "b"}]
        service.interview_agents("sim", "req", "", 1, custom_questions=["How?"])
    assert "Question X:" in sent["prompt"]


def test_english_responses_produce_key_quotes_and_english_labels(monkeypatch):
    answer = (
        "Question 1: I love the idea of a second branch near the station because "
        "it will shorten the morning queue for commuters. I only worry that the "
        "original shop could lose its calm character as it grows."
    )
    monkeypatch.setattr(
        runner_module.SimulationRunner,
        "interview_agents_batch",
        classmethod(
            lambda _cls, **_kw: {
                "success": True,
                "result": {"results": {"twitter_0": {"response": answer}}},
            }
        ),
    )
    service = _service()
    service._load_agent_profiles = lambda _id: [
        {"realname": "Ann", "profession": "Barista", "bio": "b"}
    ]
    with _locale("en"):
        result = service.interview_agents("sim", "req", "", 1, custom_questions=["How?"])
        text = result.to_text()

    interview = result.interviews[0]
    assert interview.key_quotes
    assert all(not CJK.search(q) for q in interview.key_quotes)
    assert not any("Question 1" in q for q in interview.key_quotes)
    assert "[Twitter platform answer]" in interview.response
    assert not CJK.search(text), text


def test_interview_result_labels_follow_locale():
    result = InterviewResult(interview_topic="t", interview_questions=[])
    with _locale("zh"):
        assert "深度采访报告" in result.to_text()
    with _locale("en"):
        assert "In-depth Interview Report" in result.to_text()
