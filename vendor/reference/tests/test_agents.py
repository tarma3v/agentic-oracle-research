"""Tests for agent data structures and LLM agents"""

import pytest

from src.agents.base import AgentResolution, BaseAgent, Decision
from src.retrieval.evidence import EvidencePacket, Source


def _sample_evidence() -> EvidencePacket:
    return EvidencePacket(
        question_id="TEST001",
        question_text="Will X happen?",
        resolution_criteria="Resolves Yes if...",
        resolution_date="2024-12-31T23:59:59Z",
        retrieval_timestamp="2024-01-01T00:00:00Z",
        query_used="test query",
        sources=[Source(title="Test", url="https://test.com", published_date=None, text="Content")],
    )


def test_decision_enum():
    assert Decision.YES.value == "YES"
    assert Decision.NO.value == "NO"


def test_agent_resolution_to_dict():
    resolution = AgentResolution(
        agent_name="test_agent",
        question_id="Q1",
        decision=Decision.YES,
        confidence=0.9,
        reasoning="Because.",
        timestamp="2024-01-01T00:00:00Z",
        latency_ms=123,
        prompt_tokens=10,
        completion_tokens=20,
    )
    data = resolution.to_dict()
    assert data["agent_name"] == "test_agent"
    assert data["decision"] == "YES"
    assert data["confidence"] == 0.9
    assert data["prompt_tokens"] == 10


def test_agent_resolution_error():
    resolution = AgentResolution(
        agent_name="test_agent",
        question_id="Q1",
        decision=None,
        confidence=None,
        reasoning="",
        timestamp="2024-01-01T00:00:00Z",
        latency_ms=5,
        error="boom",
    )
    assert not resolution.is_successful
    assert resolution.to_dict()["error"] == "boom"


def test_base_agent_error_handling():
    class FailingAgent(BaseAgent):
        AGENT_NAME = "fail"
        MODEL_ID = "none"

        def _get_api_key(self) -> str:
            return "x"

        def _call_api(self, evidence):
            raise RuntimeError("failure")

    agent = FailingAgent()

    class DummyEvidence:
        question_id = "Q1"

    result = agent.resolve(DummyEvidence())
    assert result.error == "failure"
    assert result.decision is None


def test_gpt4o_agent_parses_response(monkeypatch):
    pytest.importorskip("openai")
    from src.agents import openai_agent

    monkeypatch.setenv("OPENAI_API_KEY", "test")

    class DummyMessage:
        content = '{"decision":"YES","confidence":0.7,"reasoning":"ok"}'

    class DummyChoice:
        message = DummyMessage()

    class DummyUsage:
        prompt_tokens = 11
        completion_tokens = 22

    class DummyResponse:
        choices = [DummyChoice()]
        usage = DummyUsage()

    class DummyCompletions:
        @staticmethod
        def create(**kwargs):
            return DummyResponse()

    class DummyChat:
        completions = DummyCompletions()

    class DummyOpenAI:
        def __init__(self, api_key):
            self.chat = DummyChat()

    monkeypatch.setattr(openai_agent, "OpenAI", DummyOpenAI)

    agent = openai_agent.GPT4oAgent()
    decision, confidence, reasoning, metadata = agent._call_api(_sample_evidence())

    assert decision == Decision.YES
    assert confidence == 0.7
    assert reasoning == "ok"
    assert metadata["prompt_tokens"] == 11


def test_openai_agent_omits_temperature_for_gpt5_models(monkeypatch):
    pytest.importorskip("openai")
    from src.agents import openai_agent

    monkeypatch.setenv("OPENAI_API_KEY", "test")

    class DummyMessage:
        content = '{"decision":"YES","confidence":0.7,"reasoning":"ok"}'

    class DummyChoice:
        message = DummyMessage()

    class DummyUsage:
        prompt_tokens = 1
        completion_tokens = 1

    class DummyResponse:
        choices = [DummyChoice()]
        usage = DummyUsage()

    captured_kwargs = {}

    class DummyCompletions:
        @staticmethod
        def create(**kwargs):
            captured_kwargs.update(kwargs)
            return DummyResponse()

    class DummyChat:
        completions = DummyCompletions()

    class DummyOpenAI:
        def __init__(self, api_key):
            self.chat = DummyChat()

    monkeypatch.setattr(openai_agent, "OpenAI", DummyOpenAI)

    agent = openai_agent.GPT4oAgent()
    _ = agent._call_api(_sample_evidence())

    assert captured_kwargs["model"].startswith("gpt-5")
    assert "temperature" not in captured_kwargs


def test_claude_agent_parses_response(monkeypatch):
    pytest.importorskip("anthropic")
    from src.agents import anthropic_agent

    monkeypatch.setenv("ANTHROPIC_API_KEY", "test")

    class DummyToolBlock:
        type = "tool_use"
        input = {"decision": "NO", "confidence": 0.4, "reasoning": "nope"}

    class DummyUsage:
        input_tokens = 9
        output_tokens = 18

    class DummyResponse:
        content = [DummyToolBlock()]
        usage = DummyUsage()

    class DummyMessages:
        @staticmethod
        def create(**kwargs):
            return DummyResponse()

    class DummyAnthropic:
        def __init__(self, api_key):
            self.messages = DummyMessages()

    monkeypatch.setattr(anthropic_agent, "Anthropic", DummyAnthropic)

    agent = anthropic_agent.ClaudeAgent()
    decision, confidence, reasoning, metadata = agent._call_api(_sample_evidence())

    assert decision == Decision.NO
    assert confidence == 0.4
    assert reasoning == "nope"
    assert metadata["completion_tokens"] == 18


def test_gemini_agent_parses_response(monkeypatch):
    pytest.importorskip("google.genai")
    from src.agents import google_agent

    monkeypatch.setenv("GOOGLE_API_KEY", "test")

    class DummyUsage:
        prompt_token_count = 7
        candidates_token_count = 14
        output_token_count = 14

    class DummyResponse:
        text = '{"decision":"YES","confidence":1.0,"reasoning":"clear"}'
        usage_metadata = DummyUsage()

    def dummy_generate_content(self, *args, **kwargs):
        return DummyResponse()

    class DummyModels:
        def generate_content(self, *args, **kwargs):
            return DummyResponse()

    class DummyClient:
        models = DummyModels()

    def dummy_client(api_key=None):
        return DummyClient()

    import google.genai as genai_module
    monkeypatch.setattr(genai_module, "Client", dummy_client)

    agent = google_agent.GeminiAgent()
    decision, confidence, reasoning, metadata = agent._call_api(_sample_evidence())

    assert decision == Decision.YES
    assert confidence == 1.0
    assert reasoning == "clear"
    assert metadata["completion_tokens"] == 14


def test_deepseek_agent_parses_response(monkeypatch):
    pytest.importorskip("openai")
    from src.agents import deepseek_agent

    monkeypatch.setenv("TOGETHER_API_KEY", "test")
    monkeypatch.setenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")

    class DummyMessage:
        content = '{"decision":"YES","confidence":0.62,"reasoning":"sufficient evidence"}'

    class DummyChoice:
        message = DummyMessage()

    class DummyUsage:
        prompt_tokens = 13
        completion_tokens = 27

    class DummyResponse:
        choices = [DummyChoice()]
        usage = DummyUsage()

    class DummyCompletions:
        @staticmethod
        def create(**kwargs):
            return DummyResponse()

    class DummyChat:
        completions = DummyCompletions()

    class DummyOpenAI:
        def __init__(self, api_key, base_url):
            self.chat = DummyChat()

    monkeypatch.setattr(deepseek_agent, "OpenAI", DummyOpenAI)

    agent = deepseek_agent.DeepSeekAgent()
    decision, confidence, reasoning, metadata = agent._call_api(_sample_evidence())

    assert decision == Decision.YES
    assert confidence == 0.62
    assert reasoning == "sufficient evidence"
    assert metadata["prompt_tokens"] == 13


def test_deepseek_agent_requires_api_key(monkeypatch):
    pytest.importorskip("openai")
    from src.agents import deepseek_agent

    monkeypatch.delenv("TOGETHER_API_KEY", raising=False)

    with pytest.raises(ValueError, match="TOGETHER_API_KEY"):
        deepseek_agent.DeepSeekAgent()
