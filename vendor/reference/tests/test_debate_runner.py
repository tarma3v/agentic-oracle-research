"""Tests for Architecture B debate runner orchestration and tie-break behavior."""

from src.resolution.debate.runner import MultiLLMDebateRunner
from src.resolution.debate.prompts import OPENAI_ROUND_1_SCHEMA, OPENAI_ROUND_2_SCHEMA
from src.retrieval.evidence import EvidencePacket, Source


class DummyDebateAgent:
    MODEL_ID = "dummy"

    def __init__(
        self,
        agent_name,
        round1_payload,
        round2_payload,
        fail_round1=False,
        fail_round2=False,
    ):
        self.AGENT_NAME = agent_name
        self.round1_payload = round1_payload
        self.round2_payload = round2_payload
        self.fail_round1 = fail_round1
        self.fail_round2 = fail_round2
        self.round1_prompts = []
        self.round2_prompts = []

    def generate_structured(self, system_prompt, user_prompt, schema):
        is_round2 = "This is the FINAL round" in user_prompt
        if is_round2:
            self.round2_prompts.append(user_prompt)
            if self.fail_round2:
                raise RuntimeError("round2 failure")
            return self.round2_payload, {"prompt_tokens": 10, "completion_tokens": 20}

        self.round1_prompts.append(user_prompt)
        if self.fail_round1:
            raise RuntimeError("round1 failure")
        return self.round1_payload, {"prompt_tokens": 10, "completion_tokens": 20}


def _sample_evidence():
    return EvidencePacket(
        question_id="Q1",
        question_text="Will X happen?",
        resolution_criteria="Resolves Yes if X happens by date D.",
        resolution_date="2024-12-31T23:59:59Z",
        retrieval_timestamp="2024-01-01T00:00:00Z",
        query_used="query",
        sources=[Source(title="Source", url="https://example.com", published_date=None, text="text")],
    )


def test_debate_runner_round2_majority_and_anonymized_peers():
    agents = [
        DummyDebateAgent(
            "gpt-4o",
            {"decision": "YES", "confidence": 0.9, "reasoning": "gpt r1"},
            {
                "decision": "NO",
                "confidence": 0.8,
                "reasoning": "gpt r2",
                "revised": True,
                "convergence_notes": "Changed after peer evidence",
            },
        ),
        DummyDebateAgent(
            "claude-haiku",
            {"decision": "NO", "confidence": 0.7, "reasoning": "claude r1"},
            {
                "decision": "NO",
                "confidence": 0.8,
                "reasoning": "claude r2",
                "revised": False,
                "convergence_notes": "Stayed with NO",
            },
        ),
        DummyDebateAgent(
            "llama-3.3-70b-turbo",
            {"decision": "YES", "confidence": 0.75, "reasoning": "gemini r1"},
            {
                "decision": "YES",
                "confidence": 0.8,
                "reasoning": "gemini r2",
                "revised": False,
                "convergence_notes": "Stayed with YES",
            },
        ),
    ]

    runner = MultiLLMDebateRunner(agents=agents)
    result = runner.resolve(_sample_evidence())

    assert result.final_decision.value == "NO"
    assert result.tie_break_method == "none"
    assert result.round1_yes_votes == 2
    assert result.round1_no_votes == 1
    assert result.round2_yes_votes == 1
    assert result.round2_no_votes == 2

    for agent in agents:
        prompt = agent.round2_prompts[0]
        assert "AGENT A:" in prompt
        assert "AGENT B:" in prompt
        assert "EVIDENCE (SAME SHARED PACKET FROM ROUND 1):" in prompt
        assert "--- Source 1 ---" in prompt
        assert "text" in prompt

    prompt = agents[0].round2_prompts[0]
    assert "claude-haiku" not in prompt
    assert "llama-3.3-70b-turbo" not in prompt


def test_debate_runner_tie_fallback_round1_majority():
    agents = [
        DummyDebateAgent(
            "gpt-4o",
            {"decision": "YES", "confidence": 0.9, "reasoning": "gpt r1"},
            {
                "decision": "YES",
                "confidence": 0.8,
                "reasoning": "gpt r2",
                "revised": False,
                "convergence_notes": "Stable",
            },
        ),
        DummyDebateAgent(
            "claude-haiku",
            {"decision": "YES", "confidence": 0.8, "reasoning": "claude r1"},
            {
                "decision": "NO",
                "confidence": 0.8,
                "reasoning": "claude r2",
                "revised": True,
                "convergence_notes": "Changed",
            },
        ),
        DummyDebateAgent(
            "llama-3.3-70b-turbo",
            {"decision": "NO", "confidence": 0.8, "reasoning": "gemini r1"},
            {
                "decision": "YES",
                "confidence": 0.8,
                "reasoning": "unused",
                "revised": False,
                "convergence_notes": "unused",
            },
            fail_round2=True,
        ),
    ]

    runner = MultiLLMDebateRunner(agents=agents)
    result = runner.resolve(_sample_evidence())

    assert result.round1_yes_votes == 2
    assert result.round1_no_votes == 1
    assert result.round2_yes_votes == 1
    assert result.round2_no_votes == 1
    assert result.tie_break_method == "round1_majority"
    assert result.final_decision.value == "YES"


def test_debate_runner_double_tie_defaults_no():
    agents = [
        DummyDebateAgent(
            "gpt-4o",
            {"decision": "YES", "confidence": 0.8, "reasoning": "gpt r1"},
            {
                "decision": "YES",
                "confidence": 0.8,
                "reasoning": "unused",
                "revised": False,
                "convergence_notes": "unused",
            },
            fail_round2=True,
        ),
        DummyDebateAgent(
            "claude-haiku",
            {"decision": "NO", "confidence": 0.8, "reasoning": "claude r1"},
            {
                "decision": "NO",
                "confidence": 0.8,
                "reasoning": "unused",
                "revised": False,
                "convergence_notes": "unused",
            },
            fail_round2=True,
        ),
        DummyDebateAgent(
            "llama-3.3-70b-turbo",
            {"decision": "YES", "confidence": 0.8, "reasoning": "unused"},
            {
                "decision": "YES",
                "confidence": 0.8,
                "reasoning": "unused",
                "revised": False,
                "convergence_notes": "unused",
            },
            fail_round1=True,
        ),
    ]

    runner = MultiLLMDebateRunner(agents=agents)
    result = runner.resolve(_sample_evidence())

    assert result.round1_yes_votes == 1
    assert result.round1_no_votes == 1
    assert result.round2_yes_votes == 0
    assert result.round2_no_votes == 0
    assert result.tie_break_method == "default_no"
    assert result.final_decision.value == "NO"


def test_schema_for_deepseek_uses_openai_schema():
    assert MultiLLMDebateRunner._schema_for_agent("deepseek-v3", 1) == OPENAI_ROUND_1_SCHEMA
    assert MultiLLMDebateRunner._schema_for_agent("deepseek-v3", 2) == OPENAI_ROUND_2_SCHEMA


def test_schema_for_llama_uses_openai_schema():
    assert MultiLLMDebateRunner._schema_for_agent("llama-3.3-70b-turbo", 1) == OPENAI_ROUND_1_SCHEMA
    assert MultiLLMDebateRunner._schema_for_agent("llama-3.3-70b-turbo", 2) == OPENAI_ROUND_2_SCHEMA
