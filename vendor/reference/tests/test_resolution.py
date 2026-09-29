"""Tests for resolution aggregation and runner"""

import pytest

from src.agents.base import AgentResolution, Decision
from src.resolution.aggregation import AggregatedResolution, MajorityVoteStrategy
from src.retrieval.evidence import EvidencePacket, Source


def _make_resolution(decision=None, error=None, question_id="Q1"):
    return AgentResolution(
        agent_name="agent",
        question_id=question_id,
        decision=decision,
        confidence=0.5 if decision else None,
        reasoning="reasoning" if decision else "",
        timestamp="2024-01-01T00:00:00Z",
        latency_ms=10,
        error=error,
    )


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


def test_aggregated_resolution_to_dict():
    agg = AggregatedResolution(
        question_id="Q1",
        final_decision=Decision.YES,
        agent_resolutions=[_make_resolution(Decision.YES)],
        aggregation_method="majority_vote",
        yes_votes=1,
        no_votes=0,
        total_latency_ms=50,
        timestamp="2024-01-01T00:00:00Z",
    )
    data = agg.to_dict()
    assert data["final_decision"] == "YES"
    assert data["yes_votes"] == 1


def test_majority_vote_strategy_yes():
    strategy = MajorityVoteStrategy()
    resolutions = [
        _make_resolution(Decision.YES),
        _make_resolution(Decision.YES),
        _make_resolution(Decision.NO),
    ]
    result = strategy.aggregate(resolutions, total_latency_ms=100)
    assert result.final_decision == Decision.YES
    assert result.yes_votes == 2
    assert result.no_votes == 1


def test_majority_vote_strategy_tie_defaults_no():
    strategy = MajorityVoteStrategy()
    resolutions = [
        _make_resolution(Decision.YES),
        _make_resolution(Decision.NO),
        _make_resolution(decision=None, error="boom"),
    ]
    result = strategy.aggregate(resolutions, total_latency_ms=100)
    assert result.final_decision == Decision.NO
    assert result.yes_votes == 1
    assert result.no_votes == 1


def test_multi_agent_runner_with_mock_agents():
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    pytest.importorskip("google.generativeai")
    from src.resolution.runner import MultiAgentRunner

    class DummyAgent:
        def __init__(self, decision):
            self.decision = decision

        def resolve(self, evidence):
            return _make_resolution(self.decision, question_id=evidence.question_id)

    agents = [DummyAgent(Decision.YES), DummyAgent(Decision.NO), DummyAgent(Decision.YES)]
    runner = MultiAgentRunner(agents=agents)

    result = runner.resolve(_sample_evidence())
    assert result.final_decision == Decision.YES
    assert result.yes_votes == 2
    assert result.no_votes == 1


def test_multi_agent_runner_resolve_single_bounds():
    pytest.importorskip("openai")
    pytest.importorskip("anthropic")
    pytest.importorskip("google.generativeai")
    from src.resolution.runner import MultiAgentRunner

    class DummyAgent:
        def resolve(self, evidence):
            return _make_resolution(Decision.YES, question_id=evidence.question_id)

    runner = MultiAgentRunner(agents=[DummyAgent()])

    with pytest.raises(ValueError):
        runner.resolve_single(_sample_evidence(), agent_index=3)
