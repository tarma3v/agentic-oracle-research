"""Tests for Architecture B debate dataclasses."""

from src.agents.base import Decision
from src.resolution.debate.types import (
    AgentDebateTrace,
    DebateAggregatedResolution,
    DebateRoundResponse,
)


def test_round_response_to_dict():
    response = DebateRoundResponse(
        decision=Decision.YES,
        confidence=0.9,
        reasoning="Detailed reasoning",
        revised=False,
        convergence_notes="Converged",
        timestamp="2024-01-01T00:00:00Z",
        latency_ms=100,
        prompt_tokens=10,
        completion_tokens=20,
    )

    data = response.to_dict()
    assert data["decision"] == "YES"
    assert data["confidence"] == 0.9
    assert data["revised"] is False
    assert data["error"] is None


def test_agent_trace_revision_detection():
    round1 = DebateRoundResponse(
        decision=Decision.YES,
        confidence=0.8,
        reasoning="r1",
        timestamp="2024-01-01T00:00:00Z",
        latency_ms=10,
    )
    round2 = DebateRoundResponse(
        decision=Decision.NO,
        confidence=0.7,
        reasoning="r2",
        revised=True,
        convergence_notes="Changed",
        timestamp="2024-01-01T00:00:01Z",
        latency_ms=12,
    )

    trace = AgentDebateTrace(agent_name="gpt-4o", round1=round1, round2=round2)
    assert trace.revised_decision is True


def test_aggregated_resolution_to_dict():
    round1 = DebateRoundResponse(
        decision=Decision.YES,
        confidence=0.8,
        reasoning="r1",
        timestamp="2024-01-01T00:00:00Z",
        latency_ms=10,
    )
    round2 = DebateRoundResponse(
        decision=Decision.YES,
        confidence=0.8,
        reasoning="r2",
        revised=False,
        convergence_notes="Stable",
        timestamp="2024-01-01T00:00:01Z",
        latency_ms=12,
    )

    agg = DebateAggregatedResolution(
        question_id="Q1",
        final_decision=Decision.YES,
        round1_yes_votes=2,
        round1_no_votes=1,
        round2_yes_votes=2,
        round2_no_votes=1,
        tie_break_method="none",
        agent_traces=[AgentDebateTrace(agent_name="gpt-4o", round1=round1, round2=round2)],
        total_latency_ms=123,
        timestamp="2024-01-01T00:00:02Z",
    )

    data = agg.to_dict()
    assert data["final_decision"] == "YES"
    assert data["round2_yes_votes"] == 2
    assert len(data["agent_traces"]) == 1
