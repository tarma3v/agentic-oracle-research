"""Data structures for Architecture B multi-LLM debate."""

from dataclasses import dataclass
from datetime import datetime
from typing import Optional, List

from src.agents.base import Decision


@dataclass
class DebateRoundResponse:
    decision: Optional[Decision]
    confidence: Optional[float]
    reasoning: str
    timestamp: str
    latency_ms: int
    revised: Optional[bool] = None
    convergence_notes: Optional[str] = None
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.error is None and self.decision is not None

    def to_dict(self) -> dict:
        return {
            "decision": self.decision.value if self.decision else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "revised": self.revised,
            "convergence_notes": self.convergence_notes,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


@dataclass
class AgentDebateTrace:
    agent_name: str
    round1: DebateRoundResponse
    round2: DebateRoundResponse

    @property
    def revised_decision(self) -> Optional[bool]:
        if not self.round1.is_successful or not self.round2.is_successful:
            return None
        return self.round1.decision != self.round2.decision

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "round1": self.round1.to_dict(),
            "round2": self.round2.to_dict(),
            "revised_decision": self.revised_decision,
        }


@dataclass
class DebateAggregatedResolution:
    question_id: str
    final_decision: Decision
    round1_yes_votes: int
    round1_no_votes: int
    round2_yes_votes: int
    round2_no_votes: int
    tie_break_method: str
    agent_traces: List[AgentDebateTrace]
    total_latency_ms: int
    timestamp: str

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "final_decision": self.final_decision.value,
            "round1_yes_votes": self.round1_yes_votes,
            "round1_no_votes": self.round1_no_votes,
            "round2_yes_votes": self.round2_yes_votes,
            "round2_no_votes": self.round2_no_votes,
            "tie_break_method": self.tie_break_method,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
            "agent_traces": [trace.to_dict() for trace in self.agent_traces],
        }


def now_timestamp() -> str:
    return datetime.utcnow().isoformat() + "Z"
