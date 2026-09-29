"""Aggregation strategies for multi-agent resolution"""

from dataclasses import dataclass
from datetime import datetime
from typing import List

from src.agents.base import AgentResolution, Decision


@dataclass
class AggregatedResolution:
    """
    Container for the final aggregated resolution from multiple agents.
    Stores individual agent resolutions plus the aggregated decision.
    """

    question_id: str
    final_decision: Decision
    agent_resolutions: List[AgentResolution]
    aggregation_method: str  # "majority_vote"
    yes_votes: int  # Count of YES votes
    no_votes: int  # Count of NO votes
    total_latency_ms: int  # Wall-clock time for full resolution
    timestamp: str

    def to_dict(self) -> dict:
        """Serialize full result for JSON export (thesis analysis)"""
        return {
            "question_id": self.question_id,
            "final_decision": self.final_decision.value,
            "aggregation_method": self.aggregation_method,
            "yes_votes": self.yes_votes,
            "no_votes": self.no_votes,
            "total_latency_ms": self.total_latency_ms,
            "timestamp": self.timestamp,
            "agent_resolutions": [r.to_dict() for r in self.agent_resolutions],
        }


class MajorityVoteStrategy:
    """
    Simple majority vote aggregation strategy.

    """

    def aggregate(
        self, resolutions: List[AgentResolution], total_latency_ms: int
    ) -> AggregatedResolution:
        """
        Aggregate agent resolutions using majority vote.

    
        """
        successful = [r for r in resolutions if r.is_successful]

        yes_votes = sum(1 for r in successful if r.decision == Decision.YES)
        no_votes = len(successful) - yes_votes

        final_decision = Decision.YES if yes_votes > no_votes else Decision.NO

        return AggregatedResolution(
            question_id=resolutions[0].question_id if resolutions else "unknown",
            final_decision=final_decision,
            agent_resolutions=resolutions,
            aggregation_method="majority_vote",
            yes_votes=yes_votes,
            no_votes=no_votes,
            total_latency_ms=total_latency_ms,
            timestamp=datetime.utcnow().isoformat() + "Z",
        )
