"""Resolution strategies for multi-agent prediction"""

from .aggregation import AggregatedResolution, MajorityVoteStrategy
from .runner import MultiAgentRunner

__all__ = [
    "AggregatedResolution",
    "MajorityVoteStrategy",
    "MultiAgentRunner",
]

try:
    from .debate import MultiLLMDebateRunner, DebateAggregatedResolution

    __all__.extend(["MultiLLMDebateRunner", "DebateAggregatedResolution"])
except Exception:  # pragma: no cover - optional import path in constrained envs
    pass
