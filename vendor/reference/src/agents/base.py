from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from datetime import datetime
from enum import Enum
from typing import Optional, Tuple, Dict, Any


class Decision(str, Enum):
    YES = "YES"
    NO = "NO"


@dataclass
class AgentResolution:
    agent_name: str
    question_id: str
    decision: Optional[Decision]
    confidence: Optional[float]
    reasoning: str
    timestamp: str
    latency_ms: int
    prompt_tokens: Optional[int] = None
    completion_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        return {
            "agent_name": self.agent_name,
            "question_id": self.question_id,
            "decision": self.decision.value if self.decision else None,
            "confidence": self.confidence,
            "reasoning": self.reasoning,
            "timestamp": self.timestamp,
            "latency_ms": self.latency_ms,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "error": self.error,
        }


class BaseAgent(ABC):
    AGENT_NAME: str
    MODEL_ID: str

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._get_api_key()
        self._client = None

    @abstractmethod
    def _get_api_key(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def _call_api(self, evidence) -> Tuple[Decision, float, str, Dict[str, Any]]:
        raise NotImplementedError

    def resolve(self, evidence) -> AgentResolution:
        start = time.perf_counter()
        timestamp = datetime.utcnow().isoformat() + "Z"

        try:
            decision, confidence, reasoning, metadata = self._call_api(evidence)
            latency_ms = int((time.perf_counter() - start) * 1000)
            return AgentResolution(
                agent_name=self.AGENT_NAME,
                question_id=evidence.question_id,
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                timestamp=timestamp,
                latency_ms=latency_ms,
                prompt_tokens=metadata.get("prompt_tokens"),
                completion_tokens=metadata.get("completion_tokens"),
            )
        except Exception as exc:  # pragma: no cover - error path
            latency_ms = int((time.perf_counter() - start) * 1000)
            return AgentResolution(
                agent_name=self.AGENT_NAME,
                question_id=getattr(evidence, "question_id", "unknown"),
                decision=None,
                confidence=None,
                reasoning="",
                timestamp=timestamp,
                latency_ms=latency_ms,
                error=str(exc),
            )
