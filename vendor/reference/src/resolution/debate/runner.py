"""Architecture B runner: two-round multi-LLM debate with cross-examination."""

from __future__ import annotations

import json
import random
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Dict, List, Optional, Tuple

from src.agents.base import BaseAgent, Decision
from src.retrieval.evidence import EvidencePacket, Source

from .prompts import (
    ANTHROPIC_ROUND_1_TOOL,
    ANTHROPIC_ROUND_2_TOOL,
    GeminiRound1Output,
    GeminiRound2Output,
    OPENAI_ROUND_1_SCHEMA,
    OPENAI_ROUND_2_SCHEMA,
    ROUND_1_SYSTEM_PROMPT,
    ROUND_1_USER_TEMPLATE,
    ROUND_2_SYSTEM_PROMPT,
    ROUND_2_USER_TEMPLATE,
)
from .types import AgentDebateTrace, DebateAggregatedResolution, DebateRoundResponse, now_timestamp


class MultiLLMDebateRunner:
    """Run two debate rounds across heterogeneous agents with shared evidence."""

    def __init__(self, agents: Optional[List[BaseAgent]] = None):
        if agents is None:
            from src.agents import GPT4oAgent, DeepSeekAgent, LlamaAgent

            self.agents = [GPT4oAgent(), DeepSeekAgent(), LlamaAgent()]
        else:
            self.agents = agents

    def resolve(self, evidence: EvidencePacket) -> DebateAggregatedResolution:
        start = time.perf_counter()

        round1_by_agent = self._run_round1(evidence)
        round2_by_agent = self._run_round2(evidence, round1_by_agent)

        round1_yes_votes, round1_no_votes = self._count_votes(round1_by_agent)
        round2_yes_votes, round2_no_votes = self._count_votes(round2_by_agent)

        final_decision, tie_break_method = self._aggregate_decision(
            round1_yes_votes=round1_yes_votes,
            round1_no_votes=round1_no_votes,
            round2_yes_votes=round2_yes_votes,
            round2_no_votes=round2_no_votes,
        )

        traces = [
            AgentDebateTrace(
                agent_name=agent.AGENT_NAME,
                round1=round1_by_agent.get(
                    agent.AGENT_NAME,
                    self._error_response("Round 1 response missing"),
                ),
                round2=round2_by_agent.get(
                    agent.AGENT_NAME,
                    self._error_response("Round 2 response missing"),
                ),
            )
            for agent in self.agents
        ]

        total_latency_ms = int((time.perf_counter() - start) * 1000)
        return DebateAggregatedResolution(
            question_id=evidence.question_id,
            final_decision=final_decision,
            round1_yes_votes=round1_yes_votes,
            round1_no_votes=round1_no_votes,
            round2_yes_votes=round2_yes_votes,
            round2_no_votes=round2_no_votes,
            tie_break_method=tie_break_method,
            agent_traces=traces,
            total_latency_ms=total_latency_ms,
            timestamp=now_timestamp(),
        )

    def _run_round1(self, evidence: EvidencePacket) -> Dict[str, DebateRoundResponse]:
        if not self.agents:
            return {}

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(self._resolve_round1_for_agent, agent, evidence): agent.AGENT_NAME
                for agent in self.agents
            }
            return {agent_name: future.result() for future, agent_name in futures.items()}

    def _run_round2(
        self,
        evidence: EvidencePacket,
        round1_by_agent: Dict[str, DebateRoundResponse],
    ) -> Dict[str, DebateRoundResponse]:
        if not self.agents:
            return {}

        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(
                    self._resolve_round2_for_agent,
                    agent,
                    evidence,
                    round1_by_agent,
                ): agent.AGENT_NAME
                for agent in self.agents
            }
            return {agent_name: future.result() for future, agent_name in futures.items()}

    def _resolve_round1_for_agent(
        self,
        agent: BaseAgent,
        evidence: EvidencePacket,
    ) -> DebateRoundResponse:
        exa_evidence = self._format_exa_sources(evidence.sources)
        user_prompt = ROUND_1_USER_TEMPLATE.format(
            question=evidence.question_text,
            criteria=evidence.resolution_criteria,
            exa_evidence=exa_evidence,
        )

        return self._run_structured_round(
            agent=agent,
            round_number=1,
            system_prompt=ROUND_1_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def _resolve_round2_for_agent(
        self,
        agent: BaseAgent,
        evidence: EvidencePacket,
        round1_by_agent: Dict[str, DebateRoundResponse],
    ) -> DebateRoundResponse:
        own_round1 = round1_by_agent.get(agent.AGENT_NAME)
        if own_round1 is None or not own_round1.is_successful:
            return self._error_response("Round 1 failed for this agent; skipping round 2")

        your_round_1_response = json.dumps(
            {
                "decision": own_round1.decision.value if own_round1.decision else None,
                "confidence": own_round1.confidence,
                "reasoning": own_round1.reasoning,
            },
            indent=2,
        )

        peers = [
            (name, response)
            for name, response in round1_by_agent.items()
            if name != agent.AGENT_NAME and response.is_successful
        ]
        random.Random(f"{evidence.question_id}:{agent.AGENT_NAME}").shuffle(peers)

        if peers:
            sections = []
            for idx, (peer_name, peer_response) in enumerate(peers, start=1):
                label = f"AGENT {chr(ord('A') + idx - 1)}"
                sections.append(
                    "\n".join(
                        [
                            f"{label}:",
                            f"Answer: {peer_response.decision.value if peer_response.decision else 'N/A'}",
                            f"Confidence: {peer_response.confidence}",
                            f"Reasoning: {peer_response.reasoning}",
                        ]
                    )
                )
            other_agents_section = "\n\n".join(sections)
        else:
            other_agents_section = "No successful peer responses available."

        exa_evidence = self._format_exa_sources(evidence.sources)
        user_prompt = ROUND_2_USER_TEMPLATE.format(
            question=evidence.question_text,
            criteria=evidence.resolution_criteria,
            exa_evidence=exa_evidence,
            your_round_1_response=your_round_1_response,
            other_agents_section=other_agents_section,
        )

        return self._run_structured_round(
            agent=agent,
            round_number=2,
            system_prompt=ROUND_2_SYSTEM_PROMPT,
            user_prompt=user_prompt,
        )

    def _run_structured_round(
        self,
        agent: BaseAgent,
        round_number: int,
        system_prompt: str,
        user_prompt: str,
    ) -> DebateRoundResponse:
        start = time.perf_counter()
        timestamp = now_timestamp()

        try:
            schema = self._schema_for_agent(agent.AGENT_NAME, round_number)
            generator = getattr(agent, "generate_structured", None)
            if generator is None:
                raise ValueError(f"{agent.AGENT_NAME} does not support generate_structured")

            parsed, metadata = generator(system_prompt, user_prompt, schema)
            decision, confidence, reasoning = self._parse_common_fields(parsed)

            revised = None
            convergence_notes = None
            if round_number == 2:
                revised = bool(parsed["revised"])
                convergence_notes = str(parsed["convergence_notes"])

            latency_ms = int((time.perf_counter() - start) * 1000)
            return DebateRoundResponse(
                decision=decision,
                confidence=confidence,
                reasoning=reasoning,
                revised=revised,
                convergence_notes=convergence_notes,
                timestamp=timestamp,
                latency_ms=latency_ms,
                prompt_tokens=metadata.get("prompt_tokens"),
                completion_tokens=metadata.get("completion_tokens"),
            )
        except Exception as exc:
            latency_ms = int((time.perf_counter() - start) * 1000)
            return DebateRoundResponse(
                decision=None,
                confidence=None,
                reasoning="",
                revised=None,
                convergence_notes=None,
                timestamp=timestamp,
                latency_ms=latency_ms,
                error=str(exc),
            )

    @staticmethod
    def _schema_for_agent(agent_name: str, round_number: int) -> Any:
        from src.agents.openai_agent import GPT4oAgent

        if agent_name in {
            GPT4oAgent.AGENT_NAME,
            "gpt-4o",
            "deepseek-v3",
            "llama-3.3-70b-turbo",
        }:
            return OPENAI_ROUND_1_SCHEMA if round_number == 1 else OPENAI_ROUND_2_SCHEMA
        if agent_name == "claude-haiku":
            return ANTHROPIC_ROUND_1_TOOL if round_number == 1 else ANTHROPIC_ROUND_2_TOOL
        if agent_name == "gemini-2.0-flash":
            return GeminiRound1Output if round_number == 1 else GeminiRound2Output
        raise ValueError(f"Unsupported agent_name: {agent_name}")

    @staticmethod
    def _parse_common_fields(parsed: Dict[str, Any]) -> Tuple[Decision, float, str]:
        decision_raw = str(parsed["decision"]).strip().upper()
        if decision_raw not in {"YES", "NO"}:
            raise ValueError(f"Invalid decision: {parsed['decision']}")

        confidence = float(parsed["confidence"])
        if not (0.0 <= confidence <= 1.0):
            raise ValueError(f"Confidence out of range: {confidence}")

        reasoning = str(parsed["reasoning"])
        return Decision(decision_raw), confidence, reasoning

    @staticmethod
    def _count_votes(by_agent: Dict[str, DebateRoundResponse]) -> Tuple[int, int]:
        yes_votes = sum(
            1
            for response in by_agent.values()
            if response.is_successful and response.decision == Decision.YES
        )
        no_votes = sum(
            1
            for response in by_agent.values()
            if response.is_successful and response.decision == Decision.NO
        )
        return yes_votes, no_votes

    @staticmethod
    def _aggregate_decision(
        round1_yes_votes: int,
        round1_no_votes: int,
        round2_yes_votes: int,
        round2_no_votes: int,
    ) -> Tuple[Decision, str]:
        if round2_yes_votes > round2_no_votes:
            return Decision.YES, "none"
        if round2_no_votes > round2_yes_votes:
            return Decision.NO, "none"

        if round1_yes_votes > round1_no_votes:
            return Decision.YES, "round1_majority"
        if round1_no_votes > round1_yes_votes:
            return Decision.NO, "round1_majority"

        return Decision.NO, "default_no"

    @staticmethod
    def _format_exa_sources(sources: List[Source]) -> str:
        if not sources:
            return "No evidence sources were retrieved."

        lines: List[str] = []
        for idx, source in enumerate(sources, start=1):
            lines.append(f"--- Source {idx} ---")
            lines.append(f"Title: {source.title}")
            lines.append(f"URL: {source.url}")
            if source.published_date:
                lines.append(f"Published: {source.published_date}")
            lines.append(source.text)
            lines.append("")
        return "\n".join(lines)

    @staticmethod
    def _error_response(error: str) -> DebateRoundResponse:
        return DebateRoundResponse(
            decision=None,
            confidence=None,
            reasoning="",
            timestamp=now_timestamp(),
            latency_ms=0,
            error=error,
        )
