"""Multi-agent runner for parallel resolution execution"""

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Optional

from src.agents import GPT4oAgent, DeepSeekAgent, LlamaAgent
from src.agents.base import BaseAgent
from src.retrieval.evidence import EvidencePacket

from .aggregation import AggregatedResolution, MajorityVoteStrategy


class MultiAgentRunner:
    """
    Orchestrates multiple LLM agents to resolve prediction market questions.

    Runs agents in parallel using ThreadPoolExecutor and aggregates
    results using a configurable strategy (default: majority vote).
    """

    def __init__(
        self,
        agents: Optional[List[BaseAgent]] = None,
        strategy: Optional[MajorityVoteStrategy] = None,
    ):
        """
        Initialize the multi-agent runner.

        Args:
            agents: List of agent instances. Defaults to GPT4oAgent, DeepSeekAgent, LlamaAgent.
            strategy: Aggregation strategy. Defaults to MajorityVoteStrategy.
        """
        self.agents = agents or [GPT4oAgent(), DeepSeekAgent(), LlamaAgent()]
        self.strategy = strategy or MajorityVoteStrategy()

    def resolve(self, evidence: EvidencePacket) -> AggregatedResolution:
        """
        Resolve a prediction market question using all agents in parallel.

        Args:
            evidence: EvidencePacket containing question and retrieved sources

        Returns:
            AggregatedResolution with final decision and individual agent responses
        """
        start = time.perf_counter()

        # Run agents in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=len(self.agents)) as executor:
            futures = {
                executor.submit(agent.resolve, evidence): agent for agent in self.agents
            }
            resolutions = [f.result() for f in as_completed(futures)]

        total_latency_ms = int((time.perf_counter() - start) * 1000)
        return self.strategy.aggregate(resolutions, total_latency_ms)

    def resolve_single(self, evidence: EvidencePacket, agent_index: int):
        """
        Resolve using a single agent (useful for testing/debugging).


        """
        if agent_index < 0 or agent_index >= len(self.agents):
            raise ValueError(f"agent_index must be between 0 and {len(self.agents) - 1}")
        return self.agents[agent_index].resolve(evidence)
