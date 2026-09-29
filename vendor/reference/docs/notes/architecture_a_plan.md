# Architecture A: Independent LLM Agents with Aggregation

## Overview

Implement three independent LLM agents (GPT-4o, Claude Haiku, Gemini 2.0) that receive identical `EvidencePacket` inputs and produce structured YES/NO decisions with reasoning. Outputs are aggregated via majority vote.

## File Structure

```
src/agents/
    __init__.py              # Exports
    base.py                  # Decision enum, AgentResolution dataclass, BaseAgent ABC
    prompts.py               # System prompt + user message template
    openai_agent.py          # GPT4oAgent (json_schema structured output)
    anthropic_agent.py       # ClaudeAgent (tool_use structured output)
    google_agent.py          # GeminiAgent (response_schema structured output)

src/resolution/
    __init__.py              # Exports
    aggregation.py           # AggregatedResolution, MajorityVoteStrategy
    runner.py                # MultiAgentRunner (parallel execution)
```

## Phase 1: Core Data Structures

### File: `src/agents/base.py`

```python
from enum import Enum
from dataclasses import dataclass
from typing import Optional
from abc import ABC, abstractmethod

class Decision(str, Enum):
    YES = "YES"
    NO = "NO"

@dataclass
class AgentResolution:
    agent_name: str
    question_id: str
    decision: Optional[Decision]  # None if error
    confidence: Optional[float]   # 0.0-1.0 scale, None if error
    reasoning: str
    timestamp: str
    latency_ms: int                      # Measured by your code
    prompt_tokens: Optional[int] = None  # From API response metadata
    completion_tokens: Optional[int] = None
    error: Optional[str] = None

    @property
    def is_successful(self) -> bool:
        return self.error is None

    def to_dict(self) -> dict:
        """Serialize for JSON export (thesis analysis)"""
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
            "error": self.error
        }

class BaseAgent(ABC):
    AGENT_NAME: str
    MODEL_ID: str

    def __init__(self, api_key: Optional[str] = None):
        self.api_key = api_key or self._get_api_key()
        self._client = None

    @abstractmethod
    def _get_api_key(self) -> str: ...

    @abstractmethod
    def _call_api(self, evidence) -> tuple[Decision, str, dict]: ...

    def resolve(self, evidence) -> AgentResolution:
        # Wraps _call_api with timing, error handling
        ...
```

### File: `src/agents/prompts.py`

```python
SYSTEM_PROMPT = """You are an expert prediction market resolution agent. Your task is to determine whether a prediction market question should resolve to YES or NO based on the provided evidence.

## Instructions
1. Read the question and resolution criteria carefully
2. Analyze ALL provided sources for relevant information
3. Make your decision based on definitive evidence
4. If evidence is ambiguous, use your best judgment
5. Rate your confidence in your decision from 0.0 (very uncertain) to 1.0 (absolutely certain)

## Output
- decision: YES or NO
- confidence: A number from 0.0 to 1.0 indicating how confident you are
- reasoning: Your explanation referencing specific sources"""

USER_MESSAGE_TEMPLATE = """Please analyze the following prediction market question and evidence, then provide your resolution decision.

{evidence_text}

Based on the evidence above, should this question resolve to YES or NO?"""
```

### Evidence → Prompt Integration

`EvidencePacket` already has a `format_for_prompt()` method (in `src/retrieval/evidence.py`) that formats evidence for LLM consumption:

```python
# EvidencePacket.format_for_prompt() returns:
"""
Question: Will X happen by December 2024?
Resolution Criteria: Resolves Yes if X occurs before midnight ET on Dec 31, 2024.
Resolution Date: 2024-12-31T23:59:59Z

Retrieved 10 sources:

--- Source 1 ---
Title: X happened today, officials confirm
URL: https://example.com/article
Published: 2024-12-15

[Full article text here...]

--- Source 2 ---
...
"""
```

**Usage in agents:**

```python
from src.agents.prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE

def _call_api(self, evidence: EvidencePacket) -> tuple[Decision, float, str, dict]:
    user_message = USER_MESSAGE_TEMPLATE.format(
        evidence_text=evidence.format_for_prompt()
    )
    # Now pass system_prompt + user_message to the LLM API
```

## Phase 2: Agent Implementations

### File: `src/agents/openai_agent.py` (GPT-4o)

**Structured output approach**: `response_format` with `json_schema` and `strict: True`

```python
RESOLUTION_SCHEMA = {
    "name": "resolution",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"}
        },
        "required": ["decision", "confidence", "reasoning"],
        "additionalProperties": False
    }
}

response = client.chat.completions.create(
    model="gpt-4o",
    messages=[...],
    response_format={"type": "json_schema", "json_schema": RESOLUTION_SCHEMA},
    temperature=0.0
)
parsed = json.loads(response.choices[0].message.content)
# parsed = {"decision": "YES", "confidence": 8.5, "reasoning": "..."}
```

### File: `src/agents/anthropic_agent.py` (Claude Haiku)

**Structured output approach**: Forced tool use with `tool_choice`

```python
RESOLUTION_TOOL = {
    "name": "submit_resolution",
    "description": "Submit your resolution decision",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"}
        },
        "required": ["decision", "confidence", "reasoning"]
    }
}

response = client.messages.create(
    model="claude-3-5-haiku-20241022",
    tools=[RESOLUTION_TOOL],
    tool_choice={"type": "tool", "name": "submit_resolution"},
    messages=[...]
)
tool_input = response.content[0].input  # {"decision": "YES", "confidence": 8.5, "reasoning": "..."}
```

### File: `src/agents/google_agent.py` (Gemini 2.0)

**Structured output approach**: `response_mime_type` + `response_schema`

```python
from google.ai.generativelanguage import Schema, Type

RESOLUTION_SCHEMA = Schema(
    type=Type.OBJECT,
    properties={
        "decision": Schema(type=Type.STRING, enum=["YES", "NO"]),
        "confidence": Schema(type=Type.NUMBER),  # 0.0-1.0
        "reasoning": Schema(type=Type.STRING)
    },
    required=["decision", "confidence", "reasoning"]
)

response = model.generate_content(
    user_message,
    generation_config=GenerationConfig(
        response_mime_type="application/json",
        response_schema=RESOLUTION_SCHEMA,
        temperature=0.0
    )
)
parsed = json.loads(response.text)  # {"decision": "YES", "confidence": 8.5, "reasoning": "..."}
```

## Phase 3: Aggregation Layer

### File: `src/resolution/aggregation.py`

```python
@dataclass
class AggregatedResolution:
    question_id: str
    final_decision: Decision
    agent_resolutions: List[AgentResolution]
    aggregation_method: str  # "majority_vote"
    yes_votes: int           # Count of YES votes
    no_votes: int            # Count of NO votes
    total_latency_ms: int    # Wall-clock time for full resolution
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
            "agent_resolutions": [r.to_dict() for r in self.agent_resolutions]
        }

class MajorityVoteStrategy:
    """
    Simple majority vote: if more agents say YES, final is YES; otherwise NO.
    With 3 agents: need 2+ YES for YES, otherwise NO.
    """
    def aggregate(self, resolutions: List[AgentResolution], total_latency_ms: int) -> AggregatedResolution:
        successful = [r for r in resolutions if r.is_successful]
        yes_votes = sum(1 for r in successful if r.decision == Decision.YES)
        no_votes = len(successful) - yes_votes

        # Majority wins: if yes_votes > no_votes => YES, else NO
        final_decision = Decision.YES if yes_votes > no_votes else Decision.NO

        return AggregatedResolution(
            question_id=resolutions[0].question_id,
            final_decision=final_decision,
            agent_resolutions=resolutions,
            aggregation_method="majority_vote",
            yes_votes=yes_votes,
            no_votes=no_votes,
            total_latency_ms=total_latency_ms,
            timestamp=datetime.utcnow().isoformat() + "Z"
        )
```

### File: `src/resolution/runner.py`

```python
class MultiAgentRunner:
    def __init__(self, agents=None, strategy=None):
        self.agents = agents or [GPT4oAgent(), ClaudeAgent(), GeminiAgent()]
        self.strategy = strategy or MajorityVoteStrategy()

    def resolve(self, evidence: EvidencePacket) -> AggregatedResolution:
        start = time.perf_counter()

        # Run agents in parallel with ThreadPoolExecutor
        with ThreadPoolExecutor(max_workers=3) as executor:
            futures = {executor.submit(agent.resolve, evidence): agent
                       for agent in self.agents}
            resolutions = [f.result() for f in as_completed(futures)]

        total_latency_ms = int((time.perf_counter() - start) * 1000)
        return self.strategy.aggregate(resolutions, total_latency_ms)
```

## Phase 4: Testing

### File: `tests/test_agents.py`

- Test `AgentResolution` dataclass creation
- Test `Decision` enum
- Mock API responses for each agent
- Test error handling (failed API calls)

### File: `tests/test_resolution.py`

- Test `MajorityVoteStrategy` with various vote distributions
- Test `AggregatedResolution` properties
- Test `MultiAgentRunner` with mocked agents

### Integration Test

```python
# test_integration.py
evidence = ExaOracleRetriever.load_from_cache("some_question_id")
runner = MultiAgentRunner()
result = runner.resolve(evidence)
print(f"Decision: {result.final_decision}")
```

## Implementation Order

1. **`src/agents/base.py`** - Decision, AgentResolution, BaseAgent ABC
2. **`src/agents/prompts.py`** - System prompt and template
3. **`src/agents/openai_agent.py`** - GPT4oAgent
4. **`src/agents/anthropic_agent.py`** - ClaudeAgent
5. **`src/agents/google_agent.py`** - GeminiAgent
6. **`src/agents/__init__.py`** - Exports
7. **`src/resolution/aggregation.py`** - AggregatedResolution, MajorityVoteStrategy
8. **`src/resolution/runner.py`** - MultiAgentRunner
9. **`src/resolution/__init__.py`** - Exports
10. **`tests/test_agents.py`** - Unit tests
11. **`tests/test_resolution.py`** - Unit tests

## Environment Variables Required

```bash
# .env
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...
GOOGLE_API_KEY=AI...
```

## Verification

1. Run unit tests: `pytest tests/test_agents.py tests/test_resolution.py -v`
2. Run integration test with cached evidence:
   ```python
   from src.resolution import MultiAgentRunner
   from src.retrieval import ExaOracleRetriever

   evidence = ExaOracleRetriever.load_from_cache("question_id")
   runner = MultiAgentRunner()
   result = runner.resolve(evidence)
   ```
3. Verify all three agents return valid decisions
4. Verify aggregation produces correct majority vote

## Key Design Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Temperature | 0.0 | Deterministic for reproducibility |
| Parallel execution | ThreadPoolExecutor | Simple, works with sync API clients |
| Error handling | Return AgentResolution with error field | Graceful degradation, don't block on single agent failure |
| Aggregation | Simple majority vote | YES if majority says YES, else NO |
| Confidence scale | 0.0-1.0 | Agent self-reported confidence for analysis |

## Confirmed Choices

- **Claude model**: `claude-3-5-haiku-20241022`
- **Gemini model**: `gemini-2.0-flash`
- **Retry logic**: Yes - 3 retries with exponential backoff (1s, 2s, 4s delays)
