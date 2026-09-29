"""Prompt and schema definitions for Architecture B debate resolution."""

from pydantic import BaseModel, Field

ROUND_1_SYSTEM_PROMPT = """ You are an expert prediction market resolution agent. Your task is to determine whether a prediction market question should resolve to YES or NO based on the provided evidence.

Determine whether the question should resolve to YES or NO based only on the evidence provided."""

ROUND_1_USER_TEMPLATE = """QUESTION: {question}

RESOLUTION CRITERIA: {criteria}

EVIDENCE PROVIDED from reputable sources:
{exa_evidence}

Provide your answer in JSON with these fields:
- decision: YES or NO
- confidence: number between 0.0 and 1.0
- reasoning: detailed step-by-step reasoning that cites relevant evidence and criteria interpretation

Think carefully about:
1. Read the question and resolution criteria carefully
2. Analyze ALL provided sources for relevant information
3. Make your decision based on definitive evidence. You should prioritize information that describe an outcome that happened.
4. If evidence is ambiguous, use your best judgment
5. Rate your confidence in your decision from 0.0 (very uncertain) to 1.0 (absolutely certain)
"""


ROUND_2_SYSTEM_PROMPT = """You are in the final cross-examination round of a multi-agent resolution debate.
You will see how other agents reasoned about the same question. Your job is to CRITICALLY EVALUATE their reasoning against the evidence — not to defer to them.

IMPORTANT: You should only change your answer if you identify a SPECIFIC factual error in your own round 1 reasoning, or if another agent points to SPECIFIC EVIDENCE in the shared packet that you misread or overlooked. Do NOT change your answer simply because other agents disagree with you or sound confident."""

ROUND_2_USER_TEMPLATE = """You previously resolved this prediction market question.
Now you will see how two other expert agents reasoned about the same question.

QUESTION: {question}

RESOLUTION CRITERIA: {criteria}

EVIDENCE (SAME SHARED PACKET FROM ROUND 1):
{exa_evidence}

YOUR PREVIOUS ANSWER:
{your_round_1_response}

OTHER AGENTS' REASONING:
{other_agents_section}

This is the FINAL round. Critically evaluate the other agents' reasoning:

1. For each agent that DISAGREES with you: Do they cite specific evidence from the shared packet that contradicts your reasoning? Or are they asserting a conclusion without evidentiary support?
2. Re-read the specific pieces of evidence that are most relevant to the disagreement.
3. ONLY change your decision if you can identify a concrete error in your own round 1 analysis — for example, you misread a date, overlooked a source, or misinterpreted the resolution criteria.
4. If agents agree with you, do NOT increase your confidence unless they provide additional evidence-based reasoning you hadn't considered.

DEFAULT BEHAVIOR: Change your decision ONLY if another agent identifies specific evidence 
or because there is aconcrete flaw in your reasoning — not simply because they reached 
a different conclusion.

Provide your FINAL answer in JSON with these fields:
- decision: YES or NO
- confidence: number between 0.0 and 1.0
- reasoning: explain your final reasoning, explicitly addressing each disagreeing agent's key claim
- revised: true if you changed your decision from round 1, else false
- convergence_notes: short note on agreement/disagreement and why
"""

OPENAI_ROUND_1_SCHEMA = {
    "name": "debate_round_1_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
        },
        "required": ["decision", "confidence", "reasoning"],
        "additionalProperties": False,
    },
}

OPENAI_ROUND_2_SCHEMA = {
    "name": "debate_round_2_response",
    "strict": True,
    "schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
            "revised": {"type": "boolean"},
            "convergence_notes": {"type": "string"},
        },
        "required": ["decision", "confidence", "reasoning", "revised", "convergence_notes"],
        "additionalProperties": False,
    },
}

ANTHROPIC_ROUND_1_TOOL = {
    "name": "submit_round_1_resolution",
    "description": "Submit round 1 resolution decision",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
        },
        "required": ["decision", "confidence", "reasoning"],
    },
}

ANTHROPIC_ROUND_2_TOOL = {
    "name": "submit_round_2_resolution",
    "description": "Submit final round resolution decision",
    "input_schema": {
        "type": "object",
        "properties": {
            "decision": {"type": "string", "enum": ["YES", "NO"]},
            "confidence": {"type": "number", "minimum": 0.0, "maximum": 1.0},
            "reasoning": {"type": "string"},
            "revised": {"type": "boolean"},
            "convergence_notes": {"type": "string"},
        },
        "required": ["decision", "confidence", "reasoning", "revised", "convergence_notes"],
    },
}


class GeminiRound1Output(BaseModel):
    decision: str = Field(description="YES or NO")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    reasoning: str = Field(description="Step-by-step reasoning")


class GeminiRound2Output(BaseModel):
    decision: str = Field(description="YES or NO")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    reasoning: str = Field(description="Final reasoning")
    revised: bool = Field(description="Whether answer changed from round 1")
    convergence_notes: str = Field(description="Agreement/disagreement summary")
