"""Anthropic Claude Haiku agent implementation"""
import os
import time
from typing import Dict, Any, Tuple

from anthropic import Anthropic

from .base import BaseAgent, Decision
from .prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE

RESOLUTION_TOOL = {
    "name": "submit_resolution",
    "description": "Submit your resolution decision",
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


class ClaudeAgent(BaseAgent):
    AGENT_NAME = "claude-haiku"
    MODEL_ID = "claude-haiku-4-5-20251001"

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        self._client = Anthropic(api_key=self.api_key)

    def _get_api_key(self) -> str:
        api_key = os.getenv("ANTHROPIC_API_KEY")
        if not api_key:
            raise ValueError(
                "Anthropic API key required. Set ANTHROPIC_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return api_key

    def _generate_structured_tool(
        self, system_prompt: str, user_message: str, tool_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call Anthropic API with forced tool use for structured output.
        Implements retry logic: 3 retries with exponential backoff (1s, 2s, 4s).
        """
        max_retries = 3
        delays = [1.0, 2.0, 4.0]

        for attempt in range(max_retries):
            try:
                tool_name = tool_schema["name"]
                response = self._client.messages.create(
                    model=self.MODEL_ID,
                    max_tokens=4096,
                    temperature=0.0,
                    tools=[tool_schema],
                    tool_choice={"type": "tool", "name": tool_name},
                    system=system_prompt,
                    messages=[
                        {"role": "user", "content": user_message},
                    ],
                )

                # Extract tool use result
                if not response.content or len(response.content) == 0:
                    raise ValueError("Empty response from Anthropic API")

                tool_block = next(
                    (block for block in response.content if block.type == "tool_use"), None
                )
                if tool_block is None:
                    raise ValueError("No tool_use block found in response content")

                tool_input = tool_block.input

                metadata = {
                    "prompt_tokens": response.usage.input_tokens if response.usage else None,
                    "completion_tokens": response.usage.output_tokens if response.usage else None,
                }

                return tool_input, metadata

            except Exception as exc:
                if attempt == max_retries - 1:
                    raise exc
                time.sleep(delays[attempt])

        raise RuntimeError("Failed after all retries")  # pragma: no cover

    def generate_structured(
        self, system_prompt: str, user_message: str, tool_schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._generate_structured_tool(
            system_prompt=system_prompt,
            user_message=user_message,
            tool_schema=tool_schema,
        )

    def _call_api(self, evidence) -> Tuple[Decision, float, str, Dict[str, Any]]:
        user_message = USER_MESSAGE_TEMPLATE.format(evidence_text=evidence.format_for_prompt())
        parsed, metadata = self._generate_structured_tool(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            tool_schema=RESOLUTION_TOOL,
        )

        decision = Decision(parsed["decision"])
        confidence = float(parsed["confidence"])
        reasoning = parsed["reasoning"]
        return decision, confidence, reasoning, metadata
