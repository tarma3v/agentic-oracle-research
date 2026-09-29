"""OpenAI agent implementation (default: GPT-5 nano)."""

import json
import os
import time
from typing import Dict, Any, Tuple

from openai import OpenAI

from .base import BaseAgent, Decision
from .prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE

RESOLUTION_SCHEMA = {
    "name": "resolution",
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


class GPT4oAgent(BaseAgent):
    AGENT_NAME = "gpt-5-nano"
    MODEL_ID = "gpt-5-nano"

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        self._client = OpenAI(api_key=self.api_key)

    def _get_api_key(self) -> str:
        api_key = os.getenv("OPENAI_API_KEY")
        if not api_key:
            raise ValueError(
                "OpenAI API key required. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return api_key

    def _generate_structured_json(
        self, system_prompt: str, user_message: str, schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call OpenAI API with structured JSON output.
        Implements retry logic: 3 retries with exponential backoff (1s, 2s, 4s).
        """
        max_retries = 3
        delays = [1.0, 2.0, 4.0]

        for attempt in range(max_retries):
            try:
                request_kwargs: Dict[str, Any] = {
                    "model": self.MODEL_ID,
                    "messages": [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_message},
                    ],
                    "response_format": {"type": "json_schema", "json_schema": schema},
                }
                # GPT-5 nano rejects temperature=0.0; keep deterministic temp for older models.
                if not self.MODEL_ID.startswith("gpt-5"):
                    request_kwargs["temperature"] = 0.0

                response = self._client.chat.completions.create(
                    **request_kwargs,
                )

                content = response.choices[0].message.content
                parsed = json.loads(content)

                metadata = {
                    "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
                    "completion_tokens": response.usage.completion_tokens if response.usage else None,
                }

                return parsed, metadata

            except Exception as exc:
                if attempt == max_retries - 1:
                    raise exc
                time.sleep(delays[attempt])

        raise RuntimeError("Failed after all retries")  # pragma: no cover

    def generate_structured(
        self, system_prompt: str, user_message: str, schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._generate_structured_json(
            system_prompt=system_prompt,
            user_message=user_message,
            schema=schema,
        )

    def _call_api(self, evidence) -> Tuple[Decision, float, str, Dict[str, Any]]:
        user_message = USER_MESSAGE_TEMPLATE.format(evidence_text=evidence.format_for_prompt())
        parsed, metadata = self._generate_structured_json(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            schema=RESOLUTION_SCHEMA,
        )

        decision = Decision(parsed["decision"])
        confidence = float(parsed["confidence"])
        reasoning = parsed["reasoning"]
        return decision, confidence, reasoning, metadata
