"""Google Gemini 2.0 agent implementation (google-genai SDK)"""

import json
import os
import time
from typing import Dict, Any, Tuple

from pydantic import BaseModel, Field

from .base import BaseAgent, Decision
from .prompts import SYSTEM_PROMPT, USER_MESSAGE_TEMPLATE


class ResolutionOutput(BaseModel):
    """Structured output schema for resolution decision."""

    decision: str = Field(description="YES or NO")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence 0.0-1.0")
    reasoning: str = Field(description="Explanation referencing sources")


class GeminiAgent(BaseAgent):
    AGENT_NAME = "gemini-2.0-flash"
    MODEL_ID = "gemini-2.0-flash"

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        from google import genai

        self._client = genai.Client(api_key=self.api_key)

    def _get_api_key(self) -> str:
        api_key = os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError(
                "Google API key required. Set GOOGLE_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return api_key

    def _generate_structured_content(
        self, system_prompt: str, user_message: str, response_schema: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call Google Gemini API with structured JSON output (google-genai SDK).
        Implements retry logic: 3 retries with exponential backoff (1s, 2s, 4s).
        """
        full_prompt = f"{system_prompt}\n\n{user_message}"

        max_retries = 3
        delays = [1.0, 2.0, 4.0]

        for attempt in range(max_retries):
            try:
                from google.genai import types

                config = types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=response_schema,
                    temperature=0.0,
                )
                response = self._client.models.generate_content(
                    model=self.MODEL_ID,
                    contents=full_prompt,
                    config=config,
                )

                text = getattr(response, "text", None)
                if not text:
                    raise ValueError("Empty response from Gemini API")

                parsed = json.loads(text)

                usage = getattr(response, "usage_metadata", None)
                metadata = {
                    "prompt_tokens": getattr(usage, "prompt_token_count", None) if usage else None,
                    "completion_tokens": getattr(usage, "candidates_token_count", None)
                    or getattr(usage, "output_token_count", None)
                    if usage
                    else None,
                }

                return parsed, metadata

            except Exception as exc:
                if attempt == max_retries - 1:
                    raise exc
                time.sleep(delays[attempt])

        raise RuntimeError("Failed after all retries")  # pragma: no cover

    def generate_structured(
        self, system_prompt: str, user_message: str, response_schema: Any
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        return self._generate_structured_content(
            system_prompt=system_prompt,
            user_message=user_message,
            response_schema=response_schema,
        )

    def _call_api(self, evidence) -> Tuple[Decision, float, str, Dict[str, Any]]:
        user_message = USER_MESSAGE_TEMPLATE.format(evidence_text=evidence.format_for_prompt())
        parsed, metadata = self._generate_structured_content(
            system_prompt=SYSTEM_PROMPT,
            user_message=user_message,
            response_schema=ResolutionOutput,
        )

        decision = Decision(parsed["decision"])
        confidence = float(parsed["confidence"])
        reasoning = parsed["reasoning"]
        return decision, confidence, reasoning, metadata
