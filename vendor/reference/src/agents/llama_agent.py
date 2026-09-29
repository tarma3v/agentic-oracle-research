"""Llama 3.3 70B agent implementation via Together's OpenAI-compatible API."""

import json
import os
import time
from typing import Any, Dict, Tuple

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


class LlamaAgent(BaseAgent):
    AGENT_NAME = "llama-3.3-70b-turbo"
    MODEL_ID = "meta-llama/Llama-3.3-70B-Instruct-Turbo"

    def __init__(self, api_key: str | None = None):
        super().__init__(api_key)
        base_url = os.getenv("TOGETHER_BASE_URL", "https://api.together.xyz/v1")
        self._client = OpenAI(api_key=self.api_key, base_url=base_url)

    def _get_api_key(self) -> str:
        api_key = os.getenv("TOGETHER_API_KEY")
        if not api_key:
            raise ValueError(
                "Together API key required. Set TOGETHER_API_KEY environment variable "
                "or pass api_key parameter."
            )
        return api_key

    @staticmethod
    def _extract_text_content(response: Any) -> str:
        content = response.choices[0].message.content
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            parts = []
            for item in content:
                if isinstance(item, dict) and "text" in item:
                    parts.append(str(item["text"]))
                elif hasattr(item, "text"):
                    parts.append(str(item.text))
            return "".join(parts)
        return str(content)

    @staticmethod
    def _is_schema_rejection(exc: Exception) -> bool:
        message = str(exc).lower()
        tokens = (
            "json_schema",
            "response_format",
            "unsupported",
            "not supported",
            "invalid parameter",
            "unknown parameter",
        )
        return any(token in message for token in tokens)

    def _request_structured(
        self,
        system_prompt: str,
        user_message: str,
        schema: Dict[str, Any],
        use_json_schema: bool,
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        response_format: Dict[str, Any]
        if use_json_schema:
            response_format = {"type": "json_schema", "json_schema": schema}
        else:
            response_format = {"type": "json_object"}

        response = self._client.chat.completions.create(
            model=self.MODEL_ID,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            response_format=response_format,
            temperature=0.0,
        )

        content = self._extract_text_content(response)
        if not content:
            raise ValueError("Empty response from Together API")

        parsed = json.loads(content)
        metadata = {
            "prompt_tokens": response.usage.prompt_tokens if response.usage else None,
            "completion_tokens": response.usage.completion_tokens if response.usage else None,
        }
        return parsed, metadata

    def _generate_structured_json(
        self, system_prompt: str, user_message: str, schema: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """
        Call Together's OpenAI-compatible API with structured JSON output.
        Tries json_schema first, then falls back to json_object if schema mode is unsupported.
        Retries 3 times with exponential backoff (1s, 2s, 4s).
        """
        max_retries = 3
        delays = [1.0, 2.0, 4.0]

        for attempt in range(max_retries):
            last_exc: Exception | None = None
            try:
                return self._request_structured(
                    system_prompt=system_prompt,
                    user_message=user_message,
                    schema=schema,
                    use_json_schema=True,
                )
            except Exception as exc:
                last_exc = exc
                if self._is_schema_rejection(exc):
                    try:
                        return self._request_structured(
                            system_prompt=system_prompt,
                            user_message=user_message,
                            schema=schema,
                            use_json_schema=False,
                        )
                    except Exception as fallback_exc:
                        last_exc = fallback_exc

            if attempt == max_retries - 1:
                raise last_exc if last_exc is not None else RuntimeError("Structured call failed")
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

        decision = Decision(str(parsed["decision"]).upper())
        confidence = float(parsed["confidence"])
        reasoning = str(parsed["reasoning"])
        return decision, confidence, reasoning, metadata
