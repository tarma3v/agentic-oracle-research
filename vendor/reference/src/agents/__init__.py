"""LLM agent implementations"""

from .base import Decision, AgentResolution, BaseAgent

__all__ = [
    "Decision",
    "AgentResolution",
    "BaseAgent",
]

try:
    from .openai_agent import GPT4oAgent

    __all__.append("GPT4oAgent")
except Exception:  # pragma: no cover - optional dependency
    pass

try:
    from .anthropic_agent import ClaudeAgent

    __all__.append("ClaudeAgent")
except Exception:  # pragma: no cover - optional dependency
    pass

try:
    from .google_agent import GeminiAgent

    __all__.append("GeminiAgent")
except Exception:  # pragma: no cover - optional dependency
    pass

try:
    from .deepseek_agent import DeepSeekAgent

    __all__.append("DeepSeekAgent")
except Exception:  # pragma: no cover - optional dependency
    pass

try:
    from .llama_agent import LlamaAgent

    __all__.append("LlamaAgent")
except Exception:  # pragma: no cover - optional dependency
    pass
