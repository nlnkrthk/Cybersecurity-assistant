"""
Chat model used by the agents, chosen by the shared LLM settings in .env
(see llm.settings for all variables):

    LLM_PROVIDER=nvidia | google | anthropic     LLM_MODEL=...     <PROVIDER>_API_KEY=...

The model must support tool calling. LLM_REQUESTS_PER_MINUTE adds a client-side
rate limiter, so calls wait instead of failing with 429 on free tiers.
"""
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_core.rate_limiters import InMemoryRateLimiter

from llm.settings import API_KEY_VARS, PROVIDERS, llm_settings as _resolved_settings, public_settings

AGENT_MAX_TOKENS = 4000
REQUEST_TIMEOUT = 120

__all__ = ["get_chat_model", "llm_settings", "PROVIDERS"]


def llm_settings() -> Dict[str, Any]:
    """Resolved provider, model and effort (never includes the API key)."""
    return public_settings()


def _rate_limiter(rpm: Optional[float]) -> Optional[InMemoryRateLimiter]:
    """Space calls evenly (bucket of 1) so no rolling minute exceeds the provider quota."""
    if not rpm:
        return None
    return InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.2, max_bucket_size=1)


@lru_cache(maxsize=1)
def get_chat_model():
    """LangChain chat model for the configured provider (cached)."""
    settings = _resolved_settings()
    provider, model, effort = settings["provider"], settings["model"], settings["reasoning_effort"]
    api_key = settings["api_key"]
    limiter = _rate_limiter(settings["requests_per_minute"])
    if not api_key:
        raise RuntimeError(
            f"No API key for LLM_PROVIDER={provider}. Set {' or '.join(API_KEY_VARS[provider])} in .env."
        )

    if provider == "google":
        from langchain_google_genai import ChatGoogleGenerativeAI

        # Native Gemini client: keeps the thought signatures Gemini needs across tool-call turns.
        return ChatGoogleGenerativeAI(
            model=model,
            google_api_key=api_key,
            max_output_tokens=AGENT_MAX_TOKENS,
            reasoning_effort=effort if effort in ("minimal", "low", "medium", "high") else "low",
            timeout=REQUEST_TIMEOUT,
            max_retries=6,  # retries 429s with backoff when another client shares the quota
            rate_limiter=limiter,
        )

    if provider == "nvidia":
        from langchain_openai import ChatOpenAI

        # NIM serves an OpenAI-compatible API.
        return ChatOpenAI(
            model=model,
            api_key=api_key,
            base_url=settings["base_url"],
            max_tokens=AGENT_MAX_TOKENS,
            reasoning_effort=effort,
            timeout=REQUEST_TIMEOUT * 2,  # free-tier queueing can be slow
            max_retries=2,
            rate_limiter=limiter,
        )

    from langchain_anthropic import ChatAnthropic

    # No temperature: current Claude models reject non-default sampling parameters.
    return ChatAnthropic(
        model=model,
        api_key=api_key,
        max_tokens=AGENT_MAX_TOKENS,
        timeout=REQUEST_TIMEOUT,
        max_retries=2,
        rate_limiter=limiter,
    )
