"""
Chat models used by the agents and by LLM-as-Judge, chosen by the shared LLM settings in
.env (see llm.settings for all variables):

    LLM_PROVIDER=nvidia | google | anthropic     LLM_MODEL=...     <PROVIDER>_API_KEY=...
    JUDGE_PROVIDER / JUDGE_MODEL                 optional separate judge model

The assistant model must support tool calling. LLM_REQUESTS_PER_MINUTE adds a client-side
rate limiter, so calls wait instead of failing with 429 on free tiers.
"""
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_core.rate_limiters import InMemoryRateLimiter

from llm.settings import (
    API_KEY_VARS,
    PROVIDERS,
    judge_settings as _judge_settings,
    llm_settings as _resolved_settings,
    public_settings,
)

AGENT_MAX_TOKENS = 4000
REQUEST_TIMEOUT = 120

__all__ = ["get_chat_model", "get_judge_model", "llm_settings", "judge_model_settings", "PROVIDERS"]


def llm_settings() -> Dict[str, Any]:
    """Resolved provider, model and effort of the assistant model (never includes the API key)."""
    return public_settings()


def judge_model_settings() -> Dict[str, Any]:
    """Resolved settings of the judge model (never includes the API key)."""
    return public_settings(_judge_settings())


def _rate_limiter(rpm: Optional[float]) -> Optional[InMemoryRateLimiter]:
    """Space calls evenly (bucket of 1) so no rolling minute exceeds the provider quota."""
    if not rpm:
        return None
    return InMemoryRateLimiter(requests_per_second=rpm / 60, check_every_n_seconds=0.2, max_bucket_size=1)


def build_chat_model(settings: Dict[str, Any], limiter: Optional[InMemoryRateLimiter] = None):
    """LangChain chat model for one resolved settings dict (see llm.settings)."""
    provider, model, effort = settings["provider"], settings["model"], settings["reasoning_effort"]
    api_key = settings["api_key"]
    if not api_key:
        raise RuntimeError(
            f"No API key for provider {provider}. Set {' or '.join(API_KEY_VARS[provider])} in .env."
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


@lru_cache(maxsize=1)
def _shared_limiter() -> Optional[InMemoryRateLimiter]:
    return _rate_limiter(_resolved_settings()["requests_per_minute"])


@lru_cache(maxsize=1)
def get_chat_model():
    """Assistant chat model for the configured provider (cached)."""
    return build_chat_model(_resolved_settings(), _shared_limiter())


@lru_cache(maxsize=1)
def get_judge_model():
    """LLM-as-Judge model (cached). The same instance as the assistant model unless
    JUDGE_PROVIDER / JUDGE_MODEL select another one; the same provider shares one rate limiter."""
    main, judge = _resolved_settings(), _judge_settings()
    if (judge["provider"], judge["model"]) == (main["provider"], main["model"]):
        return get_chat_model()
    limiter = _shared_limiter() if judge["provider"] == main["provider"] else _rate_limiter(judge["requests_per_minute"])
    return build_chat_model(judge, limiter)


def clear_model_cache() -> None:
    """Forget cached models (after changing .env settings in tests)."""
    for cached in (get_chat_model, get_judge_model, _shared_limiter):
        cached.cache_clear()
