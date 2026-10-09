"""
Chat model used by the agents. Reads the same .env settings as llm.client.LLMClient
(LLM_API_KEY / ANTHROPIC_API_KEY, LLM_MODEL / ANTHROPIC_MODEL).
"""
from functools import lru_cache

from langchain_anthropic import ChatAnthropic

from llm.client import LLMClient

AGENT_MAX_TOKENS = 4000


@lru_cache(maxsize=1)
def get_chat_model() -> ChatAnthropic:
    client = LLMClient()
    if not client.api_key:
        raise RuntimeError("No LLM API key configured. Set LLM_API_KEY or ANTHROPIC_API_KEY in .env.")
    provider = client._provider()
    if provider != "anthropic":
        raise RuntimeError(
            f"The agents need tool calling via Anthropic; configured provider is {provider!r}. "
            "Set ANTHROPIC_API_KEY and a Claude model in .env."
        )
    # No temperature: current Claude models reject non-default sampling parameters.
    return ChatAnthropic(
        model=client.model,
        api_key=client.api_key,
        max_tokens=AGENT_MAX_TOKENS,
        timeout=120,
        max_retries=2,
    )
