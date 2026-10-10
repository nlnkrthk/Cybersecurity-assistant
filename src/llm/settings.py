"""
LLM provider settings shared by the whole project, read from .env:

    LLM_PROVIDER=nvidia      NVIDIA_API_KEY=nvapi-...  LLM_MODEL=openai/gpt-oss-20b
                             NVIDIA_BASE_URL=https://integrate.api.nvidia.com/v1 (optional)
    LLM_PROVIDER=google      GOOGLE_API_KEY=...        LLM_MODEL=gemini-3.5-flash
    LLM_PROVIDER=anthropic   ANTHROPIC_API_KEY=...     LLM_MODEL=claude-sonnet-5-5

    LLM_REASONING_EFFORT=low   thinking effort for reasoning models (default low)
    LLM_REQUESTS_PER_MINUTE=35 optional client-side rate limit for the agents

Used by the agents (agents.llm) and by llm.client.LLMClient (standalone RAG answers,
metric extraction), so one setting switches the provider everywhere. Without
LLM_PROVIDER the provider is inferred from which API key is set.
"""
import os
from pathlib import Path
from typing import Any, Dict, Optional

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False

load_dotenv(Path(__file__).resolve().parents[2] / ".env")

PROVIDERS = ("nvidia", "google", "anthropic")

DEFAULT_MODELS = {
    "nvidia": "openai/gpt-oss-20b",
    "google": "gemini-3.5-flash",
    "anthropic": "claude-sonnet-5-5",
}
API_KEY_VARS = {
    "nvidia": ("NVIDIA_API_KEY",),
    "google": ("GOOGLE_API_KEY", "GEMINI_API_KEY"),
    "anthropic": ("ANTHROPIC_API_KEY", "LLM_API_KEY"),
}
# OpenAI-compatible endpoints (Anthropic uses its own SDK).
DEFAULT_BASE_URLS = {
    "nvidia": "https://integrate.api.nvidia.com/v1",
    "google": "https://generativelanguage.googleapis.com/v1beta/openai/",
}
BASE_URL_VARS = {"nvidia": "NVIDIA_BASE_URL", "google": "GOOGLE_BASE_URL"}


def first_env(names) -> Optional[str]:
    for name in names:
        value = os.getenv(name)
        if value and value.strip():
            return value.strip()
    return None


def configured_provider() -> Optional[str]:
    """LLM_PROVIDER if set (validated), else None."""
    provider = (os.getenv("LLM_PROVIDER") or "").strip().lower()
    if not provider:
        return None
    if provider not in PROVIDERS:
        raise RuntimeError(f"Unknown LLM_PROVIDER {provider!r}; use one of {', '.join(PROVIDERS)}.")
    return provider


def requests_per_minute() -> Optional[float]:
    value = (os.getenv("LLM_REQUESTS_PER_MINUTE") or "").strip()
    if not value:
        return None
    try:
        rpm = float(value)
    except ValueError:
        raise RuntimeError(f"LLM_REQUESTS_PER_MINUTE must be a number, got {value!r}")
    return rpm if rpm > 0 else None


def llm_settings() -> Dict[str, Any]:
    """Resolved provider settings. Includes the key under "api_key"; use public_settings() to show them."""
    provider = configured_provider()
    if provider is None:
        provider = next((p for p in ("anthropic", "nvidia", "google") if first_env(API_KEY_VARS[p])), "nvidia")
    base_url = None
    if provider in DEFAULT_BASE_URLS:
        base_url = (os.getenv(BASE_URL_VARS[provider]) or DEFAULT_BASE_URLS[provider]).strip()
    return {
        "provider": provider,
        "model": (os.getenv("LLM_MODEL") or DEFAULT_MODELS[provider]).strip(),
        "api_key": first_env(API_KEY_VARS[provider]),
        "base_url": base_url,
        "reasoning_effort": (os.getenv("LLM_REASONING_EFFORT") or "low").strip().lower(),
        "requests_per_minute": requests_per_minute(),
    }


def public_settings() -> Dict[str, Any]:
    """Settings safe to log or return from an API (no key)."""
    settings = llm_settings()
    key = settings.pop("api_key")
    return {**settings, "api_key_set": bool(key)}
