"""Small LLM abstraction for OpenAI-compatible and Anthropic providers."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable, List, Optional, Sequence, Union

try:
    from dotenv import load_dotenv
except ImportError:  # pragma: no cover - optional dependency
    def load_dotenv(*args, **kwargs):
        return False


PROJECT_ROOT = Path(__file__).resolve().parents[2]
if not load_dotenv(PROJECT_ROOT / ".env"):
    env_file = PROJECT_ROOT / ".env"
    if env_file.exists():
        for raw_line in env_file.read_text(encoding="utf-8").splitlines():
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = [part.strip() for part in line.split("=", 1)]
            os.environ.setdefault(key, value)


class LLMClient:
    """Minimal client wrapper that supports the project's configured LLM provider."""

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: Optional[str] = None,
        base_url: Optional[str] = None,
        temperature: float = 0.0,
        max_tokens: int = 1024,
    ) -> None:
        self.api_key = api_key or os.getenv("LLM_API_KEY") or os.getenv("ANTHROPIC_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("LLM_MODEL") or os.getenv("ANTHROPIC_MODEL") or "claude-sonnet-5-5"
        self.base_url = base_url or os.getenv("LLM_BASE_URL") or os.getenv("OPENROUTER_BASE_URL")
        self.temperature = float(temperature if temperature is not None else os.getenv("LLM_TEMPERATURE", "0"))
        self.max_tokens = int(max_tokens if max_tokens is not None else os.getenv("LLM_MAX_TOKENS", "1024"))

    def _provider(self) -> str:
        if self.base_url and "openrouter" in self.base_url.lower():
            return "openrouter"
        if self.api_key and self.model.lower().startswith("claude"):
            return "anthropic"
        if self.api_key and (self.base_url or os.getenv("OPENAI_API_KEY")):
            return "openai"
        if os.getenv("ANTHROPIC_API_KEY"):
            return "anthropic"
        return "openai" if self.api_key else "none"

    def generate(
        self,
        prompt: Union[str, Sequence[dict]],
        system_prompt: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        """Generate a response from the configured LLM provider."""
        if not self.api_key:
            raise ValueError(
                "No LLM API key configured. Set LLM_API_KEY, ANTHROPIC_API_KEY, or OPENAI_API_KEY."
            )

        provider = self._provider()
        messages: List[dict] = []
        if isinstance(prompt, str):
            content = prompt
            messages.append({"role": "user", "content": content})
        else:
            messages = list(prompt)

        if provider == "anthropic":
            try:
                from anthropic import Anthropic
            except ImportError as exc:
                raise RuntimeError(
                    "Anthropic SDK is not installed. Install with: pip install anthropic"
                ) from exc
            client = Anthropic(api_key=self.api_key)
            request = {
                "model": self.model,
                "max_tokens": kwargs.get("max_tokens", self.max_tokens),
                "messages": messages,
            }
            if system_prompt:
                request["system"] = system_prompt
            response = client.messages.create(**request)
            return response.content[0].text

        if provider in {"openai", "openrouter"}:
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise RuntimeError(
                    "OpenAI SDK is not installed. Install with: pip install openai"
                ) from exc

            client = OpenAI(api_key=self.api_key, base_url=self.base_url)
            request_messages: List[dict] = []
            if system_prompt:
                request_messages.append({"role": "system", "content": system_prompt})
            request_messages.extend(messages)
            response = client.chat.completions.create(
                model=self.model,
                messages=request_messages,
                temperature=kwargs.get("temperature", self.temperature),
                max_tokens=kwargs.get("max_tokens", self.max_tokens),
            )
            return response.choices[0].message.content or ""

        raise RuntimeError(f"Unsupported LLM provider: {provider!r}")


__all__ = ["LLMClient"]
