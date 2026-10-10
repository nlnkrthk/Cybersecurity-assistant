"""
Provider selection for the agents' chat model (agents.llm). No network calls.
"""
import pytest

from agents import llm as llm_module

ENV_VARS = (
    "LLM_PROVIDER", "LLM_MODEL", "LLM_REASONING_EFFORT", "LLM_REQUESTS_PER_MINUTE", "LLM_API_KEY",
    "GOOGLE_API_KEY", "GEMINI_API_KEY", "GOOGLE_BASE_URL", "NVIDIA_API_KEY", "NVIDIA_BASE_URL",
    "ANTHROPIC_API_KEY", "OPENAI_API_KEY", "LLM_BASE_URL", "OPENROUTER_BASE_URL", "ANTHROPIC_MODEL",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    llm_module.get_chat_model.cache_clear()
    yield monkeypatch
    llm_module.get_chat_model.cache_clear()


@pytest.mark.parametrize("provider, key_var, cls, model", [
    ("google", "GOOGLE_API_KEY", "ChatGoogleGenerativeAI", "gemini-3.5-flash"),
    ("nvidia", "NVIDIA_API_KEY", "ChatOpenAI", "openai/gpt-oss-20b"),
    ("anthropic", "ANTHROPIC_API_KEY", "ChatAnthropic", "claude-sonnet-5-5"),
])
def test_provider_builds_matching_client_with_default_model(clean_env, provider, key_var, cls, model):
    clean_env.setenv("LLM_PROVIDER", provider)
    clean_env.setenv(key_var, "test-key")
    chat = llm_module.get_chat_model()
    assert type(chat).__name__ == cls
    assert model in str(getattr(chat, "model", None) or getattr(chat, "model_name", ""))


def test_model_override_and_settings_hide_key(clean_env):
    clean_env.setenv("LLM_PROVIDER", "google")
    clean_env.setenv("GOOGLE_API_KEY", "secret-value")
    clean_env.setenv("LLM_MODEL", "gemini-3.8-flash")
    settings = llm_module.llm_settings()
    assert settings["model"] == "gemini-3.8-flash" and settings["api_key_set"]
    assert "secret-value" not in str(settings)


def test_provider_inferred_from_available_key(clean_env):
    clean_env.setenv("GOOGLE_API_KEY", "k")
    assert llm_module.llm_settings()["provider"] == "google"
    clean_env.setenv("ANTHROPIC_API_KEY", "k")
    assert llm_module.llm_settings()["provider"] == "anthropic"


def test_missing_key_and_unknown_provider_raise(clean_env):
    clean_env.setenv("LLM_PROVIDER", "nvidia")
    with pytest.raises(RuntimeError, match="NVIDIA_API_KEY"):
        llm_module.get_chat_model()
    clean_env.setenv("LLM_PROVIDER", "openrouter")
    with pytest.raises(RuntimeError, match="Unknown LLM_PROVIDER"):
        llm_module.llm_settings()


def test_rate_limiter_from_env(clean_env):
    clean_env.setenv("LLM_PROVIDER", "google")
    clean_env.setenv("GOOGLE_API_KEY", "k")
    clean_env.setenv("LLM_REQUESTS_PER_MINUTE", "5")
    chat = llm_module.get_chat_model()
    assert chat.rate_limiter is not None
    assert chat.rate_limiter.requests_per_second == pytest.approx(5 / 60)

    clean_env.setenv("LLM_REQUESTS_PER_MINUTE", "fast")
    with pytest.raises(RuntimeError, match="must be a number"):
        llm_module.llm_settings()


def test_no_rate_limiter_by_default(clean_env):
    clean_env.setenv("LLM_PROVIDER", "anthropic")
    clean_env.setenv("ANTHROPIC_API_KEY", "k")
    assert llm_module.get_chat_model().rate_limiter is None


# ---------------------------------------------------------------- shared settings in LLMClient (RAG, extraction)
@pytest.mark.parametrize("provider, key_var, expected_provider, base_url_part", [
    ("nvidia", "NVIDIA_API_KEY", "openai", "integrate.api.nvidia.com"),
    ("google", "GOOGLE_API_KEY", "openai", "generativelanguage.googleapis.com"),
    ("anthropic", "ANTHROPIC_API_KEY", "anthropic", None),
])
def test_llm_client_follows_llm_provider(clean_env, provider, key_var, expected_provider, base_url_part):
    from llm.client import LLMClient

    clean_env.setenv("LLM_PROVIDER", provider)
    clean_env.setenv(key_var, "test-key")
    clean_env.setenv("ANTHROPIC_API_KEY" if provider != "anthropic" else "NVIDIA_API_KEY", "other-key")
    client = LLMClient()
    assert client.api_key == "test-key"
    assert client._provider() == expected_provider
    assert (client.base_url is None) if base_url_part is None else (base_url_part in client.base_url)


def test_llm_client_legacy_behaviour_without_llm_provider(clean_env):
    from llm.client import LLMClient

    clean_env.setenv("ANTHROPIC_API_KEY", "k")
    client = LLMClient()
    assert client._provider() == "anthropic" and client.model == "claude-sonnet-5-5"


def test_llm_client_missing_key_names_provider_variable(clean_env):
    from llm.client import LLMClient

    clean_env.setenv("LLM_PROVIDER", "nvidia")
    with pytest.raises(ValueError, match="NVIDIA_API_KEY"):
        LLMClient().generate("hi")
