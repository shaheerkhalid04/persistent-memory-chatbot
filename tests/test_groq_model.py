"""The Groq adapter after the model retirement. Groq itself is faked."""

import sys
import types
from pathlib import Path
from types import SimpleNamespace

import config
import llm_connector


def _settings(model: str) -> config.Settings:
    return config.Settings(
        provider="groq", api_key="test", model=model, db_path=Path("unused.db"),
        memory_backend="sqlite", mem0_api_key=None, user_id="u", multi_user=False,
        max_memories_in_context=5,
    )


def _fake_groq(monkeypatch, calls):
    module = types.ModuleType("groq")

    class Groq:
        def __init__(self, api_key=None):
            def create(**kwargs):
                calls.append(kwargs)
                return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content="{}"))])

            self.chat = SimpleNamespace(completions=SimpleNamespace(create=create))

    module.Groq = Groq
    monkeypatch.setitem(sys.modules, "groq", module)


def test_default_groq_model_is_not_retired():
    assert config.DEFAULT_MODELS["groq"] != "llama-3.3-70b-versatile"


def test_gpt_oss_gets_low_reasoning(monkeypatch):
    calls = []
    _fake_groq(monkeypatch, calls)
    llm_connector.GroqProvider(_settings("openai/gpt-oss-120b")).json("sys", "user")
    assert calls[0]["extra_body"] == {"reasoning_effort": "low"}


def test_other_models_get_no_reasoning_param(monkeypatch):
    calls = []
    _fake_groq(monkeypatch, calls)
    llm_connector.GroqProvider(_settings("some-other-model")).json("sys", "user")
    assert "extra_body" not in calls[0]
