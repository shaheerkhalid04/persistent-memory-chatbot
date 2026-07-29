"""Runtime configuration.

Resolution order for every setting: real environment variable, then ``.env``,
then Streamlit secrets (so the same code runs locally and on Streamlit Cloud).
Nothing here reads a value at import time — settings are resolved on first use
and cached, which keeps the module importable in tests with no environment.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

try:  # optional: absent in a bare test environment
    from dotenv import load_dotenv

    load_dotenv(Path(__file__).with_name(".env"), override=False)
except ImportError:  # pragma: no cover
    pass


PROJECT_ROOT = Path(__file__).parent
DEFAULT_DB_PATH = PROJECT_ROOT / "data" / "recall.db"


def _secret(name: str) -> str | None:
    """Read one setting from the environment, falling back to Streamlit secrets.

    Args:
        name: Setting name, e.g. ``"GROQ_API_KEY"``.

    Returns:
        The value, or ``None`` when unset everywhere.
    """
    value = os.environ.get(name)
    if value:
        return value.strip()

    # st.secrets raises if no secrets file exists, which is the normal local case.
    try:
        import streamlit as st

        if name in st.secrets:
            return str(st.secrets[name]).strip()
    except Exception:
        pass
    return None


@dataclass(frozen=True)
class Settings:
    """Everything the app needs to know about its environment.

    Attributes:
        provider: Which LLM backend is active — ``groq``, ``gemini``, ``openai``
            or ``demo`` when no key is configured anywhere.
        api_key: Key for the active provider; empty in demo mode.
        model: Chat model id for the active provider.
        db_path: Where the SQLite file lives.
        memory_backend: ``sqlite`` (default) or ``mem0``.
        mem0_api_key: Key for the hosted Mem0 platform, when that backend is on.
        user_id: Namespaces memories, so one database can serve several people.
        max_memories_in_context: Cap on facts injected into the system prompt.
    """

    provider: str
    api_key: str
    model: str
    db_path: Path
    memory_backend: str
    mem0_api_key: str | None
    user_id: str
    max_memories_in_context: int

    @property
    def is_demo(self) -> bool:
        """True when no LLM key is configured and canned replies are used."""
        return self.provider == "demo"

    @property
    def label(self) -> str:
        """Short human-readable status for the sidebar footer."""
        if self.is_demo:
            return "demo mode · no API key"
        return f"{self.provider} · {self.model}"


#: Default chat model per provider. Override with ``<PROVIDER>_MODEL``.
DEFAULT_MODELS: dict[str, str] = {
    "groq": "llama-3.3-70b-versatile",
    "gemini": "gemini-2.0-flash",
    "openai": "gpt-4o-mini",
}


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Resolve settings once per process.

    Provider detection is by key presence, in preference order Groq → Gemini →
    OpenAI, unless ``LLM_PROVIDER`` pins one explicitly.

    Returns:
        The resolved settings. Never raises — a missing key yields demo mode so
        the UI still runs.
    """
    forced = (_secret("LLM_PROVIDER") or "").lower().strip()

    candidates = [
        ("groq", _secret("GROQ_API_KEY")),
        ("gemini", _secret("GEMINI_API_KEY") or _secret("GOOGLE_API_KEY")),
        ("openai", _secret("OPENAI_API_KEY")),
    ]
    if forced:
        candidates = [(name, key) for name, key in candidates if name == forced]

    provider, api_key = "demo", ""
    for name, key in candidates:
        if key:
            provider, api_key = name, key
            break

    model = (
        _secret(f"{provider.upper()}_MODEL")
        or DEFAULT_MODELS.get(provider, "")
    )

    db_path = Path(_secret("RECALL_DB_PATH") or DEFAULT_DB_PATH)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    mem0_key = _secret("MEM0_API_KEY")
    backend = (_secret("MEMORY_BACKEND") or ("mem0" if mem0_key else "sqlite")).lower()

    return Settings(
        provider=provider,
        api_key=api_key or "",
        model=model,
        db_path=db_path,
        memory_backend=backend,
        mem0_api_key=mem0_key,
        user_id=_secret("USER_ID") or "local",
        max_memories_in_context=int(_secret("MAX_MEMORIES_IN_CONTEXT") or 40),
    )


def reset_settings_cache() -> None:
    """Drop the cached settings. Used by tests that patch the environment."""
    get_settings.cache_clear()
