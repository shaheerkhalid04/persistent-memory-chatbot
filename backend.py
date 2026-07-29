"""The seam the UI talks to.

``app.py`` imports only from here, so the UI never learns which model provider
or memory backend is active. Everything below is a thin delegation to
:mod:`storage`, :mod:`memory_handler` and :mod:`llm_connector`.

Wiring is built lazily and once per process — Streamlit imports this module a
single time and reruns the script against it, so a module-level singleton is
the right lifetime for the database handle and the provider client.
"""

from __future__ import annotations

import dataclasses
from typing import Iterator

from config import Settings, get_settings
from llm_connector import LLMConnector
from memory_handler import MemoryEvent, MemoryHandler, build_memory_store
from models import (
    CATEGORIES,
    Conversation,
    LLMError,
    Memory,
    Message,
    StorageError,
)
from storage import SQLiteStore

__all__ = [
    "CATEGORIES",
    "Conversation",
    "LLMError",
    "Memory",
    "MemoryEvent",
    "Message",
    "StorageError",
    "clear_memories",
    "delete_conversation",
    "delete_memory",
    "load_conversations",
    "load_memories",
    "observe_turn",
    "save_conversation",
    "status_label",
    "stream_reply",
    "suggest_title",
    "upsert_memory",
]

# Streamlit serves every browser session from one process, so the wiring is
# cached per user id rather than globally. With MULTI_USER off there is exactly
# one entry; with it on, each visitor gets an isolated store.
_stores: dict[str, SQLiteStore] = {}
_handlers: dict[str, MemoryHandler] = {}
_llm: LLMConnector | None = None


def _settings() -> Settings:
    return get_settings()


def _resolve(user_id: str | None) -> str:
    """Fall back to the configured default namespace."""
    return user_id or _settings().user_id


def store(user_id: str | None = None) -> SQLiteStore:
    """The SQLite store for one user, opened and migrated on first use."""
    key = _resolve(user_id)
    if key not in _stores:
        _stores[key] = SQLiteStore(_settings().db_path, key)
    return _stores[key]


def llm() -> LLMConnector:
    """The shared LLM connector. Stateless, so one instance serves everyone."""
    global _llm
    if _llm is None:
        _llm = LLMConnector(_settings())
    return _llm


def memory(user_id: str | None = None) -> MemoryHandler:
    """The memory handler for one user, bound to the configured backend."""
    key = _resolve(user_id)
    if key not in _handlers:
        settings = _settings()
        scoped = replace_settings(settings, user_id=key)
        _handlers[key] = MemoryHandler(
            store=build_memory_store(scoped, store(key)),
            llm=llm(),
            settings=scoped,
        )
    return _handlers[key]


def replace_settings(settings: Settings, **changes) -> Settings:
    """Copy settings with overrides. Settings is frozen, so this is the seam."""
    return dataclasses.replace(settings, **changes)


def reset() -> None:
    """Drop the cached wiring. Used by tests that change the environment."""
    global _llm
    _stores.clear()
    _handlers.clear()
    _llm = None


def status_label() -> str:
    """Short description of the active model and memory backend."""
    settings = _settings()
    backend = "mem0" if settings.memory_backend == "mem0" else "sqlite"
    return f"Memory · {backend}  ·  Model · {settings.label}"


# --------------------------------------------------------------------------- #
# Conversations
# --------------------------------------------------------------------------- #


def load_conversations(user_id: str | None = None) -> list[Conversation]:
    """Every stored conversation, newest activity first.

    Returns:
        Conversations read from disk. An empty list is valid and means a first
        run, not an error.

    Raises:
        StorageError: If the database cannot be read.
    """
    return store(user_id).load_conversations()


def save_conversation(conv: Conversation, user_id: str | None = None) -> None:
    """Insert or replace a conversation and its messages.

    Args:
        conv: The conversation to persist.

    Raises:
        StorageError: If the write fails.
    """
    store(user_id).save_conversation(conv)


def delete_conversation(conversation_id: str, user_id: str | None = None) -> None:
    """Remove a conversation and its messages.

    Args:
        conversation_id: Id to drop. Unknown ids are a no-op.

    Raises:
        StorageError: If the delete fails.
    """
    store(user_id).delete_conversation(conversation_id)


# --------------------------------------------------------------------------- #
# Memories
# --------------------------------------------------------------------------- #


def load_memories(user_id: str | None = None) -> list[Memory]:
    """Every stored fact about the user.

    Raises:
        StorageError: If the store cannot be read.
    """
    return memory(user_id).all()


def upsert_memory(m: Memory, user_id: str | None = None) -> None:
    """Insert a fact, or overwrite the one with the same id.

    Args:
        m: The fact to write. Reusing an existing id is what makes an update
            replace the old text instead of adding a second row.

    Raises:
        StorageError: If the write fails.
    """
    memory(user_id).upsert(m)


def delete_memory(memory_id: str, user_id: str | None = None) -> None:
    """Forget a single fact.

    Args:
        memory_id: Id to remove. Unknown ids are a no-op.

    Raises:
        StorageError: If the delete fails.
    """
    memory(user_id).delete(memory_id)


def clear_memories(user_id: str | None = None) -> None:
    """Forget everything.

    Raises:
        StorageError: If the wipe fails.
    """
    memory(user_id).clear()


def observe_turn(user_text: str, conversation_id: str | None = None,
                 user_id: str | None = None) -> list[MemoryEvent]:
    """Decide what to remember from a user turn, and persist it.

    Args:
        user_text: The raw user message.
        conversation_id: Attributed as the source of any new fact.

    Returns:
        Changes applied, for the UI to surface. Empty on most turns.

    Raises:
        StorageError: If a write fails.
    """
    return memory(user_id).observe(user_text, conversation_id)


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #


def stream_reply(messages: list[dict], memories: list[Memory],
                 user_id: str | None = None) -> Iterator[str]:
    """Stream the assistant's reply with the user's facts in context.

    Args:
        messages: Turn history as ``{"role", "content"}`` dicts, oldest first,
            ending with the user turn to answer.
        memories: Facts available for injection; ranked for relevance here.

    Yields:
        Text chunks. Concatenated they form the complete reply.

    Raises:
        LLMError: If the provider rejects the call.
    """
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    relevant = memory(user_id).recall(last_user, memories)
    yield from llm().stream_chat(messages, relevant)


def suggest_title(first_message: str) -> str:
    """Derive a conversation title from its opening message.

    Trimmed locally rather than asked of the model — a title is not worth an
    extra round trip on every new chat.

    Args:
        first_message: The user's opening turn.

    Returns:
        A title of at most 40 characters.
    """
    cleaned = " ".join(first_message.strip().split())
    if not cleaned:
        return "New chat"
    if len(cleaned) <= 40:
        return cleaned
    return cleaned[:37].rsplit(" ", 1)[0] + "..."
