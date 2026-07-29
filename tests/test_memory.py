"""Tests for the memory layer.

Covers the two behaviours the brief calls out: facts survive a restart, and new
information overrides old rather than accumulating beside it.

No network access. The LLM is replaced by a fake whose JSON responses are
scripted, so extraction logic is tested without a key or a live model.
"""

from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from config import Settings
from memory_handler import MemoryHandler, SQLiteMemoryStore
from models import Memory
from storage import SQLiteStore


def make_settings(db_path) -> Settings:
    return Settings(
        provider="demo",
        api_key="",
        model="",
        db_path=db_path,
        memory_backend="sqlite",
        mem0_api_key=None,
        user_id="tester",
        multi_user=False,
        max_memories_in_context=40,
    )


class FakeLLM:
    """Returns scripted extraction decisions, one per call."""

    def __init__(self, *responses: dict) -> None:
        self._responses = list(responses)
        self.calls: list[str] = []

    def complete_json(self, system: str, user: str) -> dict:
        self.calls.append(user)
        return self._responses.pop(0) if self._responses else {"operations": []}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "test.db"


@pytest.fixture
def handler(db_path):
    settings = make_settings(db_path)
    store = SQLiteStore(db_path, settings.user_id)
    return MemoryHandler(SQLiteMemoryStore(store), llm=None, settings=settings)


# --------------------------------------------------------------------------- #
# Persistence
# --------------------------------------------------------------------------- #


def test_memories_survive_a_restart(db_path):
    """A fact written by one process is readable by the next."""
    settings = make_settings(db_path)

    first = SQLiteStore(db_path, settings.user_id)
    first.upsert_memory(Memory.new("Is 25 years old", "Identity", "conv-1"))

    # A fresh store object on the same file stands in for an app restart.
    second = SQLiteStore(db_path, settings.user_id)
    memories = second.load_memories()

    assert [m.text for m in memories] == ["Is 25 years old"]
    assert memories[0].source_conversation_id == "conv-1"


def test_conversations_survive_a_restart(db_path):
    """Threads and their messages round-trip through disk."""
    from models import Conversation, Message

    store = SQLiteStore(db_path, "tester")
    conv = Conversation.new("Weekend plans")
    conv.messages = [Message("user", "hi"), Message("assistant", "hello")]
    store.save_conversation(conv)

    reloaded = SQLiteStore(db_path, "tester").load_conversations()
    assert len(reloaded) == 1
    assert reloaded[0].title == "Weekend plans"
    assert [m.role for m in reloaded[0].messages] == ["user", "assistant"]
    assert [m.content for m in reloaded[0].messages] == ["hi", "hello"]


def test_conversations_are_namespaced_by_user(db_path):
    """A public deployment must not leak one visitor's threads to another."""
    from models import Conversation, Message

    alice = SQLiteStore(db_path, "alice")
    conv = Conversation.new("Alice's private thread")
    conv.messages = [Message("user", "something personal")]
    alice.save_conversation(conv)

    assert [c.title for c in alice.load_conversations()] == ["Alice's private thread"]
    assert SQLiteStore(db_path, "bob").load_conversations() == []

    # Bob cannot delete what he cannot see.
    SQLiteStore(db_path, "bob").delete_conversation(conv.id)
    assert len(alice.load_conversations()) == 1


def test_legacy_database_without_user_column_is_migrated(db_path):
    """A database written before namespacing must stay readable."""
    import sqlite3

    conn = sqlite3.connect(db_path)
    conn.executescript(
        """CREATE TABLE conversations (
               id TEXT PRIMARY KEY, title TEXT NOT NULL,
               created_at TEXT NOT NULL, updated_at TEXT NOT NULL);
           INSERT INTO conversations VALUES
               ('old1', 'Legacy thread', '2026-01-01T00:00:00', '2026-01-01T00:00:00');"""
    )
    conn.commit()
    conn.close()

    store = SQLiteStore(db_path, "local")
    assert [c.title for c in store.load_conversations()] == ["Legacy thread"]


def test_memories_are_namespaced_by_user(db_path):
    """One database file can hold separate profiles."""
    SQLiteStore(db_path, "alice").upsert_memory(Memory.new("Likes tea", "Preferences"))
    SQLiteStore(db_path, "bob").upsert_memory(Memory.new("Likes coffee", "Preferences"))

    assert [m.text for m in SQLiteStore(db_path, "alice").load_memories()] == ["Likes tea"]
    assert [m.text for m in SQLiteStore(db_path, "bob").load_memories()] == ["Likes coffee"]


# --------------------------------------------------------------------------- #
# Update vs. accumulate — the "I am 26 now" case
# --------------------------------------------------------------------------- #


def test_update_replaces_the_old_fact(db_path):
    """'I'm 26 now' overwrites the stored age instead of adding a second row."""
    settings = make_settings(db_path)
    store = SQLiteStore(db_path, settings.user_id)
    memory_store = SQLiteMemoryStore(store)

    original = Memory.new("Is 25 years old", "Identity", "conv-1")
    # Backdate so the updated_at bump is unambiguous.
    original.created_at = original.updated_at = datetime.now() - timedelta(days=2)
    memory_store.upsert(original)

    llm = FakeLLM(
        {"operations": [{"op": "UPDATE", "id": original.id,
                         "text": "Is 26 years old", "category": "Identity"}]}
    )
    handler = MemoryHandler(memory_store, llm=llm, settings=settings)
    events = handler.observe("I'm 26 now", "conv-2")

    stored = SQLiteStore(db_path, settings.user_id).load_memories()
    assert len(stored) == 1, "update must not create a second row"
    assert stored[0].id == original.id
    assert stored[0].text == "Is 26 years old"
    assert stored[0].created_at == original.created_at, "created_at is preserved"
    assert stored[0].updated_at > stored[0].created_at, "updated_at is bumped"
    assert stored[0].was_revised

    assert [(e.kind, e.memory.text) for e in events] == [("updated", "Is 26 years old")]
    assert events[0].toast == "Updated: Is 26 years old"


def test_add_creates_a_new_fact(db_path):
    settings = make_settings(db_path)
    store = SQLiteMemoryStore(SQLiteStore(db_path, settings.user_id))
    llm = FakeLLM({"operations": [{"op": "ADD", "text": "Plays football on weekends",
                                   "category": "Interests"}]})

    events = MemoryHandler(store, llm=llm, settings=settings).observe("I like football", "c1")

    assert [(e.kind, e.memory.text) for e in events] == [("added", "Plays football on weekends")]
    assert store.load()[0].category == "Interests"
    assert store.load()[0].source_conversation_id == "c1"


def test_verbatim_duplicates_are_skipped(db_path):
    """The same fact stated twice must not produce two rows."""
    settings = make_settings(db_path)
    store = SQLiteMemoryStore(SQLiteStore(db_path, settings.user_id))
    store.upsert(Memory.new("Lives in Lahore", "Location"))

    llm = FakeLLM({"operations": [{"op": "ADD", "text": "lives in lahore",
                                   "category": "Location"}]})
    events = MemoryHandler(store, llm=llm, settings=settings).observe("I live in Lahore", "c1")

    assert events == []
    assert len(store.load()) == 1


def test_extraction_payload_includes_existing_ids(db_path):
    """The model must see current ids, or it can never choose UPDATE."""
    settings = make_settings(db_path)
    store = SQLiteMemoryStore(SQLiteStore(db_path, settings.user_id))
    existing = Memory.new("Is 25 years old", "Identity")
    store.upsert(existing)

    llm = FakeLLM({"operations": []})
    MemoryHandler(store, llm=llm, settings=settings).observe("I'm 26 now", "c1")

    payload = llm.calls[0]
    assert existing.id in payload
    assert "Is 25 years old" in payload
    assert "I'm 26 now" in payload


# --------------------------------------------------------------------------- #
# Robustness — a bad extraction must never break the conversation
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "response",
    [
        {},
        {"operations": None},
        {"operations": [{"op": "ADD"}]},                       # no text
        {"operations": [{"op": "ADD", "text": "   "}]},        # blank text
        {"operations": ["not a dict"]},
        {"operations": [{"op": "NONSENSE", "text": "x", "category": "Identity"}]},
    ],
)
def test_malformed_extraction_is_survivable(db_path, response):
    settings = make_settings(db_path)
    store = SQLiteMemoryStore(SQLiteStore(db_path, settings.user_id))
    handler = MemoryHandler(store, llm=FakeLLM(response), settings=settings)

    assert handler.observe("something", "c1") == []
    assert store.load() == []


def test_unknown_category_falls_back(db_path):
    settings = make_settings(db_path)
    store = SQLiteMemoryStore(SQLiteStore(db_path, settings.user_id))
    llm = FakeLLM({"operations": [{"op": "ADD", "text": "Owns a cat", "category": "Pets"}]})

    MemoryHandler(store, llm=llm, settings=settings).observe("I have a cat", "c1")
    assert store.load()[0].category == "Identity"


def test_update_with_unknown_id_degrades_to_add(db_path):
    settings = make_settings(db_path)
    store = SQLiteMemoryStore(SQLiteStore(db_path, settings.user_id))
    llm = FakeLLM({"operations": [{"op": "UPDATE", "id": "does-not-exist",
                                   "text": "Is 30", "category": "Identity"}]})

    events = MemoryHandler(store, llm=llm, settings=settings).observe("x", "c1")
    assert [e.kind for e in events] == ["added"]
    assert len(store.load()) == 1


def test_observe_without_llm_is_a_noop(handler):
    assert handler.observe("I am 26 years old", "c1") == []


# --------------------------------------------------------------------------- #
# Recall
# --------------------------------------------------------------------------- #


def test_recall_returns_everything_below_the_cap(db_path):
    settings = make_settings(db_path)
    store = SQLiteMemoryStore(SQLiteStore(db_path, settings.user_id))
    for i in range(5):
        store.upsert(Memory.new(f"Fact number {i}", "Identity"))

    handler = MemoryHandler(store, llm=None, settings=settings)
    assert len(handler.recall("anything at all")) == 5


def test_recall_ranks_by_overlap_when_over_the_cap(db_path):
    settings = make_settings(db_path)
    settings = Settings(**{**settings.__dict__, "max_memories_in_context": 3})
    store = SQLiteMemoryStore(SQLiteStore(db_path, settings.user_id))

    store.upsert(Memory.new("Plays football on weekends", "Interests"))
    for i in range(6):
        store.upsert(Memory.new(f"Unrelated trivia {i}", "Identity"))

    handler = MemoryHandler(store, llm=None, settings=settings)
    top = handler.recall("what sport should I try")

    assert len(top) == 3
    assert any("football" in m.text.lower() for m in top), "the relevant fact must survive"


# --------------------------------------------------------------------------- #
# JSON parsing
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "raw,expected",
    [
        ('{"operations": []}', {"operations": []}),
        ('```json\n{"operations": []}\n```', {"operations": []}),
        ('Sure! Here you go: {"operations": [{"op": "ADD"}]}', {"operations": [{"op": "ADD"}]}),
        ("not json at all", {}),
        ("", {}),
        ("[1, 2, 3]", {}),
    ],
)
def test_json_recovery(raw, expected):
    from llm_connector import parse_json_object

    assert parse_json_object(raw) == expected
