"""MemoryHandler — stores and retrieves persistent facts about the user.

The memory strategy lives here, in three parts:

**What to store.** After every user turn, :meth:`MemoryHandler.observe` asks the
model to pull out durable facts — age, name, location, work, preferences,
interests, goals — and to ignore questions, small talk and anything about other
people. Most turns yield nothing, which is correct.

**How updates work.** The same call sees the facts already stored and may return
``UPDATE`` carrying an existing id instead of ``ADD``. The handler then rewrites
that row in place, so "I'm 26 now" replaces "Is 25 years old" rather than piling
up beside it.

**How recall works.** :meth:`MemoryHandler.recall` scores stored facts against
the incoming question and returns the best ones, which the connector renders
into the system prompt. Below the cap it simply returns everything — with a few
dozen facts that is both cheaper and more accurate than any ranking.

Two backends implement the same small protocol: SQLite on disk (the default),
and the hosted Mem0 platform.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, replace
from datetime import datetime
from typing import Iterable, Literal, Protocol

from config import Settings, get_settings
from llm_connector import EXTRACTION_SYSTEM, LLMConnector
from models import CATEGORIES, LLMError, Memory, StorageError
from storage import SQLiteStore

#: Words too common to signal relevance when matching a question to a fact.
STOPWORDS = frozenset(
    """a an and are as at be but by can could did do does for from had has have
    how i if in is it its me my of on or should so than that the their them then
    there these they this to us was we were what when where which who why will
    with would you your""".split()
)


@dataclass(frozen=True)
class MemoryEvent:
    """One change the handler made to the store, for the UI to report."""

    kind: Literal["added", "updated"]
    memory: Memory

    @property
    def toast(self) -> str:
        verb = "Remembered" if self.kind == "added" else "Updated"
        return f"{verb}: {self.memory.text}"


class MemoryStore(Protocol):
    """The persistence contract. Both backends satisfy exactly this."""

    def load(self) -> list[Memory]: ...

    def upsert(self, memory: Memory) -> None: ...

    def delete(self, memory_id: str) -> None: ...

    def clear(self) -> None: ...


class SQLiteMemoryStore:
    """Default backend: facts written to a SQLite file on disk.

    Args:
        store: The shared :class:`~storage.SQLiteStore`.
    """

    def __init__(self, store: SQLiteStore) -> None:
        self._store = store

    def load(self) -> list[Memory]:
        return self._store.load_memories()

    def upsert(self, memory: Memory) -> None:
        self._store.upsert_memory(memory)

    def delete(self, memory_id: str) -> None:
        self._store.delete_memory(memory_id)

    def clear(self) -> None:
        self._store.clear_memories()


class Mem0MemoryStore:
    """Adapter for the hosted Mem0 platform.

    Selected with ``MEMORY_BACKEND=mem0`` plus a ``MEM0_API_KEY``. Mem0 runs its
    own extraction and deduplication server-side, so :meth:`MemoryHandler.observe`
    hands it the raw turn and reads back the resulting facts rather than doing
    its own ADD/UPDATE reasoning.

    Args:
        settings: Resolved configuration carrying the key and user id.

    Raises:
        StorageError: If the SDK is missing or the key is absent.
    """

    def __init__(self, settings: Settings) -> None:
        try:
            from mem0 import MemoryClient
        except ImportError as exc:
            raise StorageError("mem0ai is not installed — run: pip install mem0ai") from exc
        if not settings.mem0_api_key:
            raise StorageError("MEMORY_BACKEND=mem0 needs a MEM0_API_KEY")

        self._client = MemoryClient(api_key=settings.mem0_api_key)
        self._user_id = settings.user_id

    @staticmethod
    def _to_memory(record: dict) -> Memory:
        """Map one Mem0 record onto the local dataclass."""

        def _parse(value: str | None) -> datetime:
            if not value:
                return datetime.now()
            try:
                return datetime.fromisoformat(value.replace("Z", "+00:00")).replace(tzinfo=None)
            except ValueError:
                return datetime.now()

        metadata = record.get("metadata") or {}
        category = metadata.get("category") or "Identity"
        return Memory(
            id=str(record.get("id")),
            text=record.get("memory") or record.get("text") or "",
            category=category if category in CATEGORIES else "Identity",
            created_at=_parse(record.get("created_at")),
            updated_at=_parse(record.get("updated_at") or record.get("created_at")),
            source_conversation_id=metadata.get("source_conversation_id"),
        )

    def load(self) -> list[Memory]:
        try:
            records = self._client.get_all(user_id=self._user_id)
        except Exception as exc:
            raise StorageError(f"mem0 read failed: {exc}") from exc
        if isinstance(records, dict):
            records = records.get("results", [])
        return [self._to_memory(r) for r in records]

    def upsert(self, memory: Memory) -> None:
        try:
            self._client.add(
                [{"role": "user", "content": memory.text}],
                user_id=self._user_id,
                metadata={
                    "category": memory.category,
                    "source_conversation_id": memory.source_conversation_id,
                },
            )
        except Exception as exc:
            raise StorageError(f"mem0 write failed: {exc}") from exc

    def delete(self, memory_id: str) -> None:
        try:
            self._client.delete(memory_id=memory_id)
        except Exception as exc:
            raise StorageError(f"mem0 delete failed: {exc}") from exc

    def clear(self) -> None:
        try:
            self._client.delete_all(user_id=self._user_id)
        except Exception as exc:
            raise StorageError(f"mem0 wipe failed: {exc}") from exc


class MemoryHandler:
    """Extraction, update decisions, recall and persistence of user facts.

    Args:
        store: Where facts live.
        llm: Connector used for extraction. Optional — without it, ``observe``
            is a no-op and the rest of the handler still works.
        settings: Resolved configuration.
    """

    def __init__(
        self,
        store: MemoryStore,
        llm: LLMConnector | None = None,
        settings: Settings | None = None,
    ) -> None:
        self.store = store
        self.llm = llm
        self.settings = settings or get_settings()

    # ------------------------------------------------------------------ reads

    def all(self) -> list[Memory]:
        """Every stored fact, most recently revised first."""
        return self.store.load()

    def recall(self, query: str, memories: list[Memory] | None = None) -> list[Memory]:
        """Pick the facts worth injecting for this question.

        Below the cap every fact is returned, which is the normal case for a
        personal assistant and strictly better than any ranking.

        Above the cap, facts are scored on keyword overlap plus recency and then
        drawn round-robin **across categories** rather than taken off one global
        ranking. That matters: lexical overlap cannot connect "what sport should
        I try" to "Plays football on weekends", so a global cut would silently
        drop the one fact the question was about. Round-robin guarantees every
        category keeps representation, and the scoring only decides the order
        within each.

        Semantic matching would need embeddings; that is the upgrade path if a
        store ever grows past a few hundred facts.

        Args:
            query: The user's message.
            memories: Facts to rank. Loaded from the store when omitted.

        Returns:
            At most ``max_memories_in_context`` facts.
        """
        pool = self.all() if memories is None else list(memories)
        cap = self.settings.max_memories_in_context
        if len(pool) <= cap:
            return pool

        terms = self._terms(query)
        now = datetime.now()

        def score(memory: Memory) -> float:
            overlap = len(terms & self._terms(memory.text))
            age_days = max(0.0, (now - memory.updated_at).total_seconds() / 86400)
            return overlap + 1.0 / (1.0 + age_days / 30.0)

        buckets: dict[str, list[Memory]] = {}
        for memory in sorted(pool, key=score, reverse=True):
            buckets.setdefault(memory.category, []).append(memory)

        # Categories holding a strong match go first; ties fall back to size.
        order = sorted(
            buckets,
            key=lambda c: (max(score(m) for m in buckets[c]), len(buckets[c])),
            reverse=True,
        )

        picked: list[Memory] = []
        depth = 0
        while len(picked) < cap and any(len(buckets[c]) > depth for c in order):
            for category in order:
                if depth < len(buckets[category]):
                    picked.append(buckets[category][depth])
                    if len(picked) == cap:
                        break
            depth += 1
        return picked

    @staticmethod
    def _terms(text: str) -> set[str]:
        """Lowercased content words, for cheap keyword overlap."""
        words = re.findall(r"[a-z0-9']+", text.lower())
        return {w for w in words if len(w) > 2 and w not in STOPWORDS}

    # ----------------------------------------------------------------- writes

    def observe(self, user_text: str, conversation_id: str | None = None) -> list[MemoryEvent]:
        """Decide what to remember from a user turn, and persist it.

        Args:
            user_text: The raw user message.
            conversation_id: Attributed as the source of any new fact.

        Returns:
            The changes applied, for the UI to surface as toasts. Empty on most
            turns, and always empty when no connector is configured.

        Raises:
            StorageError: If a write fails.
        """
        if self.llm is None or not user_text.strip():
            return []

        existing = self.all()

        if isinstance(self.store, Mem0MemoryStore):
            return self._observe_via_mem0(user_text, existing)

        try:
            payload = self._extraction_payload(user_text, existing)
            response = self.llm.complete_json(
                EXTRACTION_SYSTEM.format(categories=", ".join(CATEGORIES)),
                payload,
            )
        except LLMError:
            # Extraction is best-effort: never lose the user's reply over it.
            return []

        return self._apply(response.get("operations", []), existing, conversation_id)

    def _observe_via_mem0(self, user_text: str, before: list[Memory]) -> list[MemoryEvent]:
        """Let Mem0 do its own extraction, then diff to report what changed."""
        self.store.upsert(Memory.new(user_text, "Identity"))
        known = {m.id for m in before}
        return [
            MemoryEvent("added", memory)
            for memory in self.all()
            if memory.id not in known
        ]

    @staticmethod
    def _extraction_payload(user_text: str, existing: list[Memory]) -> str:
        """Render stored facts plus the new turn for the extraction call."""
        if existing:
            stored = json.dumps(
                [{"id": m.id, "text": m.text, "category": m.category} for m in existing],
                indent=None,
            )
        else:
            stored = "[]"
        return f"Stored facts:\n{stored}\n\nUser's latest message:\n{user_text.strip()}"

    def _apply(
        self,
        operations: Iterable[dict],
        existing: list[Memory],
        conversation_id: str | None,
    ) -> list[MemoryEvent]:
        """Turn the model's decisions into store writes.

        Unknown ids on an UPDATE degrade to an ADD rather than being dropped,
        and anything malformed is skipped — a bad extraction must never break
        the conversation.
        """
        by_id = {m.id: m for m in existing}
        events: list[MemoryEvent] = []

        for operation in operations or []:
            if not isinstance(operation, dict):
                continue

            text = str(operation.get("text") or "").strip()
            if not text:
                continue

            category = str(operation.get("category") or "").strip()
            if category not in CATEGORIES:
                category = "Identity"

            op = str(operation.get("op") or "ADD").upper()
            target = by_id.get(str(operation.get("id") or ""))

            if op == "UPDATE" and target is not None:
                memory = replace(
                    target,
                    text=text,
                    category=category,
                    updated_at=datetime.now(),
                    source_conversation_id=conversation_id or target.source_conversation_id,
                )
                self.store.upsert(memory)
                events.append(MemoryEvent("updated", memory))
            elif op in {"ADD", "UPDATE"}:
                if any(m.text.lower() == text.lower() for m in existing):
                    continue  # already known verbatim
                memory = Memory.new(text, category, conversation_id)
                self.store.upsert(memory)
                events.append(MemoryEvent("added", memory))

        return events

    def upsert(self, memory: Memory) -> None:
        """Write one fact directly, bypassing extraction (manual panel edits)."""
        self.store.upsert(memory)

    def delete(self, memory_id: str) -> None:
        """Forget one fact."""
        self.store.delete(memory_id)

    def clear(self) -> None:
        """Forget everything."""
        self.store.clear()


def build_memory_store(settings: Settings, sqlite_store: SQLiteStore) -> MemoryStore:
    """Pick the memory backend named by configuration.

    Falls back to SQLite if Mem0 is requested but unavailable, so a missing
    optional dependency degrades instead of breaking the app.

    Args:
        settings: Resolved configuration.
        sqlite_store: The shared SQLite store, used for the default backend.

    Returns:
        A ready memory store.
    """
    if settings.memory_backend == "mem0":
        try:
            return Mem0MemoryStore(settings)
        except StorageError:
            return SQLiteMemoryStore(sqlite_store)
    return SQLiteMemoryStore(sqlite_store)
