"""SQLite persistence for conversations and memories.

This is the layer that makes the assistant survive a restart. Everything is
written to a single file on disk (``data/recall.db`` by default); nothing is
held only in process memory.

A fresh connection is opened per operation rather than shared. Streamlit runs
each rerun on its own thread, and a module-level connection would trip
sqlite3's thread check; per-call connections cost microseconds at this scale
and remove the problem entirely.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime
from pathlib import Path
from typing import Iterator

from models import Conversation, Memory, Message, StorageError

SCHEMA = """
CREATE TABLE IF NOT EXISTS conversations (
    id          TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL DEFAULT 'local',
    title       TEXT NOT NULL,
    created_at  TEXT NOT NULL,
    updated_at  TEXT NOT NULL
);

CREATE TABLE IF NOT EXISTS messages (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    position        INTEGER NOT NULL,
    role            TEXT NOT NULL,
    content         TEXT NOT NULL,
    created_at      TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id, position);

CREATE TABLE IF NOT EXISTS memories (
    id                     TEXT PRIMARY KEY,
    user_id                TEXT NOT NULL,
    text                   TEXT NOT NULL,
    category               TEXT NOT NULL,
    created_at             TEXT NOT NULL,
    updated_at             TEXT NOT NULL,
    source_conversation_id TEXT
);

CREATE INDEX IF NOT EXISTS idx_memories_user ON memories(user_id, category);
"""

#: Indexes over columns that older databases may not have yet. Created only
#: after the migration below has had a chance to add those columns — an index
#: naming a missing column aborts the whole script and bricks the open.
INDEXES = """
CREATE INDEX IF NOT EXISTS idx_conversations_user ON conversations(user_id, updated_at);
"""


def _iso(value: datetime) -> str:
    # Full precision on purpose: created_at and updated_at are compared to tell
    # a revised fact from a new one, and second-granularity would make a fact
    # corrected moments after it was learned look like it had never changed.
    return value.isoformat()


def _dt(value: str) -> datetime:
    return datetime.fromisoformat(value)


class SQLiteStore:
    """Durable store for conversations and memories.

    Args:
        db_path: Location of the SQLite file. Parent directories are created.
        user_id: Namespace for memories, so one file can hold several profiles.
    """

    def __init__(self, db_path: Path | str, user_id: str = "local") -> None:
        self.db_path = Path(db_path)
        self.user_id = user_id
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._migrate()

    # ---------------------------------------------------------------- plumbing

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        """Yield a connection, committing on success and always closing."""
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
        except sqlite3.Error as exc:
            raise StorageError(f"cannot open {self.db_path.name}: {exc}") from exc

        conn.row_factory = sqlite3.Row
        try:
            conn.execute("PRAGMA foreign_keys = ON")
            yield conn
            conn.commit()
        except sqlite3.Error as exc:
            conn.rollback()
            raise StorageError(str(exc)) from exc
        finally:
            conn.close()

    def _migrate(self) -> None:
        """Create tables if they do not exist. Safe to call on every start."""
        with self._connect() as conn:
            # WAL keeps reads from blocking on the write of a streamed turn.
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(SCHEMA)

            # Databases created before conversations were namespaced predate the
            # user_id column; add it in place rather than making them unreadable.
            columns = {row["name"] for row in conn.execute("PRAGMA table_info(conversations)")}
            if "user_id" not in columns:
                conn.execute(
                    "ALTER TABLE conversations ADD COLUMN user_id TEXT NOT NULL DEFAULT 'local'"
                )

            conn.executescript(INDEXES)

    # ----------------------------------------------------------- conversations

    def load_conversations(self) -> list[Conversation]:
        """Read every conversation with its messages, newest activity first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM conversations WHERE user_id = ? ORDER BY updated_at DESC",
                (self.user_id,),
            ).fetchall()
            messages = conn.execute(
                """SELECT m.* FROM messages m
                   JOIN conversations c ON c.id = m.conversation_id
                   WHERE c.user_id = ?
                   ORDER BY m.conversation_id, m.position""",
                (self.user_id,),
            ).fetchall()

        by_conversation: dict[str, list[Message]] = {}
        for row in messages:
            by_conversation.setdefault(row["conversation_id"], []).append(
                Message(row["role"], row["content"], _dt(row["created_at"]))
            )

        return [
            Conversation(
                id=row["id"],
                title=row["title"],
                messages=by_conversation.get(row["id"], []),
                created_at=_dt(row["created_at"]),
                updated_at=_dt(row["updated_at"]),
            )
            for row in rows
        ]

    def save_conversation(self, conv: Conversation) -> None:
        """Insert or replace a conversation and its full message list.

        Messages are rewritten wholesale rather than diffed. Threads are short,
        and this keeps the operation idempotent — the caller never has to know
        whether a row already exists.
        """
        conv.updated_at = datetime.now()
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO conversations (id, user_id, title, created_at, updated_at)
                   VALUES (?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET title=excluded.title,
                                                 updated_at=excluded.updated_at""",
                (conv.id, self.user_id, conv.title, _iso(conv.created_at), _iso(conv.updated_at)),
            )
            conn.execute("DELETE FROM messages WHERE conversation_id = ?", (conv.id,))
            conn.executemany(
                """INSERT INTO messages (conversation_id, position, role, content, created_at)
                   VALUES (?, ?, ?, ?, ?)""",
                [
                    (conv.id, i, m.role, m.content, _iso(m.created_at))
                    for i, m in enumerate(conv.messages)
                ],
            )

    def delete_conversation(self, conversation_id: str) -> None:
        """Remove a conversation and, by cascade, its messages."""
        with self._connect() as conn:
            conn.execute(
                """DELETE FROM messages WHERE conversation_id IN
                   (SELECT id FROM conversations WHERE id = ? AND user_id = ?)""",
                (conversation_id, self.user_id),
            )
            conn.execute(
                "DELETE FROM conversations WHERE id = ? AND user_id = ?",
                (conversation_id, self.user_id),
            )

    # ---------------------------------------------------------------- memories

    def load_memories(self) -> list[Memory]:
        """Read every memory for this user, most recently revised first."""
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT * FROM memories WHERE user_id = ? ORDER BY updated_at DESC",
                (self.user_id,),
            ).fetchall()
        return [
            Memory(
                id=row["id"],
                text=row["text"],
                category=row["category"],
                created_at=_dt(row["created_at"]),
                updated_at=_dt(row["updated_at"]),
                source_conversation_id=row["source_conversation_id"],
            )
            for row in rows
        ]

    def upsert_memory(self, memory: Memory) -> None:
        """Insert a fact, or overwrite the existing row with the same id.

        This is the whole update mechanism: an extractor that decides "I'm 26
        now" supersedes "is 25 years old" returns a Memory carrying the old id
        and ``created_at`` with new ``text`` and ``updated_at``, and the
        conflict clause below rewrites that row in place.
        """
        with self._connect() as conn:
            conn.execute(
                """INSERT INTO memories
                       (id, user_id, text, category, created_at, updated_at,
                        source_conversation_id)
                   VALUES (?, ?, ?, ?, ?, ?, ?)
                   ON CONFLICT(id) DO UPDATE SET
                       text=excluded.text,
                       category=excluded.category,
                       updated_at=excluded.updated_at,
                       source_conversation_id=excluded.source_conversation_id""",
                (
                    memory.id,
                    self.user_id,
                    memory.text,
                    memory.category,
                    _iso(memory.created_at),
                    _iso(memory.updated_at),
                    memory.source_conversation_id,
                ),
            )

    def delete_memory(self, memory_id: str) -> None:
        """Forget a single fact."""
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM memories WHERE id = ? AND user_id = ?",
                (memory_id, self.user_id),
            )

    def clear_memories(self) -> None:
        """Forget everything for this user. Conversations are left alone."""
        with self._connect() as conn:
            conn.execute("DELETE FROM memories WHERE user_id = ?", (self.user_id,))
