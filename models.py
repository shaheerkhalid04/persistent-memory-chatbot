"""Domain types shared by every layer.

These are plain dataclasses with no knowledge of Streamlit, SQLite or any LLM
provider, so the storage and model layers can be swapped without touching them.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime
from typing import Literal

Role = Literal["user", "assistant"]

#: Buckets the extractor is allowed to emit. Deliberately short — the memory
#: panel groups by this, and a long tail makes it unreadable.
CATEGORIES: tuple[str, ...] = (
    "Identity",
    "Preferences",
    "Location",
    "Work",
    "Interests",
    "Goals",
)


def _new_id() -> str:
    return uuid.uuid4().hex[:12]


@dataclass
class Memory:
    """A single durable fact about the user.

    Attributes:
        id: Stable identifier. The update path reuses it so a revised fact
            replaces the old row instead of sitting beside it.
        text: The fact, phrased in the third person ("Lives in Lahore").
        category: One of :data:`CATEGORIES`.
        created_at: When the fact was first learned.
        updated_at: When it was last revised; equal to ``created_at`` until
            something supersedes it.
        source_conversation_id: Conversation it was extracted from, or ``None``
            when typed directly into the memory panel.
    """

    id: str
    text: str
    category: str
    created_at: datetime
    updated_at: datetime
    source_conversation_id: str | None = None

    @staticmethod
    def new(text: str, category: str, source_conversation_id: str | None = None) -> "Memory":
        """Build a Memory with a fresh id and matching timestamps."""
        now = datetime.now()
        return Memory(
            id=_new_id(),
            text=text.strip(),
            category=category if category in CATEGORIES else "Identity",
            created_at=now,
            updated_at=now,
            source_conversation_id=source_conversation_id,
        )

    @property
    def was_revised(self) -> bool:
        """True once the fact has been overwritten at least once."""
        return self.updated_at > self.created_at


@dataclass
class Message:
    """One turn in a conversation."""

    role: Role
    content: str
    created_at: datetime = field(default_factory=datetime.now)

    def as_dict(self) -> dict[str, str]:
        """Shape expected by chat-completion APIs."""
        return {"role": self.role, "content": self.content}


@dataclass
class Conversation:
    """An ordered thread of messages with a human-editable title."""

    id: str
    title: str
    messages: list[Message] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    updated_at: datetime = field(default_factory=datetime.now)

    @staticmethod
    def new(title: str = "New chat") -> "Conversation":
        now = datetime.now()
        return Conversation(id=_new_id(), title=title, created_at=now, updated_at=now)

    @property
    def preview(self) -> str:
        """First user line, for list subtitles."""
        for message in self.messages:
            if message.role == "user":
                return message.content
        return "No messages yet"


class StorageError(RuntimeError):
    """Raised when the persistence layer cannot read or write."""


class LLMError(RuntimeError):
    """Raised when the model backend is misconfigured or unreachable."""
