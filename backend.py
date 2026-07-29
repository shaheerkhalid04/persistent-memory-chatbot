"""Data model and storage/LLM seams for the chatbot.

Everything in this module is a **stub**. The UI talks to these functions only, so
swapping in a real memory backend (Mem0, Zep, OpenMemory, ...) and a real LLM
(Groq / Gemini / OpenAI) means rewriting the bodies below and nothing else.

Contract expected by the UI layer:

    stream_reply(messages, memories)  -> Iterator[str]   (token stream)
    load_conversations()              -> list[Conversation]
    save_conversation(conv)           -> None
    load_memories()                   -> list[Memory]
    upsert_memory(m)                  -> None
    delete_memory(memory_id)          -> None

Any of them may raise ``StorageError``; the UI catches it and renders a banner
instead of a traceback.
"""

from __future__ import annotations

import random
import time
import uuid
from dataclasses import dataclass, field, replace
from datetime import datetime, timedelta
from typing import Iterator, Literal

# --------------------------------------------------------------------------- #
# Types
# --------------------------------------------------------------------------- #

Role = Literal["user", "assistant"]

#: Categories the memory extractor is allowed to emit. Keep this list short —
#: the sidebar groups memories by it, and long tails make the panel unreadable.
CATEGORIES: tuple[str, ...] = (
    "Identity",
    "Preferences",
    "Location",
    "Work",
    "Interests",
    "Goals",
)


class StorageError(RuntimeError):
    """Raised when the persistence layer cannot read or write."""


@dataclass
class Memory:
    """A single durable fact about the user.

    Attributes:
        id: Stable unique identifier; also the key used by ``delete_memory``.
        text: The fact itself, phrased in third person ("Lives in Lahore").
        category: One of :data:`CATEGORIES`. Drives grouping in the memory panel.
        created_at: When the fact was first learned.
        updated_at: When the fact was last revised. Equal to ``created_at``
            until the fact is overwritten ("is 25" -> "is 26").
        source_conversation_id: Conversation the fact was extracted from, or
            ``None`` if the user typed it directly into the memory panel.
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
            id=uuid.uuid4().hex[:12],
            text=text.strip(),
            category=category,
            created_at=now,
            updated_at=now,
            source_conversation_id=source_conversation_id,
        )


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
        return Conversation(id=uuid.uuid4().hex[:12], title=title, created_at=now, updated_at=now)

    @property
    def preview(self) -> str:
        """First user line, for list subtitles."""
        for m in self.messages:
            if m.role == "user":
                return m.content
        return "No messages yet"


# --------------------------------------------------------------------------- #
# In-memory fake store
#
# Stands in for the real database so the UI is fully explorable before any
# backend exists. Replace these two globals with real reads/writes.
# --------------------------------------------------------------------------- #

_FAKE_LATENCY = 0.35  # seconds; makes the loading skeleton actually visible

_now = datetime.now()


def _seed_conversations() -> list[Conversation]:
    c1 = Conversation(
        id="c_welcome001",
        title="Weekend in Islamabad",
        created_at=_now - timedelta(days=3),
        updated_at=_now - timedelta(days=3, hours=2),
        messages=[
            Message("user", "I'm heading to Islamabad this weekend, first time.",
                    _now - timedelta(days=3)),
            Message(
                "assistant",
                "Nice — three days is enough to see the good parts without rushing.\n\n"
                "- **Daman-e-Koh** at sunset, then walk down to Pir Sohawa for dinner\n"
                "- **Saidpur Village** in the morning, before the crowds\n"
                "- **Lok Virsa** if you want the museum day\n\n"
                "Pack a light jacket — the Margalla side drops a few degrees after dark.",
                _now - timedelta(days=3),
            ),
        ],
    )
    c2 = Conversation(
        id="c_sports0002",
        title="Getting back into sport",
        created_at=_now - timedelta(days=1),
        updated_at=_now - timedelta(days=1, minutes=20),
        messages=[
            Message("user", "I've been sitting at a desk all year and it shows.",
                    _now - timedelta(days=1)),
            Message(
                "assistant",
                "Since you already play football on weekends, the easiest win is just "
                "adding structure to what you enjoy rather than starting something new. "
                "Two short sessions midweek would carry the weekend game a long way.",
                _now - timedelta(days=1),
            ),
        ],
    )
    c3 = Conversation(
        id="c_python0003",
        title="Parsing a messy CSV",
        created_at=_now - timedelta(hours=5),
        updated_at=_now - timedelta(hours=4),
        messages=[
            Message("user", "How do I skip bad rows in a CSV without the whole thing blowing up?",
                    _now - timedelta(hours=5)),
            Message(
                "assistant",
                "Read it row by row and quarantine the failures instead of aborting:\n\n"
                "```python\nimport csv\n\ngood, bad = [], []\nwith open(\"data.csv\", newline=\"\") as fh:\n"
                "    for i, row in enumerate(csv.DictReader(fh), start=2):\n"
                "        try:\n            good.append({**row, \"amount\": float(row[\"amount\"])})\n"
                "        except (ValueError, KeyError) as exc:\n            bad.append((i, exc))\n```\n\n"
                "You end up with a clean list and a report of exactly which lines to fix.",
                _now - timedelta(hours=4),
            ),
        ],
    )
    return [c3, c2, c1]


def _seed_memories() -> list[Memory]:
    def m(text, cat, days_old, days_updated=None, src=None):
        created = _now - timedelta(days=days_old)
        updated = _now - timedelta(days=days_updated if days_updated is not None else days_old)
        return Memory(uuid.uuid4().hex[:12], text, cat, created, updated, src)

    return [
        m("Is 26 years old", "Identity", 41, 2),
        m("Goes by Sam", "Identity", 41),
        m("Lives in Lahore, Pakistan", "Location", 40),
        m("Travels to Islamabad a few times a year", "Location", 3, src="c_welcome001"),
        m("Plays football on weekends", "Interests", 12, src="c_sports0002"),
        m("Reads non-fiction, mostly history", "Interests", 22),
        m("Prefers short, direct answers over long explanations", "Preferences", 30),
        m("Dislikes being asked follow-up questions before an answer", "Preferences", 9),
        m("Drinks coffee black, no sugar", "Preferences", 18),
        m("Backend engineer, works mostly in Python", "Work", 35),
        m("On a team of four, ships on Thursdays", "Work", 15),
        m("Wants to run a half marathon this year", "Goals", 7),
    ]


_conversations: list[Conversation] = _seed_conversations()
_memories: list[Memory] = _seed_memories()


# --------------------------------------------------------------------------- #
# Storage seam
# --------------------------------------------------------------------------- #


def load_conversations() -> list[Conversation]:
    """Return every stored conversation, newest activity first.

    Returns:
        Conversations sorted by ``updated_at`` descending. Empty list is valid
        and must render the empty state, not an error.

    Raises:
        StorageError: If the store is unreachable or corrupt.
    """
    time.sleep(_FAKE_LATENCY)
    return sorted(_conversations, key=lambda c: c.updated_at, reverse=True)


def save_conversation(conv: Conversation) -> None:
    """Insert or replace a conversation, keyed on ``conv.id``.

    Called after every completed turn, on rename, and on creation. Must be
    idempotent — the UI does not track whether a row already exists.

    Args:
        conv: The conversation to persist, including its full message list.

    Raises:
        StorageError: If the write fails.
    """
    conv.updated_at = datetime.now()
    for i, existing in enumerate(_conversations):
        if existing.id == conv.id:
            _conversations[i] = conv
            return
    _conversations.insert(0, conv)


def delete_conversation(conversation_id: str) -> None:
    """Remove a conversation and its messages.

    Args:
        conversation_id: Id of the conversation to drop. Unknown ids are a no-op.

    Raises:
        StorageError: If the delete fails.
    """
    global _conversations
    _conversations = [c for c in _conversations if c.id != conversation_id]


def load_memories() -> list[Memory]:
    """Return every stored memory.

    Ordering is the UI's business — it groups by category and sorts by
    ``updated_at`` itself.

    Raises:
        StorageError: If the store is unreachable or corrupt.
    """
    time.sleep(_FAKE_LATENCY * 2)  # slower on purpose: exercises the skeleton
    return list(_memories)


def upsert_memory(m: Memory) -> None:
    """Insert a memory, or overwrite the existing one with the same id.

    This is the update path for "I'm 26 now" replacing "is 25 years old": the
    caller reuses the old ``id`` and ``created_at``, bumps ``updated_at``, and
    passes the revised text.

    Args:
        m: The memory to write.

    Raises:
        StorageError: If the write fails.
    """
    for i, existing in enumerate(_memories):
        if existing.id == m.id:
            _memories[i] = m
            return
    _memories.append(m)


def delete_memory(memory_id: str) -> None:
    """Forget a single fact.

    Args:
        memory_id: Id of the memory to remove. Unknown ids are a no-op.

    Raises:
        StorageError: If the delete fails.
    """
    global _memories
    _memories = [m for m in _memories if m.id != memory_id]


def clear_memories() -> None:
    """Forget everything. Wired to the destructive 'clear all' control.

    Raises:
        StorageError: If the wipe fails.
    """
    _memories.clear()


# --------------------------------------------------------------------------- #
# Memory extraction seam
# --------------------------------------------------------------------------- #


def extract_memories(user_text: str, conversation_id: str, existing: list[Memory]) -> list[Memory]:
    """Decide what, if anything, is worth remembering from a user turn.

    The real implementation hands ``user_text`` to the memory library, which
    returns add/update/no-op decisions. Updates come back carrying the id of the
    memory they supersede, which is how "I'm 26 now" overwrites "is 25 years old"
    rather than piling up beside it.

    Args:
        user_text: The raw user message.
        conversation_id: Attributed as the memory's source.
        existing: Current memories, so the stub can return an *update* to one of
            them (same ``id``, same ``created_at``, fresh ``updated_at``).

    Returns:
        Memories to pass to :func:`upsert_memory`. Empty when nothing is
        durable — most turns.
    """
    text = user_text.lower()

    # Crude keyword rules stand in for real extraction.
    rules: list[tuple[tuple[str, ...], str, str]] = [
        (("i am", "i'm", "my name", "call me"), "Identity", "Mentioned something about who they are"),
        (("i like", "i love", "i enjoy", "favourite", "favorite"), "Interests", "Enjoys {frag}"),
        (("i live", "i'm from", "i am from", "based in"), "Location", "Location: {frag}"),
        (("i work", "my job", "my team"), "Work", "Work: {frag}"),
        (("i want", "i'd like to", "goal", "trying to"), "Goals", "Wants to {frag}"),
        (("i prefer", "i hate", "i dislike", "don't like"), "Preferences", "Preference: {frag}"),
    ]

    for triggers, category, template in rules:
        if any(t in text for t in triggers):
            frag = user_text.strip().rstrip(".")
            if len(frag) > 90:
                frag = frag[:87].rsplit(" ", 1)[0] + "..."
            new_text = template.format(frag=frag) if "{frag}" in template else frag

            # Simulate an overwrite when the category already has a fact.
            same_category = [m for m in existing if m.category == category]
            if same_category and random.random() < 0.35:
                target = same_category[0]
                return [replace(target, text=new_text, updated_at=datetime.now(),
                                source_conversation_id=conversation_id)]
            return [Memory.new(new_text, category, conversation_id)]

    return []


# --------------------------------------------------------------------------- #
# LLM seam
# --------------------------------------------------------------------------- #

_CANNED: dict[str, str] = {
    "sport": (
        "Football, without hesitation.\n\n"
        "You already play it on weekends, so there's no ramp-up and no new gear to buy — "
        "the hard part of picking up a sport is the part you've already done. If you want "
        "something to complement it, swimming is the usual pairing: same cardio load, none "
        "of the impact on your knees.\n\n"
        "Two evenings a week is enough to feel the difference in the weekend game by month two."
    ),
    "code": (
        "Here's the pattern I'd reach for:\n\n"
        "```python\nfrom dataclasses import dataclass\nfrom typing import Iterator\n\n\n"
        "@dataclass(frozen=True)\nclass Fact:\n    key: str\n    value: str\n\n\n"
        "def merge(old: list[Fact], new: Iterator[Fact]) -> list[Fact]:\n"
        "    \"\"\"Last write wins, order preserved.\"\"\"\n"
        "    index = {f.key: i for i, f in enumerate(old)}\n"
        "    out = list(old)\n"
        "    for fact in new:\n"
        "        if fact.key in index:\n"
        "            out[index[fact.key]] = fact\n"
        "        else:\n"
        "            out.append(fact)\n"
        "    return out\n```\n\n"
        "Keying on `key` rather than appending is what stops the store from filling up with "
        "stale copies of the same fact."
    ),
    "remember": (
        "Here's what I'm holding onto about you right now:\n\n"
        "{memory_lines}\n\n"
        "You can edit or delete any of it from the memory panel — the toggle is in the sidebar."
    ),
    "default": (
        "Got it. Based on what I know about you, the short version:\n\n"
        "1. Start with what you already do rather than what looks best on paper.\n"
        "2. Keep the first commitment small enough that a bad week doesn't end it.\n"
        "3. Reassess in three weeks, not three days.\n\n"
        "Want me to go deeper on any of those?"
    ),
}


def stream_reply(messages: list[dict], memories: list[Memory]) -> Iterator[str]:
    """Stream an assistant reply token by token.

    The real implementation builds a system prompt from ``memories``, appends
    ``messages``, and yields deltas from the provider's streaming endpoint.

    Args:
        messages: Full turn history as ``{"role": ..., "content": ...}`` dicts,
            oldest first, ending with the user turn to answer.
        memories: Facts to inject as context. The stub only reads them for the
            "what do you remember" path; a real backend would render them into
            the system prompt.

    Yields:
        Text chunks. Concatenating every chunk gives the complete reply.
    """
    last_user = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
    text = last_user.lower()

    if any(w in text for w in ("remember", "know about me", "my memories")):
        lines = "\n".join(f"- {m.text}" for m in memories[:6]) or "- Nothing yet."
        reply = _CANNED["remember"].format(memory_lines=lines)
    elif any(w in text for w in ("code", "python", "function", "script", "bug")):
        reply = _CANNED["code"]
    elif any(w in text for w in ("sport", "exercise", "fit", "gym", "run")):
        reply = _CANNED["sport"]
    else:
        reply = _CANNED["default"]

    for token in _tokenize(reply):
        time.sleep(random.uniform(0.012, 0.032))
        yield token


def _tokenize(text: str) -> Iterator[str]:
    """Split into word-ish chunks so the stream looks like a real one."""
    buf = ""
    for ch in text:
        buf += ch
        if ch in " \n":
            yield buf
            buf = ""
    if buf:
        yield buf


def suggest_title(first_message: str) -> str:
    """Derive a conversation title from its opening message.

    A real backend would ask the model for a 3-5 word summary; this trims.

    Args:
        first_message: The user's opening turn.

    Returns:
        A short title, never longer than 40 characters.
    """
    cleaned = " ".join(first_message.strip().split())
    if len(cleaned) <= 40:
        return cleaned or "New chat"
    return cleaned[:37].rsplit(" ", 1)[0] + "..."
