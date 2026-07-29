"""Recall — a chatbot that remembers you between sessions.

The Streamlit UI. Every data and model operation goes through ``backend.py``,
so this file knows nothing about SQLite, Groq or Mem0.

Streamlit re-executes this file top to bottom on every interaction, so the whole
app is organised around one rule: **a click writes an intent into session state
and calls st.rerun(); the next pass reads that intent and renders.** Nothing
mutates state a widget on the current pass has already consumed.

Pass order matters and is fixed:

    1. init state (guarded)          5. sidebar
    2. flush queued toasts           6. chat input (pinned; painted before we block)
    3. bootstrap storage             7. main area — messages, then streaming
    4. consume the pending prompt    8. after the stream: persist, rerun
"""

from __future__ import annotations

from typing import Callable, TypeVar

import streamlit as st

from backend import (
    Conversation,
    LLMError,
    Memory,
    Message,
    StorageError,
    clear_memories,
    delete_conversation,
    delete_memory,
    load_conversations,
    load_memories,
    observe_turn,
    save_conversation,
    status_label,
    stream_reply,
    suggest_title,
    upsert_memory,
)
from ui import components as ui
from ui.styles import inject as inject_css

T = TypeVar("T")

st.set_page_config(
    page_title="Recall",
    page_icon=":material/bookmark:",
    layout="wide",
    initial_sidebar_state="expanded",
)

DEFAULT_TITLE = "New chat"


# --------------------------------------------------------------------------- #
# 1. State
# --------------------------------------------------------------------------- #

#: Every key the app reads, with the value a first run must see. Reads go
#: through ``s()`` so a missing key can never crash the first pass.
DEFAULTS: dict[str, object] = {
    "initialized": False,
    "memories_loaded": False,
    "storage_error": None,
    "conversations": [],
    "memories": [],
    "active_id": None,
    "pending_prompt": None,
    "is_streaming": False,
    "show_memory": False,
    "renaming_id": None,
    "confirm_delete_id": None,
    "editing_memory_id": None,
    "toast_queue": [],
}


def init_state() -> None:
    """Seed session state once per browser session."""
    for key, value in DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, list) else value


def s(key: str) -> object:
    """Read session state with the declared default as a fallback."""
    return st.session_state.get(key, DEFAULTS.get(key))


def guard(fn: Callable[..., T], *args, fallback: T | None = None) -> T | None:
    """Run a storage read, converting failures into a banner instead of a trace.

    Args:
        fn: The storage function to invoke.
        *args: Positional arguments forwarded to ``fn``.
        fallback: Value to return if the call raises.

    Returns:
        The function's result, or ``fallback`` when it raised.
    """
    try:
        return fn(*args)
    except StorageError as exc:
        st.session_state["storage_error"] = str(exc)
        return fallback
    except Exception as exc:  # unexpected: still no traceback in the user's face
        st.session_state["storage_error"] = f"{type(exc).__name__}: {exc}"
        return fallback


def attempt(fn: Callable[..., object], *args) -> bool:
    """Run a storage write for its side effect.

    Args:
        fn: The storage function to invoke.
        *args: Positional arguments forwarded to ``fn``.

    Returns:
        ``True`` if the call completed, ``False`` if it failed (in which case the
        error banner is armed for this pass).
    """
    try:
        fn(*args)
        return True
    except StorageError as exc:
        st.session_state["storage_error"] = str(exc)
        return False
    except Exception as exc:
        st.session_state["storage_error"] = f"{type(exc).__name__}: {exc}"
        return False


def queue_toast(message: str, icon: str = ":material/bookmark_added:") -> None:
    """Stash a toast to fire on the next pass, after the pending rerun."""
    st.session_state["toast_queue"] = list(s("toast_queue")) + [(message, icon)]


def flush_toasts() -> None:
    """Show and clear anything queued by the previous pass."""
    for message, icon in list(s("toast_queue")):
        st.toast(message, icon=icon)
    st.session_state["toast_queue"] = []


# --------------------------------------------------------------------------- #
# 2. Conversation helpers
# --------------------------------------------------------------------------- #


def active_conversation() -> Conversation | None:
    """The conversation currently open, or ``None`` if the list is empty."""
    conversations: list[Conversation] = list(s("conversations"))
    active_id = s("active_id")
    return next((c for c in conversations if c.id == active_id), None)


def start_new_chat() -> None:
    """Create an empty conversation, make it active, and rerun."""
    conv = Conversation.new(DEFAULT_TITLE)
    st.session_state["conversations"] = [conv] + list(s("conversations"))
    st.session_state["active_id"] = conv.id
    st.session_state["renaming_id"] = None
    st.session_state["confirm_delete_id"] = None
    attempt(save_conversation, conv)
    st.rerun()


def bootstrap() -> None:
    """First-pass load of conversations and memories, with visible loading UI."""
    if not s("initialized"):
        with st.spinner("Opening your history…"):
            conversations = guard(load_conversations, fallback=[]) or []
        st.session_state["conversations"] = conversations
        st.session_state["active_id"] = conversations[0].id if conversations else None
        st.session_state["initialized"] = True

    if not s("memories_loaded"):
        slot = st.empty()
        with slot.container():
            st.markdown('<div class="rail-label">Recalling what I know about you</div>',
                        unsafe_allow_html=True)
            ui.skeleton(4)
        st.session_state["memories"] = guard(load_memories, fallback=[]) or []
        st.session_state["memories_loaded"] = True
        slot.empty()


# --------------------------------------------------------------------------- #
# 3. Turn handling
# --------------------------------------------------------------------------- #


def submit_prompt(text: str) -> None:
    """Queue a user turn and lock the input, then rerun to render it.

    Args:
        text: Raw user message. Blank input is ignored.
    """
    text = text.strip()
    if not text or s("is_streaming"):
        return
    st.session_state["pending_prompt"] = text
    st.session_state["is_streaming"] = True
    st.rerun()


def consume_pending_prompt() -> None:
    """Fold the queued prompt into the active conversation, pre-render.

    Runs before any widget is drawn so the sidebar, header and message list all
    see the same conversation state on this pass. Memory extraction happens here
    too — its toasts land immediately rather than a pass late.
    """
    prompt = s("pending_prompt")
    if not prompt:
        return

    conv = active_conversation()
    if conv is None:
        conv = Conversation.new(DEFAULT_TITLE)
        st.session_state["conversations"] = [conv] + list(s("conversations"))
        st.session_state["active_id"] = conv.id

    conv.messages.append(Message("user", str(prompt)))
    if conv.title == DEFAULT_TITLE:
        conv.title = suggest_title(str(prompt))
    attempt(save_conversation, conv)

    # The handler extracts, decides ADD vs UPDATE against what is already
    # stored, and persists. We only mirror the result into the session cache.
    for event in guard(observe_turn, str(prompt), conv.id, fallback=[]) or []:
        replace_memory(event.memory)
        icon = (":material/bookmark_added:" if event.kind == "added"
                else ":material/autorenew:")
        st.toast(event.toast, icon=icon)

    st.session_state["pending_prompt"] = None


def replace_memory(mem: Memory) -> None:
    """Insert or overwrite a memory in the in-session cache, keyed on id."""
    memories = [m for m in s("memories") if m.id != mem.id]
    st.session_state["memories"] = memories + [mem]


def run_stream(conv: Conversation) -> None:
    """Stream the assistant's reply, persist the turn, then rerun.

    Args:
        conv: The conversation whose last message is the user turn to answer.
    """
    payload = [m.as_dict() for m in conv.messages]
    with st.chat_message("assistant"):
        try:
            reply = st.write_stream(stream_reply(payload, list(s("memories"))))
        except LLMError as exc:
            st.session_state["is_streaming"] = False
            st.error(f"The model call failed — {exc}", icon=":material/error:")
            return
        except Exception as exc:
            st.session_state["is_streaming"] = False
            st.error(f"Unexpected model error — {type(exc).__name__}: {exc}",
                     icon=":material/error:")
            return

    conv.messages.append(Message("assistant", str(reply)))
    attempt(save_conversation, conv)
    st.session_state["is_streaming"] = False
    st.rerun()


# --------------------------------------------------------------------------- #
# 4. Sidebar
# --------------------------------------------------------------------------- #


def render_sidebar() -> None:
    """Brand, new chat, memory toggle, search, and the conversation rail."""
    with st.sidebar:
        ui.brand()
        st.markdown("<div style='height:0.7rem'></div>", unsafe_allow_html=True)

        if st.button("＋  New chat", key="new_chat", type="primary",
                     use_container_width=True, disabled=bool(s("is_streaming"))):
            start_new_chat()

        st.toggle("Memory panel", key="show_memory",
                  help="Split the view and show everything the bot has stored.")

        ui.rail_label("Conversations")
        query = st.text_input(
            "Search", key="conv_search", placeholder="Search titles…",
            label_visibility="collapsed",
        )

        conversations: list[Conversation] = list(s("conversations"))
        needle = (query or "").strip().lower()
        visible = [c for c in conversations if needle in c.title.lower()] if needle else conversations

        ui.conversation_list(
            visible,
            active_id=str(s("active_id")) if s("active_id") else None,
            renaming_id=str(s("renaming_id")) if s("renaming_id") else None,
            confirm_delete_id=str(s("confirm_delete_id")) if s("confirm_delete_id") else None,
            empty_message=("No conversations match." if needle
                           else "No conversations yet — start one above."),
            on_select=handle_select,
            on_rename_start=handle_rename_start,
            on_rename_commit=handle_rename_commit,
            on_rename_cancel=handle_rename_cancel,
            on_delete_request=handle_delete_request,
            on_delete_confirm=handle_delete_confirm,
            on_delete_cancel=handle_delete_cancel,
        )

        st.markdown("<div style='height:1.2rem'></div>", unsafe_allow_html=True)
        st.caption(status_label())


def handle_select(conversation_id: str) -> None:
    st.session_state["active_id"] = conversation_id
    st.session_state["renaming_id"] = None
    st.session_state["confirm_delete_id"] = None
    st.rerun()


def handle_rename_start(conversation_id: str) -> None:
    st.session_state["renaming_id"] = conversation_id
    st.session_state["confirm_delete_id"] = None
    st.rerun()


def handle_rename_commit(conversation_id: str, new_title: str) -> None:
    conv = next((c for c in s("conversations") if c.id == conversation_id), None)
    if conv is not None:
        conv.title = new_title.strip() or DEFAULT_TITLE
        attempt(save_conversation, conv)
    st.session_state["renaming_id"] = None
    st.rerun()


def handle_rename_cancel() -> None:
    st.session_state["renaming_id"] = None
    st.rerun()


def handle_delete_request(conversation_id: str) -> None:
    st.session_state["confirm_delete_id"] = conversation_id
    st.session_state["renaming_id"] = None
    st.rerun()


def handle_delete_confirm(conversation_id: str) -> None:
    attempt(delete_conversation, conversation_id)
    remaining = [c for c in s("conversations") if c.id != conversation_id]
    st.session_state["conversations"] = remaining
    if s("active_id") == conversation_id:
        st.session_state["active_id"] = remaining[0].id if remaining else None
    st.session_state["confirm_delete_id"] = None
    queue_toast("Conversation deleted", ":material/delete:")
    st.rerun()


def handle_delete_cancel() -> None:
    st.session_state["confirm_delete_id"] = None
    st.rerun()


# --------------------------------------------------------------------------- #
# 5. Memory panel handlers
# --------------------------------------------------------------------------- #


def handle_memory_edit_start(memory_id: str) -> None:
    st.session_state["editing_memory_id"] = memory_id
    st.rerun()


def handle_memory_edit_commit(memory_id: str, new_text: str) -> None:
    from dataclasses import replace as dc_replace
    from datetime import datetime

    mem = next((m for m in s("memories") if m.id == memory_id), None)
    text = new_text.strip()
    if mem is not None and text:
        updated = dc_replace(mem, text=text, updated_at=datetime.now())
        attempt(upsert_memory, updated)
        replace_memory(updated)
        queue_toast(f"Updated: {text}", ":material/edit:")
    st.session_state["editing_memory_id"] = None
    st.rerun()


def handle_memory_edit_cancel() -> None:
    st.session_state["editing_memory_id"] = None
    st.rerun()


def handle_memory_delete(memory_id: str) -> None:
    attempt(delete_memory, memory_id)
    st.session_state["memories"] = [m for m in s("memories") if m.id != memory_id]
    st.session_state["editing_memory_id"] = None
    queue_toast("Forgotten", ":material/delete:")
    st.rerun()


def handle_memory_clear_all() -> None:
    attempt(clear_memories)
    st.session_state["memories"] = []
    st.session_state["editing_memory_id"] = None
    queue_toast("Memory cleared", ":material/delete_sweep:")
    st.rerun()


# --------------------------------------------------------------------------- #
# 6. Main area
# --------------------------------------------------------------------------- #


def render_header(conv: Conversation | None) -> None:
    """Title strip with turn count, last activity, and the memory badge."""
    if conv is None:
        title, subtitle = "No conversation", "Start one from the sidebar"
    else:
        turns = len(conv.messages)
        subtitle = (f"{turns} message{'s' if turns != 1 else ''} · "
                    f"updated {ui.relative_time(conv.updated_at)}"
                    if turns else "Empty · nothing said yet")
        title = conv.title
    ui.header_strip(title, subtitle, len(list(s("memories"))))


def render_chat(conv: Conversation | None) -> None:
    """Message history, empty state, and the live stream."""
    if conv is None or not conv.messages:
        ui.empty_state(on_starter=submit_prompt)
        return

    for message in conv.messages:
        with st.chat_message(message.role):
            st.markdown(message.content)

    if s("is_streaming"):
        run_stream(conv)


def render_memory_column() -> None:
    """Right-hand panel; shows a skeleton if the store has not answered yet.

    Wrapped in a keyed container so the stylesheet can cap its height and give
    it its own scrollbar. Without that the panel grows past the viewport, the
    whole page scrolls as one, and the chat gets pushed off screen.
    """
    if not s("memories_loaded"):
        ui.skeleton(5)
        return

    ui.memory_panel(
        sorted(list(s("memories")), key=lambda m: m.updated_at, reverse=True),
        editing_id=str(s("editing_memory_id")) if s("editing_memory_id") else None,
        on_edit_start=handle_memory_edit_start,
        on_edit_commit=handle_memory_edit_commit,
        on_edit_cancel=handle_memory_edit_cancel,
        on_delete=handle_memory_delete,
        on_clear_all=handle_memory_clear_all,
    )


# --------------------------------------------------------------------------- #
# 7. Entry point
# --------------------------------------------------------------------------- #


def main() -> None:
    init_state()
    inject_css()
    flush_toasts()

    bootstrap()
    consume_pending_prompt()

    if s("storage_error"):
        st.error(f"Storage is unavailable — {s('storage_error')}", icon=":material/error:")

    render_sidebar()

    # Painted before the stream blocks the script, so the lock is visible.
    typed = st.chat_input(
        "Ask anything — it'll remember what matters",
        disabled=bool(s("is_streaming")),
        key="composer",
    )

    conv = active_conversation()
    render_header(conv)

    if s("show_memory"):
        chat_col, memory_col = st.columns([1.45, 1], gap="large")
        with chat_col:
            render_chat(conv)
        with memory_col:
            with st.container(key="memory_scroll"):
                render_memory_column()
    else:
        render_chat(conv)

    if typed:
        submit_prompt(typed)


if __name__ == "__main__":
    main()
