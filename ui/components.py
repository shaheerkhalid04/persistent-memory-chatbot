"""Reusable render functions.

These own markup and widget layout only. They never decide *what* happens — the
caller passes callbacks, and every callback in ``app.py`` follows the same shape:
write an intent into ``st.session_state``, then ``st.rerun()``. Nothing here
mutates state that a widget on the same pass has already rendered.
"""

from __future__ import annotations

from datetime import datetime
from typing import Callable, Iterable

import streamlit as st

from backend import Conversation, Memory

#: Glyph per memory category. Falls back to a neutral mark for unknown ones.
CATEGORY_GLYPH: dict[str, str] = {
    "Identity": "◈",
    "Preferences": "◉",
    "Location": "⌖",
    "Work": "▤",
    "Interests": "✦",
    "Goals": "➤",
}


def relative_time(when: datetime) -> str:
    """Render a timestamp as a coarse relative phrase.

    Args:
        when: The moment to describe. Assumed to be in the past.

    Returns:
        Strings like ``"just now"``, ``"4 hours ago"``, ``"2 days ago"``.
    """
    seconds = max(0.0, (datetime.now() - when).total_seconds())
    if seconds < 60:
        return "just now"
    minutes = seconds / 60
    if minutes < 60:
        n = int(minutes)
        return f"{n} minute{'s' if n != 1 else ''} ago"
    hours = minutes / 60
    if hours < 24:
        n = int(hours)
        return f"{n} hour{'s' if n != 1 else ''} ago"
    days = hours / 24
    if days < 7:
        n = int(days)
        return "yesterday" if n == 1 else f"{n} days ago"
    if days < 31:
        n = int(days / 7)
        return f"{n} week{'s' if n != 1 else ''} ago"
    if days < 365:
        n = max(1, int(days / 30))
        return f"{n} month{'s' if n != 1 else ''} ago"
    n = max(1, int(days / 365))
    return f"{n} year{'s' if n != 1 else ''} ago"


def brand() -> None:
    """Wordmark at the top of the sidebar."""
    st.markdown(
        '<div class="brand">'
        '<span class="brand-mark">Recall<span>.</span></span>'
        '<span class="brand-sub">memory</span>'
        "</div>",
        unsafe_allow_html=True,
    )


def rail_label(text: str) -> None:
    """Small uppercase section heading used in the sidebar."""
    st.markdown(f'<div class="rail-label">{text}</div>', unsafe_allow_html=True)


def header_strip(title: str, subtitle: str, memory_count: int) -> None:
    """Title bar above the chat, with a live memory-count badge.

    Args:
        title: Active conversation title.
        subtitle: Secondary line, e.g. turn count and last activity.
        memory_count: Number of facts currently stored.
    """
    st.markdown(
        f"""
        <div class="strip">
          <div style="min-width:0">
            <div class="strip-title">{_escape(title)}</div>
            <div class="strip-meta">{_escape(subtitle)}</div>
          </div>
          <div class="badge"><span class="dot"></span>{memory_count} remembered</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def skeleton(rows: int = 4) -> None:
    """Shimmer placeholders shown while the memory store is being read."""
    st.markdown("".join(f'<div class="skel" style="width:{w}%"></div>'
                        for w in _skeleton_widths(rows)),
                unsafe_allow_html=True)


def _skeleton_widths(rows: int) -> Iterable[int]:
    pattern = (100, 82, 94, 71, 88)
    return (pattern[i % len(pattern)] for i in range(rows))


# --------------------------------------------------------------------------- #
# Sidebar: conversation list
# --------------------------------------------------------------------------- #


def conversation_list(
    conversations: list[Conversation],
    active_id: str | None,
    renaming_id: str | None,
    confirm_delete_id: str | None,
    *,
    empty_message: str = "No conversations yet.",
    on_select: Callable[[str], None],
    on_rename_start: Callable[[str], None],
    on_rename_commit: Callable[[str, str], None],
    on_rename_cancel: Callable[[], None],
    on_delete_request: Callable[[str], None],
    on_delete_confirm: Callable[[str], None],
    on_delete_cancel: Callable[[], None],
) -> None:
    """Render the conversation rail with rename and guarded delete.

    Args:
        conversations: Already filtered and ordered for display.
        active_id: Conversation to highlight, if any.
        renaming_id: Conversation currently in inline-rename mode.
        confirm_delete_id: Conversation currently awaiting delete confirmation.
        empty_message: Shown when the list is empty. The caller distinguishes an
            empty store from a search that matched nothing.
        on_select: Called with a conversation id when a row is clicked.
        on_rename_start: Called with an id to enter rename mode.
        on_rename_commit: Called with ``(id, new_title)``.
        on_rename_cancel: Called to leave rename mode untouched.
        on_delete_request: Called with an id to arm the confirm step.
        on_delete_confirm: Called with an id once the user confirms.
        on_delete_cancel: Called to disarm the confirm step.
    """
    if not conversations:
        st.caption(empty_message)
        return

    for conv in conversations:
        if conv.id == renaming_id:
            _rename_row(conv, on_rename_commit, on_rename_cancel)
        elif conv.id == confirm_delete_id:
            _delete_row(conv, on_delete_confirm, on_delete_cancel)
        else:
            _normal_row(conv, conv.id == active_id, on_select, on_rename_start, on_delete_request)


def _normal_row(
    conv: Conversation,
    is_active: bool,
    on_select: Callable[[str], None],
    on_rename_start: Callable[[str], None],
    on_delete_request: Callable[[str], None],
) -> None:
    # Streamlit auto-closes a bare <div> written through st.markdown, so wrapper
    # divs cannot scope CSS. Keyed containers can: they emit an ``st-key-<key>``
    # class, which is what ui/styles.py targets.
    shell = st.container(key="active_conv_row") if is_active else st.container()
    with shell:
        row, edit, trash = st.columns([1, 0.17, 0.17], gap="small")
        with row:
            if st.button(_truncate(conv.title, 26), key=f"pick_{conv.id}",
                         use_container_width=True, help=_truncate(conv.preview, 90)):
                on_select(conv.id)
        with edit:
            if st.button("✎", key=f"ren_{conv.id}", help="Rename"):
                on_rename_start(conv.id)
        with trash:
            if st.button("✕", key=f"del_{conv.id}", help="Delete"):
                on_delete_request(conv.id)


def _rename_row(
    conv: Conversation,
    on_commit: Callable[[str, str], None],
    on_cancel: Callable[[], None],
) -> None:
    new_title = st.text_input(
        "Rename", value=conv.title, key=f"rename_input_{conv.id}",
        label_visibility="collapsed", placeholder="Conversation title",
    )
    save, cancel = st.columns(2, gap="small")
    with save:
        if st.button("Save", key=f"rename_ok_{conv.id}", type="primary", use_container_width=True):
            on_commit(conv.id, new_title)
    with cancel:
        if st.button("Cancel", key=f"rename_no_{conv.id}", use_container_width=True):
            on_cancel()


def _delete_row(
    conv: Conversation,
    on_confirm: Callable[[str], None],
    on_cancel: Callable[[], None],
) -> None:
    st.caption(f"Delete “{_truncate(conv.title, 22)}”?")
    yes, no = st.columns(2, gap="small")
    with yes:
        if st.button("Delete", key=f"delok_{conv.id}", use_container_width=True):
            on_confirm(conv.id)
    with no:
        if st.button("Keep", key=f"delno_{conv.id}", use_container_width=True):
            on_cancel()


# --------------------------------------------------------------------------- #
# Memory panel
# --------------------------------------------------------------------------- #


def memory_panel(
    memories: list[Memory],
    editing_id: str | None,
    *,
    on_edit_start: Callable[[str], None],
    on_edit_commit: Callable[[str, str], None],
    on_edit_cancel: Callable[[], None],
    on_delete: Callable[[str], None],
    on_clear_all: Callable[[], None],
) -> None:
    """Render stored facts grouped by category, each row inline-editable.

    Args:
        memories: Facts to display; grouped and sorted internally.
        editing_id: Memory currently in inline-edit mode, if any.
        on_edit_start: Called with a memory id to open its editor.
        on_edit_commit: Called with ``(id, new_text)``.
        on_edit_cancel: Called to close the editor without saving.
        on_delete: Called with a memory id to forget one fact.
        on_clear_all: Called once the destructive confirm is checked and clicked.
    """
    st.markdown(
        f'<div class="mem-head"><h3>Memory</h3>'
        f"<span>{len(memories)} fact{'s' if len(memories) != 1 else ''}</span></div>",
        unsafe_allow_html=True,
    )

    if not memories:
        st.caption("Nothing stored yet. Facts you share in chat land here automatically.")
        return

    for category, items in _group(memories):
        glyph = CATEGORY_GLYPH.get(category, "•")
        with st.expander(f"{glyph}  {category}  ·  {len(items)}", expanded=True):
            for mem in items:
                if mem.id == editing_id:
                    _memory_editor(mem, on_edit_commit, on_edit_cancel)
                else:
                    _memory_row(mem, on_edit_start, on_delete)

    st.markdown("<div style='height:0.6rem'></div>", unsafe_allow_html=True)
    confirmed = st.checkbox("I understand this erases every stored fact",
                            key="confirm_clear_all")
    if st.button("Clear all memory", key="clear_all_btn", disabled=not confirmed,
                 use_container_width=True):
        on_clear_all()


def _memory_row(
    mem: Memory,
    on_edit_start: Callable[[str], None],
    on_delete: Callable[[str], None],
) -> None:
    updated = mem.updated_at > mem.created_at
    stamp = (f'updated <em>{relative_time(mem.updated_at)}</em>' if updated
             else f"added {relative_time(mem.created_at)}")

    body, edit, trash = st.columns([1, 0.14, 0.14], gap="small")
    with body:
        st.markdown(
            f'<div class="mem-item"><div class="mem-text">{_escape(mem.text)}</div>'
            f'<div class="mem-time">{stamp}</div></div>',
            unsafe_allow_html=True,
        )
    with edit:
        if st.button("✎", key=f"medit_{mem.id}", help="Edit"):
            on_edit_start(mem.id)
    with trash:
        if st.button("✕", key=f"mdel_{mem.id}", help="Forget"):
            on_delete(mem.id)


def _memory_editor(
    mem: Memory,
    on_commit: Callable[[str, str], None],
    on_cancel: Callable[[], None],
) -> None:
    text = st.text_area(
        "Fact", value=mem.text, key=f"mem_edit_{mem.id}",
        label_visibility="collapsed", height=72,
    )
    save, cancel = st.columns(2, gap="small")
    with save:
        if st.button("Save", key=f"memok_{mem.id}", type="primary", use_container_width=True):
            on_commit(mem.id, text)
    with cancel:
        if st.button("Cancel", key=f"memno_{mem.id}", use_container_width=True):
            on_cancel()


def _group(memories: list[Memory]) -> list[tuple[str, list[Memory]]]:
    """Bucket memories by category, newest-updated first inside each bucket."""
    buckets: dict[str, list[Memory]] = {}
    for mem in memories:
        buckets.setdefault(mem.category, []).append(mem)
    for items in buckets.values():
        items.sort(key=lambda m: m.updated_at, reverse=True)
    return sorted(buckets.items(), key=lambda kv: (-len(kv[1]), kv[0]))


# --------------------------------------------------------------------------- #
# Empty state
# --------------------------------------------------------------------------- #

STARTERS: tuple[str, ...] = (
    "What do you already remember about me?",
    "I'm 26 and I've started running — what should I aim for?",
    "Show me a clean way to merge records without duplicating them.",
)


def empty_state(on_starter: Callable[[str], None]) -> None:
    """Hero shown when the active conversation has no messages.

    Args:
        on_starter: Called with the prompt text when a starter card is clicked.
    """
    st.markdown(
        """
        <div class="empty">
          <div class="empty-kicker">Persistent memory</div>
          <h1>Pick up where you left off.</h1>
          <p>Tell it something once — your age, where you live, what you're into —
             and it holds onto that between sessions. Nothing is re-explained twice.</p>
        </div>
        <div class="starter-label">Try one of these</div>
        """,
        unsafe_allow_html=True,
    )

    cols = st.columns(3, gap="medium")
    for index, (col, prompt) in enumerate(zip(cols, STARTERS)):
        with col:
            # Key prefix is the CSS hook for the card styling; keep it stable.
            if st.button(prompt, key=f"starter_{index}", use_container_width=True):
                on_starter(prompt)


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #


def _escape(text: str) -> str:
    """Minimal HTML escaping for values interpolated into markup."""
    return (text.replace("&", "&amp;").replace("<", "&lt;")
                .replace(">", "&gt;").replace('"', "&quot;"))


def _truncate(text: str, limit: int) -> str:
    text = text.strip() or "Untitled"
    return text if len(text) <= limit else text[: limit - 1].rstrip() + "…"
