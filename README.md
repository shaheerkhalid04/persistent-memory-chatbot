# Recall

A Streamlit chatbot that remembers you between sessions. Facts you mention once —
your age, where you live, what you're into — are extracted, stored, and injected
back into context on later turns, so nothing gets re-explained.

This repository currently contains **the frontend**. The model call and the
storage layer sit behind stub functions in `backend.py`, all with real
signatures, type hints and fake data, so the UI is fully explorable before a
backend exists.

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Layout

| Path | What's in it |
| --- | --- |
| `app.py` | State schema, pass ordering, all event handlers |
| `backend.py` | Dataclasses + the six seams to implement |
| `ui/styles.py` | The whole stylesheet, one string |
| `ui/components.py` | Render functions — markup and layout only |
| `.streamlit/config.toml` | Theme colours, kept in sync with `ui/styles.py` |

## The seams to implement

```python
stream_reply(messages: list[dict], memories: list[Memory]) -> Iterator[str]
load_conversations() -> list[Conversation]
save_conversation(conv: Conversation) -> None
load_memories() -> list[Memory]
upsert_memory(m: Memory) -> None
delete_memory(memory_id: str) -> None
```

Plus two conveniences the UI also calls: `extract_memories()` (decides what is
worth keeping from a user turn) and `suggest_title()`.

Any of them may raise `StorageError`; the UI catches it and renders a banner
rather than a traceback.

## How the Streamlit reruns are handled

Streamlit re-executes `app.py` top to bottom on every interaction, so the app
follows one rule throughout: **a click writes an intent into `st.session_state`
and calls `st.rerun()`; the next pass reads that intent and renders.** No state a
widget has already consumed is mutated later in the same pass.

The pass order in `main()` is fixed:

1. `init_state()` — every key seeded behind a guard, so a first run cannot
   `KeyError`. Reads go through `s(key)`, which falls back to the declared default.
2. `flush_toasts()` — toasts queued by the previous pass fire here.
3. `bootstrap()` — conversations load under a spinner, memories under a shimmer
   skeleton.
4. `consume_pending_prompt()` — the queued user turn is folded into the active
   conversation *before* anything renders, so the sidebar, header and message
   list all agree.
5. Sidebar.
6. `st.chat_input`, painted before the stream blocks the script so the disabled
   state is actually visible while a reply is being generated.
7. Main area — message history, then `st.write_stream`.
8. Persist the completed turn, then `st.rerun()`.

## Memory strategy

**What's stored.** Durable facts only, in six buckets: Identity, Preferences,
Location, Work, Interests, Goals. Phrased in the third person ("Plays football on
weekends") so they read cleanly when concatenated into a system prompt. Passing
remarks and anything scoped to the current turn are not stored.

**How it's recalled.** `stream_reply()` receives the full memory list alongside
the message history; the implementation renders it into the system prompt so the
model answers with the facts already in view. *Yesterday: "I like football."
Today: "What sport should I try?" → football.*

**How updates work.** Overwrites, not appends. When a new statement supersedes an
old fact, `extract_memories()` returns a `Memory` carrying the **existing id and
`created_at`** with fresh `text` and `updated_at`. `upsert_memory()` keys on `id`,
so "I'm 26 now" replaces "is 25 years old" instead of sitting next to it. The
memory panel marks these rows "updated 2 days ago" rather than "added".

**Where it lives.** Behind `load_memories` / `upsert_memory` / `delete_memory`.
The stub keeps a list in the module; a real implementation writes to the memory
library's own store (or SQLite/JSON) so the facts survive a restart.

## Memory tooling surveyed

| Tool | Model | Fits when |
| --- | --- | --- |
| **Mem0** | LLM extracts facts, embeds them, dedupes against existing entries with add/update/delete decisions | Default choice — the add/update semantics map straight onto `upsert_memory` |
| **Zep** | Temporal knowledge graph; facts carry validity intervals and can be invalidated rather than overwritten | You need "was true then, is true now" history, not just latest value |
| **OpenMemory** | Self-hosted, local-first store with a shared memory API across apps | Nothing may leave the machine |
| **LangMem** | Memory primitives (semantic / episodic / procedural) for LangGraph agents | Already building on LangGraph |
| **Letta** | Agent runtime with self-editing memory blocks in the context window | The agent should manage its own memory budget |
| **Cognee** | Builds a graph + vector index over ingested data | Memory is a document corpus, not a fact list |

The frontend is agnostic — swapping between them is a rewrite of six function
bodies in `backend.py` and nothing else.

## Testing checklist

- Facts survive a restart: state a fact, restart the app, ask about it.
- Overwrite: "I'm 25" → "I'm 26 now" leaves **one** Identity row, marked updated.
- First run with an empty store renders the empty state, not an error.
- A raised `StorageError` shows a banner, not a traceback.
- The chat input is locked for the whole duration of a stream.
