# Recall — a chatbot with persistent memory

A Streamlit assistant that remembers you between sessions. Facts you mention
once — your age, where you live, what you're into — are extracted by the model,
written to disk, and injected back into context on later turns. Close the app,
reopen it tomorrow, and it still knows.

![stack](https://img.shields.io/badge/Streamlit-1.40+-FF6A3D) ![llm](https://img.shields.io/badge/LLM-Groq%20%7C%20Gemini%20%7C%20OpenAI-444) ![store](https://img.shields.io/badge/Memory-SQLite%20%7C%20Mem0-444)

```bash
pip install -r requirements.txt
cp .env.example .env        # add your GROQ_API_KEY
streamlit run app.py
```

Without a key it still runs, in an obvious demo mode with canned replies, so the
UI is explorable before you sign up for anything.

---

## The scenario it was built for

> **Yesterday:** "I'm 25 and I like football"
> **Today:** "What sport should I try?" → *"You're a big football fan, so you
> might enjoy a similar team sport. Have you considered rugby?"*
> **Later:** "Actually I'm 26 now" → the stored age is **replaced**, not duplicated.

That is a real transcript from `tests/` plus the live check in
[Testing](#testing), not an illustration.

---

## Memory tooling surveyed

| Tool | Model | Fits when | Used here |
| --- | --- | --- | --- |
| **Mem0** | LLM extracts facts, embeds them, dedupes against existing entries with add/update/delete decisions | Default choice — its add/update semantics map straight onto a keyed upsert | ✅ adapter included (`MEMORY_BACKEND=mem0`) |
| **Zep** | Temporal knowledge graph; facts carry validity intervals and are invalidated rather than overwritten | You need "was true then, is true now" history | Surveyed |
| **OpenMemory** | Self-hosted, local-first store with a shared memory API across apps | Nothing may leave the machine | Surveyed |
| **LangMem** | Semantic / episodic / procedural memory primitives for LangGraph agents | Already building on LangGraph | Surveyed |
| **Letta** (MemGPT) | Agent runtime with self-editing memory blocks inside the context window | The agent should manage its own memory budget | Surveyed |
| **Cognee** | Builds a graph plus vector index over ingested data | Memory is a document corpus, not a fact list | Surveyed |

**Why the default is a built-in SQLite store rather than Mem0's OSS mode.**
Mem0 open-source needs a vector database plus an embedding model; with
`sentence-transformers` that is roughly 2 GB of PyTorch, which does not fit
Streamlit Community Cloud's resource limits. So this project implements Mem0's
*algorithm* — LLM-driven extract → compare against stored facts → ADD or UPDATE
— over plain SQLite, in ~80 lines and with no extra dependencies, and ships a
Mem0 adapter behind the same interface for anyone who wants the hosted platform.
Switching is one environment variable; no other code changes.

---

## The three tools

| Tool | File | Responsibility |
| --- | --- | --- |
| **MemoryHandler** | [`memory_handler.py`](memory_handler.py) | Decides what to store, whether it's an add or an update, and what to recall |
| **LLMConnector** | [`llm_connector.py`](llm_connector.py) | Streams from Groq / Gemini / OpenAI with memory injected into the system prompt |
| **StreamlitChatUI** | [`app.py`](app.py) + [`ui/`](ui/) | Conversation history, input box, memory panel |

Supporting: [`storage.py`](storage.py) (SQLite), [`models.py`](models.py)
(dataclasses), [`config.py`](config.py) (env resolution),
[`backend.py`](backend.py) (the single seam the UI imports).

```
     app.py  ──imports only──▶  backend.py
                                    │
                ┌───────────────────┼───────────────────┐
                ▼                   ▼                   ▼
         MemoryHandler        LLMConnector          SQLiteStore
                │                   │                   │
        ┌───────┴───────┐    ┌──────┼──────┐         recall.db
        ▼               ▼    ▼      ▼      ▼
   SQLiteMemory    Mem0Memory  Groq Gemini OpenAI
```

---

## Memory strategy

**What is stored.** Durable facts only, in six buckets: Identity, Preferences,
Location, Work, Interests, Goals. Written in the third person ("Plays football on
weekends") so they concatenate cleanly into a prompt. Questions, small talk,
hypotheticals and facts about other people are explicitly excluded — most turns
store nothing, which is correct.

**How it decides.** After each user turn, `MemoryHandler.observe()` sends the
model the new message *plus every fact already stored, with its id*, and asks for
operations:

```json
{"operations": [{"op": "UPDATE", "id": "a1b2c3", "text": "Is 26 years old",
                 "category": "Identity"}]}
```

Because the model sees the existing ids, it can choose `UPDATE` over `ADD`. The
handler then rewrites that row in place — same `id`, same `created_at`, new
`text` and `updated_at`. That is the entire mechanism behind "I'm 26 now"
replacing "Is 25 years old", and it's why the memory panel can label a row
"updated 2 days ago" instead of "added".

Extraction is best-effort: a malformed response, an unknown category, or an
`UPDATE` naming an id that no longer exists all degrade gracefully rather than
breaking the reply. Ten tests cover exactly those paths.

**How recall works.** Below the context cap (40 facts by default) every fact is
injected — simpler and more accurate than any ranking at that size. Above it,
facts are scored on keyword overlap plus recency and drawn **round-robin across
categories** rather than off one global list. That detail matters: lexical
overlap cannot connect "what sport should I try" to "Plays football on weekends",
so a global cut would silently drop the one fact the question was about.
Semantic matching would need embeddings — the upgrade path past a few hundred
facts.

**Where it lives.** `data/recall.db`, gitignored. Conversations, messages and
memories are all on disk; nothing important lives only in session state.

---

## Configuration

Resolution order: real environment variable → `.env` → `st.secrets`, so the same
code runs locally and on Streamlit Cloud. See [`.env.example`](.env.example).

| Variable | Default | Purpose |
| --- | --- | --- |
| `GROQ_API_KEY` | — | Enables Groq. Also `GEMINI_API_KEY`, `OPENAI_API_KEY` |
| `LLM_PROVIDER` | auto | Pin a provider instead of auto-detecting by key |
| `GROQ_MODEL` | `llama-3.3-70b-versatile` | Per-provider model override |
| `MEMORY_BACKEND` | `sqlite` | Or `mem0` with a `MEM0_API_KEY` |
| `RECALL_DB_PATH` | `data/recall.db` | Where the database lives |
| `USER_ID` | `local` | Namespaces memories, so one file serves several profiles |
| `MULTI_USER` | `0` | Give each browser session its own namespace (public deploys) |
| `MAX_MEMORIES_IN_CONTEXT` | `40` | Cap on facts injected per turn |

Providers are auto-detected by key presence in the order Groq → Gemini → OpenAI.
With none set, the app runs in demo mode and says so in the sidebar.

---

## Testing

```bash
pytest -q          # 26 tests, no network, no API key needed
```

The suite replaces the model with a fake whose JSON decisions are scripted, so
extraction logic is testable offline. It covers the brief's two required
behaviours directly:

- `test_memories_survive_a_restart` — a second store object on the same file
  (standing in for an app restart) reads back what the first one wrote.
- `test_update_replaces_the_old_fact` — asserts **one** row remains, `created_at`
  is preserved, `updated_at` is bumped, and the text now says 26.

Plus per-user isolation of both memories and conversations, schema migration of
a database predating namespacing, verbatim-duplicate rejection, recall ranking,
and six parametrised malformed-response cases.

Against the live API, the full scenario was verified end to end: two facts
extracted from one sentence with correct categories, survived a process restart,
the model referenced football when asked about sport, and "I'm 26 now" collapsed
the store back to a single age fact marked as revised.

---

## How the Streamlit reruns are handled

Streamlit re-executes `app.py` top to bottom on every interaction, so the UI
follows one rule: **a click writes an intent into `st.session_state` and calls
`st.rerun()`; the next pass reads that intent and renders.** No state a widget
has already consumed is mutated later in the same pass.

The pass order in `main()` is fixed:

1. `init_state()` — every key seeded behind a guard; reads go through `s(key)`,
   which falls back to a declared default, so a first run cannot `KeyError`.
2. `flush_toasts()` — toasts queued by the previous pass fire here.
3. `bootstrap()` — conversations load under a spinner, memories under a skeleton.
4. `consume_pending_prompt()` — the queued turn is folded into the conversation
   and passed to the memory handler *before* anything renders, so the sidebar,
   header and message list all agree.
5. Sidebar.
6. `st.chat_input`, painted before the stream blocks the script so its disabled
   state is visible while a reply generates.
7. Main area — history, then `st.write_stream`.
8. Persist the completed turn, then `st.rerun()`.

Two Streamlit specifics worth knowing if you edit the CSS: a bare `<div>` written
through `st.markdown` is auto-closed and wraps nothing, so wrapper divs cannot
scope styles — use keyed containers and their `st-key-*` class. And a button with
`help=` nests inside a tooltip wrapper, so `.stButton > button` misses it.

---

## Deployment

Live: **https://recall-memory-chatbot.streamlit.app**

Deploys from `main` on Streamlit Community Cloud. Add `GROQ_API_KEY` under app
settings → Secrets to take the hosted instance out of demo mode.

Set `MULTI_USER=1` there too. Without it every visitor shares one namespace and
would read each other's conversations and memories; with it, each browser session
gets its own isolated store. Conversations and memories are both namespaced, and
the delete paths are scoped as well, so one visitor cannot remove another's data.

One caveat that remains: **the container's disk is ephemeral.** A Streamlit Cloud
rebuild wipes `data/recall.db`. Persistence across *restarts* works; persistence
across *redeploys* needs a hosted store — point `MEMORY_BACKEND=mem0` at the
platform, or move the database to managed Postgres. And in multi-user mode a
visitor's namespace lives in session state, so their memory lasts as long as
their tab.

Neither affects local single-user use, which is what the project is built for:
there `USER_ID` stays `local` and memories persist indefinitely.
