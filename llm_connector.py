"""LLMConnector — sends queries to Groq / Gemini / OpenAI with memory injected.

One class, three provider adapters, and a keyless fallback. Provider SDKs are
imported lazily inside their adapter so the app starts with only the one you
actually installed.

Two capabilities are needed by the rest of the app:

    stream_chat(messages, memories) -> Iterator[str]   the visible reply
    complete_json(system, user)     -> dict            structured extraction
"""

from __future__ import annotations

import json
import random
import re
import time
from typing import Iterator, Protocol

from config import Settings, get_settings
from models import LLMError, Memory

SYSTEM_PROMPT = """You are Recall, a personal assistant with long-term memory of \
this specific user.

{memory_block}

How to use what you know:
- Weave relevant facts in naturally, as a friend who remembers would. Never say \
"according to my memory" or "based on your stored facts".
- Do not recite facts that are not relevant to the question.
- If the user contradicts something you know, trust what they just said.
- If you know nothing relevant, just answer normally. Do not invent facts about \
them.

Keep answers direct and concrete. Use markdown, and fenced code blocks with a \
language tag whenever you show code."""

NO_MEMORY_BLOCK = "You have not learned anything about this user yet."

EXTRACTION_SYSTEM = """You maintain a long-term memory store about a user. \
Given their latest message and the facts already stored, decide what should \
change.

Return ONLY a JSON object of this shape:
{{"operations": [{{"op": "ADD", "text": "...", "category": "..."}},
                 {{"op": "UPDATE", "id": "<existing id>", "text": "...", \
"category": "..."}}]}}

Rules:
- category must be exactly one of: {categories}
- ADD only durable facts about the user worth recalling weeks later: age, name, \
location, job, preferences, interests, relationships, goals.
- UPDATE when the new message supersedes a stored fact. Reuse that fact's exact \
id. "I'm 26 now" must UPDATE the stored age, never ADD a second one.
- Write facts in the third person, one fact per operation, under 100 characters. \
Example: "Is 26 years old", "Plays football on weekends".
- Ignore questions, small talk, hypotheticals and anything about other people.
- If nothing is worth storing, return {{"operations": []}}. This is the common case."""


class Provider(Protocol):
    """What every backend adapter must offer."""

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]: ...

    def json(self, system: str, user: str) -> str: ...


# --------------------------------------------------------------------------- #
# Adapters
# --------------------------------------------------------------------------- #


class GroqProvider:
    """Groq — the default. Fast, free tier, OpenAI-compatible chat API."""

    def __init__(self, settings: Settings) -> None:
        try:
            from groq import Groq
        except ImportError as exc:  # pragma: no cover
            raise LLMError("groq is not installed — run: pip install groq") from exc
        self._client = Groq(api_key=settings.api_key)
        self._model = settings.model

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, *messages],
            stream=True,
            temperature=0.6,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def json(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content or "{}"


class GeminiProvider:
    """Google Gemini via the google-genai SDK."""

    def __init__(self, settings: Settings) -> None:
        try:
            from google import genai
        except ImportError as exc:  # pragma: no cover
            raise LLMError("google-genai is not installed — run: pip install google-genai") from exc
        self._genai = genai
        self._client = genai.Client(api_key=settings.api_key)
        self._model = settings.model

    @staticmethod
    def _contents(messages: list[dict]) -> list[dict]:
        """Gemini calls the assistant role 'model' and nests text in parts."""
        return [
            {
                "role": "model" if m["role"] == "assistant" else "user",
                "parts": [{"text": m["content"]}],
            }
            for m in messages
        ]

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        stream = self._client.models.generate_content_stream(
            model=self._model,
            contents=self._contents(messages),
            config={"system_instruction": system, "temperature": 0.6},
        )
        for chunk in stream:
            if chunk.text:
                yield chunk.text

    def json(self, system: str, user: str) -> str:
        response = self._client.models.generate_content(
            model=self._model,
            contents=self._contents([{"role": "user", "content": user}]),
            config={
                "system_instruction": system,
                "response_mime_type": "application/json",
                "temperature": 0,
            },
        )
        return response.text or "{}"


class OpenAIProvider:
    """OpenAI chat completions."""

    def __init__(self, settings: Settings) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:  # pragma: no cover
            raise LLMError("openai is not installed — run: pip install openai") from exc
        self._client = OpenAI(api_key=settings.api_key)
        self._model = settings.model

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        stream = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "system", "content": system}, *messages],
            stream=True,
            temperature=0.6,
        )
        for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    def json(self, system: str, user: str) -> str:
        response = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            response_format={"type": "json_object"},
            temperature=0,
        )
        return response.choices[0].message.content or "{}"


class DemoProvider:
    """Keyless fallback so the UI is usable with nothing configured.

    Replies are canned and extraction is a small set of keyword rules. The
    sidebar shows "demo mode · no API key" whenever this is active, so nobody
    mistakes it for a real model.
    """

    _REPLIES: dict[tuple[str, ...], str] = {
        ("remember", "know about me", "my memories"): (
            "Here is what I have stored about you:\n\n{memory_lines}\n\n"
            "You can edit or delete any of it from the memory panel.\n\n"
            "*(Demo mode — no API key configured, so this reply is canned. "
            "Add a `GROQ_API_KEY` to talk to a real model.)*"
        ),
        ("code", "python", "function", "script"): (
            "```python\ndef merge(old: dict, new: dict) -> dict:\n"
            "    \"\"\"Last write wins, order preserved.\"\"\"\n"
            "    return {**old, **new}\n```\n\n"
            "*(Demo mode — no API key configured. Add a `GROQ_API_KEY` for real answers.)*"
        ),
    }

    _RULES: tuple[tuple[tuple[str, ...], str], ...] = (
        (("i am ", "i'm ", "my name", "call me"), "Identity"),
        (("i live", "i'm from", "i am from", "based in"), "Location"),
        (("i work", "my job", "my team"), "Work"),
        (("i like", "i love", "i enjoy", "favourite", "favorite"), "Interests"),
        (("i want", "i'd like to", "my goal", "trying to"), "Goals"),
        (("i prefer", "i hate", "i dislike", "don't like"), "Preferences"),
    )

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._memory_lines = "- Nothing yet."

    def set_memory_lines(self, lines: str) -> None:
        self._memory_lines = lines or "- Nothing yet."

    def stream(self, system: str, messages: list[dict]) -> Iterator[str]:
        last = next((m["content"] for m in reversed(messages) if m["role"] == "user"), "")
        lowered = last.lower()

        reply = None
        for triggers, template in self._REPLIES.items():
            if any(t in lowered for t in triggers):
                reply = template.format(memory_lines=self._memory_lines)
                break
        if reply is None:
            reply = (
                "I have noted that. Ask me what I remember about you, or add a "
                "`GROQ_API_KEY` to `.env` and restart to get real answers.\n\n"
                "*(Demo mode — no API key configured.)*"
            )

        for token in re.findall(r"\S+\s*", reply):
            time.sleep(random.uniform(0.010, 0.028))
            yield token

    def json(self, system: str, user: str) -> str:
        """Keyword extraction standing in for the model's JSON decision."""
        lowered = user.lower()
        # The user message is the last line of the extraction prompt payload.
        for triggers, category in self._RULES:
            if any(t in lowered for t in triggers):
                fact = user.strip().splitlines()[-1].strip()
                if len(fact) > 96:
                    fact = fact[:93].rsplit(" ", 1)[0] + "..."
                return json.dumps(
                    {"operations": [{"op": "ADD", "text": fact, "category": category}]}
                )
        return json.dumps({"operations": []})


PROVIDERS = {
    "groq": GroqProvider,
    "gemini": GeminiProvider,
    "openai": OpenAIProvider,
    "demo": DemoProvider,
}


# --------------------------------------------------------------------------- #
# Connector
# --------------------------------------------------------------------------- #


class LLMConnector:
    """Sends queries to the configured provider with memory context injected.

    Args:
        settings: Resolved configuration. Defaults to the process settings.
    """

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._provider: Provider | None = None

    @property
    def provider(self) -> Provider:
        """The active adapter, constructed on first use."""
        if self._provider is None:
            self._provider = PROVIDERS[self.settings.provider](self.settings)
        return self._provider

    # ------------------------------------------------------------------ prompt

    def build_system_prompt(self, memories: list[Memory]) -> str:
        """Render stored facts into the system prompt.

        This is the "inject into context" half of the memory strategy: facts are
        grouped by category so related ones read together, and capped so a large
        store cannot crowd out the conversation.

        Args:
            memories: Facts to inject, already filtered for relevance.

        Returns:
            The full system prompt.
        """
        if not memories:
            return SYSTEM_PROMPT.format(memory_block=NO_MEMORY_BLOCK)

        capped = memories[: self.settings.max_memories_in_context]
        grouped: dict[str, list[str]] = {}
        for memory in capped:
            grouped.setdefault(memory.category, []).append(memory.text)

        lines = ["What you know about this user:"]
        for category, facts in grouped.items():
            lines.append(f"\n{category}:")
            lines.extend(f"- {fact}" for fact in facts)

        return SYSTEM_PROMPT.format(memory_block="\n".join(lines))

    # ------------------------------------------------------------------- calls

    def stream_chat(self, messages: list[dict], memories: list[Memory]) -> Iterator[str]:
        """Stream a reply, with the user's facts in the system prompt.

        Args:
            messages: Turn history as ``{"role", "content"}`` dicts, oldest
                first, ending with the user turn to answer.
            memories: Facts to inject as context.

        Yields:
            Text chunks; concatenated they form the complete reply.

        Raises:
            LLMError: If the provider rejects the call.
        """
        system = self.build_system_prompt(memories)

        if isinstance(self.provider, DemoProvider):
            self.provider.set_memory_lines("\n".join(f"- {m.text}" for m in memories[:8]))

        try:
            yield from self.provider.stream(system, messages)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.settings.provider} call failed: {exc}") from exc

    def complete_json(self, system: str, user: str) -> dict:
        """Ask for a structured JSON answer and parse it defensively.

        Models occasionally wrap JSON in prose or a code fence even in JSON
        mode, so the first ``{...}`` block is extracted as a fallback rather
        than letting a stray character lose the whole extraction.

        Args:
            system: Instructions describing the JSON shape.
            user: The payload to reason about.

        Returns:
            The parsed object, or ``{}`` when nothing parseable came back.

        Raises:
            LLMError: If the provider rejects the call.
        """
        try:
            raw = self.provider.json(system, user)
        except LLMError:
            raise
        except Exception as exc:
            raise LLMError(f"{self.settings.provider} extraction failed: {exc}") from exc

        return parse_json_object(raw)


def parse_json_object(raw: str) -> dict:
    """Parse a JSON object out of a model response.

    Args:
        raw: Whatever the model returned.

    Returns:
        The parsed object, or ``{}`` if no object could be recovered.
    """
    if not raw:
        return {}
    try:
        parsed = json.loads(raw)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, re.DOTALL)
    if not match:
        return {}
    try:
        parsed = json.loads(match.group(0))
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        return {}
