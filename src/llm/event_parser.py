from __future__ import annotations

import json
import os
from typing import Any, Protocol

from .event_schemas import EventSequence


class EventDecompositionProtocol(Protocol):
    def decompose(self, query: str) -> EventSequence:
        """Break a TRAKE query into its ordered sequence of visual events."""


EVENT_RESPONSE_JSON_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "events": {"type": "array", "items": {"type": "string"}},
        "expansions": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["events", "expansions"],
    "additionalProperties": False,
}

EVENT_SYSTEM_PROMPT = (
    "You decompose a TRAKE query (Vietnamese or English) describing an "
    "ordered sequence of visual events in one video. Return a JSON object "
    'only: {"events": [string, ...], "expansions": [string, ...]}. '
    '"events" is the ordered list of individual events the query describes, '
    "one short self-contained English sentence per event describing what is "
    'visually happening at that moment (e.g. "athlete running toward the '
    'bar"), in the same order they occur in the source query — whether the '
    "query numbers them explicitly (E1, E2, ...) or just narrates them as "
    "connected prose. Never merge two distinct events into one sentence, "
    "never invent an event the query does not imply, never add commentary "
    "or explanations. If the query only describes a single moment with no "
    'sequence, return that single moment as the only item in "events". '
    '"expansions" is 1 to 3 alternate short English phrasings of the whole '
    "query as a single sentence (the full sequence, not one event) suitable "
    "for a CLIP-style whole-clip search; do not duplicate an item already "
    'in "events".'
)


class QwenEventDecomposer:
    """Event decomposition via a Qwen model on an OpenAI-compatible endpoint
    (Alibaba Cloud Model Studio / DashScope by default).

    Requires the optional `openai` package and API credentials; a `client`
    can be injected for testing without any network access or dependency.
    """

    DEFAULT_BASE_URL = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"

    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "qwen3.8-max",
        base_url: str | None = None,
        api_key: str | None = None,
        max_tokens: int = 1024,
    ) -> None:
        if client is None:
            import openai  # local import: optional dependency

            resolved_key = (
                api_key or os.environ.get("DASHSCOPE_API_KEY") or os.environ.get("QWEN_API_KEY")
            )
            if not resolved_key:
                raise ValueError(
                    "No Qwen/DashScope API key found: pass api_key= or set "
                    "DASHSCOPE_API_KEY (or QWEN_API_KEY) in the environment"
                )
            client = openai.OpenAI(
                api_key=resolved_key,
                base_url=base_url or os.environ.get("DASHSCOPE_BASE_URL") or self.DEFAULT_BASE_URL,
            )
        self._client = client
        self._model = model
        self._max_tokens = max_tokens

    def decompose(self, query: str) -> EventSequence:
        if not isinstance(query, str) or not query.strip():
            raise ValueError("query must be a nonblank string")
        response = self._client.chat.completions.create(
            model=self._model,
            max_tokens=self._max_tokens,
            response_format={"type": "json_object"},
            # Reasoning models can spend the whole max_tokens budget on
            # hidden chain-of-thought and return empty content otherwise —
            # not needed for this short structured-output task.
            extra_body={"enable_thinking": False},
            messages=[
                {"role": "system", "content": EVENT_SYSTEM_PROMPT},
                {"role": "user", "content": query},
            ],
        )
        choice = response.choices[0]
        finish_reason = getattr(choice, "finish_reason", None)
        if finish_reason == "content_filter":
            raise RuntimeError(f"Qwen declined event decomposition for query={query!r}")
        payload_text = choice.message.content
        if not payload_text or not payload_text.strip():
            raise RuntimeError(f"Qwen returned an empty response for query={query!r}")
        payload = json.loads(payload_text)

        events = tuple(
            dict.fromkeys(str(value).strip() for value in payload.get("events", ()) if str(value).strip())
        )
        if not events:
            # Abstain to the whole query as a single event rather than crash
            # on a malformed/empty LLM response.
            events = (query,)

        seen = {query.strip().casefold()} | {event.casefold() for event in events}
        expansions: list[str] = []
        for value in payload.get("expansions", ()):
            text = str(value).strip()
            key = text.casefold()
            if not text or key in seen:
                continue
            seen.add(key)
            expansions.append(text)

        return EventSequence(query=query, events=events, expansions=tuple(expansions))


__all__ = [
    "EVENT_RESPONSE_JSON_SCHEMA",
    "EVENT_SYSTEM_PROMPT",
    "EventDecompositionProtocol",
    "QwenEventDecomposer",
]
