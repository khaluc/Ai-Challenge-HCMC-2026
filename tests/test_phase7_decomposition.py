from __future__ import annotations

import json

import pytest

from llm.event_parser import QwenEventDecomposer
from llm.event_schemas import EventSequence


def test_event_sequence_rejects_empty_events() -> None:
    with pytest.raises(ValueError):
        EventSequence(query="q", events=())


def test_event_sequence_rejects_duplicate_events() -> None:
    with pytest.raises(ValueError):
        EventSequence(query="q", events=("a", "a"))


def test_event_sequence_rejects_blank_query() -> None:
    with pytest.raises(ValueError):
        EventSequence(query="  ", events=("a",))


class _FakeChatMessage:
    def __init__(self, content: str | None) -> None:
        self.content = content


class _FakeChatChoice:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.message = _FakeChatMessage(content)
        self.finish_reason = finish_reason


class _FakeChatCompletion:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.choices = [_FakeChatChoice(content, finish_reason)]


class _FakeChatCompletions:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self._content = content
        self._finish_reason = finish_reason
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeChatCompletion(self._content, self._finish_reason)


class _FakeChat:
    def __init__(self, content: str | None, finish_reason: str) -> None:
        self.completions = _FakeChatCompletions(content, finish_reason)


class _FakeOpenAICompatibleClient:
    def __init__(self, content: str | None, finish_reason: str = "stop") -> None:
        self.chat = _FakeChat(content, finish_reason)


def test_qwen_event_decomposer_parses_ordered_events() -> None:
    payload = {
        "events": [
            "athlete running toward the bar",
            "athlete taking off",
            "athlete clearing the bar",
            "athlete landing on the mat",
        ],
        "expansions": ["a high jump attempt from run-up to landing"],
    }
    client = _FakeOpenAICompatibleClient(json.dumps(payload))
    decomposer = QwenEventDecomposer(client=client, model="qwen3.8-max")
    query = "E1: chạy đà E2: giậm nhảy E3: bay qua xà E4: tiếp đất"

    result = decomposer.decompose(query)

    assert result.query == query
    assert result.events == (
        "athlete running toward the bar",
        "athlete taking off",
        "athlete clearing the bar",
        "athlete landing on the mat",
    )
    assert result.expansions == ("a high jump attempt from run-up to landing",)
    assert client.chat.completions.last_kwargs["model"] == "qwen3.8-max"
    assert client.chat.completions.last_kwargs["response_format"] == {"type": "json_object"}


def test_qwen_event_decomposer_abstains_to_whole_query_on_empty_events() -> None:
    client = _FakeOpenAICompatibleClient(json.dumps({"events": [], "expansions": []}))
    decomposer = QwenEventDecomposer(client=client)

    result = decomposer.decompose("một câu hỏi bất kỳ")

    assert result.events == ("một câu hỏi bất kỳ",)
    assert result.expansions == ()


def test_qwen_event_decomposer_drops_expansions_duplicating_query_or_events() -> None:
    payload = {
        "events": ["event a"],
        "expansions": ["Event A", "the query", "a new phrasing"],
    }
    client = _FakeOpenAICompatibleClient(json.dumps(payload))
    decomposer = QwenEventDecomposer(client=client)

    result = decomposer.decompose("the query")

    assert result.expansions == ("a new phrasing",)


def test_qwen_event_decomposer_raises_on_content_filter() -> None:
    client = _FakeOpenAICompatibleClient(None, finish_reason="content_filter")
    decomposer = QwenEventDecomposer(client=client)
    with pytest.raises(RuntimeError):
        decomposer.decompose("bất kỳ câu hỏi nào")


def test_qwen_event_decomposer_requires_api_key_without_injected_client() -> None:
    with pytest.raises(ValueError):
        QwenEventDecomposer(api_key=None)
