from __future__ import annotations

import json

import pytest

from retrieval.processing import RuleBasedObjectParser
from retrieval.hybrid_schemas import HybridHit
from llm.expansion_fusion import fuse_expansions
from llm.expansion_schemas import QueryStructure, QueryUnderstanding
from llm.query_expansion import (
    AnthropicQueryUnderstanding,
    QwenQueryUnderstanding,
    RuleBasedQueryUnderstanding,
)


def _parser() -> RuleBasedObjectParser:
    return RuleBasedObjectParser(
        [
            "Person",
            "Man",
            "Woman",
            "Boy",
            "Girl",
            "Car",
            "Vehicle",
            "Dog",
            "Bowl",
            "Food",
            "Table",
            "Orange",
        ]
    )


def test_rule_based_understanding_splits_objects_attributes_and_relation() -> None:
    understanding = RuleBasedQueryUnderstanding(_parser(), max_expansions=4)
    text = "một người đàn ông mặc áo đỏ đứng cạnh một chiếc ô tô màu trắng"

    result = understanding.understand(text)

    assert result.structure.objects == ("person", "car")
    assert result.structure.attributes["person"] == ("red", "male")
    assert result.structure.attributes["car"] == ("white",)
    assert result.structure.relation == "next_to"
    assert result.expansions == (
        text,
        "red male person next to white car",
        "red male person beside white car",
        "red male person close to white car",
    )


def test_rule_based_understanding_abstains_when_no_objects_detected() -> None:
    understanding = RuleBasedQueryUnderstanding(_parser(), max_expansions=4)
    text = "60 Giây Sáng 01082024 HTV Tin Tức"

    result = understanding.understand(text)

    assert result.structure.objects == ()
    assert result.structure.relation is None
    assert result.expansions == (text,)


def test_query_structure_rejects_duplicate_objects() -> None:
    with pytest.raises(ValueError):
        QueryStructure(objects=("person", "person"))


def test_query_understanding_requires_at_least_one_expansion() -> None:
    with pytest.raises(ValueError):
        QueryUnderstanding(text="hello", structure=QueryStructure(), expansions=())


def _hit(query_id: str, rank: int, video_id: str, frame_id: int, faiss_index: int) -> HybridHit:
    return HybridHit(
        query_id=query_id,
        rank=rank,
        video_id=video_id,
        frame_id=frame_id,
        score=1.0 / rank,
        faiss_index=faiss_index,
        keyframe_index=faiss_index + 1,
        timestamp=float(faiss_index),
        keyframe_path=f"keyframes/{faiss_index:03d}.jpg",
        keyframe_available=True,
        fusion_method="rrf",
    )


def test_fuse_expansions_combines_rankings_by_submit_key() -> None:
    rankings = {
        "e0": [
            _hit("q::e0", 1, "L21_V001", 10, 0),
            _hit("q::e0", 2, "L21_V001", 20, 1),
        ],
        "e1": [
            _hit("q::e1", 1, "L21_V001", 20, 1),
            _hit("q::e1", 2, "L21_V002", 5, 2),
        ],
    }
    texts = {"e0": "text a", "e1": "text b"}

    fused = fuse_expansions(rankings, texts, rrf_k=60, limit=100)

    assert [(item.hit.video_id, item.hit.frame_id) for item in fused] == [
        ("L21_V001", 20),
        ("L21_V001", 10),
        ("L21_V002", 5),
    ]
    top = fused[0]
    assert len(top.evidence) == 2
    assert {evidence.expansion_id for evidence in top.evidence} == {"e0", "e1"}
    assert top.score == pytest.approx(1 / 62 + 1 / 61)


def test_fuse_expansions_rejects_sparse_ranks() -> None:
    with pytest.raises(ValueError):
        fuse_expansions(
            {"e0": [_hit("q::e0", 2, "L21_V001", 10, 0)]},
            {"e0": "text"},
        )


class _FakeTextBlock:
    def __init__(self, text: str) -> None:
        self.type = "text"
        self.text = text


class _FakeResponse:
    def __init__(self, payload: dict, stop_reason: str) -> None:
        self.content = [_FakeTextBlock(json.dumps(payload))]
        self.stop_reason = stop_reason


class _FakeMessages:
    def __init__(self, payload: dict, stop_reason: str) -> None:
        self._payload = payload
        self._stop_reason = stop_reason
        self.last_kwargs: dict | None = None

    def create(self, **kwargs):
        self.last_kwargs = kwargs
        return _FakeResponse(self._payload, self._stop_reason)


class _FakeAnthropicClient:
    def __init__(self, payload: dict, stop_reason: str = "end_turn") -> None:
        self.messages = _FakeMessages(payload, stop_reason)


def test_anthropic_understanding_parses_structured_response() -> None:
    payload = {
        "objects": ["person", "car"],
        "attributes": [
            {"concept": "person", "values": ["male", "red"]},
            {"concept": "car", "values": ["white"]},
        ],
        "relation": "next_to",
        "expansions": ["a man in red beside a white car", "person near white car"],
    }
    client = _FakeAnthropicClient(payload)
    adapter = AnthropicQueryUnderstanding(client=client, model="claude-opus-5", max_expansions=3)
    text = "một người đàn ông mặc áo đỏ đứng cạnh một chiếc ô tô màu trắng"

    result = adapter.understand(text)

    assert result.structure.objects == ("person", "car")
    assert result.structure.attributes["person"] == ("male", "red")
    assert result.structure.attributes["car"] == ("white",)
    assert result.structure.relation == "next_to"
    assert result.expansions[0] == text
    assert len(result.expansions) == 3
    assert client.messages.last_kwargs["model"] == "claude-opus-5"


def test_anthropic_understanding_raises_on_refusal() -> None:
    client = _FakeAnthropicClient({}, stop_reason="refusal")
    adapter = AnthropicQueryUnderstanding(client=client)
    with pytest.raises(RuntimeError):
        adapter.understand("bất kỳ câu hỏi nào")


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


def test_qwen_understanding_parses_structured_response() -> None:
    payload = {
        "objects": ["person", "car"],
        "attributes": [
            {"concept": "person", "values": ["male", "red"]},
            {"concept": "car", "values": ["white"]},
        ],
        "relation": "next_to",
        "expansions": ["a man in red beside a white car", "person near white car"],
    }
    client = _FakeOpenAICompatibleClient(json.dumps(payload))
    adapter = QwenQueryUnderstanding(client=client, model="qwen3.8-max", max_expansions=3)
    text = "một người đàn ông mặc áo đỏ đứng cạnh một chiếc ô tô màu trắng"

    result = adapter.understand(text)

    assert result.structure.objects == ("person", "car")
    assert result.structure.attributes["person"] == ("male", "red")
    assert result.structure.attributes["car"] == ("white",)
    assert result.structure.relation == "next_to"
    assert result.expansions[0] == text
    assert len(result.expansions) == 3
    assert client.chat.completions.last_kwargs["model"] == "qwen3.8-max"
    assert client.chat.completions.last_kwargs["response_format"] == {"type": "json_object"}


def test_qwen_understanding_raises_on_content_filter() -> None:
    client = _FakeOpenAICompatibleClient(None, finish_reason="content_filter")
    adapter = QwenQueryUnderstanding(client=client)
    with pytest.raises(RuntimeError):
        adapter.understand("bất kỳ câu hỏi nào")


def test_qwen_understanding_requires_api_key_without_injected_client() -> None:
    with pytest.raises(ValueError):
        QwenQueryUnderstanding(api_key=None)
