from __future__ import annotations

import json
import zipfile
from dataclasses import dataclass
from pathlib import Path

import pytest

from qna.candidates import CandidateConfig, QACandidateSearch, narrow_candidates
from qna.images import load_keyframe_bytes
from qna.io import write_qa_results
from qna.pipeline import KISVideoQA
from qna.routing import is_speech_question
from qna.schemas import QACandidate, QAResult, VLMAnswer
from vlm.image_qa import QwenVLAnswerer


@dataclass(frozen=True)
class _FakeHit:
    video_id: str
    frame_id: int
    score: float
    rank: int
    faiss_index: int
    keyframe_index: int
    timestamp: float
    keyframe_path: str
    keyframe_available: bool


def _hit(video_id: str, frame_id: int, rank: int, *, available: bool = True) -> _FakeHit:
    return _FakeHit(
        video_id=video_id,
        frame_id=frame_id,
        score=1.0 / rank,
        rank=rank,
        faiss_index=rank,
        keyframe_index=rank,
        timestamp=float(rank),
        keyframe_path=f"Keyframes_L21.zip::keyframes/{video_id}/{rank:03d}.jpg",
        keyframe_available=available,
    )


def test_narrow_candidates_keeps_top_videos_then_top_frames() -> None:
    hits = [
        _hit("L21_V001", 10, 1),
        _hit("L21_V002", 20, 2),
        _hit("L21_V001", 11, 3),
        _hit("L21_V003", 30, 4),  # 3rd distinct video, video_limit=2 excludes it
        _hit("L21_V002", 21, 5),
    ]
    config = CandidateConfig(search_top_k=100, video_limit=2, frame_limit=3)

    candidates = narrow_candidates(hits, config)

    assert [c.video_id for c in candidates] == ["L21_V001", "L21_V002", "L21_V001"]
    assert all(c.video_id != "L21_V003" for c in candidates)
    assert len(candidates) == 3  # frame_limit caps it even though 4 hits matched


def test_narrow_candidates_skips_frames_without_keyframe_image() -> None:
    hits = [
        _hit("L21_V001", 10, 1, available=False),
        _hit("L21_V001", 11, 2, available=True),
    ]
    config = CandidateConfig(video_limit=5, frame_limit=5)

    candidates = narrow_candidates(hits, config)

    assert [c.frame_id for c in candidates] == [11]


class _FakeSearcher:
    def __init__(self, hits: list[_FakeHit]) -> None:
        self._hits = hits
        self.last_call: dict | None = None

    def search(self, query, *, top_k=100, query_id=None):
        self.last_call = {"query": query, "top_k": top_k, "query_id": query_id}
        return self._hits


def test_qa_candidate_search_delegates_to_searcher() -> None:
    hits = [_hit("L21_V001", 10, 1)]
    searcher = _FakeSearcher(hits)
    search = QACandidateSearch(searcher, config=CandidateConfig(search_top_k=50, video_limit=5, frame_limit=5))

    candidates = search.find_candidates("một người", query_id="q1")

    assert searcher.last_call == {"query": "một người", "top_k": 50, "query_id": "q1"}
    assert [c.video_id for c in candidates] == ["L21_V001"]


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Diễn giả nhắc đến công nghệ nào?", True),
        ("What did the speaker mention about AI?", True),
        ("Người phụ nữ đang cầm vật gì?", False),
        ("What color is the car?", False),
    ],
)
def test_is_speech_question(question: str, expected: bool) -> None:
    assert is_speech_question(question) is expected


class _FakeVLM:
    def __init__(self, answers: dict[tuple[str, int], VLMAnswer]) -> None:
        self._answers = answers
        self.calls: list[tuple[bytes, str, str]] = []

    def answer(self, image_bytes: bytes, question: str, *, question_type: str = "open_ended") -> VLMAnswer:
        self.calls.append((image_bytes, question, question_type))
        key = json.loads(image_bytes.decode("utf-8"))
        return self._answers[(key["video_id"], key["frame_id"])]


class _FakeCandidateSearch:
    def __init__(self, candidates: list[QACandidate]) -> None:
        self._candidates = candidates
        self.last_query: str | None = None

    def find_candidates(self, question: str, *, query_id: str | None = None) -> list[QACandidate]:
        self.last_query = question
        return self._candidates


def _candidate(video_id: str, frame_id: int, rank: int) -> QACandidate:
    return QACandidate(
        video_id=video_id,
        frame_id=frame_id,
        keyframe_path=f"fake::{video_id}/{frame_id}",
        keyframe_available=True,
        faiss_index=rank,
        keyframe_index=rank,
        timestamp=float(rank),
        retrieval_rank=rank,
        retrieval_score=1.0 / rank,
    )


def _patch_fake_keyframe_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "qna.pipeline.load_keyframe_bytes",
        lambda keyframe_path, *, data_root: json.dumps(
            {"video_id": keyframe_path.split("::")[1].split("/")[0], "frame_id": int(keyframe_path.split("/")[1])}
        ).encode("utf-8"),
    )


def test_kis_video_qa_reranks_by_vlm_confidence_when_retrieval_ties(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Both candidates share retrieval_rank=1 -> equal retrieval_score, so
    # video_score/frame_score both normalize to 1.0 and joint_confidence
    # collapses to the VLM's own confidence: retrieval evidence is a tie, so
    # confidence alone should decide.
    candidates = [_candidate("L21_V001", 10, 1), _candidate("L21_V002", 20, 1)]
    _patch_fake_keyframe_loader(monkeypatch)
    vlm = _FakeVLM(
        {
            ("L21_V001", 10): VLMAnswer(answer="trophy", confidence=0.4),
            ("L21_V002", 20): VLMAnswer(answer="cup", confidence=0.9),
        }
    )
    qa = KISVideoQA(_FakeCandidateSearch(candidates), vlm, data_root=".")

    results = qa.ask("Người phụ nữ đang cầm vật gì?", query_id="q1", top_k=2)

    assert [r.video_id for r in results] == ["L21_V002", "L21_V001"]
    assert results[0].answer == "cup"
    assert results[0].confidence == pytest.approx(0.9)
    assert results[0].joint_confidence == pytest.approx(0.9)
    assert results[0].video_score == pytest.approx(1.0)
    assert results[0].frame_score == pytest.approx(1.0)
    assert results[0].question_type == "open_ended"
    # Regression: QAResult must carry the candidate's timestamp/faiss/keyframe
    # index forward, or the web UI can't seek the video to the right frame.
    assert results[0].timestamp == pytest.approx(1.0)
    assert results[0].faiss_index == 1
    assert results[0].keyframe_index == 1
    assert results[0].keyframe_available is True
    assert results[0].rank == 1
    assert results[1].rank == 2
    assert all(r.route == "visual" for r in results)
    assert len(vlm.calls) == 2
    assert all(call[2] == "open_ended" for call in vlm.calls)


def test_kis_video_qa_joint_confidence_weighs_retrieval_score(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    # L21_V001 outranks L21_V002 on retrieval (rank 1 vs 2, score 1.0 vs 0.5)
    # even though the VLM is more confident about L21_V002 (0.9 vs 0.4) —
    # the joint product should let retrieval evidence flip the pure-VLM order.
    candidates = [_candidate("L21_V001", 10, 1), _candidate("L21_V002", 20, 2)]
    _patch_fake_keyframe_loader(monkeypatch)
    vlm = _FakeVLM(
        {
            ("L21_V001", 10): VLMAnswer(answer="trophy", confidence=0.4),
            ("L21_V002", 20): VLMAnswer(answer="cup", confidence=0.9),
        }
    )
    qa = KISVideoQA(_FakeCandidateSearch(candidates), vlm, data_root=".")

    results = qa.ask("Người phụ nữ đang cầm vật gì?", query_id="q1", top_k=2)

    assert [r.video_id for r in results] == ["L21_V001", "L21_V002"]
    assert results[0].video_score == pytest.approx(1.0)
    assert results[0].frame_score == pytest.approx(1.0)
    assert results[0].joint_confidence == pytest.approx(0.4)
    assert results[1].video_score == pytest.approx(0.5)
    assert results[1].frame_score == pytest.approx(0.5)
    assert results[1].joint_confidence == pytest.approx(0.225)


def test_kis_video_qa_uses_question_splitter_for_retrieval_only(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from qna.query_split import QuerySplit

    original_question = "Có bao nhiêu người lên sân khấu?"

    class _FakeSplitter:
        def split(self, text: str) -> QuerySplit:
            return QuerySplit(
                original=text,
                scene_description="an awards ceremony with people on stage",
                question=text,
            )

    candidates = [_candidate("L21_V001", 10, 1)]
    search = _FakeCandidateSearch(candidates)
    _patch_fake_keyframe_loader(monkeypatch)
    vlm = _FakeVLM({("L21_V001", 10): VLMAnswer(answer="two", confidence=0.8)})
    qa = KISVideoQA(search, vlm, data_root=".", question_splitter=_FakeSplitter())

    results = qa.ask(original_question, query_id="q1", top_k=1)

    # Retrieval saw the scene description, not the raw question...
    assert search.last_query == "an awards ceremony with people on stage"
    # ...while the VLM still saw the original question text.
    assert vlm.calls[0][1] == original_question
    assert results[0].scene_description == "an awards ceremony with people on stage"
    assert results[0].question_type == "closed_set"


def test_kis_video_qa_speech_question_without_transcript_backend_is_honest() -> None:
    qa = KISVideoQA(_FakeCandidateSearch([]), _FakeVLM({}), data_root=".")

    results = qa.ask("Diễn giả nhắc đến công nghệ nào?", query_id="q2")

    assert len(results) == 1
    assert results[0].route == "transcript"
    assert results[0].answer is None
    assert results[0].video_id is None
    assert "no transcript" in results[0].note.lower()


def test_kis_video_qa_no_candidates_returns_honest_result() -> None:
    qa = KISVideoQA(_FakeCandidateSearch([]), _FakeVLM({}), data_root=".")

    results = qa.ask("Người phụ nữ đang cầm vật gì?", query_id="q3")

    assert len(results) == 1
    assert results[0].answer is None
    assert results[0].video_id is None
    assert results[0].note is not None


def test_load_keyframe_bytes_reads_member_from_archive(tmp_path: Path) -> None:
    archive_path = tmp_path / "Keyframes_L21.zip"
    with zipfile.ZipFile(archive_path, "w") as archive:
        archive.writestr("keyframes/L21_V001/001.jpg", b"fake-jpeg-bytes")

    data = load_keyframe_bytes("Keyframes_L21.zip::keyframes/L21_V001/001.jpg", data_root=tmp_path)

    assert data == b"fake-jpeg-bytes"


def test_load_keyframe_bytes_rejects_unqualified_path(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        load_keyframe_bytes("keyframes/L21_V001/001.jpg", data_root=tmp_path)


def test_load_keyframe_bytes_missing_archive_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_keyframe_bytes("Missing.zip::a.jpg", data_root=tmp_path)


def test_write_qa_results_round_trips(tmp_path: Path) -> None:
    results = [
        QAResult(
            query_id="q1",
            question="Người phụ nữ đang cầm vật gì?",
            route="visual",
            rank=1,
            video_id="L21_V001",
            frame_id=30005,
            answer="trophy",
            confidence=0.91,
        )
    ]

    path = write_qa_results(tmp_path / "qa.csv", results)
    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == 2
    assert "trophy" in lines[1]


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


def test_qwen_vl_answerer_parses_structured_response() -> None:
    client = _FakeOpenAICompatibleClient(json.dumps({"answer": "trophy", "confidence": 0.91}))
    answerer = QwenVLAnswerer(client=client, model="qwen3.8-max")

    result = answerer.answer(b"fake-jpeg-bytes", "Người phụ nữ đang cầm vật gì?")

    assert result.answer == "trophy"
    assert result.confidence == pytest.approx(0.91)
    payload = client.chat.completions.last_kwargs
    assert payload["model"] == "qwen3.8-max"
    content = payload["messages"][1]["content"]
    assert content[0]["type"] == "image_url"
    assert content[0]["image_url"]["url"].startswith("data:image/jpeg;base64,")
    assert content[1] == {"type": "text", "text": "Người phụ nữ đang cầm vật gì?"}


def test_qwen_vl_answerer_raises_on_content_filter() -> None:
    client = _FakeOpenAICompatibleClient(None, finish_reason="content_filter")
    answerer = QwenVLAnswerer(client=client)
    with pytest.raises(RuntimeError):
        answerer.answer(b"fake-jpeg-bytes", "câu hỏi bất kỳ")


def test_qwen_vl_answerer_requires_api_key_without_injected_client() -> None:
    with pytest.raises(ValueError):
        QwenVLAnswerer(api_key=None)


def test_qwen_vl_answerer_uses_closed_set_prompt_for_closed_set_questions() -> None:
    from vlm.image_qa import CLOSED_SET_SYSTEM_PROMPT, OPEN_ENDED_SYSTEM_PROMPT

    client = _FakeOpenAICompatibleClient(json.dumps({"answer": "3", "confidence": 0.8}))
    answerer = QwenVLAnswerer(client=client)

    answerer.answer(b"fake-jpeg-bytes", "Có bao nhiêu người?", question_type="closed_set")
    assert client.chat.completions.last_kwargs["messages"][0]["content"] == CLOSED_SET_SYSTEM_PROMPT

    answerer.answer(b"fake-jpeg-bytes", "Người này đang làm gì?", question_type="open_ended")
    assert client.chat.completions.last_kwargs["messages"][0]["content"] == OPEN_ENDED_SYSTEM_PROMPT


def test_qwen_vl_answerer_rejects_unknown_question_type() -> None:
    client = _FakeOpenAICompatibleClient(json.dumps({"answer": "x", "confidence": 0.5}))
    answerer = QwenVLAnswerer(client=client)
    with pytest.raises(ValueError):
        answerer.answer(b"fake-jpeg-bytes", "câu hỏi", question_type="multiple_choice")


@pytest.mark.parametrize(
    "question,expected",
    [
        ("Có bao nhiêu người trên sân khấu?", "closed_set"),
        ("How many people are on stage?", "closed_set"),
        ("Chiếc áo đó màu gì?", "closed_set"),
        ("What color is the car?", "closed_set"),
        ("Có phải người đàn ông đang cầm micro không", "closed_set"),
        ("Is the man holding a microphone", "closed_set"),
        ("Người phụ nữ đang cầm vật gì?", "open_ended"),
        ("Why did the crowd start cheering?", "open_ended"),
    ],
)
def test_classify_question_type(question: str, expected: str) -> None:
    from qna.question_type import classify_question_type

    assert classify_question_type(question) == expected


def test_qwen_question_splitter_parses_structured_response() -> None:
    from qna.query_split import QwenQuestionSplitter

    client = _FakeOpenAICompatibleClient(
        json.dumps(
            {
                "scene_description": "an awards ceremony with people on stage",
                "question": "Có bao nhiêu người lên sân khấu nhận giải lớn nhất?",
            }
        )
    )
    splitter = QwenQuestionSplitter(client=client, model="qwen3.8-max")

    result = splitter.split("Trong video lễ trao giải, có bao nhiêu người lên sân khấu nhận giải lớn nhất?")

    assert result.scene_description == "an awards ceremony with people on stage"
    assert result.question == "Có bao nhiêu người lên sân khấu nhận giải lớn nhất?"
    assert result.original == "Trong video lễ trao giải, có bao nhiêu người lên sân khấu nhận giải lớn nhất?"


def test_qwen_question_splitter_falls_back_to_original_text_when_blank() -> None:
    from qna.query_split import QwenQuestionSplitter

    client = _FakeOpenAICompatibleClient(json.dumps({"scene_description": "", "question": ""}))
    splitter = QwenQuestionSplitter(client=client)

    result = splitter.split("một câu hỏi ngắn")

    assert result.scene_description == "một câu hỏi ngắn"
    assert result.question == "một câu hỏi ngắn"


class _FakeTextEmbedder:
    """Deterministic bag-of-words-ish embedder: exact text match -> identical
    vector, otherwise vectors share no dimensions -> similarity 0."""

    def embed_texts(self, texts):
        vocabulary = sorted({text.strip().casefold() for text in texts})
        return [
            [1.0 if text.strip().casefold() == word else 0.0 for word in vocabulary] for text in texts
        ]


def test_score_answers_matches_identical_text() -> None:
    from qna.answer_scoring import score_answers

    results = score_answers(
        [("trophy", "trophy"), ("cup", "trophy")],
        embedder=_FakeTextEmbedder(),
        threshold=0.85,
    )

    assert results[0].matched is True
    assert results[0].similarity == pytest.approx(1.0)
    assert results[1].matched is False
    assert results[1].similarity == pytest.approx(0.0)


def test_score_answers_rejects_blank_rows() -> None:
    from qna.answer_scoring import score_answers

    with pytest.raises(ValueError):
        score_answers([("", "trophy")], embedder=_FakeTextEmbedder())
