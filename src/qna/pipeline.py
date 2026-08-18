from __future__ import annotations

import numbers
from pathlib import Path

from .candidates import QACandidateSearch
from .images import load_keyframe_bytes
from .query_split import QuestionSplitterProtocol
from .question_type import classify_question_type
from .routing import TranscriptQAProtocol, is_speech_question
from .schemas import QACandidate, QAResult, VLMAnswer
from vlm.image_qa import VLMAnswererProtocol


def _normalize(values: list[float]) -> list[float]:
    """Scale to [0, 1] relative to the batch's best score (`value / max`), not
    min-max: min-max forces the single worst candidate in *any* batch to
    exactly 0.0 regardless of how close it actually was to the best one,
    which would zero out its joint confidence outright. Dividing by the max
    keeps every candidate's relative standing without that cliff."""
    if not values:
        return []
    best = max(values)
    if best <= 0:
        return [1.0 for _ in values]
    return [max(0.0, value) / best for value in values]


class KISVideoQA:
    """Question -> [optional scene/question split] -> candidate narrowing ->
    VLM per candidate -> joint-confidence rerank -> answers.

    Speech-content questions ("what did the speaker mention?") are routed to
    an optional TranscriptQAProtocol implementation instead of the VLM, since
    a VLM only sees pixels and cannot know what was said. Without a
    transcript backend configured, such a question gets an honest "not
    available" result instead of a hallucinated visual guess.
    """

    def __init__(
        self,
        candidate_search: QACandidateSearch,
        vlm: VLMAnswererProtocol,
        *,
        data_root: str | Path = ".",
        transcript_qa: TranscriptQAProtocol | None = None,
        question_splitter: QuestionSplitterProtocol | None = None,
    ) -> None:
        self.candidate_search = candidate_search
        self.vlm = vlm
        self.data_root = Path(data_root)
        self.transcript_qa = transcript_qa
        self.question_splitter = question_splitter

    def ask(self, question: str, *, query_id: str = "qa", top_k: int = 1) -> list[QAResult]:
        if not isinstance(question, str) or not question.strip():
            raise ValueError("question must be a nonblank string")
        if isinstance(top_k, bool) or not isinstance(top_k, numbers.Integral) or top_k < 1:
            raise ValueError("top_k must be a positive integer")

        if is_speech_question(question):
            return self._answer_speech(question, query_id=query_id)
        return self._answer_visual(question, query_id=query_id, top_k=int(top_k))

    def _answer_speech(self, question: str, *, query_id: str) -> list[QAResult]:
        if self.transcript_qa is None:
            return [
                QAResult(
                    query_id=query_id,
                    question=question,
                    route="transcript",
                    rank=1,
                    video_id=None,
                    frame_id=None,
                    answer=None,
                    confidence=0.0,
                    note=(
                        "Speech/ASR question detected but this dataset has no "
                        "transcript index (BTC provides no ASR for this stage). "
                        "Configure a TranscriptQAProtocol backend to answer it."
                    ),
                )
            ]
        return [self.transcript_qa.answer(question, query_id=query_id)]

    def _answer_visual(self, question: str, *, query_id: str, top_k: int) -> list[QAResult]:
        scene_description = question
        vlm_question = question
        if self.question_splitter is not None:
            split = self.question_splitter.split(question)
            scene_description = split.scene_description
            vlm_question = split.question

        question_type = classify_question_type(question)

        candidates = self.candidate_search.find_candidates(scene_description, query_id=query_id)
        if not candidates:
            return [
                QAResult(
                    query_id=query_id,
                    question=question,
                    route="visual",
                    rank=1,
                    video_id=None,
                    frame_id=None,
                    answer=None,
                    confidence=0.0,
                    note="No candidate keyframes with available images were retrieved for this question.",
                    scene_description=scene_description,
                    question_type=question_type,
                )
            ]

        # Joint confidence = P(video) * P(frame) * P(answer): retrieval score
        # is a frame-level number, so P(video) reuses each video's single
        # best frame score (how strongly the video as a whole matched the
        # scene description) and P(frame) normalizes each candidate's own
        # score within the batch (how distinctive this exact frame is).
        video_best_score: dict[str, float] = {}
        for candidate in candidates:
            current = video_best_score.get(candidate.video_id)
            if current is None or candidate.retrieval_score > current:
                video_best_score[candidate.video_id] = candidate.retrieval_score
        video_ids = list(video_best_score.keys())
        video_score_norm = dict(zip(video_ids, _normalize([video_best_score[v] for v in video_ids])))
        frame_score_norm = _normalize([candidate.retrieval_score for candidate in candidates])

        scored: list[tuple[QACandidate, VLMAnswer, float, float, float]] = []
        for candidate, frame_score in zip(candidates, frame_score_norm):
            image_bytes = load_keyframe_bytes(candidate.keyframe_path, data_root=self.data_root)
            vlm_answer = self.vlm.answer(image_bytes, vlm_question, question_type=question_type)
            video_score = video_score_norm[candidate.video_id]
            joint_confidence = video_score * frame_score * vlm_answer.confidence
            scored.append((candidate, vlm_answer, video_score, frame_score, joint_confidence))

        scored.sort(key=lambda item: -item[4])
        return [
            QAResult(
                query_id=query_id,
                question=question,
                route="visual",
                rank=rank,
                video_id=candidate.video_id,
                frame_id=candidate.frame_id,
                answer=vlm_answer.answer,
                confidence=vlm_answer.confidence,
                scene_description=scene_description,
                question_type=question_type,
                video_score=video_score,
                frame_score=frame_score,
                joint_confidence=joint_confidence,
                timestamp=candidate.timestamp,
                faiss_index=candidate.faiss_index,
                keyframe_index=candidate.keyframe_index,
                keyframe_available=candidate.keyframe_available,
            )
            for rank, (candidate, vlm_answer, video_score, frame_score, joint_confidence) in enumerate(
                scored[:top_k], start=1
            )
        ]


__all__ = ["KISVideoQA"]
