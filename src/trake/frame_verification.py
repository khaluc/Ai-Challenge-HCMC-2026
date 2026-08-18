from __future__ import annotations

from typing import Mapping, Sequence

from vlm.frame_verifier import FrameEventScorerProtocol

from .frame_verification_schemas import (
    OriginalEventFrame,
    TRAKEVerificationResult,
    VerificationCandidate,
    VerifiedEvent,
)


class TRAKEVerifier:
    """VLM verification among nearby candidate frames, then a final
    monotonic-order safety check across the whole event sequence.

    If verification would break f1 < f2 < ... < fn, the whole sequence
    reverts to the pre-verification (coarse/fine) assignments rather than
    submit an invalid TRAKE answer.
    """

    def __init__(self, scorer: FrameEventScorerProtocol) -> None:
        self.scorer = scorer

    def _best_among(
        self,
        video_id: str,
        original: OriginalEventFrame,
        candidates: Sequence[VerificationCandidate],
    ) -> VerifiedEvent:
        scored = [(candidate, self.scorer.score(candidate.image_bytes, original.event_text)) for candidate in candidates]
        scored.sort(key=lambda item: -item[1].confidence)
        best_candidate, best_score = scored[0]
        return VerifiedEvent(
            event_index=original.event_index,
            event_text=original.event_text,
            video_id=video_id,
            frame_id=best_candidate.frame_id,
            timestamp=best_candidate.timestamp,
            confidence=best_score.confidence,
            reason=best_score.reason,
            candidates_considered=len(candidates),
            verified=True,
        )

    @staticmethod
    def _as_original(video_id: str, original: OriginalEventFrame, candidates_considered: int, reason: str) -> VerifiedEvent:
        return VerifiedEvent(
            event_index=original.event_index,
            event_text=original.event_text,
            video_id=video_id,
            frame_id=original.frame_id,
            timestamp=original.timestamp,
            confidence=0.0,
            reason=reason,
            candidates_considered=candidates_considered,
            verified=False,
        )

    @staticmethod
    def is_monotonic(events: Sequence[VerifiedEvent]) -> bool:
        timestamps = [event.timestamp for event in events]
        return all(a < b for a, b in zip(timestamps, timestamps[1:]))

    def verify_sequence(
        self,
        video_id: str,
        originals: Sequence[OriginalEventFrame],
        candidates_by_event: Mapping[int, Sequence[VerificationCandidate]],
    ) -> TRAKEVerificationResult:
        if not video_id.strip():
            raise ValueError("video_id must not be blank")
        if not originals:
            raise ValueError("originals must not be empty")

        verified_events: list[VerifiedEvent] = []
        for original in originals:
            candidates = candidates_by_event.get(original.event_index, ())
            if not candidates:
                verified_events.append(
                    self._as_original(
                        video_id, original, 0, "no verification candidates given; kept original"
                    )
                )
                continue
            verified_events.append(self._best_among(video_id, original, candidates))

        if self.is_monotonic(verified_events):
            return TRAKEVerificationResult(video_id=video_id, events=tuple(verified_events), monotonic=True)

        reverted_events = tuple(
            self._as_original(
                video_id,
                original,
                len(candidates_by_event.get(original.event_index, ())),
                "verification broke monotonic order; reverted to original assignment",
            )
            for original in originals
        )
        return TRAKEVerificationResult(
            video_id=video_id,
            events=reverted_events,
            monotonic=self.is_monotonic(reverted_events),
        )


__all__ = ["TRAKEVerifier"]
