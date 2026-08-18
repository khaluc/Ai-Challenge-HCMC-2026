from __future__ import annotations

import pytest

from vlm.frame_verifier import FrameEventScore
from trake.frame_verification_schemas import OriginalEventFrame, VerificationCandidate
from trake.frame_verification import TRAKEVerifier


class _FakeScorer:
    def __init__(self, scores: dict[tuple[int, str], FrameEventScore]) -> None:
        self._scores = scores

    def score(self, image_bytes: bytes, event_text: str) -> FrameEventScore:
        frame_id = int(image_bytes.decode("utf-8"))
        return self._scores[(frame_id, event_text)]


def test_verifier_picks_best_candidate_and_stays_monotonic() -> None:
    originals = [
        OriginalEventFrame(0, "e0", frame_id=100, timestamp=10.0),
        OriginalEventFrame(1, "e1", frame_id=200, timestamp=20.0),
    ]
    candidates_by_event = {
        0: [
            VerificationCandidate(99, 9.5, b"99"),
            VerificationCandidate(100, 10.0, b"100"),
            VerificationCandidate(101, 10.5, b"101"),
        ],
        1: [
            VerificationCandidate(199, 19.5, b"199"),
            VerificationCandidate(200, 20.0, b"200"),
            VerificationCandidate(201, 20.5, b"201"),
        ],
    }
    scores = {
        (99, "e0"): FrameEventScore(False, 0.2, "early"),
        (100, "e0"): FrameEventScore(True, 0.5, "ok"),
        (101, "e0"): FrameEventScore(True, 0.9, "best"),
        (199, "e1"): FrameEventScore(True, 0.95, "best"),
        (200, "e1"): FrameEventScore(True, 0.6, "ok"),
        (201, "e1"): FrameEventScore(False, 0.1, "late"),
    }
    verifier = TRAKEVerifier(_FakeScorer(scores))

    result = verifier.verify_sequence("V1", originals, candidates_by_event)

    assert result.monotonic
    assert result.events[0].frame_id == 101
    assert result.events[1].frame_id == 199
    assert all(event.verified for event in result.events)


def test_verifier_reverts_to_original_when_verification_breaks_order() -> None:
    originals = [
        OriginalEventFrame(0, "e0", frame_id=100, timestamp=10.0),
        OriginalEventFrame(1, "e1", frame_id=200, timestamp=20.0),
    ]
    candidates_by_event = {
        0: [VerificationCandidate(150, 15.0, b"150")],
        1: [VerificationCandidate(50, 5.0, b"50")],  # would land BEFORE event 0
    }
    scores = {
        (150, "e0"): FrameEventScore(True, 0.9, "best"),
        (50, "e1"): FrameEventScore(True, 0.9, "best"),
    }
    verifier = TRAKEVerifier(_FakeScorer(scores))

    result = verifier.verify_sequence("V1", originals, candidates_by_event)

    assert result.monotonic  # reverting to originals (10.0 < 20.0) restores order
    assert result.events[0].frame_id == 100
    assert result.events[1].frame_id == 200
    assert all(not event.verified for event in result.events)


def test_verifier_keeps_original_when_no_candidates_given() -> None:
    originals = [OriginalEventFrame(0, "e0", frame_id=100, timestamp=10.0)]
    verifier = TRAKEVerifier(_FakeScorer({}))

    result = verifier.verify_sequence("V1", originals, {})

    assert result.events[0].frame_id == 100
    assert result.events[0].verified is False
    assert result.monotonic


def test_verifier_rejects_empty_originals() -> None:
    verifier = TRAKEVerifier(_FakeScorer({}))
    with pytest.raises(ValueError):
        verifier.verify_sequence("V1", [], {})


def test_verifier_rejects_blank_video_id() -> None:
    verifier = TRAKEVerifier(_FakeScorer({}))
    with pytest.raises(ValueError):
        verifier.verify_sequence("  ", [OriginalEventFrame(0, "e0", 1, 1.0)], {})
