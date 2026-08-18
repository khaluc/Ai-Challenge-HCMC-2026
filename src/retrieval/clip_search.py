from __future__ import annotations

from typing import Protocol, Sequence

import numpy as np

from .clip_store import Phase1ClipStore
from .schemas import Query, RetrievalHit


class TextEncoderProtocol(Protocol):
    """Minimal interface needed by the textual KIS baseline."""

    def encode(self, texts: list[str]) -> np.ndarray:
        """Encode texts into one CLIP vector per input string."""


# Short alias for callers that prefer the model-agnostic name.
TextEncoder = TextEncoderProtocol


class TextualKIS:
    """End-to-end text -> CLIP -> FAISS -> BTC frame-id baseline."""

    def __init__(self, store: Phase1ClipStore, encoder: TextEncoderProtocol) -> None:
        self.store = store
        self.encoder = encoder

    def search(
        self,
        query: Query | str,
        top_k: int = 100,
        *,
        query_id: str | None = None,
    ) -> list[RetrievalHit]:
        """Retrieve one query and return 1-based, submission-ready hits.

        Passing a :class:`Query` is preferred. A plain string is accepted for
        interactive use and receives ``query_id='query'`` unless explicitly
        overridden.
        """

        if isinstance(query, str):
            normalized_query = Query(query_id=query_id or "query", text=query)
        elif isinstance(query, Query):
            if query_id is not None and query_id != query.query_id:
                raise ValueError(
                    "query_id cannot override the id of an existing Query object"
                )
            normalized_query = query
        else:
            raise TypeError("query must be a Query or string")
        return self.search_batch([normalized_query], top_k=top_k)

    def search_batch(
        self,
        queries: Sequence[Query],
        top_k: int = 100,
    ) -> list[RetrievalHit]:
        """Encode a query batch once and return a flat list of ranked hits.

        The flat representation is intentional: every hit carries ``query_id``
        and ``rank``, so it can be streamed directly to prediction/submission
        writers while preserving the input query order.
        """

        # Validate through the store before invoking a potentially expensive
        # encoder. A correctly shaped dummy is sufficient for top_k checking.
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)):
            raise TypeError("top_k must be an integer")
        top_k = int(top_k)
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_list = list(queries)
        for position, query in enumerate(query_list):
            if not isinstance(query, Query):
                raise TypeError(
                    f"queries[{position}] must be a Query, got {type(query).__name__}"
                )
        if not query_list:
            return []

        encoded = self.encoder.encode([query.text for query in query_list])
        vectors = np.asarray(encoded)
        if not np.issubdtype(vectors.dtype, np.number) or np.issubdtype(
            vectors.dtype, np.complexfloating
        ):
            raise ValueError("Text encoder must return real numeric vectors")
        expected_shape = (len(query_list), self.store.dimension)
        if vectors.ndim != 2 or vectors.shape != expected_shape:
            raise ValueError(
                "Text encoder output shape mismatch: "
                f"expected {expected_shape}, got {vectors.shape}"
            )

        results: list[RetrievalHit] = []
        for query, vector in zip(query_list, vectors):
            mapped_hits = self.store.search(vector, top_k=top_k)
            for rank, mapped in enumerate(mapped_hits, start=1):
                frame = mapped.frame
                results.append(
                    RetrievalHit(
                        query_id=query.query_id,
                        rank=rank,
                        video_id=frame.video_id,
                        # This is the exact submit frame id, not the keyframe
                        # index and not a timestamp-derived approximation.
                        frame_id=frame.video_frame_id,
                        score=mapped.score,
                        faiss_index=frame.faiss_index,
                        keyframe_index=frame.keyframe_index,
                        timestamp=frame.timestamp,
                        keyframe_path=frame.keyframe_path,
                        keyframe_available=frame.keyframe_available,
                    )
                )
        return results


__all__ = ["TextEncoder", "TextEncoderProtocol", "TextualKIS"]
