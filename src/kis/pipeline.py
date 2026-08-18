from __future__ import annotations

from pathlib import Path

from retrieval.clip_encoder import HFCLIPTextEncoder
from retrieval.clip_store import Phase1ClipStore
from retrieval.hybrid_search import HybridConfig, HybridSearchResult, HybridTextualKIS
from retrieval.hybrid_store import Phase1HybridStore
from retrieval.schemas import Query


class KISPipeline:
    """Loads the hybrid KIS engine once and exposes a plain search() call.

    Loading the CLIP checkpoint and FAISS index takes a few seconds, so the
    web app builds one instance at startup and reuses it across requests
    instead of re-running `kis.hybrid_cli`'s per-invocation `_load_engine`.
    """

    def __init__(
        self,
        *,
        artifacts_root: str | Path,
        model: str,
        cache_dir: str | Path,
        device: str = "cpu",
        allow_download: bool = False,
        batch_size: int = 32,
        config: HybridConfig | None = None,
    ) -> None:
        self.encoder = HFCLIPTextEncoder(
            model_id=model,
            cache_dir=cache_dir,
            device=device,
            local_files_only=not allow_download,
            batch_size=batch_size,
        )
        root = Path(artifacts_root)
        clip_store = Phase1ClipStore(
            root / "clip" / "faiss.index",
            root / "catalog" / "frame_mapping.parquet",
            expected_dimension=self.encoder.dimension,
            metadata_path=root / "clip" / "clip_index_meta.json",
        )
        self.store = Phase1HybridStore(root, clip_store)
        self.engine = HybridTextualKIS(self.store, self.encoder, config=config or HybridConfig())

    def search(
        self, query_text: str, *, query_id: str = "web", top_k: int = 20
    ) -> HybridSearchResult:
        query = Query(query_id, query_text)
        return self.engine.search_detailed_batch([query], top_k=top_k)[0]


__all__ = ["KISPipeline"]
