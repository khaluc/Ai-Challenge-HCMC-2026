from __future__ import annotations

import os
import json
import numbers
from pathlib import Path
from typing import Sequence

import numpy as np
import torch


DEFAULT_MODEL_ID = "openai/clip-vit-base-patch32"
DEFAULT_MODEL_DIR = Path("models/openai-clip-vit-base-patch32")
DEFAULT_CACHE_DIR = Path("models/huggingface")
DEFAULT_MODEL_REVISION = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"


def resolve_local_hf_commit(model_source: Path | str) -> str | None:
    metadata = (
        Path(model_source)
        / ".cache"
        / "huggingface"
        / "download"
        / "config.json.metadata"
    )
    if not metadata.is_file():
        return None
    try:
        first_line = metadata.read_text(encoding="utf-8").splitlines()[0].strip()
    except (OSError, IndexError):
        return None
    return first_line or None


def read_local_model_manifest(model_source: Path | str) -> dict[str, object] | None:
    path = Path(model_source) / "MODEL_MANIFEST.json"
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) else None


class EncoderLoadError(RuntimeError):
    """Raised when the configured CLIP checkpoint is unavailable or incompatible."""


class HFCLIPTextEncoder:
    """OpenAI CLIP text tower with its retrieval projection and L2 normalization."""

    def __init__(
        self,
        model_id: str = str(DEFAULT_MODEL_DIR),
        *,
        cache_dir: Path | str = DEFAULT_CACHE_DIR,
        revision: str | None = None,
        device: str = "cpu",
        local_files_only: bool = True,
        batch_size: int = 32,
    ) -> None:
        if (
            isinstance(batch_size, bool)
            or not isinstance(batch_size, numbers.Integral)
            or batch_size <= 0
        ):
            raise ValueError("batch_size must be positive")
        if device.startswith("cuda") and not torch.cuda.is_available():
            raise ValueError("CUDA was requested but this PyTorch installation has no CUDA device")

        # Transformers otherwise probes TensorFlow, adding startup latency and noisy logs.
        os.environ.setdefault("USE_TF", "0")
        os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
        if local_files_only:
            os.environ.setdefault("HF_HUB_OFFLINE", "1")
        try:
            from transformers import CLIPTextModelWithProjection, CLIPTokenizerFast

            common = {
                "cache_dir": str(cache_dir),
                "local_files_only": local_files_only,
            }
            if revision:
                common["revision"] = revision
            self.tokenizer = CLIPTokenizerFast.from_pretrained(model_id, **common)
            self.model = CLIPTextModelWithProjection.from_pretrained(model_id, **common)
        except Exception as exc:
            mode = "local cache" if local_files_only else "Hugging Face"
            raise EncoderLoadError(
                f"Could not load {model_id!r} from {mode}. "
                "Run `python -m kis.baseline_cli prepare-model`, or pass a Hugging Face repo ID "
                "with --allow-download."
            ) from exc

        self.model_id = model_id
        self.cache_dir = Path(cache_dir)
        self.revision = revision
        self.device = torch.device(device)
        self.batch_size = batch_size
        self.model.eval().to(self.device)
        self.dimension = int(self.model.config.projection_dim)
        self.max_length = min(77, int(self.tokenizer.model_max_length))

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        values = list(texts)
        if not values:
            return np.empty((0, self.dimension), dtype=np.float32)
        for index, value in enumerate(values):
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"Text at batch position {index} must be a nonblank string")

        batches: list[np.ndarray] = []
        with torch.inference_mode():
            for start in range(0, len(values), self.batch_size):
                tokens = self.tokenizer(
                    values[start : start + self.batch_size],
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                tokens = {name: tensor.to(self.device) for name, tensor in tokens.items()}
                output = self.model(**tokens)
                embeddings = output.text_embeds.float()
                expected_rows = len(values[start : start + self.batch_size])
                if (
                    embeddings.ndim != 2
                    or embeddings.shape != (expected_rows, self.dimension)
                ):
                    raise RuntimeError(
                        f"CLIP returned shape {tuple(embeddings.shape)}; "
                        f"expected ({expected_rows}, {self.dimension})"
                    )
                norms64 = torch.linalg.vector_norm(embeddings.double(), dim=1)
                if not torch.isfinite(embeddings).all() or not torch.isfinite(norms64).all() or torch.any(norms64 <= 0):
                    raise RuntimeError("CLIP produced a non-finite or zero text vector")
                embeddings = embeddings / norms64.to(embeddings.dtype).unsqueeze(1)
                if not torch.isfinite(embeddings).all():
                    raise RuntimeError("CLIP normalization produced a non-finite text vector")
                batches.append(embeddings.cpu().numpy().astype(np.float32, copy=False))
        return np.ascontiguousarray(np.concatenate(batches, axis=0), dtype=np.float32)

    def describe(self) -> dict[str, object]:
        commit_hash = getattr(self.model.config, "_commit_hash", None) or resolve_local_hf_commit(
            self.model_id
        )
        description = {
            "backend": "transformers.CLIPTextModelWithProjection",
            "model_id": self.model_id,
            "revision": self.revision,
            "resolved_commit": commit_hash,
            "dimension": self.dimension,
            "max_length": self.max_length,
            "device": str(self.device),
            "dtype": "float32",
            "l2_normalized": True,
            "cache_dir": str(self.cache_dir.resolve()),
        }
        manifest = read_local_model_manifest(self.model_id)
        if manifest:
            description["weights_sha256"] = manifest.get("weights_sha256")
            description["checkpoint_repo_id"] = manifest.get("repo_id")
        return description
