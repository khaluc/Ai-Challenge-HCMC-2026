from __future__ import annotations

import io
import json
import os
from contextlib import ExitStack
from pathlib import Path
from zipfile import ZipFile

import numpy as np
import torch
import torch.nn.functional as functional
from PIL import Image

from data_processing.layout import DatasetLayout, indexed_members
from data_processing.source import load_clip_array, zip_members_by_video

from .clip_encoder import (
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    read_local_model_manifest,
    resolve_local_hf_commit,
)


def verify_image_feature_compatibility(
    data_root: Path | str,
    *,
    model_id: str = str(DEFAULT_MODEL_DIR),
    cache_dir: Path | str = DEFAULT_CACHE_DIR,
    revision: str | None = None,
    local_files_only: bool = True,
    samples: int = 5,
    min_cosine: float = 0.99,
    device: str = "cpu",
) -> dict:
    """Re-encode real keyframes and compare them to organizer-provided vectors."""
    if samples <= 0:
        raise ValueError("samples must be positive")
    if not -1.0 <= min_cosine <= 1.0:
        raise ValueError("min_cosine must be in [-1, 1]")
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise ValueError("CUDA was requested but is unavailable")

    layout = DatasetLayout.discover(data_root)
    keyframes = indexed_members(layout.keyframe_archives, ".jpg")
    available = sorted(
        (video_id, frame_index, source)
        for video_id, by_index in keyframes.items()
        for frame_index, source in by_index.items()
    )
    if not available:
        raise ValueError("No available keyframe images to verify")
    positions = np.linspace(0, len(available) - 1, min(samples, len(available)), dtype=int)
    selected = [available[int(position)] for position in positions]
    clip_members = zip_members_by_video(layout.clip_archive, ".npy")

    images: list[Image.Image] = []
    expected_vectors: list[np.ndarray] = []
    sample_metadata: list[dict] = []
    with ExitStack() as stack:
        archive_handles: dict[Path, ZipFile] = {}
        clip_handle = stack.enter_context(ZipFile(layout.clip_archive))
        clip_cache: dict[str, np.ndarray] = {}
        for video_id, keyframe_index, (archive, member) in selected:
            if archive not in archive_handles:
                archive_handles[archive] = stack.enter_context(ZipFile(archive))
            image = Image.open(io.BytesIO(archive_handles[archive].read(member))).convert("RGB")
            images.append(image)
            if video_id not in clip_cache:
                clip_cache[video_id] = load_clip_array(clip_handle, clip_members[video_id])
            local_index = keyframe_index - 1
            if not 0 <= local_index < len(clip_cache[video_id]):
                raise ValueError(f"No CLIP row {local_index} for {video_id}/{keyframe_index}")
            expected_vectors.append(
                clip_cache[video_id][local_index].astype(np.float32, copy=False)
            )
            sample_metadata.append(
                {
                    "video_id": video_id,
                    "keyframe_index": keyframe_index,
                    "keyframe_source": f"{archive.name}::{member}",
                }
            )

    os.environ.setdefault("USE_TF", "0")
    os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
    if local_files_only:
        os.environ.setdefault("HF_HUB_OFFLINE", "1")
    from transformers import CLIPImageProcessor, CLIPModel

    common: dict[str, object] = {
        "cache_dir": str(cache_dir),
        "local_files_only": local_files_only,
    }
    if revision:
        common["revision"] = revision
    processor = CLIPImageProcessor.from_pretrained(model_id, **common)
    model = CLIPModel.from_pretrained(model_id, **common).eval().to(device)
    pixel_values = processor(images=images, return_tensors="pt")["pixel_values"].to(device)
    with torch.inference_mode():
        actual = model.get_image_features(pixel_values=pixel_values).float()
        actual = functional.normalize(actual, p=2, dim=1).cpu().numpy()
    expected = np.stack(expected_vectors).astype(np.float32, copy=False)
    expected /= np.linalg.norm(expected, axis=1, keepdims=True)
    if actual.shape != expected.shape:
        raise ValueError(
            f"Candidate model features {actual.shape} do not match BTC features {expected.shape}"
        )
    cosines = np.sum(actual * expected, axis=1)
    for item, cosine in zip(sample_metadata, cosines):
        item["cosine_similarity"] = float(cosine)

    report = {
        "compatible": bool(np.all(cosines >= min_cosine)),
        "model_id": model_id,
        "revision": revision,
        "resolved_commit": getattr(model.config, "_commit_hash", None)
        or resolve_local_hf_commit(model_id),
        "threshold": min_cosine,
        "sample_count": len(sample_metadata),
        "cosine_min": float(cosines.min()),
        "cosine_mean": float(cosines.mean()),
        "cosine_max": float(cosines.max()),
        "samples": sample_metadata,
        "interpretation": (
            "A high cosine confirms that the checkpoint and image preprocessing match "
            "the organizer-provided CLIP feature space."
        ),
    }
    manifest = read_local_model_manifest(model_id)
    if manifest:
        report["weights_sha256"] = manifest.get("weights_sha256")
        report["checkpoint_repo_id"] = manifest.get("repo_id")
    return report


def write_compatibility_report(report: dict, path: Path | str) -> Path:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix(output.suffix + ".tmp")
    tmp.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    tmp.replace(output)
    return output
