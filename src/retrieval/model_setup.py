from __future__ import annotations

import hashlib
import json
from pathlib import Path

from .clip_encoder import DEFAULT_MODEL_DIR, DEFAULT_MODEL_ID, DEFAULT_MODEL_REVISION


MODEL_ALLOW_PATTERNS = (
    "config.json",
    "preprocessor_config.json",
    "merges.txt",
    "vocab.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "special_tokens_map.json",
    "pytorch_model.bin",
    "model.safetensors",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_model(
    *,
    repo_id: str = DEFAULT_MODEL_ID,
    output_dir: Path | str = DEFAULT_MODEL_DIR,
    revision: str | None = DEFAULT_MODEL_REVISION,
) -> dict:
    """Download the minimal PyTorch CLIP snapshot into a Windows-safe local directory."""
    from huggingface_hub import snapshot_download

    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    snapshot_download(
        repo_id=repo_id,
        revision=revision,
        local_dir=output,
        allow_patterns=list(MODEL_ALLOW_PATTERNS),
        max_workers=1,
    )
    weights = next(
        (path for path in (output / "model.safetensors", output / "pytorch_model.bin") if path.exists()),
        None,
    )
    if weights is None:
        raise FileNotFoundError(f"Downloaded snapshot at {output} has no PyTorch weights")
    metadata_path = output / ".cache" / "huggingface" / "download" / "config.json.metadata"
    resolved_revision = revision
    if metadata_path.exists():
        lines = metadata_path.read_text(encoding="utf-8").splitlines()
        if lines:
            resolved_revision = lines[0].strip() or resolved_revision
    manifest = {
        "repo_id": repo_id,
        "requested_revision": revision,
        "revision": resolved_revision,
        "weights_file": weights.name,
        "weights_bytes": weights.stat().st_size,
        "weights_sha256": _sha256(weights),
    }
    manifest_path = output / "MODEL_MANIFEST.json"
    tmp = manifest_path.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    tmp.replace(manifest_path)
    manifest["model_dir"] = str(output)
    manifest["manifest"] = str(manifest_path)
    return manifest
