"""Textual KIS command-line entry points: baseline, hybrid, and LLM-expanded search.

The reusable logic lives in `retrieval` (baseline + hybrid CLIP/BM25/objects
search) and `llm` (query expansion); this package holds the three matching
CLIs (`baseline_cli.py`, `hybrid_cli.py`, `expansion_cli.py`) plus their CSV
submission writers. `pipeline.py` wraps hybrid engine construction + search
for long-lived callers (the Flask web app) that load the engine once instead
of per-invocation like the CLIs do.
"""

__all__: list[str] = []
