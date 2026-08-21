from __future__ import annotations

import base64
import dataclasses
import io
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()  # picks up DASHSCOPE_API_KEY from a local .env
except ImportError:
    pass

# `kis.pipeline` pulls in torch (via retrieval.clip_encoder); it must be
# imported before pyarrow.parquet. On this Windows environment, whichever of
# torch/pyarrow loads its bundled C++ runtime DLLs first wins the
# process-wide DLL cache, and pyarrow-before-torch reliably breaks torch's
# DLL init with `OSError: [WinError 1114]`.
from kis.pipeline import KISPipeline
from llm.expansion_retrieval import ExpandedHybridSearch
from llm.query_expansion import QwenQueryUnderstanding
from qna.images import load_keyframe_bytes
from qna.candidates import CandidateConfig, QACandidateSearch
from qna.pipeline import KISVideoQA
from qna.query_split import QwenQuestionSplitter
from qna.question_type import classify_question_type
from trake.aggregation import VideoRetrievalConfig
from trake.dense_frame_search import extract_video_to_cache
from trake.video_catalog import VideoCatalog
from trake.web_pipeline import TRAKEWebPipeline
from vlm.image_qa import QwenVLAnswerer
from retrieval.io import plan_btc_submission_paths
from submission.queue import SubmissionQueue
from submission.ranking_optimizer import RankableHit, rerank_with_tiered_diversity

import shutil

import pyarrow.parquet as pq
import yaml
from flask import Flask, abort, jsonify, render_template, request, send_file

BASE_DIR = Path(__file__).resolve().parent
CONFIG_PATH = BASE_DIR / "config.yaml"


def load_config() -> dict:
    with CONFIG_PATH.open("r", encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def build_frame_index(mapping_path: Path) -> dict[tuple[str, int], str]:
    """Map (video_id, Video_Frame_ID) -> Keyframe_Path for the thumbnail route.

    `Video_Frame_ID` is not the catalog's primary key (a handful of frames
    repeat within a video), so the first available row wins; that is fine
    here since a thumbnail lookup only needs *a* valid image for that frame,
    not the single canonical mapping row.
    """
    table = pq.read_table(
        mapping_path,
        columns=["Video_ID", "Video_Frame_ID", "Keyframe_Path", "Keyframe_Available"],
    )
    index: dict[tuple[str, int], str] = {}
    for video_id, frame_id, path, available in zip(
        table.column("Video_ID").to_pylist(),
        table.column("Video_Frame_ID").to_pylist(),
        table.column("Keyframe_Path").to_pylist(),
        table.column("Keyframe_Available").to_pylist(),
    ):
        key = (video_id, frame_id)
        if available and key not in index:
            index[key] = path
    return index


def build_video_frames_index(mapping_path: Path) -> dict[str, list[dict]]:
    """Map Video_ID -> every available frame in that video, ordered by
    Video_Frame_ID, for the "all frames of this video" browser."""
    table = pq.read_table(
        mapping_path,
        columns=["Video_ID", "Video_Frame_ID", "Timestamp", "Keyframe_Index", "Keyframe_Available"],
    )
    grouped: dict[str, dict[int, dict]] = {}
    for video_id, frame_id, timestamp, keyframe_index, available in zip(
        table.column("Video_ID").to_pylist(),
        table.column("Video_Frame_ID").to_pylist(),
        table.column("Timestamp").to_pylist(),
        table.column("Keyframe_Index").to_pylist(),
        table.column("Keyframe_Available").to_pylist(),
    ):
        if not available:
            continue
        bucket = grouped.setdefault(video_id, {})
        if frame_id not in bucket:
            bucket[frame_id] = {
                "frame_id": frame_id,
                "timestamp": timestamp,
                "keyframe_index": keyframe_index,
            }
    return {
        video_id: sorted(frames.values(), key=lambda frame: frame["frame_id"])
        for video_id, frames in grouped.items()
    }


config = load_config()
paths_cfg = config["paths"]
model_cfg = config["model"]
retrieval_cfg = config["retrieval"]
llm_cfg = config["llm"]
qna_cfg = config["qna"]
trake_cfg = config["trake"]

INDEXES_ROOT = BASE_DIR / paths_cfg["indexes"]
DATA_ROOT = BASE_DIR / paths_cfg["data"]
VIDEO_CACHE_DIR = BASE_DIR / paths_cfg["video_cache_dir"]
SUBMISSIONS_DIR = BASE_DIR / "submissions"

pipeline = KISPipeline(
    artifacts_root=INDEXES_ROOT,
    model=str(BASE_DIR / paths_cfg["model_dir"]),
    cache_dir=BASE_DIR / paths_cfg["cache_dir"],
    device=model_cfg["device"],
    allow_download=model_cfg["allow_download"],
    batch_size=model_cfg["batch_size"],
)
frame_index = build_frame_index(INDEXES_ROOT / "catalog" / "frame_mapping.parquet")
video_frames_index = build_video_frames_index(INDEXES_ROOT / "catalog" / "frame_mapping.parquet")
video_catalog = VideoCatalog(INDEXES_ROOT / "catalog" / "frame_mapping.parquet")
submission_queue = SubmissionQueue()

# Q&A, TRAKE and LLM query expansion need a Qwen (DASHSCOPE_API_KEY) client.
# Built lazily so a missing/invalid key only breaks those specific routes
# with a clear 503, never the already-working plain KIS search at startup.
_qna_service: KISVideoQA | None = None
_trake_service: TRAKEWebPipeline | None = None
_expansion_service: ExpandedHybridSearch | None = None


def get_qna_service() -> KISVideoQA:
    global _qna_service
    if _qna_service is None:
        candidate_search = QACandidateSearch(
            pipeline.engine,
            config=CandidateConfig(
                search_top_k=qna_cfg["search_top_k"],
                video_limit=qna_cfg["video_limit"],
                frame_limit=qna_cfg["frame_limit"],
            ),
        )
        vlm = QwenVLAnswerer(model=llm_cfg["vlm_model"], base_url=llm_cfg["base_url"])
        splitter = QwenQuestionSplitter(model=llm_cfg["model"], base_url=llm_cfg["base_url"])
        _qna_service = KISVideoQA(
            candidate_search, vlm, data_root=DATA_ROOT, question_splitter=splitter
        )
    return _qna_service


def get_expansion_service() -> ExpandedHybridSearch:
    global _expansion_service
    if _expansion_service is None:
        understanding = QwenQueryUnderstanding(model=llm_cfg["model"], base_url=llm_cfg["base_url"])
        _expansion_service = ExpandedHybridSearch(pipeline.engine, understanding)
    return _expansion_service


def apply_ranking(results: list[dict]) -> list[dict]:
    """Tiered MMR rerank: Top-5 order is untouched, diversity only past rank 6."""
    if not results:
        return results
    pool = [
        RankableHit(
            video_id=item["video_id"],
            frame_id=item["frame_id"],
            timestamp=item["timestamp"],
            score=item["score"],
            extra={"i": str(index)},
        )
        for index, item in enumerate(results)
    ]
    reranked = rerank_with_tiered_diversity(pool)
    ordered = []
    for new_rank, item in enumerate(reranked, start=1):
        original = dict(results[int(item.extra["i"])])
        original["rank"] = new_rank
        ordered.append(original)
    return ordered


def get_trake_service() -> TRAKEWebPipeline:
    global _trake_service
    if _trake_service is None:
        _trake_service = TRAKEWebPipeline(
            hybrid_engine=pipeline.engine,
            hybrid_store=pipeline.store,
            encoder=pipeline.encoder,
            video_catalog=video_catalog,
            data_root=DATA_ROOT,
            llm_model=llm_cfg["model"],
            vlm_model=llm_cfg["vlm_model"],
            llm_base_url=llm_cfg["base_url"],
            video_retrieval_config=VideoRetrievalConfig(video_limit=trake_cfg["video_limit"]),
        )
    return _trake_service


app = Flask(__name__)


@app.get("/")
def index():
    return render_template("index.html")


@app.get("/api/system/status")
def system_status():
    usage = shutil.disk_usage(DATA_ROOT.anchor or DATA_ROOT)
    return jsonify(
        {
            "status": "ready",
            "storage_used": usage.used,
            "storage_total": usage.total,
        }
    )


# --- KIS -------------------------------------------------------------------


@app.post("/search/kis")
def search_kis():
    payload = request.get_json(silent=True) or {}
    query_text = str(payload.get("query") or "").strip()
    if not query_text:
        return jsonify({"error": "query must not be blank"}), 400
    top_k = int(payload.get("top_k") or retrieval_cfg["default_top_k"])
    top_k = max(1, min(top_k, 100))
    use_expansion = bool(payload.get("expand"))

    if use_expansion:
        try:
            expansion = get_expansion_service()
            hits = expansion.search(query_text, top_k=top_k)
        except ValueError as exc:
            return jsonify({"error": str(exc)}), 503
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 503
        results = [hit.as_dict() for hit in hits]
    else:
        result = pipeline.search(query_text, top_k=top_k)
        results = [hit.as_dict() for hit in result.hits]

    results = apply_ranking(results)
    return jsonify(
        {
            "query": query_text,
            "top_k": top_k,
            "expand": use_expansion,
            "results": results,
        }
    )


@app.get("/frame/<video_id>/<int:frame_id>")
def serve_frame(video_id: str, frame_id: int):
    keyframe_path = frame_index.get((video_id, frame_id))
    if keyframe_path is None:
        abort(404)
    try:
        image_bytes = load_keyframe_bytes(keyframe_path, data_root=DATA_ROOT)
    except FileNotFoundError:
        abort(404)

    width = request.args.get("w", type=int)
    if width:
        from PIL import Image

        img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
        ratio = width / img.width
        img = img.resize((width, max(1, round(img.height * ratio))))
        buffer = io.BytesIO()
        img.save(buffer, format="JPEG", quality=85)
        image_bytes = buffer.getvalue()

    return send_file(io.BytesIO(image_bytes), mimetype="image/jpeg")


# --- Video player (Milestone 2) --------------------------------------------


@app.get("/video/<video_id>")
def serve_video(video_id: str):
    try:
        info = video_catalog.get(video_id)
        path = extract_video_to_cache(info, data_root=DATA_ROOT, cache_dir=VIDEO_CACHE_DIR)
    except (KeyError, FileNotFoundError) as exc:
        abort(404, description=str(exc))
    return send_file(path, mimetype="video/mp4", conditional=True)


@app.get("/api/video/<video_id>/info")
def video_info(video_id: str):
    try:
        info = video_catalog.get(video_id)
    except (KeyError, FileNotFoundError) as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(
        {"video_id": info.video_id, "fps": info.fps, "video_available": info.video_available}
    )


@app.get("/api/video/<video_id>/frames")
def video_frames(video_id: str):
    """Every available keyframe in this video, for the "all frames of this
    video" browser in the Preview & Details panel."""
    frames = video_frames_index.get(video_id)
    if frames is None:
        abort(404, description=f"Unknown video_id: {video_id!r}")
    return jsonify({"video_id": video_id, "count": len(frames), "frames": frames})


# --- Q&A (Milestone 4) -------------------------------------------------------


@app.post("/search/qna")
def search_qna():
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("query") or payload.get("question") or "").strip()
    if not question:
        return jsonify({"error": "question must not be blank"}), 400
    top_k = max(1, int(payload.get("top_k") or qna_cfg["answers_top_k"]))
    query_id = str(payload.get("query_id") or "qa")
    try:
        qa = get_qna_service()
        results = qa.ask(question, query_id=query_id, top_k=top_k)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify({"question": question, "results": [result.as_dict() for result in results]})


@app.post("/vlm/ask")
def vlm_ask():
    """Manual multi-frame ask (spec section 10): user-ticked frames, no retrieval narrowing."""
    payload = request.get_json(silent=True) or {}
    question = str(payload.get("question") or "").strip()
    video_id = str(payload.get("video_id") or "").strip()
    frame_ids = payload.get("frame_ids") or []
    if not question or not video_id or not frame_ids:
        return jsonify({"error": "question, video_id and frame_ids are required"}), 400
    try:
        qa = get_qna_service()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503

    question_type = classify_question_type(question)
    answers = []
    for raw_frame_id in frame_ids:
        frame_id = int(raw_frame_id)
        keyframe_path = frame_index.get((video_id, frame_id))
        if keyframe_path is None:
            answers.append({"frame_id": frame_id, "error": "keyframe not available"})
            continue
        image_bytes = load_keyframe_bytes(keyframe_path, data_root=DATA_ROOT)
        vlm_answer = qa.vlm.answer(image_bytes, question, question_type=question_type)
        answers.append(
            {
                "frame_id": frame_id,
                "answer": vlm_answer.answer,
                "confidence": vlm_answer.confidence,
            }
        )
    return jsonify(
        {"video_id": video_id, "question": question, "question_type": question_type, "answers": answers}
    )


# --- TRAKE (Milestones 5-6) -------------------------------------------------


@app.post("/search/trake")
def search_trake():
    payload = request.get_json(silent=True) or {}
    query_text = str(payload.get("query") or "").strip()
    if not query_text:
        return jsonify({"error": "query must not be blank"}), 400
    query_id = str(payload.get("query_id") or "trake")
    try:
        trake = get_trake_service()
        result = trake.search(query_text, query_id=query_id)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    return jsonify(result)


@app.post("/trake/align")
def trake_align():
    """Coarse-align an ordered event list against a specific candidate video —
    lets the UI switch the Event Panel to any candidate chip, not just the
    top-ranked one `/search/trake` aligns automatically."""
    payload = request.get_json(silent=True) or {}
    video_id = str(payload.get("video_id") or "").strip()
    events = [str(event).strip() for event in (payload.get("events") or []) if str(event).strip()]
    if not video_id or not events:
        return jsonify({"error": "video_id and events are required"}), 400
    try:
        trake = get_trake_service()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    try:
        result = trake.align(video_id, events)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify(result)


@app.post("/refine")
def refine_frame():
    """Dense frame grid around a coarse event position (spec section 13), no VLM yet."""
    payload = request.get_json(silent=True) or {}
    video_id = str(payload.get("video_id") or "").strip()
    coarse_timestamp = payload.get("coarse_timestamp")
    if not video_id or coarse_timestamp is None:
        return jsonify({"error": "video_id and coarse_timestamp are required"}), 400
    window_seconds = float(payload.get("window_seconds") or trake_cfg["window_seconds"])
    step_seconds = float(payload.get("step_seconds") or trake_cfg["step_seconds"])
    try:
        trake = get_trake_service()
        frames = trake.dense_frames(
            video_id,
            float(coarse_timestamp),
            window_seconds=window_seconds,
            step_seconds=step_seconds,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(
        {
            "video_id": video_id,
            "frames": [
                {
                    "timestamp": frame.timestamp,
                    "image_base64": base64.b64encode(frame.image_bytes).decode("ascii"),
                }
                for frame in frames
            ],
        }
    )


@app.post("/vlm/verify")
def vlm_verify():
    """VLM-picks the best candidate near a coarse/refined event position (section 14)."""
    payload = request.get_json(silent=True) or {}
    video_id = str(payload.get("video_id") or "").strip()
    event_text = str(payload.get("event_text") or "").strip()
    coarse_timestamp = payload.get("coarse_timestamp")
    if not video_id or not event_text or coarse_timestamp is None:
        return jsonify({"error": "video_id, event_text and coarse_timestamp are required"}), 400
    window_seconds = float(payload.get("window_seconds") or trake_cfg["window_seconds"])
    step_seconds = float(payload.get("step_seconds") or trake_cfg["step_seconds"])
    max_candidate_frames = int(
        payload.get("max_candidate_frames") or trake_cfg["max_candidate_frames"]
    )
    try:
        trake = get_trake_service()
        result = trake.verify(
            video_id,
            event_text,
            float(coarse_timestamp),
            window_seconds=window_seconds,
            step_seconds=step_seconds,
            max_candidate_frames=max_candidate_frames,
        )
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 503
    except (KeyError, FileNotFoundError, RuntimeError) as exc:
        return jsonify({"error": str(exc)}), 404
    return jsonify(dataclasses.asdict(result))


# --- Submission Queue (Milestones 3, 7) -------------------------------------


@app.get("/submission")
def get_submission():
    return jsonify({"items": submission_queue.list()})


@app.post("/submission/add")
def add_submission():
    payload = request.get_json(silent=True) or {}
    try:
        item = submission_queue.add(
            video_id=str(payload.get("video_id") or "").strip(),
            frame_id=int(payload.get("frame_id")),
            route=str(payload.get("route") or "kis"),
            answer=payload.get("answer"),
        )
    except (ValueError, TypeError) as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"item": item, "items": submission_queue.list()})


@app.post("/submission/remove")
def remove_submission():
    payload = request.get_json(silent=True) or {}
    item_id = payload.get("id")
    if item_id is None:
        return jsonify({"error": "id is required"}), 400
    submission_queue.remove(int(item_id))
    return jsonify({"items": submission_queue.list()})


@app.post("/submission/move")
def move_submission():
    payload = request.get_json(silent=True) or {}
    item_id = payload.get("id")
    direction = payload.get("direction")
    if item_id is None or direction not in ("up", "down"):
        return jsonify({"error": "id and direction ('up'/'down') are required"}), 400
    submission_queue.move(int(item_id), direction)
    return jsonify({"items": submission_queue.list()})


@app.post("/submission/export")
def export_submission():
    payload = request.get_json(silent=True) or {}
    query_id = str(payload.get("query_id") or "submission").strip()
    try:
        target = plan_btc_submission_paths(SUBMISSIONS_DIR, [query_id])[query_id]
        path = submission_queue.export_csv(target)
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"path": str(path.resolve()), "count": len(submission_queue.list())})


if __name__ == "__main__":
    server_cfg = config.get("server", {})
    app.run(
        host=server_cfg.get("host", "127.0.0.1"),
        port=server_cfg.get("port", 5000),
        debug=server_cfg.get("debug", False),
        # The watchdog reloader recursively watches the whole working
        # directory by default, including `data/batch_*/*.zip` (100+ GB) and
        # every read seems to register as a "change" — it kept restarting
        # the server mid-session and eventually crashed it. debug=True still
        # gives the interactive debugger/stack traces; only the reloader is
        # disabled.
        use_reloader=False,
    )
