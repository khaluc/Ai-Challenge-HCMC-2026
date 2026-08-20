// ---------- shared state ----------

const state = {
  videoId: null,
  frameId: null,
  fps: 25,
  trake: null, // { videoId, events: [{event_index, event_text, frame_id, timestamp}] }
  queryType: "kis",
  kisResults: [],
  kisPage: 1,
  currentHit: null,
  expandOn: false,
};

const PAGE_SIZE = 30;

// ---------- DOM refs ----------

const form = document.getElementById("search-form");
const queryInput = document.getElementById("query-input");
const trakeQueryBuilder = document.getElementById("trake-query-builder");
const trakeEventRows = document.getElementById("trake-event-rows");
const trakeAddEventBtn = document.getElementById("trake-add-event");
const trakeHint = document.getElementById("trake-hint");
const statusEl = document.getElementById("search-status");
const grid = document.getElementById("candidate-grid");
const qnaPanel = document.getElementById("qna-panel");
const qnaAnswers = document.getElementById("qna-answers");
const qnaSceneDescription = document.getElementById("qna-scene-description");
const qnaCandidateGrid = document.getElementById("qna-candidate-grid");
const qnaAskBtn = document.getElementById("qna-ask-btn");
const qnaAskStatus = document.getElementById("qna-ask-status");
const trakePanel = document.getElementById("trake-panel");
const trakeVideoCandidates = document.getElementById("trake-video-candidates");
const trakeEvents = document.getElementById("trake-events");
const timelineContent = document.getElementById("timeline-content");
const videoEl = document.getElementById("video-el");
const infoVideoId = document.getElementById("info-video-id");
const infoKeyframeBadge = document.getElementById("info-keyframe-badge");
const infoFrameId = document.getElementById("info-frame-id");
const infoTime = document.getElementById("info-time");
const infoKeyframeIndex = document.getElementById("info-keyframe-index");
const infoFaissIndex = document.getElementById("info-faiss-index");

const resultsList = document.getElementById("results-list");
const resultsPagination = document.getElementById("results-pagination");
const resultsMeta = document.getElementById("results-meta");
const resultsTopkBadge = document.getElementById("results-topk-badge");
const csvExportBtn = document.getElementById("csv-export-btn");
const viewStatsBtn = document.getElementById("view-stats-btn");
const statsModal = document.getElementById("stats-modal");
const statsModalBody = document.getElementById("stats-modal-body");
const statsModalClose = document.getElementById("stats-modal-close");

const scoreBreakdownBody = document.getElementById("score-breakdown-body");
const matchInfoBody = document.getElementById("match-info-body");
const objectsBody = document.getElementById("objects-body");

const filtersBtn = document.getElementById("filters-btn");
const filtersPopover = document.getElementById("filters-popover");
const chipExpansion = document.getElementById("chip-expansion");

const themeToggle = document.getElementById("theme-toggle");
const appShell = document.getElementById("app-shell");
const previewExpand = document.getElementById("preview-expand");

const systemStatusEl = document.getElementById("system-status");
const systemStatusText = document.getElementById("system-status-text");

const playToggle = document.getElementById("play-toggle");
const muteToggle = document.getElementById("mute-toggle");
const seekBar = document.getElementById("seek-bar");
const timeLabel = document.getElementById("time-label");
const fullscreenToggle = document.getElementById("fullscreen-toggle");

// ---------- helpers ----------

function setStatus(text, isError) {
  statusEl.textContent = text;
  statusEl.classList.toggle("error", Boolean(isError));
}

async function postJSON(url, body) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body || {}),
  });
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

async function getJSON(url) {
  const response = await fetch(url);
  const data = await response.json().catch(() => ({}));
  if (!response.ok) {
    throw new Error(data.error || `HTTP ${response.status}`);
  }
  return data;
}

function fmtTime(seconds) {
  if (seconds == null || Number.isNaN(seconds)) return "—";
  const m = Math.floor(seconds / 60);
  const s = (seconds % 60).toFixed(2).padStart(5, "0");
  return `${String(m).padStart(2, "0")}:${s}`;
}

function fmtScore(v) {
  return v == null ? "—" : v.toFixed(3);
}

// ---------- video player / frame navigation ----------

async function loadVideo(videoId, frameId, timestamp) {
  state.videoId = videoId;
  state.frameId = frameId ?? null;

  if (videoEl.dataset.currentVideo !== videoId) {
    videoEl.src = `/video/${encodeURIComponent(videoId)}`;
    videoEl.dataset.currentVideo = videoId;
    try {
      const info = await getJSON(`/api/video/${encodeURIComponent(videoId)}/info`);
      state.fps = info.fps || 25;
    } catch (error) {
      state.fps = 25;
    }
  }
  if (typeof timestamp === "number") {
    const seekTo = () => {
      videoEl.currentTime = timestamp;
      videoEl.removeEventListener("loadedmetadata", seekTo);
    };
    if (videoEl.readyState >= 1) {
      videoEl.currentTime = timestamp;
    } else {
      videoEl.addEventListener("loadedmetadata", seekTo);
    }
  }
  infoVideoId.textContent = videoId;
  infoFrameId.textContent = frameId != null ? frameId : "—";
  updateTimeInfo();
}

function updateTimeInfo() {
  infoTime.textContent = fmtTime(videoEl.currentTime);
  timeLabel.textContent = `${fmtTime(videoEl.currentTime)} / ${fmtTime(videoEl.duration || 0)}`;
  if (!seekBar.dragging && videoEl.duration) {
    seekBar.value = Math.round((videoEl.currentTime / videoEl.duration) * 1000);
  }
}

videoEl.addEventListener("timeupdate", updateTimeInfo);
videoEl.addEventListener("loadedmetadata", updateTimeInfo);

function stepFrames(n) {
  if (!state.videoId) return;
  videoEl.currentTime = Math.max(0, videoEl.currentTime + n / state.fps);
}

function stepSeconds(n) {
  if (!state.videoId) return;
  videoEl.currentTime = Math.max(0, videoEl.currentTime + n);
}

document.querySelectorAll(".frame-nav button").forEach((button) => {
  button.addEventListener("click", () => {
    const nav = button.dataset.nav;
    if (nav === "play") {
      videoEl.paused ? videoEl.play() : videoEl.pause();
    } else if (nav === "-1s") {
      stepSeconds(-1);
    } else if (nav === "+1s") {
      stepSeconds(1);
    } else {
      stepFrames(parseInt(nav, 10));
    }
  });
});

document.addEventListener("keydown", (event) => {
  const tag = (event.target.tagName || "").toLowerCase();
  if (tag === "input" || tag === "textarea" || tag === "select") return;
  if (event.code === "Space") {
    event.preventDefault();
    videoEl.paused ? videoEl.play() : videoEl.pause();
  } else if (event.key === "a" || event.key === "A") {
    stepFrames(event.shiftKey ? -10 : -1);
  } else if (event.key === "d" || event.key === "D") {
    stepFrames(event.shiftKey ? 10 : 1);
  }
});

// custom transport controls (play/mute/seek/fullscreen)

playToggle.addEventListener("click", () => {
  videoEl.paused ? videoEl.play() : videoEl.pause();
});
videoEl.addEventListener("play", () => {
  playToggle.textContent = "❚❚";
});
videoEl.addEventListener("pause", () => {
  playToggle.textContent = "▶";
});

muteToggle.addEventListener("click", () => {
  videoEl.muted = !videoEl.muted;
  muteToggle.textContent = videoEl.muted ? "🔇" : "🔊";
});

seekBar.addEventListener("mousedown", () => {
  seekBar.dragging = true;
});
seekBar.addEventListener("input", () => {
  if (videoEl.duration) {
    videoEl.currentTime = (seekBar.value / 1000) * videoEl.duration;
  }
});
seekBar.addEventListener("change", () => {
  seekBar.dragging = false;
});

fullscreenToggle.addEventListener("click", () => {
  if (videoEl.requestFullscreen) videoEl.requestFullscreen();
});

// ---------- detail panel (Score Breakdown / Match Info / Objects & Concepts) ----------

function isExpandedHit(hit) {
  return hit && hit.best_expansion_id !== undefined;
}

function showDetails(hit) {
  state.currentHit = hit || null;

  infoKeyframeIndex.textContent = hit && hit.keyframe_index != null ? hit.keyframe_index : "—";
  infoFaissIndex.textContent = hit && hit.faiss_index != null ? hit.faiss_index : "—";
  infoKeyframeBadge.innerHTML = hit
    ? hit.keyframe_available
      ? '<span class="badge-available">Keyframe Available</span>'
      : '<span class="badge-unavailable">Keyframe Unavailable</span>'
    : "";

  if (!hit) {
    scoreBreakdownBody.innerHTML = '<div class="empty-note">Chọn một kết quả để xem chi tiết điểm số</div>';
    matchInfoBody.innerHTML = '<div class="empty-note">—</div>';
    objectsBody.innerHTML = '<div class="empty-note">—</div>';
    return;
  }

  if (isExpandedHit(hit)) {
    scoreBreakdownBody.innerHTML = `
      <div class="score-breakdown-row">
        <div class="sb-top"><span class="sb-label">Final Score (RRF across expansions)</span><span class="sb-value">${fmtScore(hit.score)}</span></div>
        <div class="sb-bar-track"><div class="sb-bar-fill" style="width:100%;background:var(--accent-strong)"></div></div>
      </div>
      <div class="detail-row"><span class="k">Best Expansion</span><span class="v">${hit.best_expansion_id}</span></div>
      <div class="detail-row"><span class="k">Best Expansion Rank</span><span class="v">#${hit.best_expansion_rank}</span></div>
      <div class="detail-row"><span class="k">Num Expansions Matched</span><span class="v">${hit.num_expansions_matched}</span></div>
      <div class="detail-row" style="align-items:flex-start"><span class="k">Text</span><span class="v" style="text-align:right;max-width:65%">${hit.best_expansion_text || "—"}</span></div>
    `;

    matchInfoBody.innerHTML = `
      <div class="detail-row"><span class="k">Relation</span><span class="v">${hit.relation || "—"}</span></div>
      <div class="detail-row" style="align-items:flex-start"><span class="k">Attributes</span><span class="v" style="text-align:right">${
        hit.attributes && Object.keys(hit.attributes).length
          ? Object.entries(hit.attributes)
              .map(([k, v]) => `${k}: ${v.join(", ")}`)
              .join("<br>")
          : "—"
      }</span></div>
    `;

    const concepts = hit.object_concepts || [];
    const matched = hit.matched_objects || [];
    objectsBody.innerHTML = `
      <div class="detail-row"><span class="k">Matched Objects</span><span class="v">${matched.length ? "Verified" : "Not Verified"}</span></div>
      <div class="concept-tags" style="margin-top:6px">${
        concepts.length
          ? concepts.map((c) => `<span class="concept-tag">${c}</span>`).join("")
          : '<span class="empty-note">Không có concept nào</span>'
      }</div>
    `;
    return;
  }

  const rows = [
    { label: "CLIP Score", value: hit.semantic_score, color: "var(--accent-blue)" },
    { label: "BM25 Score", value: hit.metadata_bm25_score, color: "var(--accent-teal)" },
    { label: "Object Score", value: hit.object_score, color: "var(--warning)" },
    { label: "Final Score", value: hit.score, color: "var(--accent-strong)" },
  ];
  const maxVal = Math.max(0.001, ...rows.map((r) => r.value || 0));
  scoreBreakdownBody.innerHTML = rows
    .map(
      (r) => `
    <div class="score-breakdown-row">
      <div class="sb-top"><span class="sb-label">${r.label}</span><span class="sb-value">${fmtScore(r.value)}</span></div>
      <div class="sb-bar-track"><div class="sb-bar-fill" style="width:${r.value ? (r.value / maxVal) * 100 : 0}%;background:${r.color}"></div></div>
    </div>`
    )
    .join("");

  matchInfoBody.innerHTML = `
    <div class="detail-row"><span class="k">Fusion Method</span><span class="v">${hit.fusion_method || "—"}</span></div>
    <div class="detail-row"><span class="k">Semantic Rank</span><span class="v">${hit.semantic_rank ?? "—"}</span></div>
    <div class="detail-row"><span class="k">Metadata Rank</span><span class="v">${hit.metadata_rank ?? "—"}</span></div>
    <div class="detail-row"><span class="k">Object Rank</span><span class="v">${hit.object_rank ?? "—"}</span></div>
    <div class="detail-row"><span class="k">Metadata Match Mode</span><span class="v">${hit.metadata_match_mode || "—"}</span></div>
  `;

  const matched = hit.matched_objects || [];
  objectsBody.innerHTML = `
    <div class="detail-row"><span class="k">Mean Confidence</span><span class="v">${hit.object_mean_confidence != null ? hit.object_mean_confidence.toFixed(2) : "—"}</span></div>
    <div class="concept-tags" style="margin-top:6px">${
      matched.length
        ? matched.map((m) => `<span class="concept-tag">${m}</span>`).join("")
        : '<span class="empty-note">Không có object khớp</span>'
    }</div>
  `;
}

// ---------- KIS candidate rows ----------

function selectRow(card) {
  const container = card.closest(".results-list, #candidate-grid, #qna-candidate-grid");
  if (container) {
    container.querySelectorAll(".result-tile.selected").forEach((el) => el.classList.remove("selected"));
  }
  card.classList.add("selected");
}

function candidateCard(hit, { checkbox = false, rank = null } = {}) {
  const card = document.createElement("div");
  card.className = "result-tile";
  card.dataset.videoId = hit.video_id;
  card.dataset.frameId = hit.frame_id;
  card.dataset.timestamp = hit.timestamp;

  const img = document.createElement("img");
  img.className = "tile-thumb";
  img.loading = "lazy";
  img.src = `/frame/${hit.video_id}/${hit.frame_id}?w=240`;
  img.alt = `${hit.video_id} frame ${hit.frame_id}`;
  card.appendChild(img);

  if (rank != null) {
    const rankEl = document.createElement("div");
    rankEl.className = `tile-rank${rank === 1 ? " top1" : ""}`;
    rankEl.textContent = rank;
    card.appendChild(rankEl);
  }

  if (checkbox) {
    const tick = document.createElement("input");
    tick.type = "checkbox";
    tick.className = "candidate-tick";
    tick.dataset.frameId = hit.frame_id;
    card.appendChild(tick);
  } else {
    const badge = document.createElement("span");
    badge.className = `tile-badge ${hit.keyframe_available ? "available" : "unavailable"}`;
    badge.textContent = hit.keyframe_available ? "✓" : "✕";
    badge.title = hit.keyframe_available ? "Keyframe Available" : "Keyframe Unavailable";
    card.appendChild(badge);
  }

  const maxScore = state.kisResults.length ? state.kisResults[0].score || 1 : hit.score || 1;
  const pct = hit.score != null && maxScore ? Math.min(100, (hit.score / maxScore) * 100) : 0;
  const caption = document.createElement("div");
  caption.className = "tile-caption";
  caption.innerHTML = `
    <div class="tile-video-id">${hit.video_id}</div>
    <div class="tile-meta">Frame ${hit.frame_id} &middot; ${fmtScore(hit.score)}</div>
    <div class="tile-score-bar"><div class="tile-score-fill" style="width:${pct}%"></div></div>
  `;
  card.appendChild(caption);

  if (!checkbox) {
    card.addEventListener("click", () => {
      selectRow(card);
      loadVideo(hit.video_id, hit.frame_id, hit.timestamp);
      showDetails(hit);
    });
  }
  return card;
}

function renderCandidates(container, results, options) {
  container.innerHTML = "";
  results.forEach((hit, index) =>
    container.appendChild(candidateCard(hit, { ...options, rank: (options && options.rankOffset ? options.rankOffset : 0) + index + 1 }))
  );
}

function renderKisPage() {
  const start = (state.kisPage - 1) * PAGE_SIZE;
  const pageItems = state.kisResults.slice(start, start + PAGE_SIZE);
  renderCandidates(grid, pageItems, { rankOffset: start });
  renderPagination();
}

function renderPagination() {
  const total = state.kisResults.length;
  const totalPages = Math.max(1, Math.ceil(total / PAGE_SIZE));
  resultsPagination.innerHTML = "";
  if (totalPages <= 1) {
    resultsPagination.classList.add("hidden");
    return;
  }
  resultsPagination.classList.remove("hidden");

  const makeBtn = (label, page, opts = {}) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.textContent = label;
    if (opts.active) btn.classList.add("active");
    if (opts.disabled) btn.disabled = true;
    if (!opts.disabled && !opts.active) {
      btn.addEventListener("click", () => {
        state.kisPage = page;
        renderKisPage();
      });
    }
    return btn;
  };

  resultsPagination.appendChild(makeBtn("‹", state.kisPage - 1, { disabled: state.kisPage <= 1 }));

  const pagesToShow = new Set([1, totalPages, state.kisPage, state.kisPage - 1, state.kisPage + 1]);
  let prevPage = 0;
  for (let p = 1; p <= totalPages; p += 1) {
    if (!pagesToShow.has(p)) continue;
    if (p - prevPage > 1) {
      const ell = document.createElement("span");
      ell.className = "ellipsis";
      ell.textContent = "…";
      resultsPagination.appendChild(ell);
    }
    resultsPagination.appendChild(makeBtn(String(p), p, { active: p === state.kisPage }));
    prevPage = p;
  }

  resultsPagination.appendChild(makeBtn("›", state.kisPage + 1, { disabled: state.kisPage >= totalPages }));
}

// ---------- Q&A ----------

function renderQnaAnswers(results) {
  qnaAnswers.innerHTML = "";
  results.forEach((result) => {
    const card = document.createElement("div");
    card.className = "qna-answer-card";
    if (result.video_id == null) {
      card.innerHTML = `<div>${result.note || "Không có câu trả lời"}</div>`;
      qnaAnswers.appendChild(card);
      return;
    }
    const img = document.createElement("img");
    img.src = `/frame/${result.video_id}/${result.frame_id}?w=140`;
    card.appendChild(img);

    card.addEventListener("click", () => {
      loadVideo(result.video_id, result.frame_id, result.timestamp);
      showDetails(result);
    });

    const body = document.createElement("div");
    body.style.flex = "1";
    const jointLine =
      result.joint_confidence != null
        ? `<div class="qna-joint-line">joint ${result.joint_confidence.toFixed(2)} = video ${result.video_score.toFixed(2)} &times; frame ${result.frame_score.toFixed(2)} &times; answer ${result.confidence.toFixed(2)}</div>`
        : "";
    body.innerHTML = `
      <div>${result.video_id} · frame ${result.frame_id} · conf ${result.confidence.toFixed(2)}${
      result.question_type ? ` <span class="qna-type-badge ${result.question_type}">${result.question_type === "closed_set" ? "closed-set" : "open-ended"}</span>` : ""
    }</div>
      <div class="answer-text">${result.answer || "—"}</div>
      ${jointLine}
    `;
    card.appendChild(body);

    qnaAnswers.appendChild(card);
  });
}

qnaAskBtn.addEventListener("click", async () => {
  const ticked = Array.from(qnaCandidateGrid.querySelectorAll(".candidate-tick:checked"));
  if (ticked.length === 0) {
    qnaAskStatus.textContent = "Chưa tick frame nào";
    return;
  }
  const videoId = ticked[0].closest(".result-tile").dataset.videoId;
  const frameIds = ticked.map((el) => parseInt(el.dataset.frameId, 10));
  const timestampByFrame = {};
  ticked.forEach((el) => {
    const tile = el.closest(".result-tile");
    timestampByFrame[el.dataset.frameId] = parseFloat(tile.dataset.timestamp);
  });
  qnaAskStatus.textContent = `Đang hỏi VLM cho ${frameIds.length} frame...`;
  try {
    const data = await postJSON("/vlm/ask", {
      question: queryInput.value.trim(),
      video_id: videoId,
      frame_ids: frameIds,
    });
    renderQnaAnswers(
      data.answers.map((a) => ({
        video_id: a.error ? null : videoId,
        frame_id: a.frame_id,
        answer: a.answer,
        confidence: a.confidence || 0,
        note: a.error,
        question_type: data.question_type,
        timestamp: timestampByFrame[a.frame_id],
      }))
    );
    qnaAskStatus.textContent = `${data.answers.length} câu trả lời`;
  } catch (error) {
    qnaAskStatus.textContent = `Lỗi: ${error.message}`;
  }
});

async function runQna(question) {
  qnaPanel.classList.remove("hidden");
  grid.classList.add("hidden");
  resultsPagination.classList.add("hidden");
  qnaAnswers.innerHTML = "";
  qnaCandidateGrid.innerHTML = "";

  const [qnaResult, kisResult] = await Promise.allSettled([
    postJSON("/search/qna", { query: question, top_k: 5 }),
    postJSON("/search/kis", { query: question, top_k: 20 }),
  ]);

  if (qnaResult.status === "fulfilled") {
    const firstWithScene = qnaResult.value.results.find((r) => r.scene_description);
    qnaSceneDescription.textContent = firstWithScene
      ? `Retrieval scene: "${firstWithScene.scene_description}"`
      : "";
    renderQnaAnswers(qnaResult.value.results);
  } else {
    qnaSceneDescription.textContent = "";
    qnaAnswers.innerHTML = `<div class="status error">${qnaResult.reason.message}</div>`;
  }
  if (kisResult.status === "fulfilled") {
    state.kisResults = kisResult.value.results;
    renderCandidates(qnaCandidateGrid, kisResult.value.results, { checkbox: true });
  }
}

// ---------- TRAKE ----------

function renderTimeline(events) {
  timelineContent.innerHTML = "";
  if (!events || events.length === 0) {
    timelineContent.className = "placeholder";
    timelineContent.textContent = "Chưa có kết quả TRAKE — timeline sẽ hiện ở đây";
    return;
  }
  timelineContent.className = "";
  const track = document.createElement("div");
  track.className = "timeline-track";
  const timestamps = events.map((e) => e.timestamp);
  const min = Math.min(...timestamps);
  const max = Math.max(...timestamps);
  const span = Math.max(1, max - min);
  events.forEach((event, index) => {
    const pct = ((event.timestamp - min) / span) * 90 + 5;
    const marker = document.createElement("div");
    marker.className = "timeline-marker";
    marker.style.left = `${pct}%`;
    marker.innerHTML = `<div class="dot"></div>E${index + 1}<br>${fmtTime(event.timestamp)}`;
    marker.addEventListener("click", () => loadVideo(state.trake.videoId, event.frame_id, event.timestamp));
    track.appendChild(marker);
  });
  timelineContent.appendChild(track);
}

function eventCard(event, index, videoId) {
  const card = document.createElement("div");
  card.className = "event-card";
  card.dataset.eventIndex = event.event_index;
  card.innerHTML = `
    <h4>EVENT ${index + 1} — ${event.event_text}</h4>
    <div class="event-meta">Frame: <span class="ev-frame">${event.frame_id}</span> &middot; ${fmtTime(event.timestamp)}</div>
    <div class="event-actions">
      <button type="button" class="ev-preview">Preview</button>
      <button type="button" class="ev-refine">Refine</button>
      <button type="button" class="ev-verify">Verify with VLM</button>
    </div>
    <div class="dense-grid-wrap"></div>
    <div class="verify-result-wrap"></div>
  `;

  let currentTimestamp = event.timestamp;
  let currentFrameId = event.frame_id;

  card.querySelector(".ev-preview").addEventListener("click", () => {
    loadVideo(videoId, currentFrameId, currentTimestamp);
  });

  card.querySelector(".ev-refine").addEventListener("click", async () => {
    const wrap = card.querySelector(".dense-grid-wrap");
    wrap.innerHTML = "Đang decode...";
    try {
      const data = await postJSON("/refine", {
        video_id: videoId,
        coarse_timestamp: currentTimestamp,
      });
      wrap.innerHTML = "";
      const denseGrid = document.createElement("div");
      denseGrid.className = "dense-grid";
      data.frames.forEach((frame) => {
        const cell = document.createElement("div");
        const img = document.createElement("img");
        img.src = `data:image/jpeg;base64,${frame.image_base64}`;
        if (Math.abs(frame.timestamp - currentTimestamp) < 1e-6) img.classList.add("selected");
        img.addEventListener("click", () => {
          denseGrid.querySelectorAll("img").forEach((el) => el.classList.remove("selected"));
          img.classList.add("selected");
          currentTimestamp = frame.timestamp;
          currentFrameId = Math.round(frame.timestamp * state.fps);
          card.querySelector(".ev-frame").textContent = currentFrameId;
        });
        cell.appendChild(img);
        const label = document.createElement("div");
        label.className = "frame-label";
        label.textContent = fmtTime(frame.timestamp);
        cell.appendChild(label);
        denseGrid.appendChild(cell);
      });
      wrap.appendChild(denseGrid);
    } catch (error) {
      wrap.innerHTML = `<span class="status error">${error.message}</span>`;
    }
  });

  card.querySelector(".ev-verify").addEventListener("click", async () => {
    const wrap = card.querySelector(".verify-result-wrap");
    wrap.innerHTML = "Đang verify với VLM...";
    try {
      const data = await postJSON("/vlm/verify", {
        video_id: videoId,
        event_text: event.event_text,
        coarse_timestamp: currentTimestamp,
      });
      currentTimestamp = data.timestamp;
      currentFrameId = Math.round(data.timestamp * state.fps);
      card.querySelector(".ev-frame").textContent = currentFrameId;
      wrap.innerHTML = `<div class="verify-result">Best frame: ${currentFrameId} (${fmtTime(currentTimestamp)}) &middot; confidence ${data.confidence.toFixed(2)} &middot; matches=${data.matches} &middot; "${data.reason}"</div>`;
    } catch (error) {
      wrap.innerHTML = `<span class="status error">${error.message}</span>`;
    }
  });

  return card;
}

function renderTrakeAlignment(videoId, alignment) {
  trakeEvents.innerHTML = "";
  if (!alignment || !alignment.feasible) {
    trakeEvents.innerHTML = `<div class="status error">Không tìm được alignment khả thi cho ${videoId}.</div>`;
    renderTimeline([]);
    return;
  }
  state.trake.videoId = videoId;
  state.trake.assignments = alignment.assignments;
  alignment.assignments.forEach((event, index) => {
    trakeEvents.appendChild(eventCard(event, index, videoId));
  });
  renderTimeline(alignment.assignments);
  loadVideo(videoId, alignment.assignments[0].frame_id, alignment.assignments[0].timestamp);
}

function setActiveTrakeChip(videoId) {
  trakeVideoCandidates.querySelectorAll(".trake-video-card").forEach((chip) => {
    chip.classList.toggle("active", chip.dataset.videoId === videoId);
  });
}

async function runTrake(query) {
  trakePanel.classList.remove("hidden");
  grid.classList.add("hidden");
  resultsPagination.classList.add("hidden");
  trakeEvents.innerHTML = "Đang decompose + tìm video...";
  trakeVideoCandidates.innerHTML = "";

  try {
    const data = await postJSON("/search/trake", { query });
    trakeEvents.innerHTML = "";

    state.trake = { videoId: null, assignments: [], eventTexts: data.event_sequence.events };

    trakeVideoCandidates.innerHTML = "<strong>Candidate videos:</strong> ";
    data.candidates.forEach((candidate) => {
      const chip = document.createElement("button");
      chip.type = "button";
      chip.className = "trake-video-card";
      chip.dataset.videoId = candidate.video_id;
      chip.textContent = `${candidate.video_id} (score ${candidate.total_score.toFixed(3)})`;
      chip.addEventListener("click", async () => {
        if (chip.classList.contains("active")) return;
        const originalText = chip.textContent;
        chip.textContent = `${candidate.video_id} — đang align...`;
        try {
          const alignment = await postJSON("/trake/align", {
            video_id: candidate.video_id,
            events: state.trake.eventTexts,
          });
          setActiveTrakeChip(candidate.video_id);
          renderTrakeAlignment(candidate.video_id, alignment);
        } catch (error) {
          trakeEvents.innerHTML = `<div class="status error">${error.message}</div>`;
        } finally {
          chip.textContent = originalText;
        }
      });
      trakeVideoCandidates.appendChild(chip);
    });

    if (!data.top_alignment || !data.top_alignment.feasible) {
      trakeEvents.innerHTML = '<div class="status error">Không tìm được alignment khả thi cho video candidate hàng đầu.</div>';
      renderTimeline([]);
      return;
    }

    const videoId = data.top_alignment.video_id;
    setActiveTrakeChip(videoId);
    renderTrakeAlignment(videoId, data.top_alignment);
  } catch (error) {
    trakeEvents.innerHTML = `<div class="status error">${error.message}</div>`;
  }
}

// ---------- results header: CSV export + stats ----------

csvExportBtn.addEventListener("click", () => {
  if (!state.kisResults.length) {
    setStatus("Chưa có kết quả để export", true);
    return;
  }
  const header = "video_id,frame_id,score,timestamp\n";
  const body = state.kisResults.map((r) => `${r.video_id},${r.frame_id},${r.score ?? ""},${r.timestamp}`).join("\n");
  const blob = new Blob([header + body], { type: "text/csv" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `kis_results_${Date.now()}.csv`;
  document.body.appendChild(a);
  a.click();
  a.remove();
  URL.revokeObjectURL(url);
});

function computeStats(rows) {
  if (!rows || rows.length === 0) return null;
  const scores = rows.map((r) => r.score).filter((v) => v != null);
  const avg = scores.reduce((a, b) => a + b, 0) / scores.length;
  return {
    count: rows.length,
    avg,
    min: Math.min(...scores),
    max: Math.max(...scores),
    withKeyframe: rows.filter((r) => r.keyframe_available).length,
    withSemantic: rows.filter((r) => r.semantic_rank != null).length,
    withMetadata: rows.filter((r) => r.metadata_rank != null).length,
    withObject: rows.filter((r) => r.object_rank != null).length,
  };
}

viewStatsBtn.addEventListener("click", () => {
  const stats = computeStats(state.kisResults);
  statsModalBody.innerHTML = stats
    ? `
    <div class="detail-row"><span class="k">Results</span><span class="v">${stats.count}</span></div>
    <div class="detail-row"><span class="k">Avg score</span><span class="v">${stats.avg.toFixed(4)}</span></div>
    <div class="detail-row"><span class="k">Min / Max</span><span class="v">${stats.min.toFixed(4)} / ${stats.max.toFixed(4)}</span></div>
    <div class="detail-row"><span class="k">Keyframe available</span><span class="v">${stats.withKeyframe}/${stats.count}</span></div>
    <div class="detail-row"><span class="k">CLIP branch hits</span><span class="v">${stats.withSemantic}/${stats.count}</span></div>
    <div class="detail-row"><span class="k">BM25 branch hits</span><span class="v">${stats.withMetadata}/${stats.count}</span></div>
    <div class="detail-row"><span class="k">Object branch hits</span><span class="v">${stats.withObject}/${stats.count}</span></div>
  `
    : '<div class="empty-note">Chưa có kết quả để thống kê</div>';
  statsModal.classList.remove("hidden");
});
statsModalClose.addEventListener("click", () => statsModal.classList.add("hidden"));
statsModal.addEventListener("click", (event) => {
  if (event.target === statsModal) statsModal.classList.add("hidden");
});

// ---------- filters popover ----------

filtersBtn.addEventListener("click", (event) => {
  event.stopPropagation();
  filtersPopover.classList.toggle("hidden");
});
document.addEventListener("click", (event) => {
  if (!filtersPopover.contains(event.target) && event.target !== filtersBtn) {
    filtersPopover.classList.add("hidden");
  }
});

chipExpansion.addEventListener("click", () => {
  state.expandOn = !state.expandOn;
  chipExpansion.classList.toggle("on", state.expandOn);
});

document.getElementById("mode-debug").addEventListener("change", (event) => {
  document.body.classList.toggle("debug-mode", event.target.checked);
});
document.getElementById("mode-competition").addEventListener("change", (event) => {
  document.body.classList.toggle("competition-mode", event.target.checked);
});

// ---------- theme + preview expand ----------

if (localStorage.getItem("aic-theme") === "light") {
  document.documentElement.classList.add("theme-light");
}
themeToggle.addEventListener("click", () => {
  document.documentElement.classList.toggle("theme-light");
  localStorage.setItem("aic-theme", document.documentElement.classList.contains("theme-light") ? "light" : "dark");
});

previewExpand.addEventListener("click", () => {
  appShell.classList.toggle("preview-expanded");
});

// ---------- system status ----------

async function loadSystemStatus() {
  try {
    await getJSON("/api/system/status");
    systemStatusText.textContent = "SYSTEM READY";
    systemStatusEl.classList.remove("offline");
  } catch (error) {
    systemStatusText.textContent = "SYSTEM OFFLINE";
    systemStatusEl.classList.add("offline");
  }
}
loadSystemStatus();
setInterval(loadSystemStatus, 30000);

// ---------- TRAKE query builder (multi-event input) ----------

function renumberTrakeEvents() {
  const rows = Array.from(trakeEventRows.children);
  rows.forEach((row, index) => {
    row.querySelector(".trake-event-num").textContent = `E${index + 1}`;
    row.querySelector(".trake-event-input").placeholder = `Event ${index + 1}: mô tả sự kiện...`;
  });
  const removeButtons = trakeEventRows.querySelectorAll(".trake-event-remove");
  removeButtons.forEach((btn) => {
    btn.disabled = removeButtons.length <= 1;
  });
}

function addTrakeEventRow(value = "") {
  const row = document.createElement("div");
  row.className = "trake-event-row";
  row.innerHTML = `
    <span class="trake-event-num"></span>
    <input type="text" class="trake-event-input" autocomplete="off">
    <button type="button" class="trake-event-remove" title="Xoá event">&times;</button>
  `;
  row.querySelector(".trake-event-input").value = value;
  row.querySelector(".trake-event-remove").addEventListener("click", () => {
    if (trakeEventRows.children.length <= 1) return;
    row.remove();
    renumberTrakeEvents();
  });
  trakeEventRows.appendChild(row);
  renumberTrakeEvents();
}

function buildTrakeQuery() {
  const values = Array.from(trakeEventRows.querySelectorAll(".trake-event-input"))
    .map((input) => input.value.trim())
    .filter(Boolean);
  return values.map((text, index) => `E${index + 1}: ${text}`).join(" ");
}

trakeAddEventBtn.addEventListener("click", () => addTrakeEventRow());
addTrakeEventRow();
addTrakeEventRow();

// ---------- query type tabs ----------

const tabButtons = document.querySelectorAll("#query-tabs button");
tabButtons.forEach((button) => {
  button.addEventListener("click", () => {
    tabButtons.forEach((b) => b.classList.remove("active"));
    button.classList.add("active");
    state.queryType = button.dataset.type;
    const isTrake = state.queryType === "trake";
    queryInput.classList.toggle("hidden", isTrake);
    trakeQueryBuilder.classList.toggle("hidden", !isTrake);
    trakeHint.classList.toggle("hidden", !isTrake);
  });
});

// ---------- main search dispatch ----------

form.addEventListener("submit", async (event) => {
  event.preventDefault();
  const type = state.queryType;
  const query = type === "trake" ? buildTrakeQuery() : queryInput.value.trim();
  if (!query) {
    if (type === "trake") setStatus("Nhập ít nhất 1 event", true);
    return;
  }

  grid.classList.add("hidden");
  resultsPagination.classList.add("hidden");
  qnaPanel.classList.add("hidden");
  trakePanel.classList.add("hidden");
  setStatus("Đang tìm...");

  const startedAt = performance.now();
  try {
    if (type === "kis") {
      grid.classList.remove("hidden");
      const topK = 100;
      resultsTopkBadge.textContent = `Top ${topK}`;
      const data = await postJSON("/search/kis", { query, top_k: topK, expand: state.expandOn });
      const elapsed = ((performance.now() - startedAt) / 1000).toFixed(3);
      state.kisResults = data.results;
      state.kisExpanded = Boolean(data.expand);
      state.kisPage = 1;
      renderKisPage();
      resultsMeta.textContent = `Found ${data.results.length} results (${elapsed}s)${data.expand ? " · LLM expanded" : ""}`;
      setStatus(`${data.results.length} kết quả${data.expand ? " (đã expand qua LLM)" : ""}`);
    } else if (type === "qna") {
      resultsTopkBadge.textContent = "Q&A";
      resultsMeta.textContent = "";
      await runQna(query);
      setStatus("Đã trả lời Q&A");
    } else if (type === "trake") {
      resultsTopkBadge.textContent = "TRAKE";
      resultsMeta.textContent = "";
      await runTrake(query);
      setStatus("Đã dựng Event Panel TRAKE");
    }
  } catch (error) {
    setStatus(error.message, true);
  }
});
