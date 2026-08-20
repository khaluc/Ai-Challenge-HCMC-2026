# Kiến trúc luồng xử lý — KIS / Q&A / TRAKE

Tài liệu này ghi lại **luồng thật đang chạy** trong code (không phải spec đề
xuất) cho 3 kiểu truy vấn của web app, tính tới thời điểm viết (sau khi đã nối
LLM Expansion, tách scene/question, phân loại closed-set/open-ended, joint
confidence, và multi-candidate align cho TRAKE). Mỗi bước đều trỏ tới
file:line thật trong `src/` và `app.py` để tra cứu lại khi cần.

---

## 1. KIS (Known-Item Search)

```
Query (text)
     │
     ▼
chip "LLM Expansion" tắt (mặc định)          chip "LLM Expansion" bật
     │                                              │
     ▼                                              ▼
pipeline.search()                        get_expansion_service().search()
KISPipeline → HybridTextualKIS           ExpandedHybridSearch
[kis/pipeline.py:48]                     [llm/expansion_retrieval.py:68]
     │                                              │
     │                              QwenQueryUnderstanding.understand()
     │                              → 2-4 cách diễn đạt lại (LLM, Qwen)
     │                              [llm/query_expansion.py:390]
     │                                              │
     │                              mỗi expansion search riêng, RRF gộp lại
     │                              [llm/expansion_fusion.py]
     │                                              │
     └──────────────────┬───────────────────────────┘
                         ▼
          HybridTextualKIS.search_batch()
          [retrieval/hybrid_search.py]
                         │
          ┌──────────────┼───────────────┐
          ▼               ▼               ▼
        CLIP            BM25          Objects
      (semantic)      (metadata)   (object detector concepts)
          │               │               │
          └───────────────┼───────────────┘
                           ▼
                RRF/weighted fusion → HybridHit / ExpandedHit
                           │
                           ▼
       apply_ranking() — tiered MMR diversity rerank
       [app.py:137, submission/ranking_optimizer.py:77]
       rank 1-5 giữ nguyên thứ tự (diversity_weight=0),
       diversity tăng dần từ rank 6 trở đi
                           │
                           ▼
       JSON → frontend render lưới ảnh (candidateCard)
       click → showDetails(): Score Breakdown / Match Info /
               Objects & Concepts (hoặc Best Expansion/Relation/
               Attributes nếu đang ở chế độ Expansion)
```

**Route:** `POST /search/kis` — [app.py:200](app.py#L200)

**Trạng thái các nhánh CLIP/BM25/Objects:** luôn bật cùng lúc cho mọi truy
vấn (không có cách tắt riêng từng nhánh qua UI) — 3 chip đó trong giao diện
chỉ là hiển thị trạng thái, không phải toggle thật.

---

## 2. Q&A (Video Question Answering)

```
Question (text)
     │
     ▼
is_speech_question(question)?  [qna/routing.py:45] — khớp từ khóa
     │
     ├── CÓ (hỏi về lời nói) ─────► route="transcript"
     │                              chưa có transcript/ASR backend
     │                              → trả honest note, không đoán mò
     │                              [qna/pipeline.py:65-84]
     │
     └── KHÔNG (route="visual")
              │
              ▼
     QwenQuestionSplitter.split(question)  [qna/query_split.py]
     → scene_description (dùng để retrieval)
     → question (giữ nguyên, gửi cho VLM)
              │
              ▼
     classify_question_type(question)  [qna/question_type.py]
     → "closed_set" (đếm/màu sắc/có-không) hoặc "open_ended"
     (rule-based, không tốn gọi LLM)
              │
              ▼
     QACandidateSearch.find_candidates(scene_description)
     [qna/candidates.py:96] — dùng chung HybridTextualKIS/
     ExpandedHybridSearch với KIS, thu hẹp còn top video_limit
     video × top frame_limit keyframe (config.yaml: 5×5 mặc định)
              │
              ▼
     với TỪNG keyframe (tuần tự, từng ảnh một):
         load_keyframe_bytes() → 1 ảnh JPEG tĩnh
         QwenVLAnswerer.answer(image, question, question_type)
         [vlm/image_qa.py:102] — chọn CLOSED_SET_SYSTEM_PROMPT
         hoặc OPEN_ENDED_SYSTEM_PROMPT tùy question_type
         → {answer, confidence}
              │
              ▼
     joint_confidence = video_score × frame_score × answer_confidence
     [qna/pipeline.py:105-134]
     - video_score  = retrieval_score tốt nhất của video đó, chuẩn hóa
                       theo video điểm cao nhất trong batch
     - frame_score  = retrieval_score của chính keyframe đó, chuẩn hóa
                       theo keyframe điểm cao nhất trong batch
              │
              ▼
     sort theo joint_confidence giảm dần → top_k QAResult
     (mang theo timestamp/faiss_index/keyframe_index để UI seek
     đúng video)
```

**Route:** `POST /search/qna` — [app.py:285](app.py#L285)
**Route phụ:** `POST /vlm/ask` — hỏi thủ công N frame đã tick, cùng
`QwenVLAnswerer` + `classify_question_type`, bỏ qua bước retrieval/split.

**Giới hạn đã biết:** VLM chỉ nhìn **đúng 1 ảnh tĩnh** mỗi lần gọi — không
có bước nào đưa nhiều frame/toàn bộ video vào cùng 1 lần gọi, nên câu hỏi
cần hiểu trình tự thời gian ("...sau khi... gần cuối video") không được
xử lý đúng bản chất.

---

## 3. TRAKE (chuỗi sự kiện theo thời gian)

```
Query = "E1: ... E2: ... E3: ..." (frontend ghép từ các ô Event riêng)
     │
     ▼
QwenEventDecomposer.decompose(query)  [llm/event_parser.py]
→ EventSequence(events=[...câu event tiếng Anh theo thứ tự...],
                expansions=[...])
     │
     ▼
TRAKEVideoRetrieval.find_candidate_videos()  [trake/video_retrieval.py]
→ batch search [full_query + từng event + expansions] qua
  HybridTextualKIS, rồi gộp điểm mỗi video:
     video_score = 1.0·global_similarity + 1.0·event_coverage
                 + 1.0·bm25_score + 1.0·multi_query_vote
     [trake/aggregation.py:76, trọng số mặc định đều =1.0]
→ Top 3-5 video candidate (hiện thành chip trong UI)
     │
     ▼
CoarseTemporalAligner.align(video, events)  [trake/temporal_dp.py:150]
→ CLIP text-image similarity + dynamic programming
  → gán 1 frame cho mỗi event, đúng thứ tự
→ TỰ ĐỘNG chạy sẵn cho candidate #1 (điểm cao nhất) khi search xong
     │
     ▼
Event Panel (mỗi event: Preview / Refine / Verify with VLM) + timeline
     │
     ├─ click chip candidate KHÁC (vd L26_V052)
     │       │
     │       ▼
     │  POST /trake/align {video_id, events}  [app.py, mới thêm]
     │       │
     │       ▼
     │  CoarseTemporalAligner.align(video_id_mới, events) — dùng lại
     │  đúng danh sách event đã decompose 1 lần, không gọi lại LLM
     │       │
     │       ▼
     │  Event Panel dựng lại cho video vừa chọn
     │
     ├─ "Refine" → POST /refine → decode_dense_window()
     │             (chỉ decode frame quanh timestamp, KHÔNG gọi VLM)
     │
     └─ "Verify with VLM" → POST /vlm/verify → FineFrameAligner.refine()
                → QwenFrameEventScorer chấm từng frame trong dense
                  window, chọn frame khớp nhất với đúng 1 event đó
```

**Routes:** `POST /search/trake`, `POST /trake/align`, `POST /refine`,
`POST /vlm/verify` — tất cả trong `app.py`.

---

## Bảng tóm tắt gọi LLM/VLM thật theo từng luồng

| Luồng | Bước gọi LLM/VLM | Model | Bắt buộc hay tùy chọn |
|---|---|---|---|
| KIS | Query expansion (`QwenQueryUnderstanding`) | Qwen text | Tùy chọn — chip "LLM Expansion" |
| Q&A | Scene/question split (`QwenQuestionSplitter`) | Qwen text | Luôn chạy (route visual) |
| Q&A | Trả lời từng keyframe (`QwenVLAnswerer`) | Qwen vision | Luôn chạy (route visual) |
| TRAKE | Decompose query thành event (`QwenEventDecomposer`) | Qwen text | Luôn chạy |
| TRAKE | Verify with VLM (`QwenFrameEventScorer`) | Qwen vision | Chỉ khi bấm nút "Verify with VLM" |

Tất cả đều dùng chung 1 API key `DASHSCOPE_API_KEY` (Qwen/DashScope) — dự
án hiện không có key nào khác (không có GPT-4V/Claude vision/LLaVA cục bộ)
nên bất cứ chỗ nào cần "chọn model theo loại câu hỏi" (như closed-set vs
open-ended trong Q&A) đều đổi **prompt** trên cùng 1 model Qwen, không phải
đổi model thật.
