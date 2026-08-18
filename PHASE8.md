# Giai đoạn 8 — TRAKE: Video Retrieval (Top 3–5 candidate, không chốt 1 video)

Sai video khiến toàn bộ R-Score của câu trả lời TRAKE bằng 0, nên Stage 8 **không bao giờ chốt ngay một video**. Thay vào đó: search nhiều sub-query (full query + từng event + LLM expansions), gộp điểm theo video, giữ lại Top 3–5 video ứng viên cho bước sau (định vị frame theo từng event bên trong video đã chọn — chưa nằm trong phạm vi Stage 8).

```text
Full Query + Event 1..N + LLM Expansions
                │  (mỗi sub-query search riêng qua Stage 3 hybrid)
                ▼
       Top-K video mỗi sub-query
                │
         aggregate theo video_id
                │
video_score = global_similarity + event_coverage + BM25 + multi_query_vote
                │
                ▼
          Top 3–5 video
```

## Kiến trúc (`trake/` — `video_retrieval.py`, `aggregation.py`, `video_retrieval_cli.py`)

### Batch search — `retrieval.py`

`TRAKEVideoRetrieval.find_candidate_videos(query)`:
1. Gọi `EventDecompositionProtocol.decompose(query)` (Stage 7) — 1 lần.
2. Gộp `[full_query] + events + expansions` thành một batch `Query` list, gọi **một lượt** `HybridTextualKIS.search_detailed_batch(...)` cho toàn bộ sub-query (một lần encode CLIP cho tất cả, không lặp lại việc load model — cùng pattern hiệu năng đã dùng ở Stage 4/5).
3. Gọi `store.search_metadata(...)` riêng cho full query text để lấy BM25 thô theo video (độc lập với nhánh metadata đã blend trong Stage 3, dùng đúng nghĩa "BM25_score" như công thức đề bài).
4. Gộp kết quả qua `aggregate_video_candidates`.

### Công thức gộp điểm — `aggregation.py`

```text
video_score(v) = w1·global_similarity(v) + w2·event_coverage(v) + w3·bm25_score(v) + w4·multi_query_vote(v)
```

Mỗi thành phần chuẩn hóa về [0, 1] độc lập trước khi cộng:

| Thành phần | Nguồn | Cách tính |
|---|---|---|
| `global_similarity` | Hit tốt nhất của **full query** cho video đó | `(semantic_score + 1) / 2`, clamp [0,1]; = 0 nếu video không nằm trong Top-K của full query |
| `event_coverage` | Số event (không tính expansion) mà video xuất hiện trong Top-K của sub-query event đó | `matched_events / total_events` |
| `bm25_score` | `store.search_metadata` cho full query | min-max normalize + đảo dấu (FTS5 `bm25()`: điểm càng thấp càng liên quan) |
| `multi_query_vote` | Toàn bộ sub-query (full + event + expansion) | tỉ lệ sub-query có video đó trong Top-K của chính nó |

`VideoRetrievalConfig`: `video_limit=5`, `per_query_top_k=100`, 4 trọng số mặc định `1.0` bằng nhau — **baseline bảo thủ chưa tune**, giống nguyên tắc trọng số RRF/weighted ở Stage 3.

`video_id` không xuất hiện ở bất kỳ sub-query nào và không có BM25 hit thì không được xét (không nằm trong `candidate_video_ids`).

## Chạy thử

```powershell
python -m trake.video_retrieval_cli find-videos "E1: chạy đà E2: giậm nhảy E3: bay qua xà E4: tiếp đất" --video-limit 5
```

## Đã chạy thật trên toàn bộ 177.321 vector — kết quả và giới hạn cần biết

Chạy full pipeline (Stage 7 decompose thật qua Qwen + Stage 8 batch search 8 sub-query qua Stage 3 hybrid thật), 51 giây, không lỗi. Top 5 video trả về với đủ 4 thành phần điểm, ví dụ video rank 1 (`L24_V028`): `event_coverage=1.0` (khớp cả 4 event) nhưng `global_similarity=0.0` (không nằm trong Top-K của full query gốc, vì câu gốc còn giữ nguyên nhãn `"E1:"` `"E2:"` khiến CLIP khớp kém) — **đúng minh chứng cho lý do phải tách event**: nếu chỉ search nguyên câu query một lần, video này sẽ bị bỏ sót hoàn toàn.

**Nhưng khi đối chiếu metadata thật của 5 video trả về, không video nào là nội dung nhảy cao** — 3 video là thi múa lân sư rồng (Cúp Chợ Lớn HTV 2024), 2 video là clip ôn thi THPT (Hóa học, Địa lý). Cơ chế Stage 7-8 chạy đúng kỹ thuật (decompose đúng, batch search đúng, tính điểm đúng công thức, giữ Top-5 thay vì chốt 1), nhưng:

1. Bộ dữ liệu L21–L30 hiện có (chương trình truyền hình HTV/Thanh Niên: múa lân, tin tức, ôn thi, phỏng vấn đường phố...) **nhiều khả năng không chứa cảnh nhảy cao thật** — ví dụ trong đề bài có thể không khớp với nội dung dataset này.
2. Ngay cả khi có, CLIP zero-shot trên corpus 177k frame đa dạng, không fine-tune cho thể thao điền kinh, có giới hạn về độ chính xác — đúng caveat đã lặp lại từ Stage 2 đến Stage 6.

Kết luận trung thực: **hạ tầng TRAKE đã sẵn sàng và verify đúng cơ chế**, nhưng chưa có ví dụ nào chứng minh được độ chính xác thật trên nội dung phù hợp — cần query TRAKE thật khớp với nội dung dataset (hoặc ground truth chính thức) để đánh giá đúng.

## Kiểm thử

```powershell
python -m pytest -q
```

`tests/test_phase8_video_retrieval.py` khóa: công thức gộp 4 thành phần điểm (dựng `HybridHit`/`HybridSearchResult`/`MetadataVideoHit` tay để kiểm tra chính xác từng con số), validation `VideoRetrievalConfig` (trọng số không âm, ít nhất một trọng số dương, `per_query_top_k` ≤ 100), và `TRAKEVideoRetrieval` batch đúng thứ tự sub-query (searcher/store giả lập, không cần mạng/CLIP thật).
