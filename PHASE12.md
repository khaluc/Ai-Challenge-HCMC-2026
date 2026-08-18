# Giai đoạn 12 — Tối ưu Ranking theo R@K

Diversity không nên áp dụng đều cho cả Top-100 — vì `Final Score` giảm rất nhanh theo rank (đúng ở Rank 1 = 1.0, Rank 3 = 0.8, Rank 15 = 0.6, Rank 30 = 0.4, Rank 70 = 0.2), nên mục tiêu quan trọng nhất là Top-1 và Top-5 đúng tuyệt đối — không được đánh đổi bằng diversity.

## Chính sách theo tier (`submission/ranking_optimizer.py`)

| Rank | Vai trò | `diversity_weight` mặc định |
|---|---|---|
| 1–5 | Confidence cao nhất, tuyệt đối không đánh đổi | `0.0` |
| 6–20 | Candidate mạnh | `0.15` |
| 21–50 | Alternative hypotheses | `0.4` |
| 51–100 | Diversity / exploration | `0.7` |

## Thuật toán: MMR theo từng tier

Thuật toán thuần túy, không cần LLM/VLM. Với mỗi vị trí output (rank 1→100, chọn tuần tự), lấy trọng số diversity của tier tương ứng rồi chọn candidate còn lại tối đa hóa:

```text
mmr(c) = (1 − weight) × relevance(c) − weight × max_similarity(c, đã_chọn)
```

- `relevance(c)`: điểm gốc (từ Stage 3/4) min-max normalize về [0,1] một lần trên toàn bộ pool — không đổi trong suốt quá trình chọn.
- `similarity(a, b)`: `0` nếu khác `video_id`; nếu cùng video thì suy giảm tuyến tính theo khoảng cách thời gian (`1 − gap/near_duplicate_seconds`, cắt tại 0) — hai frame cùng video, cách nhau ≥ `near_duplicate_seconds` (mặc định 5s) coi như không trùng lặp.

Vì tier 1 (rank 1-5) có `diversity_weight=0`, 5 vị trí đầu **luôn** là 5 candidate có relevance cao nhất — không bao giờ bị đánh đổi cho diversity, đúng nguyên tắc "Top-1/Top-5 quan trọng nhất". Diversity chỉ thật sự có trọng số lớn ở tier cuối (51–100).

## Chạy thử

```powershell
python -m submission.cli rerank --predictions experiments/results/kis_expansion/predictions.csv --output submissions/reranked_predictions.csv
```

Đọc trực tiếp CSV nội bộ dạng Stage 3/4 (`query_id,rank,video_id,frame_id,score,timestamp,...`), rerank theo từng `query_id`, giữ nguyên mọi cột khác, chỉ cập nhật `rank`.

## Đã chạy thật trên predictions CSV thật (300 dòng, 3 query từ Stage 4)

- **Top-10 mỗi query giữ nguyên 100% thứ tự gốc** (đúng thiết kế: tier 1-2 gần như không đổi diversity).
- **76–87% vị trí trong Top-100 bị đổi chỗ** so với ranking gốc — thuật toán thực sự chạy và tái sắp xếp, không phải no-op.
- Đo `max_same_video_run` (chuỗi dài nhất các candidate liên tiếp cùng video ở rank 51-100): dữ liệu gốc từ Stage 3/4 hóa ra **đã tương đối đa dạng sẵn** (run dài nhất chỉ 1-2 candidate liên tiếp cùng video), nên hiệu quả thấy được qua metric này không kịch tính như ví dụ "500,501,502,503 cùng giả thuyết" trong đề bài — cơ chế MMR vẫn hoạt động đúng (đã verify riêng bằng test case cố ý dựng 10 candidate gần trùng lặp cùng video, xác nhận thuật toán kéo candidate từ video khác lên sớm hơn trong tier diversity cao).

Kết luận trung thực: thuật toán đúng và đã chạy thật, nhưng để thấy tác động rõ rệt cần dữ liệu có nhiều candidate gần-trùng-lặp thật sự (ví dụ Top-100 bị dominate bởi vài frame liên tiếp cùng 1 video) — chưa có ground truth/kết quả xấu thật để đối chiếu mức cải thiện R@K cụ thể.

## Kiểm thử

```powershell
python -m pytest -q
```

`tests/test_phase12_reranking.py` khóa: Top-tier giữ nguyên thứ tự pure-relevance dù có candidate gần trùng lặp, tier diversity cao kéo candidate từ video khác lên sớm (dựng tình huống 10 candidate cùng video + 1 candidate khác video, verify vị trí xuất hiện), `RankingTier` validation, và round-trip CSV qua `rerank_predictions_csv`.
