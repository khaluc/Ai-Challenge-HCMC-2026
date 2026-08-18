# Giai đoạn 9 — TRAKE: Coarse Temporal Alignment

Với mỗi video ứng viên (từ Stage 8), tìm chuỗi frame `f1 < f2 < ... < fn` — một frame cho mỗi event, **theo đúng thứ tự thời gian** — tối đa hóa tổng độ khớp:

```text
maximize  S(E1,f1) + S(E2,f2) + ... + S(En,fn)
subject to  f1 < f2 < ... < fn
```

## Vì sao không thể chọn greedy argmax

Nếu chọn riêng frame khớp nhất cho từng event một cách độc lập, nhiều event có thể trùng vào cùng một frame (frame đó khớp tốt với cả 2-3 mô tả), hoặc thứ tự bị đảo — cả hai đều vô nghĩa cho TRAKE. Cần một thuật toán tối ưu **có ràng buộc thứ tự**.

## Baseline: Dynamic Programming (không cần DTW)

`trake/temporal_dp.py::align_events`:

```text
dp[i][j] = max frame khớp cho event i tại frame index j
         = similarity[i][j] + max( dp[i-1][j'] )  với mọi j' < j
```

Backtrack từ `argmax_j dp[N-1][j]` để lấy chuỗi frame cuối cùng. Độ phức tạp O(N·M²) (N = số event, M = số keyframe trong video) — đơn giản, không phải DTW, đúng như đề bài không bắt buộc. M thực tế chỉ vài chục đến vài trăm keyframe/video (trung bình ~200) nên O(N·M²) đủ nhanh.

Nếu video có ít keyframe hơn số event (`M < N`) → trả `feasible=False`, không cố gán ép.

### Transition penalty (tùy chọn)

```text
penalty(j, j') = transition_penalty_weight × max(0, min_frame_gap − (j − j'))
```

Mặc định `transition_penalty_weight=0` (tắt, đúng tinh thần "không bắt buộc" của đề bài). Bật lên để phạt việc gán hai event liên tiếp vào các frame quá sát nhau — tránh trường hợp DP chọn hai frame gần như trùng thời điểm cho hai event khác nhau.

## `CoarseTemporalAligner`

Dùng CLIP text encoder + vector keyframe **đã có sẵn từ Stage 1** (không cần load lại ảnh, không cần model image encoder riêng) qua `Phase1HybridStore.best_frames_in_video(video_id, vector, limit=...)` — lấy toàn bộ frame của một video kèm cosine similarity, rồi đưa vào `align_events`.

## Chạy thử

```powershell
python -m trake.temporal_dp_cli align L21_V001 "athlete running toward the bar" "athlete taking off" "athlete clearing the bar" "athlete landing on the mat"
```

## Đã chạy thật trên dữ liệu thật

```powershell
python -m trake.temporal_dp_cli align L21_V001 "a woman standing on a street" "a car parked on the street" "people walking"
```

Kết quả: `feasible=true`, gán đúng 3 frame theo thứ tự `keyframe_index` tăng dần (177 < 300 < 305, tương ứng timestamp 725.1s < 1229.07s < 1245.3s) — đúng ràng buộc `f1 < f2 < f3`.

## Kiểm thử

```powershell
python -m pytest -q
```

`tests/test_phase9_alignment.py` khóa: DP tôn trọng ràng buộc tăng dần ngay cả khi greedy argmax sẽ vi phạm (dựng similarity matrix tay để ép tình huống xung đột), `feasible=False` khi thiếu frame, transition penalty đẩy hai event ra xa nhau khi bật, validation `CoarseAlignment` từ chối assignment không tăng dần, và wiring thật qua `CoarseTemporalAligner` trên dữ liệu synthetic (CLIP vector thật, không mock similarity).
