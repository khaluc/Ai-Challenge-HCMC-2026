# Giai đoạn 11 — TRAKE: VLM Verify

Kiểm tra lại các candidate frame **gần nhau** (ví dụ frame 1520/1521/1522 quanh thời điểm "chân rời đất") bằng VLM, rồi bắt buộc kiểm tra lại thứ tự cuối cùng `f1 < f2 < ... < fn`.

## Kiến trúc (`trake/` — `frame_verification.py`, `frame_verification_schemas.py`, `verify_cli.py`)

`TRAKEVerifier` dùng chung `FrameEventScorerProtocol`/`QwenFrameEventScorer` với Stage 10 (`vlm/frame_verifier.py`) — không định nghĩa adapter LLM mới, vì bản chất là cùng một tác vụ "VLM chấm điểm 1 frame so với 1 event", chỉ khác input: Stage 10 chấm trên toàn bộ dense window, Stage 11 chấm trên một tập nhỏ candidate gần biên đã được thu hẹp thêm.

`verify_sequence(video_id, originals, candidates_by_event)`:

1. Với mỗi event, nếu có candidate → VLM chấm từng candidate, chọn confidence cao nhất (`verified=True`).
2. Event không có candidate nào được cung cấp → giữ nguyên assignment gốc (`verified=False`, không đoán mò).
3. **Kiểm tra lại `f1 < f2 < ... < fn`** trên toàn bộ chuỗi đã verify.
4. Nếu việc verify làm **vỡ thứ tự** → toàn bộ chuỗi quay về assignment gốc (trước verify) thay vì nộp một chuỗi TRAKE không hợp lệ. Đây là "safety net" bắt buộc theo đúng yêu cầu đề bài — sai thứ tự thì R-Score TRAKE cũng hỏng như sai video.

`verified=True` chỉ có nghĩa "đã chạy so sánh VLM và chọn candidate tốt nhất trong tập được đưa vào", **không** đảm bảo candidate đó thực sự khớp — VLM tự báo `matches`/`reason`/`confidence` riêng, và ngay cả candidate được chọn vẫn có thể có `reason` cho thấy không khớp hoàn toàn (xem phần chạy thật bên dưới).

## Chạy thử

```powershell
python -m trake.verify_cli verify L21_V001 "event 1" "event 2" ... --window-seconds 0.75 --step-seconds 0.25
```

Nội bộ: chạy Stage 9 (coarse) trước, rồi với mỗi event decode một cửa sổ **hẹp hơn** Stage 10 (mặc định ±0.75s, sát nghĩa "vài frame gần nhau" hơn là "dense window" ±2.5s của Stage 10) làm candidate cho verify.

## Đã chạy thật trên dữ liệu thật

Query 2 event: `"a woman standing on a street"`, `"a car parked on the street"` trên `L21_V001`. Kết quả:

- Event 1: chọn `timestamp=725.6s, confidence=0.8`, `reason="figures in rain gear standing on flooded street at night"` — khớp đúng ảnh thật (đã xác minh trực quan ở Stage 10).
- Event 2: chọn `timestamp=1229.32s, confidence=0.9`, nhưng **VLM tự báo** `reason="truck moving on street, no parked car"` — tức là trong 5 candidate được đưa vào, không candidate nào thực sự có ô tô đậu, VLM chọn cái "ít tệ nhất" và trung thực nói rõ lý do thay vì giả vờ khớp.
- `monotonic=true` (725.6 < 1229.32) — kiểm tra thứ tự cuối cùng qua.

Đây là ví dụ cho thấy: pipeline verify chạy đúng cơ chế và VLM được hiệu chỉnh khá tốt (biết tự báo khi không chắc), nhưng **chất lượng câu trả lời cuối cùng vẫn phụ thuộc vào retrieval/coarse alignment đã tìm đúng vùng thời gian hay chưa** — nếu Stage 9 gán sai vùng, Stage 11 chỉ chọn được "tốt nhất trong số các lựa chọn tồi", không tự tạo ra frame đúng nếu nó không có trong candidate.

## Kiểm thử

```powershell
python -m pytest -q
```

`tests/test_phase11_verify.py` khóa: chọn đúng candidate confidence cao nhất, **reverting về assignment gốc khi verify làm vỡ thứ tự** (test dựng tình huống cố ý gây đảo thứ tự), giữ nguyên khi không có candidate, và validation input (`originals` rỗng, `video_id` blank).
