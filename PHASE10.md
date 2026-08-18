# Giai đoạn 10 — TRAKE: Fine Frame Alignment

Keyframes BTC (precomputed CLIP, ~1 keyframe mỗi vài giây) chỉ đủ để tìm **vị trí coarse** (Stage 9). Ground-truth TRAKE có thể hẹp dưới 10 frame — cần quay lại **video gốc**, decode dense quanh vị trí coarse, rồi để VLM chọn khung hình đúng ngữ nghĩa nhất.

```text
Coarse Event Position (Stage 9, vd E2 ≈ 725.1s)
        ↓
Video gốc (Video_Path trong frame_mapping, cùng dạng Archive.zip::member như Keyframe_Path)
        ↓
Decode ±2–5 giây quanh vị trí coarse (KHÔNG decode toàn bộ video)
        ↓
Dense Candidate Frames
        ↓
VLM chấm điểm từng frame theo semantic definition của event
        ↓
Semantic Keyframe (frame + timestamp + confidence)
```

## Kiến trúc (`trake/` + `vlm/frame_verifier.py`)

### `VideoCatalog` — tra Video_Path/FPS

Đọc trực tiếp `frame_mapping.parquet` (cột `Video_ID, Video_Path, FPS, Video_Available`) một lần, tra theo `video_id`. `Video_Path` đúng dạng `"Videos_L21_a.zip::video/L21_V001.mp4"` — cùng quy ước `archive_ref` như `Keyframe_Path` của Stage 1.

### `decode_dense_window` — chỉ decode vùng cần thiết

Extract **một video duy nhất** từ ZIP ra file tạm (`zipfile.open` + `shutil.copyfileobj`, không giải nén cả archive nhiều GB), mở bằng OpenCV (`cv2.VideoCapture`), seek tới từng mốc thời gian trong cửa sổ `[center − window, center + window]` bằng `CAP_PROP_POS_MSEC`, encode mỗi frame decode được thành JPEG bytes.

Không cài `ffmpeg`/`decord` trong môi trường này — dùng `opencv-python` (đã có sẵn) làm baseline. Hạn chế đã biết: seek theo mili giây của OpenCV có thể trượt tới frame decode được gần nhất tùy codec, không chính xác tuyệt đối từng mili giây — chấp nhận được vì bước sau (VLM) chấm điểm lại từng candidate đã decode, không phụ thuộc việc seek chính xác tuyệt đối.

### VLM chấm điểm — dùng chung với Stage 11

`vlm/frame_verifier.py::QwenFrameEventScorer` — cùng cơ chế OpenAI-compatible/DashScope, model mặc định `qwen3.8-max`. Input: 1 ảnh + 1 event text. Output JSON `{"matches": bool, "confidence": number, "reason": string}`. Module này dùng chung cho cả Stage 10 (chọn 1 frame tốt nhất trong dense window) lẫn Stage 11 (so sánh candidate gần biên).

### `FineFrameAligner`

`refine(video_id, event_text, coarse_timestamp)`: decode dense window → (nếu quá `max_candidate_frames`, subsample đều để giới hạn số lệnh gọi VLM) → chấm điểm từng candidate → chọn confidence cao nhất → `SemanticKeyframe`.

Mặc định: `window_seconds=2.5` (±2.5s), `step_seconds=0.5`, `max_candidate_frames=12` — nằm trong khoảng ±2–5s đề bài yêu cầu, không decode nguyên video.

## Chạy thử

```powershell
python -m trake.frame_refinement_cli refine L21_V001 "a woman standing on a street" 725.1 --window-seconds 2.0 --step-seconds 0.5
python -m trake.frame_refinement_cli align L21_V001 "event 1" "event 2" ...   # Stage 9 + Stage 10 nối tiếp
```

## Đã chạy thật — có xác minh trực quan

```powershell
python -m trake.frame_refinement_cli refine L21_V001 "a woman standing on a street" 725.1 --window-seconds 2.0 --step-seconds 0.5
```

Kết quả thật: extract video từ `Videos_L21_a.zip`, decode 9 candidate trong cửa sổ ±2s, gọi Qwen VL 9 lần, chọn `timestamp=725.6s, confidence=0.8, matches=true, reason="figures in raincoats standing on flooded street at night"`.

**Đã trích xuất đúng frame đó và xem trực tiếp** — ảnh thật đúng là cảnh vài người mặc áo mưa đứng trên đường ngập nước ban đêm, khớp chính xác với `reason` VLM đưa ra. Xác nhận cả decode lẫn VLM scoring hoạt động đúng trên dữ liệu thật.

## Lưu ý môi trường: `python -m` đôi lúc lỗi torch DLL

Trong quá trình test, `python -m trake.frame_refinement_cli ...`/`phase11 ...` (các lệnh cần load CLIP/torch) thỉnh thoảng lỗi `OSError: [WinError 1114] DLL initialization routine failed` khi torch load — nhưng `import torch` chạy độc lập hoặc gọi trực tiếp qua script Python luôn thành công. Đây là hiện tượng flaky khi load DLL torch trên Windows (nghi do tranh chấp tài nguyên/scan antivirus sau nhiều lượt ghi file tạm khi decode video), không phải lỗi logic — thử lại lệnh (hoặc gọi trực tiếp bằng script thay vì qua `python -m`) sẽ qua.

## Kiểm thử

```powershell
python -m pytest -q
```

`tests/test_phase10_fine_alignment.py` khóa: `VideoCatalog` tra đúng/báo lỗi khi thiếu video hoặc `Video_Available=False`, `decode_dense_window` extract từ ZIP + decode thật bằng video mp4 synthetic dựng qua OpenCV (không cần dữ liệu BTC thật), `FineFrameAligner` chọn đúng confidence cao nhất và subsample đúng khi vượt cap, và adapter `QwenFrameEventScorer` (client giả lập, không cần mạng).
