# Giai đoạn 6 — OCR (module bổ sung, CHƯA triển khai)

OCR **không** phải phần bắt buộc của pipeline. Đây là ghi chú thiết kế để triển khai nhanh sau này, không phải code đã có.

## Điều kiện kích hoạt

Chỉ bắt đầu code khi có **benchmark thật** (ground truth query chính thức của BTC, hoặc ít nhất một tập query mẫu đủ lớn) cho thấy tỉ lệ đáng kể câu hỏi/query phụ thuộc vào nội dung chữ trong khung hình:

- bảng hiệu
- chữ trên màn hình (slide, banner, phụ đề cứng)
- tên người xuất hiện dạng chữ (caption, name tag)
- số áo
- biển báo
- logo/text thương hiệu

Hiện dự án **chưa có ground truth/query chính thức** (đã ghi nhận xuyên suốt PHASE2–PHASE5), nên chưa có căn cứ để bật module này. Khi có bộ query thật, việc đầu tiên là đếm tỉ lệ query rơi vào các loại trên rồi mới quyết định.

## Thiết kế dự kiến khi triển khai

```text
Keyframes
   │
  OCR
   │
  Text
   │
BM25 Index (giống metadata_fts của Stage 1, nhưng theo frame thay vì theo video)
```

Hybrid Retrieval khi đó thêm một nhánh:

```text
CLIP + Metadata + ASR + Objects + OCR  →  RRF/weighted fusion (như Stage 3/4)
```

Ghi chú thiết kế (để không lặp lại các quyết định đã rút ra ở Stage 3–5):

- OCR index nên theo **frame_id**, không phải video_id — khác với metadata BM25 (theo video) — vì chữ trên biển hiệu/slide chỉ xuất hiện ở một số frame cụ thể trong video, không đại diện cho cả video.
- Nhánh OCR nên **abstain** (đóng góp 0 vào fusion) khi không tìm thấy text khớp, giống nguyên tắc các nhánh khác trong Stage 3 — không ép frame chỉ vì trùng vài ký tự OCR nhiễu.
- VLM (Stage 5) vẫn đọc được chữ trực tiếp trên candidate frame mà không cần OCR — khác biệt là OCR cho phép **search text trên toàn bộ dataset trước khi thu hẹp candidate**, còn VLM chỉ đọc được sau khi đã có candidate. Hai cái bổ sung nhau, không thay thế nhau.
- Cần chọn engine OCR hỗ trợ tiếng Việt có dấu (ví dụ PaddleOCR, hoặc OCR qua VLM API sẵn có nếu không muốn thêm dependency mới) — quyết định cụ thể để dành đến lúc triển khai thật.
- ASR trong sơ đồ trên vẫn là placeholder — BTC hiện chưa cung cấp transcript/ASR (xem PHASE3.md), nên nhánh ASR thật cũng đang chờ dữ liệu giống OCR.

## Việc cần làm trước khi bật module này

1. Có tập query thật (ground truth BTC hoặc tối thiểu một sample đủ đại diện).
2. Đếm tỉ lệ query thuộc các loại liệt kê ở trên.
3. Nếu tỉ lệ đủ cao để đáng đầu tư — quay lại phần "Thiết kế dự kiến" ở trên và triển khai theo đúng pattern pluggable/protocol đã dùng ở Stage 3–5 (nhánh mới trong `HybridConfig`, `BranchHit`/`fuse_rankings` đã sẵn generic cho việc thêm branch).
