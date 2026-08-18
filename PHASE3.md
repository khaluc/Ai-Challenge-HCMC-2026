# Giai đoạn 3 — Hybrid Retrieval cho Textual KIS

Stage 3 giữ nguyên CLIP baseline đã xác minh ở Stage 2 rồi bổ sung hai tín hiệu độc lập từ metadata BM25 và Faster R-CNN objects:

```text
                         Query
                           │
             ┌─────────────┼─────────────┐
             ▼             ▼             ▼
     CLIP ViT-B/32    Metadata FTS5   Object parser
             │             │             │
          FAISS          BM25        Co-occurrence SQL
             │             │             │
             └─────────────┼─────────────┘
                           ▼
                   RRF / Weighted Sum
                           ▼
                 dedupe video_id + frame_id
                           ▼
                        Top-100
```

Fusion chính là bước rerank của baseline hybrid này. Chưa có VLM/cross-encoder reranker; màu sắc và quan hệ không gian vẫn do CLIP hỗ trợ mềm, không bị biến thành điều kiện Faster R-CNN bắt buộc.

## Các nhánh hiện có

### 1. Semantic CLIP

- Dùng đúng `openai/clip-vit-base-patch32` tương thích với feature BTC.
- Mặc định lấy 500 candidate từ FAISS trước fusion.
- Luôn submit `Video_Frame_ID` lấy trực tiếp từ canonical mapping.
- Không lọc frame chỉ vì `Keyframe_Available=False`; vector và mapping vẫn tồn tại.

Không được thay riêng text encoder hiện tại bằng SigLIP/OpenCLIP ViT-L/14 rồi query FAISS cũ. Guard runtime chỉ có thể bắt sai dimension; một model khác cùng 512 chiều vẫn có thể lọt qua nhưng trả kết quả sai không gian embedding. Checkpoint hiện tại đã được kiểm chứng trong `experiments/results/kis_baseline/encoder_compatibility.json`. Mỗi model mới phải encode lại ảnh, xây image index tương ứng, chạy kiểm tra compatibility và được evaluator benchmark như một nhánh riêng.

### 2. Metadata BM25

- SQLite FTS5 có 873 document cấp video: title, description, keywords và author.
- Tokenizer hiện tại fold dấu Unicode, bỏ stopword đơn giản và không cần Underthesea.
- Search thử `AND` các keyword trước; nếu quá chặt thì fallback `OR`.
- Metadata chỉ xác định video. Hệ thống lấy các frame CLIP tốt nhất bên trong video đó thay vì gán BM25 cho mọi frame một cách tùy ý.

BTC không cung cấp transcript/ASR trong bộ dữ liệu hiện có và Stage 3 chưa triển khai nhánh ASR, nên manifest ghi rõ `asr_or_transcript_available=false`. Khi có dữ liệu thật, cần build transcript index và retrieval branch mới; contract fusion có thể được giữ nguyên.

### 3. Objects

Parser mặc định là deterministic và có thể thay bằng implementation theo `ObjectParserProtocol` sau này. Ví dụ:

```text
một người mặc áo đỏ đứng cạnh ô tô
```

được phân tích thành:

```json
{
  "object_concepts": ["person", "car"],
  "person_labels": ["Person", "Man", "Woman", "Boy", "Girl"],
  "car_labels": ["Car", "Land vehicle", "Vehicle"]
}
```

Các label trong cùng concept là OR; các concept khác nhau phải cùng xuất hiện trên một keyframe. Mỗi concept chỉ lấy detection confidence cao nhất nên nhiều box của cùng một class không bị hiểu nhầm thành nhiều loại object.

Object SQLite đã được Stage 1 lọc ở confidence `0.2`; CLI mặc định dùng `0.3` và từ chối threshold thấp hơn `0.2` thay vì âm thầm trả dữ liệu thiếu. `red`, `đứng cạnh` hoặc các quan hệ tương tự không phải hard filter.

Kiểm tra query parser mà không load model CLIP:

```powershell
python -m kis.hybrid_cli analyze-query "một người mặc áo đỏ đứng cạnh ô tô"
```

## Fusion

RRF là mặc định:

```text
score(frame) = Σ weight_branch / (rrf_k + rank_branch)
rrf_k = 60
```

Trọng số khởi đầu:

```text
semantic = 1.00
metadata = 0.65
objects  = 0.80
```

Nhánh không có tín hiệu sẽ abstain và đóng góp 0. Có thể dùng weighted sum; raw score được min-max normalize riêng trong từng nhánh trước khi nhân trọng số:

```powershell
python -m kis.hybrid_cli search "a presenter in a television studio" --fusion weighted
```

Các trọng số chỉ là baseline có chủ ý bảo thủ, chưa phải optimum. Phải tune bằng ground truth và evaluator, không đánh giá bằng vài query nhìn thuận mắt.

## Chạy một query

```powershell
python -m kis.hybrid_cli search `
  "một người mặc áo đỏ đứng cạnh ô tô" `
  --query-id query-01-kis `
  --top-k 100 `
  --predictions-output experiments/results/kis_hybrid/query-01-predictions.csv `
  --submission-dir experiments/results/kis_hybrid/submission
```

Các option quan trọng:

```text
--fusion rrf|weighted
--semantic-candidates 500
--metadata-videos 30
--metadata-frames 3
--object-candidates 300
--object-min-confidence 0.30
--semantic-weight 1.0
--metadata-weight 0.65
--object-weight 0.80
--no-metadata
--no-objects
```

Hai cờ `--no-metadata` và `--no-objects` dùng để chạy ablation CLIP-only ngay trên cùng code path.

## Chạy batch và xuất submission

Input CSV vẫn strict: header phải đúng `query_id,text`, không có cột thừa, `query_id` duy nhất và không rỗng. Khi dùng query chính thức, `query_id` phải bằng đúng basename file query BTC (bỏ phần mở rộng) để output có tên `<query_id>.csv`:

```csv
query_id,text
query-01-kis,một người đứng cạnh ô tô
query-02-kis,60 Giây Sáng 01082024 HTV Tin Tức
```

Chạy:

```powershell
python -m kis.hybrid_cli retrieve `
  --queries queries.csv `
  --top-k 100 `
  --output experiments/results/kis_hybrid/predictions.csv `
  --submission-dir experiments/results/kis_hybrid/submission `
  --submission-zip experiments/results/kis_hybrid/submission.zip `
  --manifest experiments/results/kis_hybrid/retrieval_manifest.json
```

CSV nội bộ có các cột `semantic_rank`, `metadata_rank`, `object_rank`, cosine, BM25, object confidence, `metadata_match_mode` và `matched_objects`. Evaluator chỉ đọc `query_id,rank,video_id,frame_id,score` và bỏ qua các cột chẩn đoán.

CSV nộp BTC vẫn không header, chỉ có:

```csv
L21_V001,30005
L22_V012,24237
```

Mỗi query tối đa 100 dòng, video không có `.mp4`; các CSV được đặt trực tiếp ở root của ZIP. Quy cách này bám theo [hướng dẫn công khai của HCMC AI Challenge 2024](https://codalab.lisn.upsaclay.fr/competitions/20122); vẫn phải đối chiếu sample submission của đúng đợt thi hiện tại.

## Evaluator và ablation

Stage 3 tái sử dụng đúng evaluator Stage 2:

```powershell
python -m kis.hybrid_cli evaluate `
  --ground-truth ground_truth.csv `
  --predictions experiments/results/kis_hybrid/predictions.csv `
  --output experiments/results/kis_hybrid/evaluation.json
```

Quy trình benchmark đề xuất:

1. CLIP-only: thêm `--no-metadata --no-objects`.
2. CLIP + metadata: thêm `--no-objects`.
3. CLIP + objects: thêm `--no-metadata`.
4. Hybrid đầy đủ với RRF.
5. Chỉ giữ thay đổi nếu `Final Score = mean(R@1,R@5,R@20,R@50,R@100)` tăng trên cùng split.

## Smoke test hiện tại

`examples/queries.hybrid.smoke.csv` gồm ba query semantic/metadata/object. Lần chạy CPU gần nhất:

- 177.321 vector, 873 metadata document, 533 object class.
- 3 query, 300 kết quả; mỗi query đủ 100 cặp submit duy nhất.
- ZIP có 3 CSV tại archive root và vượt CRC.
- Thời gian end-to-end 27,805 giây, gồm load model/index và object SQL.

Artifacts:

- `experiments/results/kis_hybrid/smoke_predictions.csv`
- `experiments/results/kis_hybrid/retrieval_manifest.json`
- `experiments/results/kis_hybrid/submission.zip`

Ablation CLIP-only được lưu ở `experiments/results/kis_hybrid/clip_only_predictions.csv` và `clip_only_manifest.json`: 200/200 rank cùng cặp `(video_id, frame_id)` khớp chính xác Stage 2. Trong CSV Stage 3, `score` là điểm fusion/RRF; cosine gốc nằm ở `semantic_score`. Evaluator chấm theo rank nên cách đặt thang score này không đổi metric.

Đây là smoke chức năng, không phải kết quả chất lượng vì workspace vẫn chưa có query/ground truth chính thức.

## Kiểm thử

```powershell
python -m pytest -q
```

Test bao phủ tokenizer, parser object tiếng Việt, object synonym co-occurrence, metadata-conditioned CLIP, RRF, weighted sum, duplicate submit frame, exact `Video_Frame_ID`, CSV evaluator và ZIP submission.
