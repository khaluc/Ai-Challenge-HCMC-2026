# Giai đoạn 2 — Baseline Textual KIS

Baseline đã chạy end-to-end trên toàn bộ 177.321 vector:

```text
Query text
  -> OpenAI CLIP ViT-B/32 text projection
  -> L2 normalize float32
  -> FAISS IndexFlatIP
  -> join theo FAISS_Index
  -> dedupe (video_id, Video_Frame_ID)
  -> Top-100 video_id, frame_id
```

## Checkpoint CLIP

Model nằm tại `models/openai-clip-vit-base-patch32`, revision:

```text
3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268
```

SHA-256 của weights:

```text
a63082132ba4f97a80bea76823f544493bffa8082296d62d71581a4feff1576f
```

Không chỉ dựa vào tên model: pipeline đã encode lại 5 ảnh BTC thật. Cosine giữa vector encode lại và feature BTC đạt `0.998799–0.999795`, xác nhận checkpoint và preprocessing ở cùng không gian embedding. Xem `experiments/results/kis_baseline/encoder_compatibility.json`.

Nếu chuyển sang máy mới, tải snapshot đã pin bằng:

```powershell
python -m kis.baseline_cli prepare-model
```

Sau đó kiểm chứng lại:

```powershell
python -m kis.baseline_cli verify-encoder --data-root data --samples 5
```

## Chạy một query

```powershell
python -m kis.baseline_cli search `
  "a television news presenter in a studio" `
  --query-id query-01-kis `
  --top-k 100 `
  --artifacts indexes `
  --predictions-output experiments/results/kis_baseline/query-01-predictions.csv `
  --submission-dir experiments/results/kis_baseline/submission
```

`predictions-output` là bảng nội bộ có rank, score, FAISS ID, keyframe và timestamp để debug. File trong `submission-dir` là CSV BTC không header, chỉ có:

```csv
L21_V001,1234
L22_V003,5678
```

Không có `.mp4`, score hoặc rank column; thứ tự dòng chính là thứ tự xếp hạng.

## Chạy nhiều query

Input CSV:

```csv
query_id,text
query-01-kis,a person speaking in a television studio
query-02-kis,a red car moving along a city road
```

JSONL với hai field `query_id` và `text` cũng được hỗ trợ.

CSV được kiểm tra strict: header phải đúng chính xác `query_id,text` theo thứ tự, không thêm cột, và `query_id` phải duy nhất/không rỗng. Khi dùng query chính thức, đặt `query_id` bằng đúng basename của file query BTC (bỏ phần mở rộng); output sẽ có tên `<query_id>.csv`.

```powershell
python -m kis.baseline_cli retrieve `
  --queries queries.csv `
  --top-k 100 `
  --output experiments/results/kis_baseline/predictions.csv `
  --submission-dir experiments/results/kis_baseline/submission `
  --submission-zip experiments/results/kis_baseline/submission.zip
```

Mỗi query sinh tối đa 100 dòng. Retrieval tự overfetch để loại 614 cặp submit trùng trong dữ liệu mà vẫn backfill đủ Top-K. Kết quả không bị loại chỉ vì keyframe L24–L30 chưa tải; `keyframe_available` chỉ dành cho UI.

Smoke test thật hiện có tại:

- `experiments/results/kis_baseline/smoke_predictions.csv`: 200 kết quả cho 2 query.
- `experiments/results/kis_baseline/submission/`: hai CSV BTC, mỗi file đúng 100 dòng.
- `experiments/results/kis_baseline/submission.zip`: gói submission mẫu.
- `experiments/results/kis_baseline/retrieval_manifest.json`: checkpoint, index và thời gian chạy.

Trong ZIP, các file `<query_id>.csv` nằm trực tiếp ở archive root, không nằm trong thư mục con. Vẫn cần đối chiếu sample submission của đúng đợt thi hiện tại nếu BTC thay đổi quy cách đóng gói.

Quy cách này bám theo [hướng dẫn công khai của HCMC AI Challenge 2024](https://codalab.lisn.upsaclay.fr/competitions/20122): mỗi query có một CSV, tối đa 100 dòng, mỗi dòng Textual KIS là tên video không có `.mp4` và Frame Idx dạng số nguyên; ví dụ chính thức bắt đầu ngay bằng dòng dữ liệu nên writer xuất không header. Workspace chưa có sample submission của đợt hiện tại, vì vậy bước đối chiếu template ngay trước khi nộp vẫn là bắt buộc.

## Evaluator

Ground truth dùng khoảng frame inclusive:

```csv
query_id,video_id,start_frame_id,end_frame_id
query-01-kis,L21_V001,1200,1350
```

Một query có thể có nhiều dòng GT nếu có nhiều khoảng/video được chấp nhận. Với đáp án đúng một frame, có thể dùng schema rút gọn `query_id,video_id,frame_id`.

Prediction evaluator dùng bảng nội bộ:

```csv
query_id,rank,video_id,frame_id,score
query-01-kis,1,L21_V001,1234,0.82
```

Chạy:

```powershell
python -m kis.baseline_cli evaluate `
  --ground-truth ground_truth.csv `
  --predictions experiments/results/kis_baseline/predictions.csv `
  --output experiments/results/kis_baseline/evaluation.json
```

Với mỗi query:

```text
hit@K = 1 nếu trong K kết quả đầu có video_id đúng
        và frame_id nằm trong [start_frame_id, end_frame_id]
```

Evaluator tính macro:

```text
R@1, R@5, R@20, R@50, R@100
Final Score = (R@1 + R@5 + R@20 + R@50 + R@100) / 5
```

Nó cũng xuất `first_relevant_rank`, điểm từng query và `query_score_sum` để chẩn đoán. Metric dùng so sánh model vẫn là `final_score`. Rank phải liên tục từ 1, tối đa 100; duplicate `(video_id, frame_id)` hoặc query ID lạ sẽ bị báo lỗi thay vì âm thầm sửa.

`examples/ground_truth.example.csv` và `examples/predictions.example.csv` chỉ là dữ liệu giả để kiểm tra evaluator. Điểm `0.8` của ví dụ không phải kết quả benchmark trên bộ thi. Workspace hiện chưa có query/ground truth chính thức nên chưa thể báo R@K thật.

## Kiểm thử

```powershell
python -m pytest -q
```

Test bao phủ mapping bằng `FAISS_Index`, đúng `Video_Frame_ID`, dedupe/backfill, Top-100 headerless, range GT inclusive, các mốc rank 1/5/20/50/100 và công thức Final Score.

Đây là baseline CLIP thuần: query được encode nguyên văn, chưa dịch, mở rộng query, BM25 hay rerank. OpenAI CLIP thiên về tiếng Anh, nên mô tả tiếng Anh thường là mốc baseline hợp lý hơn cho tới khi bước hybrid/query processing được benchmark bằng evaluator này.
