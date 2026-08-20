# BTC/AIC Retrieval Pipeline

Pipeline này bao phủ: kiểm kê/index dữ liệu BTC/AIC, baseline Textual KIS, hybrid retrieval CLIP + metadata + objects, LLM query understanding + expansion, Q&A bằng VLM trên candidate đã thu hẹp qua KIS, TRAKE (chuỗi sự kiện: decompose → video retrieval → coarse DP alignment → fine VLM alignment → verify), và tối ưu ranking theo R@K. Artifact Giai đoạn 1 được xây trực tiếp từ ZIP; không cần giải nén video hoặc keyframe để tạo mapping, FAISS, BM25 và object store.

## Cấu trúc mã nguồn

Code nằm trong `src/`, tổ chức theo chức năng thay vì theo số giai đoạn:

```text
src/
├── data_processing/   # Giai đoạn 1 — audit ZIP, build frame_mapping/FAISS/BM25/object store
├── retrieval/         # Lõi retrieval dùng chung: CLIP store/search, hybrid CLIP+metadata+objects, RRF fusion
├── llm/                # Query expansion (Giai đoạn 4) + TRAKE event decomposition (Giai đoạn 7)
├── vlm/                # Adapter VLM (Qwen): trả lời Q&A ảnh + chấm điểm frame-event cho TRAKE
├── kis/                # CLI Textual KIS: baseline_cli, hybrid_cli, expansion_cli + io/submission writer
├── qna/                # Pipeline Q&A (Giai đoạn 5): thu hẹp candidate → VLM → rerank
├── trake/              # Giai đoạn 7-11: video retrieval, coarse DP alignment, fine alignment, verify
└── submission/         # Giai đoạn 12 — tối ưu ranking theo tier confidence/diversity
```

Package cài ở chế độ editable (`pip install -e ".[llm,video]"`) nên chạy trực tiếp bằng `python -m <package>.<module> ...` (ví dụ `python -m kis.hybrid_cli search ...`) hoặc qua console script tương ứng (`kis-hybrid`, `qna`, `trake-videos`, `data-processing`, ...). Chi tiết lệnh cụ thể từng giai đoạn nằm trong `PHASE1.md`–`PHASE12.md`.

Dữ liệu và output cũng đã tái cấu trúc theo cùng đợt này:

- `data/batch_1/{Keyframes,Videos,Objects,Metadata,CLIP_features}/` — các ZIP gốc BTC, **giữ nguyên dạng ZIP, không giải nén** (đọc trực tiếp qua `zipfile`/`archive_ref`, không cần ~107GB+ đĩa trống để giải nén). `DatasetLayout.discover()` tự tìm theo `data/batch_N/<Subfolder>/`, gộp nhiều batch nếu có, và vẫn tương thích ngược với layout phẳng (dùng trong test synthetic).
- `indexes/{catalog,clip,metadata,objects}/` — artifact Giai đoạn 1 (trước đây `artifacts/phase1/`).
- `experiments/results/{audit,kis_baseline,kis_hybrid,kis_expansion,qna}/` — output chẩn đoán/thử nghiệm từng giai đoạn (trước đây `artifacts/audit`, `artifacts/phase2-5`).
- `submissions/` — CSV Top-100 đã tối ưu ranking (Giai đoạn 12, trước đây `artifacts/phase12/`).

Vì `Keyframe_Path`/`Video_Path`/`Object_Path` trong `frame_mapping.parquet` chỉ lưu tên file ZIP (không lưu thư mục con), các loader cần mở archive trực tiếp (`qna/images.py`, `trake/dense_frame_search.py`) dùng chung `data_processing.layout.resolve_archive(data_root, archive_name)` để tìm ZIP dưới `data_root` phẳng hoặc `data_root/batch_*/*/`.

## Trạng thái dữ liệu hiện tại

- `map-keyframes`, CLIP, metadata và objects: đầy đủ, đồng bộ **873 video / 177.321 keyframe** thuộc L21–L30.
- 24/24 ZIP (10 Keyframes + 14 Videos) vượt kiểm tra CRC sâu; không có entry hỏng, trùng tên, mã hóa hoặc path không an toàn.
- Video và ảnh keyframe: đầy đủ toàn bộ L21–L30, **873 video / 177.321 ảnh**.
- Metadata có title, description, keywords và thông tin YouTube; **không có transcript/ASR**.

Chi tiết nằm trong `experiments/results/audit/data_audit.md` và `experiments/results/audit/data_audit.json`.

## Artifact đã xây

```text
indexes/
├── catalog/
│   ├── frame_mapping.parquet
│   └── frame_mapping.csv
├── clip/
│   ├── clip_vectors.f32.npy
│   ├── faiss.index
│   ├── video_offsets.parquet
│   └── clip_index_meta.json
├── metadata/
│   ├── metadata.parquet
│   ├── metadata.sqlite
│   └── metadata_index_meta.json
├── objects/
│   ├── objects_raw_nested.parquet
│   ├── objects_index.parquet
│   ├── objects.sqlite
│   └── object_store_meta.json
├── build_manifest.json
└── validation_report.json
```

`frame_mapping.parquet` là source of truth dùng chung. Bảy cột đầu đúng schema yêu cầu:

| Cột | Ý nghĩa |
|---|---|
| `Video_ID` | ID video, ví dụ `L21_V001` |
| `Keyframe_Index` | `n` trong map CSV, bắt đầu từ 1 |
| `Keyframe_Path` | Tham chiếu `archive.zip::member`; xem thêm cờ availability |
| `Video_Frame_ID` | `frame_idx` chính xác để submit |
| `Timestamp` | `pts_time` chính xác từ map CSV |
| `CLIP_Index` | Dòng CLIP cục bộ trong video, bắt đầu từ 0 |
| `Object_Path` | Tham chiếu JSON object tương ứng |

Các cột bổ sung quan trọng gồm `FAISS_Index`, `FPS`, `Keyframe_Available`, `Video_Available`, archive/member nguồn và metadata path.

> Không dùng `Keyframe_Index` để submit. Không tự tính frame bằng `Timestamp × FPS`. Luôn lấy `Video_Frame_ID` từ mapping.

Có 614 hàng dư do `Video_Frame_ID` trùng trong cùng video (192 video bị ảnh hưởng). Vì vậy khóa chính của catalog là `(Video_ID, Keyframe_Index)`, không phải `(Video_ID, Video_Frame_ID)`.

## Cách chạy lại

Môi trường hiện tại đã có đủ dependency. Nếu chuyển máy:

```powershell
python -m pip install -r requirements.txt
```

Audit toàn bộ, gồm CRC sâu:

```powershell
python -m data_processing.cli audit --data-root data --output experiments/results/audit --deep-crc
```

Build đầy đủ:

```powershell
python -m data_processing.cli build --data-root data --output indexes --object-index-min-confidence 0.2
```

Validate artifact:

```powershell
python -m data_processing.cli validate --artifacts indexes
```

Có thể build riêng từng thành phần bằng `--components mapping clip metadata objects`. Object Parquet lossless luôn giữ đủ 100 detection/keyframe; `objects_index.parquet` và SQLite chỉ giữ detection đạt threshold để truy vấn nhanh.

## Truy vấn thử

BM25 metadata hỗ trợ tìm tiếng Việt có hoặc không dấu:

```powershell
python -m data_processing.cli search-metadata "duong pho ha noi" --artifacts indexes --limit 10
```

Tìm object; kết quả đã chứa đúng `video_frame_id` để join/submit:

```powershell
python -m data_processing.cli search-objects Person --artifacts indexes --min-confidence 0.9 --limit 10
```

Kiểm tra FAISS bằng một keyframe đã biết:

```powershell
python -m data_processing.cli search-similar L21_V001 1 --artifacts indexes --limit 10
```

Text-to-image CLIP baseline đã được triển khai ở Giai đoạn 2 bằng đúng projection ViT-B/32, normalize float32 và join kết quả qua `FAISS_Index`.

## Kiểm thử

```powershell
python -m pytest -q
```

Bộ test dùng dữ liệu synthetic nhỏ, không đọc 13 GB dữ liệu thật. Nó khóa các invariant của cả ba giai đoạn: frame mapping, FAISS, BM25, objects, text retrieval, RRF/weighted fusion, ablation, Top-100 submission và evaluator R@K.

## Giai đoạn 2

Baseline Textual KIS, CSV Top-100 và evaluator R@K được hướng dẫn chi tiết trong [`PHASE2.md`](PHASE2.md).

## Giai đoạn 3

Hybrid CLIP + metadata BM25 + objects, RRF/weighted fusion và quy trình ablation được hướng dẫn trong [`PHASE3.md`](PHASE3.md).

## Giai đoạn 4

LLM (hoặc rule-based mặc định) phân tích query thành object/attribute/relation, sinh query expansion, mỗi expansion chạy qua pipeline hybrid Stage 3 rồi fusion lại bằng RRF. Kiến trúc pluggable qua `QueryUnderstandingProtocol`, chi tiết trong [`PHASE4.md`](PHASE4.md).

## Giai đoạn 5

Q&A không chạy VLM trên toàn bộ dataset: dùng KIS (Stage 3) thu hẹp còn Top 10–20 video / Top 20–50 frame, VLM chỉ chạy trên tập nhỏ đó rồi rerank theo confidence. Câu hỏi về nội dung lời nói được định tuyến sang nhánh Transcript/ASR riêng (hiện chưa có dữ liệu transcript). Chi tiết trong [`PHASE5.md`](PHASE5.md).

## Giai đoạn 6 (chưa triển khai)

OCR là module bổ sung, **chỉ code khi có benchmark thật cho thấy nhiều query liên quan đến chữ** (bảng hiệu, slide, tên người, số áo, biển báo, logo). Dự án hiện chưa có ground truth để đánh giá điều kiện này. Thiết kế dự kiến (chưa code) ghi ở [`PHASE6.md`](PHASE6.md).

## Giai đoạn 7

TRAKE decompose query nhiều sự kiện (ví dụ nhảy cao: chạy đà/giậm nhảy/bay qua xà/tiếp đất) thành từng event riêng bằng LLM, sẵn sàng cho search từng event. Chi tiết trong [`PHASE7.md`](PHASE7.md).

## Giai đoạn 8

TRAKE video retrieval: search full query + từng event + LLM expansion, gộp điểm theo video (`global_similarity + event_coverage + BM25 + multi_query_vote`), giữ Top 3–5 video ứng viên thay vì chốt ngay một video — vì sai video làm R-Score TRAKE bằng 0. Chi tiết và giới hạn đã phát hiện khi chạy thật trong [`PHASE8.md`](PHASE8.md).

## Giai đoạn 9

TRAKE coarse temporal alignment: dynamic programming tìm chuỗi frame `f1 < f2 < ... < fn` tối đa hóa tổng độ khớp CLIP, dùng vector keyframe có sẵn từ Stage 1 (không cần model image encoder riêng). Chi tiết trong [`PHASE9.md`](PHASE9.md).

## Giai đoạn 10

TRAKE fine frame alignment: chỉ decode ±2–5s quanh vị trí coarse từ video gốc (không decode cả video), để VLM chọn semantic keyframe trong các dense candidate. Đã verify trực quan khớp đúng ảnh thật. Chi tiết trong [`PHASE10.md`](PHASE10.md).

## Giai đoạn 11

VLM verify các candidate gần biên rồi bắt buộc kiểm tra lại `f1 < f2 < ... < fn`; nếu verify làm vỡ thứ tự thì quay về assignment gốc thay vì nộp chuỗi không hợp lệ. Chi tiết trong [`PHASE11.md`](PHASE11.md).

## Giai đoạn 12

Tối ưu ranking Top-100 theo tier: Rank 1-5 tuyệt đối không đánh đổi diversity, diversity chỉ tăng mạnh dần về cuối (Rank 51-100) — vì Final Score giảm rất nhanh theo rank nên Top-1/Top-5 quan trọng nhất. Thuật toán MMR thuần túy, không cần LLM. Chi tiết trong [`PHASE12.md`](PHASE12.md).

## Web UI

Flask app (`app.py`, `templates/`, `static/`) theo spec 24 mục — cả 7 milestone (KIS, video player + frame nav, submission queue, Q&A, TRAKE decompose→retrieval→align→refine→verify, ranking editor, Competition/Debug mode) đã xây và chạy thật với dữ liệu + Qwen API thật. Chi tiết kiến trúc, route, kết quả chạy thật và giới hạn đã biết trong [`WEBAPP.md`](WEBAPP.md). Luồng xử lý chi tiết của cả 3 kiểu truy vấn (KIS/Q&A/TRAKE) — từ dữ liệu BTC cấp tới lúc ra kết quả — nằm trong [`ARCHITECTURE.md`](ARCHITECTURE.md).

## Chạy trên máy khác (đồng đội đã có dataset BTC)

Repo GitHub chỉ chứa **code** (~1.3MB) — `data/` (~107GB), `indexes/` (~1.6GB), `models/` (~1.2GB) và `.env` (chứa API key) đều bị `.gitignore`, không nằm trong repo. Máy mới cần tự dựng lại 4 phần này.

1. **Clone code:**
   ```powershell
   git clone https://github.com/khaluc/Ai-Challenge-HCMC-2026.git
   cd Ai-Challenge-HCMC-2026
   ```

2. **Cài Python 3.10+ và dependency** (bản `requirements.txt` đã sửa đủ, bao gồm cả `openai`/`python-dotenv` cho Qwen — nếu dùng bản cũ hơn commit này thì thiếu 2 package đó):
   ```powershell
   python -m pip install -e ".[llm,video]"
   ```
   (hoặc `python -m pip install -r requirements.txt` nếu không cần cài editable)

3. **Đặt dataset BTC đúng layout** — copy/liên kết dataset của đồng đội vào:
   ```
   data/batch_1/{Keyframes,Videos,Objects,Metadata,CLIP_features}/*.zip
   ```
   Giữ nguyên dạng ZIP, **không giải nén** — pipeline đọc trực tiếp qua `zipfile`.

4. **Lấy checkpoint CLIP** (một trong hai cách):
   ```powershell
   python -m kis.baseline_cli prepare-model
   ```
   (tải đúng bản OpenAI CLIP ViT-B/32 đã pin sẵn), hoặc copy thẳng thư mục `models/openai-clip-vit-base-patch32/` từ máy đã có.

5. **Build index từ dataset** (chạy 1 lần, ra `indexes/`, ~1.6GB, mất vài phút tùy máy):
   ```powershell
   python -m data_processing.cli build --data-root data --output indexes --object-index-min-confidence 0.2
   python -m data_processing.cli validate --artifacts indexes
   ```
   Nếu muốn bỏ qua bước build (nhanh hơn), có thể xin trực tiếp thư mục `indexes/` đã build sẵn (chỉ 1.6GB, dễ chuyển qua USB/cloud hơn nhiều so với 107GB `data/`) rồi đặt đúng vị trí `indexes/` ở gốc repo.

6. **Tạo `.env` riêng** (không dùng chung key với máy khác — mỗi người nên có key riêng để tính phí/giới hạn tách bạch):
   ```
   DASHSCOPE_API_KEY=sk-...
   ```

7. **Chạy web app:**
   ```powershell
   python app.py
   ```
   Lần đầu load CLIP+FAISS mất ~20-40s trước khi `http://127.0.0.1:5000/` phản hồi (xem thêm mục "Chạy thử" trong [`WEBAPP.md`](WEBAPP.md) về việc debug reloader load model 2 lần).

`cache/` (video đã giải nén tạm) **không cần copy** — tự sinh lại khi phát video lần đầu, xóa an toàn bất cứ lúc nào.
