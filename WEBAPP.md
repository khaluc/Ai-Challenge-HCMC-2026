# Flask Web UI

Giao diện web theo spec bạn gửi (24 mục). **Tất cả 7 milestone đã xây và chạy thật** (không mock) — KIS, video player, frame nav, submission queue, Q&A, TRAKE (decompose → video retrieval → coarse align → dense grid refine → VLM verify), ranking editor, Competition/Debug mode.

## Kiến trúc

```text
app.py                          # Flask app: routes only, không chứa logic AI
├── kis.pipeline.KISPipeline       # hybrid CLIP+BM25+objects (Stage 3), load 1 lần
├── qna.pipeline.KISVideoQA        # candidate narrowing -> VLM -> rerank (Stage 5)
├── trake.web_pipeline.TRAKEWebPipeline  # decompose -> video retrieval -> coarse align -> refine/verify (Stage 7-10)
├── submission.queue.SubmissionQueue     # in-memory pick list cho web UI (không phải Stage 12's CSV reranker)
├── templates/index.html
└── static/{css,js}
```

Q&A và TRAKE cần Qwen (`DASHSCOPE_API_KEY`); cả hai service được dựng **lazy** (chỉ khi có request đầu tiên gọi tới), bọc trong `try/except ValueError` trả `503` rõ ràng — nếu thiếu API key thì chỉ hỏng đúng route đó, không crash cả app (KIS vẫn chạy được độc lập).

## Route đầy đủ

| Route | Milestone | Trạng thái |
|---|---|---|
| `GET /` | 1 | render `templates/index.html` |
| `POST /search/kis` | 1 | **hoạt động thật** — hybrid CLIP+BM25+objects |
| `GET /frame/<video_id>/<frame_id>` | 1 | **hoạt động thật** — đọc JPEG từ ZIP, resize bằng Pillow |
| `GET /video/<video_id>` | 2 | **hoạt động thật** — extract 1 lần vào `cache/videos/`, serve với Range request (HTML5 seek) |
| `GET /api/video/<video_id>/info` | 2-3 | **hoạt động thật** — trả FPS cho frame-nav JS |
| `POST /submission/add` `/remove` `/move` `/export` `GET /submission` | 3, 7 | **hoạt động thật** — in-memory queue, export ra CSV `video_id,frame_id` headerless đúng format BTC |
| `POST /search/qna` | 4 | **hoạt động thật** (Qwen VLM) — candidate narrowing rồi VLM rerank |
| `POST /vlm/ask` | 4 | **hoạt động thật** (Qwen VLM) — tick nhiều frame thủ công, hỏi VLM trực tiếp không qua retrieval |
| `POST /search/trake` | 5 | **hoạt động thật** (Qwen LLM + CLIP DP) — decompose + video retrieval + coarse align trên video top-1 |
| `POST /refine` | 6 | **hoạt động thật** — decode dense window quanh 1 event, trả base64 frames (không VLM) |
| `POST /vlm/verify` | 6 | **hoạt động thật** (Qwen VLM) — chấm điểm dense candidates, chọn best frame |

## Đã chạy thật — kết quả cụ thể

- **KIS**: `"duong pho ha noi"` → 200, hybrid RRF fusion đúng.
- **Video player**: `GET /video/L27_V013` trả 162MB mp4 (200 full, 206 partial với header `Range`, `Content-Range` đúng) — HTML5 `<video>` seek được nhờ Range request chuẩn.
- **Submission queue**: add/move/remove/export test đầy đủ qua Flask test client — export ra đúng `video_id,frame_id` (vd `L21_V001,999`).
- **Q&A** (`"Người phụ nữ đang cầm vật gì?"`-style câu hỏi thật): `/search/qna` trả 200 sau khi chạy VLM tuần tự trên 5 candidate frame; `/vlm/ask` (1 frame thủ công) trả `{"answer": "Không có", "confidence": 0.9}` — VLM trung thực báo không thấy kính trên khuôn mặt trong candidate đó, đúng tinh thần "không đoán mò" đã lập từ Stage 5.
- **TRAKE**: query `"E1: chay da E2: giam nhay E3: bay qua xa E4: tiep dat"` → decompose đúng 4 event tiếng Anh, video retrieval trả 5 candidate có điểm, coarse align trên video top-1 (`L25_V039`) khả thi. `/refine` decode 11 dense frame quanh 1 event trong 5s. `/vlm/verify` trên `L21_V001` + event "a woman standing on a street" tái hiện đúng phát hiện đã ghi ở `PHASE10.md`/`PHASE11.md`: confidence 0.8, matches=true, "figures in rain gear standing on flooded street at night" — cùng cảnh ngập nước ban đêm.

## ASR / câu hỏi về lời nói

Không có UI riêng cho ASR — nó nằm ngay trong `/search/qna`, dùng chung ô hỏi Q&A bình thường. `qna.routing.is_speech_question()` tự phát hiện câu hỏi kiểu "diễn giả nói gì?" / "what did the speaker say?" (danh sách cụm từ tiếng Việt + tiếng Anh trong `SPEECH_QUESTION_PHRASES`) và định tuyến sang nhánh `route="transcript"` **trước khi** chạy retrieval/VLM — vì dataset BTC ở stage này **không có transcript/ASR** (đã xác nhận từ audit Stage 1, ghi ở README). Không có `TranscriptQAProtocol` backend nào được cấu hình trong `app.py` (`get_qna_service()` không truyền `transcript_qa=`), nên kết quả luôn là:

```json
{"video_id": null, "answer": null, "route": "transcript",
 "note": "Speech/ASR question detected but this dataset has no transcript index ..."}
```

Đây là hành vi **cố ý trung thực** — trả lời "không có dữ liệu" ngay lập tức (không tốn VLM call) thay vì để VLM đoán mò nội dung lời nói từ mỗi khung hình tĩnh nó không thể biết. Frontend (`app.js::renderQnaAnswers`) đã xử lý sẵn: khi `result.video_id == null` thì hiện thẳng `result.note` thay vì card ảnh + answer bình thường. Đã verify thật qua `/search/qna` với câu hỏi `"dien gia noi gi trong bai phat bieu?"` — trả đúng note trên, tức thì (không delay VLM).

Nếu sau này có transcript/ASR thật (ví dụ chạy Whisper offline), chỉ cần viết một class implement `TranscriptQAProtocol` (`qna/routing.py`) và truyền vào `KISVideoQA(..., transcript_qa=...)` trong `get_qna_service()` — không cần đổi route hay frontend.

## Phát hiện quan trọng: Qwen vision call chậm hơn nhiều so với text-only

Cô lập bằng test riêng: gọi `qwen3.8-max` **không kèm ảnh** mất **~3.6s**; gọi **kèm 1 ảnh** (VQA-style) mất **~30-70s**, và trong một batch 5 ảnh tuần tự đã đo được trung bình **~97s/ảnh** (có thể do retry/backoff nội bộ của SDK khi có request chậm). Đây là đặc tính thật của API, không phải bug — lúc đầu tưởng nhầm là lỗi treo giống hiện tượng "torch DLL flaky" đã ghi ở Stage 10/11, nhưng cô lập bằng cách theo dõi CPU time của process (đứng yên = treo thật, không phải đang chờ mạng bình thường) rồi kiểm tra kết nối trực tiếp mới xác định đúng nguyên nhân.

Hệ quả trực tiếp: **thời gian phản hồi Q&A/TRAKE-verify tỉ lệ thuận với số lần gọi VLM tuần tự** (`qna.frame_limit`, `trake.max_candidate_frames` trong `config.yaml`). Vì vậy đã hạ mặc định cho web UI so với CLI gốc:

| Config | CLI gốc (Stage 5/10) | Web UI mặc định | Lý do |
|---|---|---|---|
| `qna.video_limit` | 15 | 5 | 30 frame tuần tự ≈ 15+ phút/câu hỏi — quá chậm để thao tác trực tiếp |
| `qna.frame_limit` | 30 | 5 | như trên |
| `trake.max_candidate_frames` | 12 (Stage 10) | 6 (khớp mặc định đã tinh chỉnh của `verify_cli.py`) | `/vlm/verify` cũng tuần tự |

Tăng lại các số này trong `config.yaml` nếu chấp nhận chờ lâu hơn để đổi lấy recall cao hơn — không đổi code, chỉ đổi config.

## Chạy thử

```powershell
python app.py
```

Debug mode (`server.debug: true` trong `config.yaml`) dùng Werkzeug reloader — **load CLIP+FAISS hai lần** (process cha theo dõi file + process con thực sự serve), nên lần khởi động đầu có thể mất 30-40s trước khi `http://127.0.0.1:5000/` phản hồi. Đặt `debug: false` nếu muốn khởi động nhanh hơn (đổi lại mất auto-reload khi sửa code).

## Lỗi môi trường đã gặp và sửa dứt điểm

`app.py` ban đầu crash 100% với `OSError: [WinError 1114]` khi import `torch` — cô lập nguyên nhân: import `pyarrow.parquet` trước `torch` khiến DLL runtime C++ mà Windows nạp trước xung đột với DLL torch cần nạp sau (tái hiện được 3/3 lần, không ngẫu nhiên). Sửa bằng cách import mọi module kéo theo torch (`kis.pipeline`, gián tiếp `retrieval.clip_encoder`) **trước** `pyarrow.parquet` trong `app.py`.

## Chưa làm / giới hạn đã biết

- **Ranking Editor** (mục 18) hiện chỉ nhóm theo band TOP1/5/20/50/100 và cho phép sắp xếp thủ công (↑/↓) — chưa có auto-sort theo score vì Submission Queue là danh sách người dùng tự chọn thủ công (không phải kết quả search thô), không có "score" chung để tự sắp.
- **Experiment Mode** (mục 20, so sánh R@1/5/20/50/100 giữa các cấu hình ablation) **không triển khai** — dự án hiện chưa có ground truth chính thức để tính R@K (đã ghi nhận từ đầu dự án). Debug Mode (checkbox "Debug scores") thay thế bằng cách hiện điểm CLIP/BM25/Objects riêng từng nhánh trên mỗi candidate — đúng những gì có sẵn để debug retrieval, không bịa ra số R@K không có căn cứ.
- **Competition Mode** chỉ ẩn `.debug-only`/`.hint` bằng CSS — không tắt được logic backend (vd vẫn tính branch scores phía server), vì tách bạch tính toán khỏi hiển thị đơn giản hơn và không ảnh hưởng hiệu năng đáng kể.
- **TRAKE Event Panel** hiện chỉ tự động chạy coarse align trên **video candidate top-1** (không phải toàn bộ Top 3-5 như Stage 8 khuyến nghị) — vì UI cần một alignment cụ thể để render Event Panel ngay khi search xong; danh sách 5 candidate video vẫn hiển thị đầy đủ (debug panel) để người dùng biết còn ứng viên khác nếu top-1 sai, nhưng phải tự đổi video thủ công (chưa có nút "align video này" cho candidate #2-5) — nếu cần, đây là điểm mở rộng rõ ràng nhất.
- Chưa có test tự động cho `app.py`/`static/js` (chỉ verify thủ công qua Flask test client + curl như ghi ở trên) — `tests/` hiện chỉ khóa invariant của `src/`, chưa có test giả lập HTTP cho web layer.
