# Giai đoạn 5 — Q&A (VLM trên candidate đã thu hẹp)

Stage 5 **không chạy VLM trên toàn bộ 177.321 keyframe**. Hệ thống KIS (Stage 3) dùng để thu hẹp candidate trước, VLM chỉ chạy trên một tập nhỏ frame đã được rank cao.

```text
                    Question
                       │
              Hybrid Retrieval (Stage 3)
                       │
              Top 10–20 Video (distinct)
                       │
              Top 20–50 Frame (trong các video đó)
                       │
                 VLM (mỗi frame)
                       │
              {answer, confidence}
                       │
             Rerank theo confidence
                       │
        video_id + frame_id + answer (Top-K)
```

Câu hỏi về **nội dung lời nói** ("Diễn giả nhắc đến công nghệ nào?") được định tuyến sang nhánh Transcript/ASR + LLM riêng, **không** đưa vào VLM — vì VLM chỉ nhìn thấy pixel, không nghe được audio, trả lời bừa sẽ là bịa. BTC hiện chưa cung cấp transcript/ASR (đã ghi nhận từ Stage 3), nên nhánh này trả về kết quả trung thực "chưa có transcript" thay vì đoán mò.

## Kiến trúc (`qna/` + `vlm/image_qa.py`)

### 1. Thu hẹp candidate — `candidates.py`

`QACandidateSearch` bọc quanh bất kỳ searcher nào có `.search(text, top_k=..., query_id=...)` — dùng được cả `HybridTextualKIS` (Stage 3, mặc định) lẫn `ExpandedHybridSearch` (Stage 4, qua cờ `--use-expansion`).

`narrow_candidates(hits, config)`:
1. Duyệt hit theo thứ tự score đã fusion (giảm dần), gom **tối đa `video_limit` video_id khác nhau** theo thứ tự xuất hiện đầu tiên (best rank).
2. Lọc lại chỉ giữ hit thuộc các video đó, **bỏ hit không có ảnh** (`keyframe_available=False`), cắt còn tối đa `frame_limit` frame — vẫn giữ nguyên thứ tự score gốc nên frame liên quan nhất luôn được ưu tiên trước khi cắt.

`CandidateConfig`: `search_top_k=100` (candidate frame-level lấy từ Stage 3 trước khi thu hẹp), `video_limit=15`, `frame_limit=30` — đúng khoảng đề bài (10–20 video, 20–50 frame).

### 2. VLM — `vlm.py`

`VLMAnswererProtocol.answer(image_bytes, question) -> VLMAnswer(answer, confidence)`.

`QwenVLAnswerer`: dùng chung cơ chế OpenAI-compatible/DashScope như Stage 4 (`openai` package, `base_url` mặc định `https://dashscope-intl.aliyuncs.com/compatible-mode/v1`, key từ `DASHSCOPE_API_KEY`). Model mặc định `qwen3.8-max` — **đã xác nhận model này hỗ trợ vision thật** (gửi `image_url` dạng `data:image/jpeg;base64,...` cùng `text` trong `content` của message, ép JSON `{"answer": string, "confidence": number}` qua `response_format={"type":"json_object"}`).

Ảnh được đọc trực tiếp từ ZIP gốc (`Keyframes_L*.zip::keyframes/VIDEO/NNN.jpg`) bằng `images.load_keyframe_bytes` — không cần giải nén ra đĩa, không cần tải thêm keyframe rời.

### 3. Q&A từ lời nói — `routing.py`

`is_speech_question(text)`: khớp câu hỏi với một bảng cụm từ khóa "nói về nội dung phát biểu" (tiếng Việt: "nhắc đến", "đề cập", "phát biểu", "diễn giả nói"...; tiếng Anh: "what did ... say", "speaker mention"...). Nếu khớp, đi qua `TranscriptQAProtocol` (Transcript/ASR → LLM) thay vì VLM.

Chưa có implementation thật cho `TranscriptQAProtocol` vì **BTC không cung cấp ASR/transcript** ở giai đoạn dữ liệu hiện tại. Không cấu hình `transcript_qa` thì `KISVideoQA` trả `QAResult(route="transcript", answer=None, note="...chưa có transcript...")` — trung thực, không dùng VLM đoán bừa nội dung âm thanh.

### 4. Orchestrator — `qa.py`

`KISVideoQA.ask(question, query_id, top_k)`:
1. Nếu là speech question → nhánh transcript (mục 3).
2. Ngược lại → `QACandidateSearch.find_candidates` → gọi VLM cho **từng** candidate frame → `sort` theo `confidence` giảm dần → trả `top_k` kết quả đầu (mỗi kết quả có `video_id, frame_id, answer, confidence, rank`).
3. Không có candidate nào (hoặc tất cả thiếu ảnh) → trả kết quả trung thực, không answer bừa.

## Cài đặt & chạy

```powershell
pip install -e ".[llm]"   # anthropic, openai, python-dotenv (đã cài sẵn trong môi trường này)
```

Dùng chung `.env`/`DASHSCOPE_API_KEY` đã cấu hình từ Stage 4.

```powershell
python -m qna.cli ask "Người phụ nữ đang cầm vật gì?" `
  --video-limit 15 --frame-limit 30 --top-k 5 `
  --output experiments/results/qna/predictions.csv
```

Cờ quan trọng:

```text
--search-top-k 100        # candidate frame-level lấy từ Stage 3 trước khi thu hẹp
--video-limit 15
--frame-limit 30
--top-k 5                 # số câu trả lời (đã rerank) trả về
--vlm-model qwen3.8-max
--vlm-base-url ...
--use-expansion            # dùng Stage 4 (LLM query expansion) thay vì Stage 3 hybrid thuần cho bước retrieval
--data-root data             # thư mục gốc chứa data/batch_N/Keyframes|Videos/*.zip
```

CSV nội bộ (`io.write_qa_results`) là diagnostic tự định nghĩa: `query_id,rank,route,video_id,frame_id,answer,confidence,note` — **chưa có định dạng submission Q&A chính thức của BTC** để đối chiếu (khác KIS đã có tham chiếu codalab), cần verify khi có đề bài chính thức.

## Đã chạy thật (không phải mock)

1. **VLM vision xác nhận hoạt động**: gửi ảnh thật (`Keyframes_L21.zip::keyframes/L21_V001/243.jpg`, ảnh một phụ nữ đeo kính, đứng ngoài đường) + câu hỏi "What is the main subject of this image?" → `{"answer": "A woman wearing glasses", "confidence": 0.9}` — **đúng khi so ảnh thật**.
2. **Định tuyến speech question**: `"diễn giả nhắc đến công nghệ nào?"` → route `"transcript"`, trả honest note, không gọi VLM.
3. **Full pipeline 2 lần trên toàn bộ 177.321 vector**:
   - `"một người phụ nữ đứng trên đường phố"` (câu mô tả, không phải câu hỏi thật) → top-1 là slide bài giảng hóa học, VLM trả `"Không"` — đúng theo đúng nội dung ảnh, cho thấy retrieval cho câu này không khớp ngữ nghĩa tốt.
   - `"Người phụ nữ đang cầm vật gì?"` (câu hỏi thật, đúng ví dụ đề bài) → top-1/2/3 đều là clip một giảng viên **nam** dạy địa lý, VLM trả `"Không có"` — hợp lý vì đúng là không có phụ nữ trong ảnh đó.
   - Cả hai lần đều chạy hết toàn bộ pipeline (retrieval thật → thu hẹp candidate → nhiều lệnh gọi VLM thật → rerank → xuất CSV) không lỗi.

**Kết luận trung thực**: cơ chế Stage 5 (thu hẹp candidate, gọi VLM, rerank, định tuyến speech) đã chạy đúng và VLM tự nó cho câu trả lời chính xác khi ảnh liên quan. Nhưng retrieval Stage 3 cho các câu hỏi tự do dạng Q&A (khác câu KIS mô tả cảnh) **chưa chắc luôn tìm đúng frame liên quan nhất trong 177k frame** — đây là giới hạn chất lượng CLIP zero-shot trên corpus đa dạng, chưa có ground truth Q&A chính thức để đo R@K/accuracy thật, và **chưa tune** `video_limit`/`frame_limit`/prompt VLM.

## Kiểm thử

```powershell
python -m pytest -q
```

`tests/test_phase5_qa.py` khóa: funnel top-video → top-frame (kể cả loại frame thiếu ảnh), delegate đúng tham số tới searcher, phát hiện speech question, rerank theo confidence trong `KISVideoQA`, honest fallback khi không có transcript backend hoặc không có candidate, đọc bytes ảnh từ ZIP đúng định dạng `Archive.zip::member`, round-trip CSV, và adapter `QwenVLAnswerer` (client giả lập, không cần mạng) bao gồm cả đường content-filter và thiếu API key.
