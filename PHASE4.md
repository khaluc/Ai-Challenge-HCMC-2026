# Giai đoạn 4 — LLM Query Understanding

Stage 4 không dùng LLM để tìm ảnh trực tiếp. LLM (hoặc bộ phân tích rule-based mặc định) chỉ dùng để **hiểu query** — tách thành object/attribute/relation và sinh ra vài query expansion — rồi mỗi expansion được search riêng qua đúng pipeline hybrid Stage 3 (CLIP + metadata BM25 + objects, đã fusion nội bộ), cuối cùng các kết quả expansion được fusion lại lần nữa bằng RRF.

```text
                              Query
                                │
                    Query Understanding (protocol)
              ┌─────────────┬───────────┴───────────┐
        RuleBasedQueryUnderstanding  QwenQueryUnderstanding  AnthropicQueryUnderstanding
        (mặc định, không cần key)   (DASHSCOPE_API_KEY)      (ANTHROPIC_API_KEY)
                                │
              {objects, attributes, relation, expansions[]}
                                │
              ┌─────────────┬──┴──┬─────────────┐
          expansion 0   expansion 1  ...   expansion N   (bản gốc luôn là expansion 0)
              │             │              │
        Stage 3 hybrid  Stage 3 hybrid  Stage 3 hybrid    (CLIP+metadata+objects → RRF nội bộ)
              │             │              │
              └─────────────┴──────┬───────┘
                                    ▼
                          RRF theo (video_id, Video_Frame_ID)
                                    ▼
                                Top-100
```

## Kiến trúc

Theo lựa chọn của người dùng: **protocol pluggable, mặc định rule-based, LLM là adapter tùy chọn** — giống hệt cách `ObjectParserProtocol` của Stage 3 đã pluggable sẵn.

### 1. `QueryUnderstandingProtocol`

```python
class QueryUnderstandingProtocol(Protocol):
    def understand(self, text: str) -> QueryUnderstanding: ...
```

Trả về `QueryUnderstanding(text, structure, expansions)` trong đó `structure` là `QueryStructure(objects, attributes, relation)`.

### 2. `RuleBasedQueryUnderstanding` (mặc định)

- Không cần API key, chạy được ngay trong `pytest` và CLI `analyze-query`.
- Tìm từ khóa quan hệ đầu tiên trong câu (bảng `RELATION_LEXICON`: "cạnh", "trước", "sau", "trên", "trong", "cầm", "lái", …) để tách câu thành 2 vế trái/phải quanh từ quan hệ đó.
- Mỗi vế được đưa qua `RuleBasedObjectParser` (tái dùng nguyên bộ parser object của Stage 3) để lấy concept.
- Attribute (màu sắc, giới tính, độ tuổi — bảng `ATTRIBUTE_LEXICON`) chỉ được gán cho một vế **khi vế đó xác định đúng một concept** — vế có 0 hoặc ≥2 concept thì bỏ qua attribute, không đoán mò.
- Nếu không tách được vế nào theo quan hệ và không phát hiện concept, hệ thống **abstain**: `expansions = (câu gốc,)`, không tự bịa ra biến thể vô nghĩa cho các query kiểu tiêu đề tin tức (`60 Giây Sáng...`).
- Sinh expansion bằng template xác định (không sáng tạo tự do): nối `attribute + concept` (tên concept vốn đã là tiếng Anh, ví dụ `"person"`, `"car"`, trùng khóa `DEFAULT_CONCEPT_LABELS` của Stage 3) qua các cụm quan hệ tiếng Anh (`RELATION_ENGLISH_PHRASES`, ví dụ `next to` / `beside` / `close to`). Câu gốc luôn là expansion đầu tiên.

Ví dụ:

```powershell
python -m kis.expansion_cli analyze-query "một người đàn ông mặc áo đỏ đứng cạnh một chiếc ô tô màu trắng"
```

```json
{
  "structure": {
    "objects": ["person", "car"],
    "attributes": { "person": ["red", "male"], "car": ["white"] },
    "relation": "next_to"
  },
  "expansions": [
    "một người đàn ông mặc áo đỏ đứng cạnh một chiếc ô tô màu trắng",
    "red male person next to white car",
    "red male person beside white car",
    "red male person close to white car"
  ]
}
```

Cả hai adapter LLM bên dưới dùng chung một JSON contract (`LLM_RESPONSE_JSON_SCHEMA`) và cùng một system prompt (`LLM_SYSTEM_PROMPT`) — logic parse payload → `QueryUnderstanding` (`_build_understanding_from_payload`) là một hàm module-level dùng chung, tránh lệch hành vi giữa hai backend. Prompt có kèm 1 ví dụ hoàn chỉnh vì lúc đầu model hay nhầm `attributes[i].concept` thành tên loại thuộc tính (`"color"`, `"gender"`) thay vì copy đúng tên object trong `"objects"` — sau khi thêm ví dụ thì đúng.

### 3. `QwenQueryUnderstanding` (mặc định cho `--llm-backend qwen`)

- Gọi Qwen qua API OpenAI-compatible của Alibaba Cloud Model Studio (DashScope), dùng package `openai` (`client.chat.completions.create(..., response_format={"type": "json_object"})`).
- `base_url` mặc định `https://dashscope-intl.aliyuncs.com/compatible-mode/v1` (endpoint quốc tế); đổi bằng `DASHSCOPE_BASE_URL` hoặc `--llm-base-url` nếu dùng endpoint Trung Quốc đại lục.
- Model mặc định `qwen3.8-max`, đổi bằng `--llm-model`.
- API key lấy từ `DASHSCOPE_API_KEY` (hoặc `QWEN_API_KEY`) trong biến môi trường/`.env`; thiếu key thì raise `ValueError` rõ ràng ngay lúc khởi tạo thay vì lỗi mơ hồ khi gọi mạng.
- `finish_reason == "content_filter"` raise `RuntimeError` rõ ràng.
- Nhận `client` qua constructor để test/inject mà không cần mạng hay API key thật.

### 4. `AnthropicQueryUnderstanding` (tùy chọn, `--llm-backend anthropic`)

- Import `anthropic` lười (lazy import) — không bắt buộc cài đặt nếu chỉ dùng rule-based/Qwen.
- Gọi `client.messages.create(..., output_config={"format": {"type": "json_schema", "schema": ...}})` để ép JSON đúng schema.
- Model mặc định `claude-opus-5`.
- `stop_reason == "refusal"` raise lỗi rõ ràng.
- API key lấy từ `ANTHROPIC_API_KEY`.

### Cài đặt và cấu hình key

```powershell
pip install -e ".[llm]"   # thêm anthropic, openai, python-dotenv
```

API key được đọc từ file `.env` ở root project (đã có `.gitignore` chặn commit):

```text
DASHSCOPE_API_KEY=sk-...
# ANTHROPIC_API_KEY=sk-ant-...
```

`kis/expansion_cli.py` tự `load_dotenv()` khi import, nên chỉ cần điền `.env` là các lệnh CLI dùng được ngay, không cần `export` thủ công.

Kích hoạt bằng cờ `--llm-backend` trên CLI:

```powershell
python -m kis.expansion_cli search "..." --llm-backend qwen --llm-model qwen3.8-max
python -m kis.expansion_cli search "..." --llm-backend anthropic --llm-model claude-opus-5
```

Đã benchmark chức năng thật với `--llm-backend qwen` (xem phần Smoke test).

## Fusion giữa các expansion

`ExpandedHybridSearch` (llm/expansion_retrieval.py):

1. Gọi `understanding.understand(text)` một lần cho mỗi query gốc.
2. Đảm bảo câu gốc luôn có mặt trong danh sách expansion (chèn vào đầu nếu backend không tự thêm), dedupe không phân biệt hoa/thường, giới hạn `--max-expansions` (mặc định 4).
3. Gộp toàn bộ (query, expansion) của cả batch thành một lượt gọi `HybridTextualKIS.search_batch` duy nhất — one encode call cho tất cả expansion, không lặp lại việc load model mỗi expansion.
4. Mỗi expansion trả về danh sách `HybridHit` đã fusion nội bộ (CLIP+metadata+objects) từ Stage 3, rank 1..N riêng.
5. `fuse_expansions` (llm/expansion_fusion.py) chạy RRF thuần túy trên các ranking đó, khóa theo `(video_id, Video_Frame_ID)`:

```text
score(frame) = Σ 1 / (rrf_k + rank_expansion)
rrf_k = 60 (mặc định, đổi bằng --expansion-rrf-k)
```

Không có trọng số riêng theo expansion — mọi expansion (kể cả câu gốc) đóng góp ngang nhau; các expansion "yếu" (0 gợi ý phù hợp) tự nhiên không đóng góp gì.

## Chạy một query

```powershell
python -m kis.expansion_cli search `
  "một người đàn ông mặc áo đỏ đứng cạnh một chiếc ô tô màu trắng" `
  --query-id query-01-kis `
  --top-k 100 `
  --predictions-output experiments/results/kis_expansion/query-01-predictions.csv `
  --submission-dir experiments/results/kis_expansion/submission
```

Các option quan trọng (kế thừa toàn bộ cờ hybrid của Stage 3, cộng thêm):

```text
--max-expansions 4
--candidates-per-expansion 100        # top-k xin từ Stage 3 cho MỖI expansion, tối đa 100
--expansion-rrf-k 60
--llm-backend {rule,anthropic,qwen}   # mặc định rule
--llm-model qwen3.8-max                # hoặc claude-opus-5 khi --llm-backend anthropic
--llm-base-url ...                     # override endpoint OpenAI-compatible (qwen)
```

## Chạy batch và xuất submission

Giống hệt định dạng input/output CSV của Stage 3 (`query_id,text` strict, submission BTC không header). CSV nội bộ Stage 4 có thêm các cột chẩn đoán: `num_expansions_matched`, `best_expansion_id`, `best_expansion_text`, `best_expansion_rank`, `matched_objects`, `object_concepts`, `attributes`, `relation`. Evaluator chỉ đọc `query_id,rank,video_id,frame_id,score`.

```powershell
python -m kis.expansion_cli retrieve `
  --queries queries.csv `
  --top-k 100 `
  --output experiments/results/kis_expansion/predictions.csv `
  --submission-dir experiments/results/kis_expansion/submission `
  --submission-zip experiments/results/kis_expansion/submission.zip `
  --manifest experiments/results/kis_expansion/retrieval_manifest.json
```

## Evaluator

Tái dùng đúng evaluator R@K của Stage 2/3:

```powershell
python -m kis.expansion_cli evaluate `
  --ground-truth ground_truth.csv `
  --predictions experiments/results/kis_expansion/predictions.csv `
  --output experiments/results/kis_expansion/evaluation.json
```

Ablation đề xuất: so sánh Stage 4 (mặc định rule-based) với việc set `--max-expansions 1` (tương đương chạy CLIP-only Stage 3 baseline, không có expansion) trên cùng ground truth; chỉ giữ lại nếu `Final Score` tăng.

## Smoke test hiện tại

Chạy thật trên toàn bộ dữ liệu L21–L30 (177.321 vector, không phải synthetic).

**Rule-based** (không cần key):

```powershell
python -m kis.expansion_cli search "một người đàn ông mặc áo đỏ đứng cạnh một chiếc ô tô màu trắng" --query-id smoke-01 --top-k 20 --predictions-output experiments/results/kis_expansion/smoke_predictions.csv --submission-dir experiments/results/kis_expansion/submission
```

- 4 expansion (câu gốc + 3 biến thể tiếng Anh theo quan hệ `next_to`), mỗi expansion search 100 candidate qua Stage 3.
- 20/20 kết quả có `keyframe_available=true`; nhiều frame được từ 3–4 expansion cùng vote (RRF cộng dồn).
- `experiments/results/kis_expansion/smoke_predictions.csv`, `experiments/results/kis_expansion/submission/smoke-01.csv`.

**Qwen thật** (`--llm-backend qwen`, model `qwen3.8-max` qua DashScope quốc tế):

```powershell
python -m kis.expansion_cli search "một người đàn ông mặc áo đỏ đứng cạnh một chiếc ô tô màu trắng" --llm-backend qwen --query-id qwen-smoke-01 --top-k 10 --predictions-output experiments/results/kis_expansion/qwen_smoke_predictions.csv --submission-dir experiments/results/kis_expansion/submission
```

- Model trả JSON đúng schema, tách `objects=["person","car"]`, `attributes={"person":["male","red"],"car":["white"]}`, `relation="next_to"`, sinh 3 expansion tiếng Anh tự nhiên (không phải template cố định như rule-based).
- Test riêng với query tiêu đề tin tức (`60 Giây Sáng 01082024 HTV Tin Tức`) trả đúng `objects=[]`, `relation=null` — model tự biết không có object cụ thể để bắt.
- 10/10 kết quả có `keyframe_available=true`, nhiều frame có 3 expansion cùng đóng góp.
- `experiments/results/kis_expansion/qwen_smoke_predictions.csv`, `experiments/results/kis_expansion/submission/qwen-smoke-01.csv`.

Đây là smoke chức năng, không phải kết quả chất lượng — chưa có ground truth chính thức để đo R@K thật, và trọng số/max-expansions chưa được tune. So sánh rule-based vs Qwen trên cùng ground truth là việc ablation cần làm tiếp.

## Kiểm thử

```powershell
python -m pytest -q
```

`tests/test_phase4_understanding.py` khóa: tách object/attribute/relation theo quan hệ, abstain khi không có object, validation của `QueryStructure`/`QueryUnderstanding`, RRF fusion qua expansion (`fuse_expansions`), và cả hai adapter LLM (client giả lập, không cần mạng) bao gồm đường refusal/content-filter và lỗi thiếu API key.

`tests/test_phase4_expanded_retrieval.py` khóa hành vi end-to-end trên dữ liệu synthetic: nhiều expansion cùng đóng góp vào một frame, rank liên tục không trùng `(video_id, frame_id)`, CSV nội bộ và submission BTC xuất đúng.
