# Giai đoạn 7 — TRAKE: Event Decomposition

TRAKE là phần khó nhất của pipeline: query mô tả một **chuỗi sự kiện** trong video (ví dụ nhảy cao: chạy đà → giậm nhảy → bay qua xà → tiếp đất), không phải một cảnh tĩnh. Stage 7 chỉ làm bước đầu tiên: tách query thành từng event riêng biệt, theo đúng thứ tự, để Stage 8 search từng event.

## Kiến trúc (`llm/` — `event_parser.py`, `event_schemas.py`, `event_parser_cli.py`)

Không có lựa chọn rule-based khả thi ở đây — tách một câu mô tả chuỗi hành động (có thể đánh số `E1/E2/...` hoặc chỉ là văn xuôi nối tiếp) thành các event rời rạc, dịch mỗi event sang một câu tiếng Anh phù hợp cho CLIP search, đòi hỏi hiểu ngôn ngữ thật — không rule/regex nào đủ tổng quát. Vì vậy Stage 7 chỉ có một implementation thật: LLM.

### `EventDecompositionProtocol`

```python
class EventDecompositionProtocol(Protocol):
    def decompose(self, query: str) -> EventSequence: ...
```

`EventSequence(query, events, expansions)`:
- `events`: tuple thứ tự, mỗi phần tử là một câu tiếng Anh ngắn tự chứa nghĩa (ví dụ `"athlete taking off"`), sẵn sàng đưa thẳng vào CLIP text encoder.
- `expansions`: 1-3 cách diễn đạt khác của **toàn bộ chuỗi** (không phải từng event), dùng cho tìm kiếm video-level ở Stage 8 — khác vai trò với `events`.

### `QwenEventDecomposer`

Dùng chung cơ chế OpenAI-compatible/DashScope như Stage 4/5 (`openai` package, model mặc định `qwen3.8-max`, key từ `DASHSCOPE_API_KEY`). Một lệnh gọi LLM duy nhất trả về cả `events` lẫn `expansions` cùng lúc (tiết kiệm hơn 2 lệnh gọi riêng).

Nếu LLM trả `events` rỗng (lỗi định dạng hiếm gặp) — **abstain về nguyên câu query làm event duy nhất** thay vì raise lỗi, để Stage 8 vẫn chạy được (ít nhất tìm theo full query).

## Chạy thử

```powershell
python -m llm.event_parser_cli decompose "E1: chạy đà E2: giậm nhảy E3: bay qua xà E4: tiếp đất"
```

**Đã chạy thật** (không mock), kết quả:

```json
{
  "events": [
    "The athlete runs up toward the bar.",
    "The athlete takes off from the ground.",
    "The athlete flies over the bar.",
    "The athlete lands on the mat."
  ],
  "expansions": [
    "A high jumper completes a run-up, takeoff, clearance over the bar, and landing.",
    "The sequence shows an athlete performing a high jump from approach to landing.",
    "An athlete runs, leaps over a bar, and lands after clearing it."
  ]
}
```

Tách đúng 4 event theo thứ tự khớp với ví dụ E1–E4, không gộp, không bịa thêm event.

## Kiểm thử

```powershell
python -m pytest -q
```

`tests/test_phase7_decomposition.py` khóa: validation `EventSequence` (không rỗng, không trùng, không blank), parse payload LLM đúng (client giả lập, không cần mạng), abstain về nguyên query khi LLM trả events rỗng, loại bỏ expansion trùng với query/event đã có, đường content-filter và thiếu API key.
