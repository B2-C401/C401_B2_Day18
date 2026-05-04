# Individual Reflection — Lab 18

**Tên:** Nguyễn Đức Dũng  
**Module phụ trách:** M4 — Evaluation

---

## 1. Đóng góp kỹ thuật

- **Module đã implement:** `src/m4_eval.py` — Module đánh giá RAG pipeline bằng thư viện RAGAS.
- **Các hàm/class chính đã viết:**
  - `evaluate_ragas(questions, answers, contexts, ground_truths)` — Chạy đánh giá 4 metrics (faithfulness, answer_relevancy, context_precision, context_recall) thông qua RAGAS, trả về dict tổng hợp và danh sách `EvalResult` per-question.
  - `failure_analysis(eval_results, bottom_n)` — Phân tích Diagnostic Tree: tính điểm trung bình từng câu hỏi, lấy bottom-N câu kém nhất, xác định metric tệ nhất và gắn nhãn chẩn đoán + gợi ý sửa.
- **Số tests pass:** Toàn bộ test trong `tests/test_m4.py` pass sau khi fix.

---

## 2. Kiến thức học được

- **RAGAS là gì:** RAGAS (Retrieval-Augmented Generation Assessment) là framework đánh giá RAG pipeline không cần ground-truth câu trả lời "hoàn hảo" cho mọi metric. Nó tách bạch đánh giá khả năng retrieval (`context_precision`, `context_recall`) với khả năng generation (`faithfulness`, `answer_relevancy`).
- **Điều bất ngờ nhất:** `ragas.evaluate()` ở version 0.4.x _không_ đính kèm lại các cột input (`question`, `answer`, `contexts`, `ground_truth`) vào DataFrame kết quả `to_pandas()`. Khi viết lần đầu (commit `00b9a23`), tôi gọi `row["question"]`, `row["answer"]`… trực tiếp từ DataFrame — dẫn đến `KeyError` ở runtime. Đây là lỗi ngầm rất khó phát hiện nếu không test kỹ.
- **Kết nối với bài giảng:** Slide về RAG Evaluation (Metric taxonomy: faithfulness vs. relevancy, precision vs. recall) — việc tách 4 metric thành 2 nhóm giúp khoanh vùng lỗi nhanh hơn trong Diagnostic Tree.

---

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:** `ragas.evaluate()` trả về `EvaluationResult` object, không phải dict thuần. Commit `00b9a23` cố dùng `result.get("faithfulness", 0.0)` như dict Python thông thường — nhưng `EvaluationResult` không implement `get()`, gây `AttributeError`. Ngoài ra, DataFrame từ `to_pandas()` thiếu cột input như đã đề cập.

- **Cách giải quyết (nếu làm lại):**
  1. **Khởi tạo LLM/Embeddings tường minh:** Truyền `llm=ChatOpenAI(model="gpt-4o-mini")` và `embeddings=OpenAIEmbeddings()` vào `evaluate()` thay vì để RAGAS tự suy, tránh `AttributeError` khi RAGAS không tìm thấy default provider.
  2. **Không đọc cột input từ DataFrame:** Thay vào đó dùng index `i` để lấy trực tiếp từ list gốc (`questions[i]`, `answers[i]`…).
  3. **Helper `_metric(row, key)`:** Xử lý `NaN` và `None` bằng `math.isnan()` trước khi ép kiểu `float`, tránh crash khi RAGAS không tính được một metric nào đó.
  4. **Helper `_agg(key)`:** Tính mean qua `df[key].fillna(0.0).mean()` thay vì gọi `.get()` trên `EvaluationResult`.

- **Thời gian debug:** ~2 giờ — phần lớn thời gian chờ `src/pipeline.py` chạy.

---

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:** Đọc changelog và test suite của thư viện (RAGAS) trước khi implement, đặc biệt với các version đang phát triển nhanh như 0.4.x. Viết một integration test nhỏ với dữ liệu mock ngay từ đầu thay vì chờ đến cuối mới chạy toàn bộ.
- **Module nào muốn thử tiếp:** M3 (Retrieval) — muốn thử thêm BM25 hybrid search vì `failure_analysis` cho thấy `context_recall` thấp thường bắt nguồn từ chunking hoặc sparse retrieval yếu.

---

## 5. Tự đánh giá

| Tiêu chí        | Tự chấm (1-5) |
| --------------- | ------------- |
| Hiểu bài giảng  | 5             |
| Code quality    | 4             |
| Teamwork        | 5             |
| Problem solving | 5             |
