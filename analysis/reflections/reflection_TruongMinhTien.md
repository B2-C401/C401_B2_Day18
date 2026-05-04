# Individual Reflection — Lab 18

**Tên:** Trương Minh Tiền  
**Module phụ trách:** M5 Enrichment Pipeline + Setup & Integration Support

---

## 1. Đóng góp kỹ thuastic

- **Module đã implement:**
  - M5 Enrichment Pipeline (full)
  - Requirements.txt management
  - Sample data creation (.md files)
  - Test setup + data fixes
- **Các hàm/class chính đã viết:**
  - `summarize_chunk()` - OpenAI-based chunk summarization with extractive fallback
  - `generate_hypothesis_questions()` - HyQA question generation for vocabulary bridging
  - `contextual_prepend()` - Anthropic-style contextual prepending
  - `extract_metadata()` - Auto metadata extraction (topic, entities, category, language)
  - `enrich_chunks()` - Full enrichment pipeline orchestration
  - `failure_analysis()` (M4 support) - Diagnostic tree-based failure analysis
- **Số tests pass:** 10/10 (M5) + 4/4 (M4 failure_analysis)

---

## 2. Kiến thức học được

- **Khái niệm mới nhất:**
  - RAG enrichment techniques: summarization, hypothesis questions, contextual prepending
  - Vocabulary gap bridging via HyQA
  - RAGAS evaluation metrics (faithfulness, answer relevancy, context precision/recall)
  - Error Tree diagnostic methodology for failure analysis
  - One-time indexing cost vs. query-time improvement ROI
- **Điều bất ngờ nhất:**
  - OpenAI API latency difference between streaming vs. batch calls (~12s for 3 enrichment steps)
  - JSON parsing robustness - LLM sometimes wraps output in markdown code blocks
  - Python 3.14 compatibility challenges with newer ML libraries
- **Kết nối với bài giảng:**
  - Slide 5: "Advanced retrieval techniques" → M5 is the enrichment layer
  - Slide 8: "Vocabulary gap in search" → HyQA addresses this directly
  - Slide 12: "Production RAG challenges" → Contextual prepending reduces retrieval failures 49%
  - Slide 13: 'Enrichment Pipeline'
  - Slide 14: 'Enrichment Techniques'

---

## 3. Khó khăn & Cách giải quyết

- **Khó khăn lớn nhất:**
  - Python 3.14 compatibility: qdrant-client 2.4.0 not available, sentence-transformers had version constraints
  - No requirements.txt in repo initially → dependency hell
  - JSON parsing errors in test_set.json (trailing comma bug)
  - Missing sample data (.md files) in /data/
- **Cách giải quyết:**
  - Created flexible requirements.txt with `>=` constraints instead of pinned versions
  - Fixed test_set.json syntax + created 10 realistic Q&A pairs
  - Generated 3 sample .md files with realistic HR/policy content
  - Implemented graceful fallbacks: extractive summarization when no API key, empty lists for non-critical enrichment steps
- **Thời gian debug:** ~45 mins (mostly environment setup, not algorithm)

---

## 4. Nếu làm lại

- **Sẽ làm khác điều gì:**
  - Start with requirements.txt immediately (not at end)
  - Use Python 3.12 instead of 3.14 to avoid compatibility issues
  - Create mock/sample data before implementing (not after baseline runs)
  - Implement M4 failure_analysis first (simpler) before M5 (requires API)
- **Module nào muốn thử tiếp:**
  - M2 Hybrid Search (BM25 + Dense) - interesting fusion strategy
  - M3 Reranking - latency vs. accuracy tradeoff is important
  - Cross-encoder vs. bi-encoder comparison benchmarking

---

## 5. Tự đánh giá


| Tiêu chí        | Tự chấm (1-5) | Giải thích                                                                             |
| --------------- | ------------- | -------------------------------------------------------------------------------------- |
| Hiểu bài giảng  | 5             | Understand RAG fundamentals well; enrichment techniques less covered in slides         |
| Code quality    | 5             | Clean code, good error handling + fallbacks; could add more type hints                 |
| Teamwork        | 5             | Proactive help with setup, created sample data for group, documented process           |
| Problem solving | 5             | Solved env issues effectively; didn't need external help; creative fallback strategies |


---

## 6. Notes cho Phần B (Nhóm)

- M5 enrich_chunks() needs M1 chunks as input → M1 must complete first
- If M1/M2/M3 incomplete, M5 can still demo with fallback enrichment
- RAGAS evaluation might be slow first run (~1-2 min download models) but cached after
- Failure analysis diagnostic tree works even with placeholder scores

---

**Lab takeaway:** Production RAG is 70% plumbing (data loading, API errors, compatibility) and 30% algorithms. M5 enrichment is high-ROI because it's one-time cost at indexing, not per-query. Recommend this for any RAG system upgrade.