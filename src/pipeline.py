"""Production RAG Pipeline — Bài tập NHÓM: ghép M1+M2+M3+M4."""

import os, sys, time
import numpy as np

# Thêm path để import các module trong src
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.m1_chunking import load_documents, chunk_hierarchical
from src.m2_search import HybridSearch
from src.m3_rerank import CrossEncoderReranker
from src.m4_eval import load_test_set, evaluate_ragas, failure_analysis, save_report
from src.m5_enrichment import enrich_chunks
from config import RERANK_TOP_K, OPENAI_API_KEY

def build_pipeline():
    """Build production RAG pipeline."""
    print("=" * 60)
    print("PRODUCTION RAG PIPELINE - INITIALIZING")
    print("=" * 60)

    # Step 1: Load & Chunk (M1)
    print("\n[1/4] Chunking documents (M1)...")
    docs = load_documents()
    raw_chunks_data = []
    
    for doc in docs:
        # parent, children là danh sách các Chunk object từ M1
        parent, _ = chunk_hierarchical(doc["text"], metadata=doc.get("metadata", {}))
        for child in parent:
            # Chuyển Chunk object thành dict để các module sau dễ xử lý
            raw_chunks_data.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id
                }
            })
    print(f"  Generated {len(raw_chunks_data)} chunks from {len(docs)} documents")

    # Step 2: Enrichment (M5)
    print("\n[2/4] Enriching chunks (M5)...")
    # Giả sử enrich_chunks nhận list[dict] và trả về list đối tượng có .enriched_text
    enriched = enrich_chunks(raw_chunks_data, methods=["contextual", "hyqa", "metadata"])
    
    all_chunks_to_index = []
    if enriched:
        for i, e in enumerate(enriched):
            # Giữ lại toàn bộ metadata gốc (bao gồm parent_id, source) và merge với auto_metadata mới
            original_meta = raw_chunks_data[i]["metadata"]
            new_meta = getattr(e, "auto_metadata", {})
            
            all_chunks_to_index.append({
                "text": getattr(e, "enriched_text", raw_chunks_data[i]["text"]),
                "metadata": {**original_meta, **new_meta}
            })
        print(f"  Enriched {len(all_chunks_to_index)} chunks successfully")
    else:
        all_chunks_to_index = raw_chunks_data
        print("  ⚠️  M5 not implemented or failed — using raw chunks")

    # Step 3: Index (M2)
    print("\n[3/4] Indexing (BM25 + Dense)...")
    search = HybridSearch()
    search.index(all_chunks_to_index)

    # Step 4: Reranker (M3)
    print("\n[4/4] Loading reranker (M3)...")
    reranker = CrossEncoderReranker()

    return search, reranker


def run_query(query: str, search: HybridSearch, reranker: CrossEncoderReranker) -> tuple[str, list[str]]:
    """Run single query through pipeline."""
    # 1. Retrieval (M2)
    results = search.search(query) 
    
    # Ép kiểu kết quả về list[dict] để Reranker (M3) xử lý đồng nhất
    docs_for_rerank = []
    for r in results:
        docs_for_rerank.append({
            "text": r.text,
            "score": r.score,
            "metadata": r.metadata
        })

    # 2. Reranking (M3)
    # Rerank trả về list các SearchResult hoặc dict đã được sắp xếp lại
    reranked = reranker.rerank(query, docs_for_rerank, top_k=RERANK_TOP_K)
    
    # 3. Lấy Context (Ưu tiên kết quả đã rerank)
    if reranked:
        contexts = [r.text for r in reranked]
    else:
        contexts = [r["text"] for r in docs_for_rerank[:3]]

    # 4. Generation (LLM)
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
        context_str = "\n\n".join(contexts)
        
        resp = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                    {"role": "system", "content": """Bạn là trợ lý ảo chỉ cung cấp câu trả lời trực tiếp. 
                QUY TẮC:
                1. Trả lời NGẮN GỌN nhất có thể. 
                2. Loại bỏ mọi từ thừa như 'Theo tài liệu...', 'Dựa trên context...'.
                3. Nếu câu hỏi là Yes/No, hãy trả lời Yes/No kèm lý do cực ngắn.

                VÍ DỤ:
                Câu hỏi: Nhân viên có bao nhiêu ngày phép?
                Trả lời: 12 ngày làm việc mỗi năm.

                Câu hỏi: Quy trình VPN là gì?
                Trả lời: Gửi yêu cầu qua cổng thông tin IT và đợi phê duyệt trong 24h."""},
                    {"role": "user", "content": f"Context:\n{context_str}\n\nCâu hỏi: {query}"},
                ],
            max_tokens=500,
            temperature=0,
        )
        answer = resp.choices[0].message.content.strip()
    except Exception as e:
        print(f"  ⚠️ LLM Error: {e}")
        answer = "Đã xảy ra lỗi khi kết nối với trí tuệ nhân tạo."
        
    return answer, contexts


def evaluate_pipeline(search: HybridSearch, reranker: CrossEncoderReranker):
    """Run evaluation on test set."""
    print("\n" + "=" * 60)
    print("RUNNING PIPELINE EVALUATION")
    print("=" * 60)
    
    test_set = load_test_set()
    if not test_set:
        print("  ❌ No test set found. Aborting evaluation.")
        return None

    questions, answers, all_contexts, ground_truths = [], [], [], []

    for i, item in enumerate(test_set):
        answer, contexts = run_query(item["question"], search, reranker)
        
        questions.append(item["question"])
        answers.append(answer)
        all_contexts.append(contexts)
        ground_truths.append(item["ground_truth"])
        
        print(f"  [{i+1}/{len(test_set)}] Q: {item['question'][:50]}...")

    # Ragas Evaluation (M4)
    print("\n[Eval] Computing Ragas metrics...")
    results = evaluate_ragas(questions, answers, all_contexts, ground_truths)

    print("\n" + "=" * 60)
    print("FINAL PERFORMANCE METRICS")
    print("=" * 60)
    
    metrics = ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]
    for m in metrics:
        s = results.get(m, 0)
        status = "✓" if s >= 0.8 else "✗"
        print(f"  {status} {m:18}: {s:.4f}")

    # Phân tích các case thất bại (M4)
    failures = failure_analysis(results.get("per_question", []))
    save_report(results, failures)
    
    return results


if __name__ == "__main__":
    start_time = time.time()
    
    try:
        # Khởi tạo pipeline
        search_engine, rerank_model = build_pipeline()
        
        # Chạy đánh giá
        evaluate_pipeline(search_engine, rerank_model)
        
    except KeyboardInterrupt:
        print("\nProcess interrupted by user.")
    except Exception as e:
        print(f"\n❌ Pipeline Crash: {e}")
        import traceback
        traceback.print_exc()
        
    print(f"\nTotal Execution Time: {time.time() - start_time:.2f}s")