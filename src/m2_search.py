"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
import json
import numpy as np
from dataclasses import dataclass
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)
from src.m1_chunking import load_documents, chunk_hierarchical, chunk_semantic


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str 

def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    return word_tokenize(text, format="text")

class BM25Search:
    def __init__(self):
        self.corpus_tokens = []
        self.documents = []
        self.bm25 = None

    def index(self, chunks: list[dict]) -> None:
        """Build BM25 index from chunks."""
        self.documents = chunks
        self.corpus_tokens = [segment_vietnamese(c["text"]).split() for c in chunks]
        self.bm25 = BM25Okapi(self.corpus_tokens)

    def search(self, query: str, top_k: int = BM25_TOP_K) -> list[SearchResult]:
        if not self.bm25: return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        
        results = []
        for i in top_indices:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[i]),
                metadata=doc.get("metadata", {}),
                method="bm25"
            ))
        return results

class DenseSearch:
    def __init__(self):
        self._encoder = None
        self.documents = []
        self.vectors = None

    def _get_encoder(self):
        if self._encoder is None:
            # Dùng model đa ngôn ngữ mạnh cho tiếng Việt
            self._encoder = SentenceTransformer("sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2")
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into memory with vector normalization."""
        self.documents = chunks
        texts = [c["text"] for c in chunks]
        
        # 1. Encode văn bản
        raw_vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        
        # 2. Chuẩn hóa vector (Normalize to unit length) ngay từ đầu
        # Việc này giúp khi search chỉ cần tính Dot Product là ra Cosine Similarity
        norms = np.linalg.norm(raw_vectors, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        self.vectors = raw_vectors / norms
        
        # 3. Lưu ra file local
        os.makedirs("reports", exist_ok=True)
        np.save(f"reports/{collection}_vectors.npy", self.vectors)
        with open(f"reports/{collection}_docs.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using pre-normalized dense vectors."""
        if self.vectors is None:
            try:
                self.vectors = np.load(f"reports/{collection}_vectors.npy")
                with open(f"reports/{collection}_docs.json", "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except:
                return []

        # 1. Encode và chuẩn hóa query vector
        query_vector = self._get_encoder().encode(query)
        q_norm = np.linalg.norm(query_vector)
        if q_norm > 0:
            query_vector = query_vector / q_norm
        
        # 2. Tính Dot Product (lúc này tương đương Cosine Similarity vì cả 2 đã chuẩn hóa)
        sims = np.dot(self.vectors, query_vector)
        
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            results.append(SearchResult(
                text=self.documents[i]["text"],
                score=float(sims[i]),
                metadata=self.documents[i].get("metadata", {}),
                method="dense"
            ))
        return results

def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 80,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF with improved key matching."""
    rrf_scores = {}  
    
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            # Tạo key duy nhất dựa trên text và nguồn để tránh gộp nhầm
            # Nếu metadata có source hoặc ID thì dùng, không thì dùng text
            content_key = f"{result.text}_{result.metadata.get('source', '')}"
            
            if content_key not in rrf_scores:
                rrf_scores[content_key] = {
                    "score": 0.0,
                    "result": SearchResult(
                        text=result.text, 
                        score=0.0, 
                        metadata=result.metadata, 
                        method="hybrid"
                    )
                }
            rrf_scores[content_key]["score"] += 1.0 / (k + rank + 1)
    
    # Cập nhật điểm và sắp xếp
    final_results = []
    for item in rrf_scores.values():
        item["result"].score = item["score"]
        final_results.append(item["result"])
        
    return sorted(final_results, key=lambda x: x.score, reverse=True)[:top_k]

class HybridSearch:
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        # BM25 và Dense trả về danh sách kết quả riêng
        bm25_res = self.bm25.search(query, top_k=20) # Lấy rộng một chút để RRF hiệu quả
        dense_res = self.dense.search(query, top_k=20)
        
        return reciprocal_rank_fusion([bm25_res, dense_res], top_k=top_k)

if __name__ == "__main__":
    # 1. Load toàn bộ document từ thư mục data
    docs = load_documents(data_dir='data')
    
    all_children_data = [] # Danh sách tổng để chứa toàn bộ con từ mọi file
    
    print(f"--- Đang xử lý {len(docs)} documents ---")
    
    for doc in docs:
        text = doc.get("text", "")
        doc_meta = doc.get("metadata", {})
        
        # 2. Sử dụng Hierarchical Chunking từ Module 1
        parents, children = chunk_hierarchical(text, metadata=doc_meta)
        
        # 3. Chuyển đổi Chunk object sang dict để tương thích với Module 2
        for child in children:
            all_children_data.append({
                "text": child.text,
                "metadata": {
                    **child.metadata,
                    "parent_id": child.parent_id # Quan trọng để sau này retrieve ngược lại parent
                }
            })

    # 4. Khởi tạo Hybrid Search và Index TOÀN BỘ dữ liệu
    hybrid = HybridSearch()
    print(f"--- Đang tạo Index cho {len(all_children_data)} chunks ---")
    hybrid.index(all_children_data)
    
    # 5. Test thử query
    query = "quy định về nghỉ phép"
    results = hybrid.search(query)
    
    print(f"\nKết quả tìm kiếm cho: '{query}'")
    for i, r in enumerate(results):
        print(f"{i+1}. [{r.score:.4f}] {r.text} (Source: {r.metadata.get('source')})")