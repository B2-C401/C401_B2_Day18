"""Module 2: Hybrid Search — BM25 (Vietnamese) + Dense + RRF."""

import os, sys
from dataclasses import dataclass
from underthesea import word_tokenize
from rank_bm25 import BM25Okapi
import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import (QDRANT_HOST, QDRANT_PORT, COLLECTION_NAME, EMBEDDING_MODEL,
                    EMBEDDING_DIM, BM25_TOP_K, DENSE_TOP_K, HYBRID_TOP_K)


@dataclass
class SearchResult:
    text: str
    score: float
    metadata: dict
    method: str  # "bm25", "dense", "hybrid"


def segment_vietnamese(text: str) -> str:
    """Segment Vietnamese text into words."""
    # TODO: Implement Vietnamese word segmentation
    # 1. from underthesea import word_tokenize
    # 2. return word_tokenize(text, format="text")
    # Why: BM25 needs word boundaries. "nghỉ phép" = 1 word, not 2.
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
        """Search using BM25."""
        if not self.bm25:
            return []
        tokenized_query = segment_vietnamese(query).split()
        scores = self.bm25.get_scores(tokenized_query)
        top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:top_k]
        print('searching')
        results = []
        for i in top_indices:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=float(scores[i]),
                metadata=doc["metadata"],
                method="bm25"
            ))
        return results



class DenseSearch:
    def __init__(self):
        self._encoder = None
        self.documents = []
        self.vectors = []

    def _get_encoder(self):
        if self._encoder is None:
            from sentence_transformers import SentenceTransformer
            self._encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        return self._encoder

    def index(self, chunks: list[dict], collection: str = COLLECTION_NAME) -> None:
        """Index chunks into memory."""
        import json
        import numpy as np
        self.documents = chunks
        texts = [c["text"] for c in chunks]
        self.vectors = self._get_encoder().encode(texts, show_progress_bar=True)
        # Lưu ra file local
        os.makedirs("reports", exist_ok=True)
        np.save(f"reports/{collection}_vectors.npy", self.vectors)
        with open(f"reports/{collection}_docs.json", "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)

    def search(self, query: str, top_k: int = DENSE_TOP_K, collection: str = COLLECTION_NAME) -> list[SearchResult]:
        """Search using dense vectors."""
        import numpy as np
        if len(self.vectors) == 0:
            try:
                self.vectors = np.load(f"reports/{collection}_vectors.npy")
                import json
                with open(f"reports/{collection}_docs.json", "r", encoding="utf-8") as f:
                    self.documents = json.load(f)
            except Exception:
                return []

        query_vector = self._get_encoder().encode(query)
        
        # Tính toán cosine similarity dùng numpy
        norms = np.linalg.norm(self.vectors, axis=1)
        norms[norms == 0] = 1.0
        q_norm = np.linalg.norm(query_vector)
        if q_norm == 0:
            q_norm = 1.0
        
        sims = np.dot(self.vectors, query_vector) / (norms * q_norm)
        
        # Sắp xếp index theo điểm giảm dần
        top_indices = np.argsort(sims)[::-1][:top_k]
        
        results = []
        for i in top_indices:
            doc = self.documents[i]
            results.append(SearchResult(
                text=doc["text"],
                score=float(sims[i]),
                metadata=doc["metadata"],
                method="dense"
            ))
        return results



def reciprocal_rank_fusion(results_list: list[list[SearchResult]], k: int = 60,
                           top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
    """Merge ranked lists using RRF: score(d) = Σ 1/(k + rank)."""
    rrf_scores = {}  # text → {"score": float, "result": SearchResult}
    for result_list in results_list:
        for rank, result in enumerate(result_list):
            if result.text not in rrf_scores:
                rrf_scores[result.text] = {
                    "score": 0.0,
                    "result": SearchResult(text=result.text, score=0.0, metadata=result.metadata, method="hybrid")
                }
            rrf_scores[result.text]["score"] += 1.0 / (k + rank + 1)
    
    # Update score in SearchResult objects
    for item in rrf_scores.values():
        item["result"].score = item["score"]
        
    sorted_items = sorted(rrf_scores.values(), key=lambda x: x["score"], reverse=True)[:top_k]
    return [item["result"] for item in sorted_items]



class HybridSearch:
    """Combines BM25 + Dense + RRF. (Đã implement sẵn — dùng classes ở trên)"""
    def __init__(self):
        self.bm25 = BM25Search()
        self.dense = DenseSearch()

    def index(self, chunks: list[dict]) -> None:
        self.bm25.index(chunks)
        self.dense.index(chunks)

    def search(self, query: str, top_k: int = HYBRID_TOP_K) -> list[SearchResult]:
        bm25_results = self.bm25.search(query, top_k=BM25_TOP_K)
        dense_results = self.dense.search(query, top_k=DENSE_TOP_K)
        return reciprocal_rank_fusion([bm25_results, dense_results], top_k=top_k)


if __name__ == "__main__":
    print(f"Original:  Nhân viên được nghỉ phép năm")
    print(f"Segmented: {segment_vietnamese('Nhân viên được nghỉ phép năm')}")
    
    # Để tìm kiếm bằng BM25, bạn cần khởi tạo chunks và index trước:
    bm25 = BM25Search()
    dense = DenseSearch()

    CHUNKS = [
        {"text": "Nhân viên được nghỉ phép năm 12 ngày.", "metadata": {"source": "policy"}},
        {"text": "Mật khẩu thay đổi mỗi 90 ngày.", "metadata": {"source": "it"}},
        {"text": "Thời gian thử việc là 60 ngày.", "metadata": {"source": "hr"}},
    ]
    dense.index(CHUNKS)
    
    # Giờ bạn đã có thể tìm kiếm:
    results = dense.search("Nhân viên được nghỉ phép năm")
    print("\nDense Search Results:")
    for r in results:
        print(f" - Score: {r.score:.4f} | Text: {r.text}")

