"""Module 3: Reranking - Cross-encoder top-20 to top-3 + latency benchmark."""

import os
import re
import sys
import time
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import RERANK_TOP_K


@dataclass
class RerankResult:
    text: str
    original_score: float
    rerank_score: float
    metadata: dict
    rank: int


class LexicalReranker:
    """Small deterministic fallback for offline tests and missing model weights."""

    @staticmethod
    def _tokens(text: str) -> list[str]:
        return re.findall(r"\w+", text.lower(), flags=re.UNICODE)

    def compute_score(self, pairs: list[tuple[str, str]]) -> list[float]:
        scores = []
        for query, document in pairs:
            q_tokens = self._tokens(query)
            d_tokens = self._tokens(document)
            if not q_tokens or not d_tokens:
                scores.append(0.0)
                continue
            q_set = set(q_tokens)
            d_set = set(d_tokens)
            overlap = len(q_set & d_set) / len(q_set)
            phrase_bonus = 0.2 if query.lower().strip("?") in document.lower() else 0.0
            scores.append(float(overlap + phrase_bonus))
        return scores


class CrossEncoderReranker:
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._backend = ""

    def _load_model(self):
        if self._model is not None:
            return self._model

        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
            self._backend = "crossencoder"
        except Exception:
            try:
                from FlagEmbedding import FlagReranker

                self._model = FlagReranker(self.model_name, use_fp16=False)
                self._backend = "flagembedding"
            except Exception:
                self._model = LexicalReranker()
                self._backend = "lexical"
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        """Rerank documents and return the top_k results sorted by rerank_score."""
        if not documents or top_k <= 0:
            return []

        pairs = [(query, doc.get("text", "")) for doc in documents]
        model = self._load_model()
        try:
            if hasattr(model, "compute_score"):
                raw_scores = model.compute_score(pairs)
            else:
                raw_scores = model.predict(pairs)
        except Exception:
            # If one real backend fails at runtime, try the other real backend once.
            if self._backend == "crossencoder":
                try:
                    from FlagEmbedding import FlagReranker

                    self._model = FlagReranker(self.model_name, use_fp16=False)
                    self._backend = "flagembedding"
                    raw_scores = self._model.compute_score(pairs)
                except Exception:
                    self._model = LexicalReranker()
                    self._backend = "lexical"
                    raw_scores = self._model.compute_score(pairs)
            elif self._backend == "flagembedding":
                try:
                    from sentence_transformers import CrossEncoder

                    self._model = CrossEncoder(self.model_name)
                    self._backend = "crossencoder"
                    raw_scores = self._model.predict(pairs)
                except Exception:
                    self._model = LexicalReranker()
                    self._backend = "lexical"
                    raw_scores = self._model.compute_score(pairs)
            else:
                raw_scores = self._model.compute_score(pairs)

        if not isinstance(raw_scores, list):
            try:
                raw_scores = raw_scores.tolist()
            except AttributeError:
                raw_scores = list(raw_scores)

        combined = sorted(
            zip(raw_scores, documents),
            key=lambda item: float(item[0]),
            reverse=True,
        )[:top_k]

        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(combined, start=1)
        ]


class FlashrankReranker:
    """Lightweight alternative using flashrank when installed."""

    def __init__(self):
        self._model = None

    def _load_model(self):
        if self._model is None:
            try:
                from flashrank import Ranker

                self._model = Ranker()
            except Exception:
                self._model = LexicalReranker()
        return self._model

    def rerank(self, query: str, documents: list[dict], top_k: int = RERANK_TOP_K) -> list[RerankResult]:
        if not documents or top_k <= 0:
            return []

        model = self._load_model()
        if isinstance(model, LexicalReranker):
            scored = model.compute_score([(query, d.get("text", "")) for d in documents])
            ranked = sorted(zip(scored, documents), key=lambda item: item[0], reverse=True)[:top_k]
        else:
            from flashrank import RerankRequest

            passages = [{"id": i, "text": d.get("text", "")} for i, d in enumerate(documents)]
            results = model.rerank(RerankRequest(query=query, passages=passages))
            ranked = [(r.get("score", 0.0), documents[int(r.get("id", 0))]) for r in results[:top_k]]

        return [
            RerankResult(
                text=doc.get("text", ""),
                original_score=float(doc.get("score", 0.0)),
                rerank_score=float(score),
                metadata=doc.get("metadata", {}),
                rank=i,
            )
            for i, (score, doc) in enumerate(ranked, start=1)
        ]


def benchmark_reranker(reranker, query: str, documents: list[dict], n_runs: int = 5) -> dict:
    """Benchmark latency over n_runs."""
    times = []
    for _ in range(max(n_runs, 1)):
        start = time.perf_counter()
        reranker.rerank(query, documents)
        times.append((time.perf_counter() - start) * 1000)
    return {
        "avg_ms": sum(times) / len(times),
        "min_ms": min(times),
        "max_ms": max(times),
    }


if __name__ == "__main__":
    query = "Nhan vien duoc nghi phep bao nhieu ngay?"
    docs = [
        {"text": "Nhan vien duoc nghi 12 ngay/nam.", "score": 0.8, "metadata": {}},
        {"text": "Mat khau thay doi moi 90 ngay.", "score": 0.7, "metadata": {}},
        {"text": "Thoi gian thu viec la 60 ngay.", "score": 0.75, "metadata": {}},
    ]
    reranker = CrossEncoderReranker()
    for r in reranker.rerank(query, docs):
        print(f"[{r.rank}] {r.rerank_score:.4f} | {r.text}")
