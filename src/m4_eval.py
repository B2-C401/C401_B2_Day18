"""Module 4: RAGAS Evaluation — 4 metrics + failure analysis."""

import os, sys, json
from dataclasses import dataclass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from config import TEST_SET_PATH


@dataclass
class EvalResult:
    question: str
    answer: str
    contexts: list[str]
    ground_truth: str
    faithfulness: float
    answer_relevancy: float
    context_precision: float
    context_recall: float


def load_test_set(path: str = TEST_SET_PATH) -> list[dict]:
    """Load test set from JSON. (Đã implement sẵn)"""
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def evaluate_ragas(questions: list[str], answers: list[str],
                   contexts: list[list[str]], ground_truths: list[str]) -> dict:
    """Run RAGAS evaluation (compatible with ragas >= 0.2)."""
    from config import OPENAI_API_KEY
    try:
        from ragas import evaluate, EvaluationDataset
        from ragas.metrics import Faithfulness, ResponseRelevancy, LLMContextPrecisionWithReference, LLMContextRecall
        from ragas.llms import LangchainLLMWrapper
        from ragas.embeddings import LangchainEmbeddingsWrapper
        from langchain_openai import ChatOpenAI, OpenAIEmbeddings

        llm = LangchainLLMWrapper(ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY))
        emb = LangchainEmbeddingsWrapper(OpenAIEmbeddings(api_key=OPENAI_API_KEY))

        metrics = [
            Faithfulness(llm=llm),
            ResponseRelevancy(llm=llm, embeddings=emb),
            LLMContextPrecisionWithReference(llm=llm),
            LLMContextRecall(llm=llm),
        ]

        dataset = EvaluationDataset.from_list([
            {"user_input": q, "response": a, "retrieved_contexts": ctx, "reference": gt}
            for q, a, ctx, gt in zip(questions, answers, contexts, ground_truths)
        ])

        result = evaluate(dataset=dataset, metrics=metrics)
        df = result.to_pandas()

        # Find columns by keyword (RAGAS 0.2+ renamed columns)
        def find_col(keywords):
            for c in df.columns:
                if any(k in c.lower() for k in keywords):
                    return c
            return None

        col_q  = find_col(["user_input", "question"])
        col_a  = find_col(["response", "answer"])
        col_ctx = find_col(["retrieved_contexts", "contexts"])
        col_gt = find_col(["reference", "ground_truth"])
        col_f  = find_col(["faithfulness"])
        col_ar = find_col(["relevancy", "relevance"])
        col_cp = find_col(["precision"])
        col_cr = find_col(["recall"])

        def safe_float(row, col):
            return float(row.get(col, 0.0) or 0.0) if col else 0.0

        def safe_mean(col):
            return float(df[col].dropna().mean()) if col and col in df.columns else 0.0

        per_question = [
            EvalResult(
                question=row.get(col_q, ""),
                answer=row.get(col_a, ""),
                contexts=row.get(col_ctx, []),
                ground_truth=row.get(col_gt, ""),
                faithfulness=safe_float(row, col_f),
                answer_relevancy=safe_float(row, col_ar),
                context_precision=safe_float(row, col_cp),
                context_recall=safe_float(row, col_cr),
            )
            for _, row in df.iterrows()
        ]

        return {
            "faithfulness":      safe_mean(col_f),
            "answer_relevancy":  safe_mean(col_ar),
            "context_precision": safe_mean(col_cp),
            "context_recall":    safe_mean(col_cr),
            "per_question":      per_question,
        }

    except Exception as e:
        print(f"  ⚠️  RAGAS evaluation failed: {e}")
        return {"faithfulness": 0.0, "answer_relevancy": 0.0,
                "context_precision": 0.0, "context_recall": 0.0, "per_question": []}


def failure_analysis(eval_results: list[EvalResult], bottom_n: int = 10) -> list[dict]:
    """Analyze bottom-N worst questions using Diagnostic Tree."""
    results_with_scores = []
    for res in eval_results:
        metrics = {
            "faithfulness": res.faithfulness,
            "answer_relevancy": res.answer_relevancy,
            "context_precision": res.context_precision,
            "context_recall": res.context_recall
        }
        avg_score = sum(metrics.values()) / 4.0
        results_with_scores.append((avg_score, res, metrics))
        
    results_with_scores.sort(key=lambda x: x[0])
    
    failures = []
    for avg_score, res, metrics in results_with_scores[:bottom_n]:
        worst_metric = min(metrics, key=metrics.get)
        score = metrics[worst_metric]
        
        diagnosis = "Unknown"
        suggested_fix = "Unknown"
        
        if worst_metric == "faithfulness" and score < 0.85:
            diagnosis = "LLM hallucinating"
            suggested_fix = "Tighten prompt, lower temperature"
        elif worst_metric == "context_recall" and score < 0.75:
            diagnosis = "Missing relevant chunks"
            suggested_fix = "Improve chunking or add BM25"
        elif worst_metric == "context_precision" and score < 0.75:
            diagnosis = "Too many irrelevant chunks"
            suggested_fix = "Add reranking or metadata filter"
        elif worst_metric == "answer_relevancy" and score < 0.80:
            diagnosis = "Answer doesn't match question"
            suggested_fix = "Improve prompt template"
            
        failures.append({
            "question": res.question,
            "worst_metric": worst_metric,
            "score": score,
            "diagnosis": diagnosis,
            "suggested_fix": suggested_fix
        })
        
    return failures


def save_report(results: dict, failures: list[dict], path: str = "ragas_report_5.json"):
    """Save evaluation report to JSON. (Đã implement sẵn)"""
    report = {
        "aggregate": {k: v for k, v in results.items() if k != "per_question"},
        "num_questions": len(results.get("per_question", [])),
        "failures": failures,
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    print(f"Report saved to {path}")


if __name__ == "__main__":
    test_set = load_test_set()
    print(f"Loaded {len(test_set)} test questions")
    print("Run pipeline.py first to generate answers, then call evaluate_ragas().")
