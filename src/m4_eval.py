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
    """Run RAGAS evaluation."""
    from ragas import evaluate
    from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
    from datasets import Dataset

    dataset = Dataset.from_dict({
        "question": questions, 
        "answer": answers,
        "contexts": contexts, 
        "ground_truth": ground_truths,
    })

    try:
        result = evaluate(dataset, metrics=[faithfulness, answer_relevancy,
                                            context_precision, context_recall])
        df = result.to_pandas()

        per_question = []
        for _, row in df.iterrows():
            per_question.append(
                EvalResult(
                    question=row["question"],
                    answer=row["answer"],
                    contexts=row["contexts"],
                    ground_truth=row["ground_truth"],
                    faithfulness=row["faithfulness"],
                    answer_relevancy=row["answer_relevancy"],
                    context_precision=row["context_precision"],
                    context_recall=row["context_recall"],
                )
            )

        return {
            "faithfulness": float(result.get("faithfulness", 0.0)),
            "answer_relevancy": float(result.get("answer_relevancy", 0.0)),
            "context_precision": float(result.get("context_precision", 0.0)),
            "context_recall": float(result.get("context_recall", 0.0)),
            "per_question": per_question,
        }
    except Exception as e:
        print(f"RAGAS evaluation failed (likely due to missing OpenAI key): {e}")
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


def save_report(results: dict, failures: list[dict], path: str = "ragas_report.json"):
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
