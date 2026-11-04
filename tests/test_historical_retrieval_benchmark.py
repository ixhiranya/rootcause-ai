
import json
from pathlib import Path

from app.historical import search_similar_historical_incidents
from app.evaluation import evaluate_historical_retrieval


DATASET_PATH = Path("data/historical_test_cases.json")
RESULTS_PATH = Path("data/historical_benchmark_results.json")

TOP_K = 3

THRESHOLDS = [
    0.60,
    0.65,
    0.70,
    0.75,
    0.80,
    0.85,
    0.90
]


def load_test_cases():
    with open(DATASET_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def run_historical_benchmark():

    print("\nROOTCAUSE AI - HISTORICAL RETRIEVAL THRESHOLD BENCHMARK")
    print("=" * 65)

    test_cases = load_test_cases()

    positive_cases = [
        case for case in test_cases
        if not case.get("is_negative", False)
    ]

    negative_cases = [
        case for case in test_cases
        if case.get("is_negative", False)
    ]

    # Retrieve all available incidents once per test case.
    # Threshold filtering will happen locally afterward.
    all_results = {}

    for case in test_cases:

        print(f"\nPreparing {case['test_id']} - {case['category']}")

        all_results[case["test_id"]] = (
            search_similar_historical_incidents(
                query=case["query"],
                top_k=100
            )
        )

    threshold_results = []

    for threshold in THRESHOLDS:

        print("\n" + "=" * 65)
        print(f"SIMILARITY THRESHOLD: {threshold:.2f}")
        print("=" * 65)

        positive_precisions = []
        positive_recalls = []
        positive_mrrs = []

        positive_hits = 0
        false_positives = 0
        negative_rejections = 0

        case_results = []

        # Evaluate positive cases
        for case in positive_cases:

            retrieved = [
                incident
                for incident in all_results[case["test_id"]]
                if incident["similarity_score"] >= threshold
            ][:TOP_K]

            relevant_ids = case["relevant_incident_ids"]

            evaluation = evaluate_historical_retrieval(
                retrieved_incidents=retrieved,
                relevant_incident_ids=relevant_ids,
                k=TOP_K
            )

            hit = any(
                incident["incident_id"] in relevant_ids
                for incident in retrieved
            )

            if hit:
                positive_hits += 1

            positive_precisions.append(
                evaluation["precision_at_k"]
            )

            positive_recalls.append(
                evaluation["recall_at_k"]
            )

            positive_mrrs.append(
                evaluation["mrr"]
            )

            case_results.append({
                "test_id": case["test_id"],
                "category": case["category"],
                "is_negative": False,
                "retrieved_incident_ids": [
                    incident["incident_id"]
                    for incident in retrieved
                ],
                "similarity_scores": [
                    incident["similarity_score"]
                    for incident in retrieved
                ],
                "precision_at_k": evaluation["precision_at_k"],
                "recall_at_k": evaluation["recall_at_k"],
                "mrr": evaluation["mrr"],
                "hit": hit
            })

            print(
                f"{case['test_id']} | "
                f"Retrieved: "
                f"{[i['incident_id'] for i in retrieved]} | "
                f"Hit: {hit}"
            )

        # Evaluate negative cases separately
        for case in negative_cases:

            retrieved = [
                incident
                for incident in all_results[case["test_id"]]
                if incident["similarity_score"] >= threshold
            ][:TOP_K]

            false_positive = len(retrieved) > 0

            if false_positive:
                false_positives += 1
            else:
                negative_rejections += 1

            case_results.append({
                "test_id": case["test_id"],
                "category": case["category"],
                "is_negative": True,
                "retrieved_incident_ids": [
                    incident["incident_id"]
                    for incident in retrieved
                ],
                "similarity_scores": [
                    incident["similarity_score"]
                    for incident in retrieved
                ],
                "false_positive": false_positive
            })

            print(
                f"{case['test_id']} | "
                f"Retrieved: "
                f"{[i['incident_id'] for i in retrieved]} | "
                f"False Positive: {false_positive}"
            )

        # Aggregate metrics
        def average(values):
            return sum(values) / len(values) if values else 0.0

        precision = average(positive_precisions)
        recall = average(positive_recalls)
        mrr = average(positive_mrrs)

        positive_hit_rate = (
            positive_hits / len(positive_cases)
            if positive_cases else 0.0
        )

        false_positive_rate = (
            false_positives / len(negative_cases)
            if negative_cases else 0.0
        )

        negative_rejection_rate = (
            negative_rejections / len(negative_cases)
            if negative_cases else 0.0
        )

        result = {
            "threshold": threshold,
            "positive_precision_at_3": precision,
            "positive_recall_at_3": recall,
            "positive_mrr": mrr,
            "positive_hit_rate": positive_hit_rate,
            "negative_false_positive_rate": false_positive_rate,
            "negative_rejection_rate": negative_rejection_rate,
            "positive_cases": len(positive_cases),
            "negative_cases": len(negative_cases),
            "case_results": case_results
        }

        threshold_results.append(result)

        print("\nThreshold metrics:")
        print(f"Precision@3:          {precision:.4f}")
        print(f"Recall@3:             {recall:.4f}")
        print(f"MRR:                  {mrr:.4f}")
        print(f"Positive hit rate:    {positive_hit_rate:.4f}")
        print(f"False positive rate:  {false_positive_rate:.4f}")
        print(f"Negative rejection:   {negative_rejection_rate:.4f}")

    # Display comparison table
    print("\n\nTHRESHOLD COMPARISON")
    print("=" * 95)

    print(
        f"{'Threshold':<12}"
        f"{'Precision':<13}"
        f"{'Recall':<11}"
        f"{'MRR':<10}"
        f"{'Hit Rate':<12}"
        f"{'False Pos.':<13}"
        f"{'Rejection':<12}"
    )

    for result in threshold_results:

        print(
            f"{result['threshold']:<12.2f}"
            f"{result['positive_precision_at_3']:<13.4f}"
            f"{result['positive_recall_at_3']:<11.4f}"
            f"{result['positive_mrr']:<10.4f}"
            f"{result['positive_hit_rate']:<12.4f}"
            f"{result['negative_false_positive_rate']:<13.4f}"
            f"{result['negative_rejection_rate']:<12.4f}"
        )

    output = {
        "benchmark": "Historical Incident Retrieval Threshold Evaluation",
        "top_k": TOP_K,
        "thresholds_tested": THRESHOLDS,
        "positive_test_count": len(positive_cases),
        "negative_test_count": len(negative_cases),
        "results": threshold_results,
        "note": (
            "Exploratory benchmark using a small synthetic dataset. "
            "Similarity thresholds are not calibrated probabilities."
        )
    }

    with open(RESULTS_PATH, "w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\nBenchmark results saved to:")
    print(RESULTS_PATH)

    print("\nHISTORICAL THRESHOLD BENCHMARK COMPLETED!")


if __name__ == "__main__":
    run_historical_benchmark()
