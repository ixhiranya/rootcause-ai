from app.evaluation import evaluate_historical_retrieval


def test_historical_retrieval_evaluation():

    retrieved_incidents = [
        {"incident_id": 3},
        {"incident_id": 8},
        {"incident_id": 2},
        {"incident_id": 9},
        {"incident_id": 5}
    ]

    relevant_incident_ids = [2, 3, 5]

    results = evaluate_historical_retrieval(
        retrieved_incidents=retrieved_incidents,
        relevant_incident_ids=relevant_incident_ids,
        k=5
    )

    print("\nHISTORICAL RETRIEVAL EVALUATION")
    print("=" * 45)

    for metric, value in results.items():
        print(f"{metric}: {value}")

    assert results["precision_at_k"] == 0.6
    assert results["recall_at_k"] == 1.0
    assert results["mrr"] == 1.0
    assert results["hit_rate_at_k"] == 1.0

    print("\nHISTORICAL EVALUATION TEST PASSED!")


if __name__ == "__main__":
    test_historical_retrieval_evaluation()