
from app.evaluation import (
    evaluate_retrieval,
    evaluate_citations,
    check_missing_information,
    evaluate_report
)


def main():

    print("=" * 65)
    print("ROOTCAUSE AI - EVALUATION TEST")
    print("=" * 65)

    # --------------------------------------------------
    # SAMPLE RETRIEVAL RESULTS
    # --------------------------------------------------

    retrieved_chunks = [
        {"chunk_id": "chunk_2"},
        {"chunk_id": "chunk_0"},
        {"chunk_id": "chunk_1"}
    ]

    # Ground truth: chunk_0 and chunk_1 contain
    # relevant evidence for this test query.
    relevant_chunk_ids = [
        "chunk_0",
        "chunk_1"
    ]

    retrieval_results = evaluate_retrieval(
        retrieved_chunks,
        relevant_chunk_ids,
        k=3
    )

    print("\n--- RETRIEVAL EVALUATION ---")
    print("Recall@3:", retrieval_results["recall_at_k"])
    print("MRR:", retrieval_results["mrr"])
    print(
        "Relevant chunks found:",
        retrieval_results["relevant_chunks_found"]
    )
    print(
        "Retrieved IDs:",
        retrieval_results["retrieved_chunk_ids"]
    )

    # --------------------------------------------------
    # SAMPLE EVIDENCE
    # --------------------------------------------------

    evidence = [
        {
            "evidence_id": "E1",
            "chunk_id": "chunk_0",
            "source": "sample_incident.log",
            "text": (
                "Connection pool exhausted. "
                "Maximum connections=10, active connections=10."
            )
        },
        {
            "evidence_id": "E2",
            "chunk_id": "chunk_1",
            "source": "sample_incident.log",
            "text": (
                "Database query failed with TimeoutError. "
                "API returned HTTP 500."
            )
        }
    ]

    context = "\n\n".join(
        item["text"] for item in evidence
    )

    # --------------------------------------------------
    # SAMPLE MODEL REPORT
    # --------------------------------------------------

    report = {
        "incident_summary": (
            "Order requests are failing due to database "
            "connection pool exhaustion."
        ),
        "probable_root_cause": (
            "The connection pool has reached maximum capacity."
        ),
        "confidence": "high",
        "supporting_evidence": [
            {
                "evidence_id": "E1",
                "observation": (
                    "Maximum and active connections are both 10."
                )
            },
            {
                "evidence_id": "E99",
                "observation": "This is an intentionally invalid citation."
            }
        ],
        "alternative_hypotheses": [],
        "recommended_actions": [
            "Investigate connection leaks.",
            "Inspect long-running database queries."
        ],
        "missing_information": [
            (
                "The maximum connections and active connections "
                "settings are unknown."
            )
        ]
    }

    # --------------------------------------------------
    # CITATION EVALUATION
    # --------------------------------------------------

    citation_results = evaluate_citations(
        report,
        evidence
    )

    print("\n--- CITATION EVALUATION ---")
    print(
        "Citation accuracy:",
        citation_results["citation_accuracy"]
    )
    print(
        "Valid citations:",
        citation_results["valid_citations"]
    )
    print(
        "Invalid citations:",
        citation_results["invalid_citations"]
    )

    # --------------------------------------------------
    # CONSISTENCY EVALUATION
    # --------------------------------------------------

    consistency_results = check_missing_information(
        report,
        context
    )

    print("\n--- CONSISTENCY EVALUATION ---")
    print(
        "Detected facts:",
        consistency_results["detected_facts"]
    )
    print(
        "Consistency passed:",
        consistency_results["consistency_passed"]
    )

    for issue in consistency_results["inconsistencies"]:
        print("\nIssue:", issue["issue"])
        print("Model claim:", issue["model_claim"])
        print(
            "Fact found:",
            issue["fact_found_in_context"]
        )

    # --------------------------------------------------
    # COMBINED EVALUATION
    # --------------------------------------------------

    final_results = evaluate_report(
        report,
        evidence,
        context
    )

    print("\n--- FINAL EVALUATION ---")
    print(
        "Overall validation passed:",
        final_results["overall_validation_passed"]
    )

    print("\n" + "=" * 65)
    print("EVALUATION TEST COMPLETED")
    print("=" * 65)


if __name__ == "__main__":
    main()
