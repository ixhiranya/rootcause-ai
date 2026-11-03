
from app.chunking import chunk_log_text, load_text_file
from app.retrieval import IncidentRetriever
from app.context_builder import build_context
from app.investigator import RootCauseInvestigator
from app.evaluation import evaluate_report


def main():

    print("=" * 70)
    print("ROOTCAUSE AI - INVESTIGATION WITH AUTOMATIC EVALUATION")
    print("=" * 70)

    # 1. Load incident logs
    text = load_text_file(
        "data/samples/sample_incident.log"
    )

    # 2. Structural-aware chunking
    chunks = chunk_log_text(
        text,
        source="sample_incident.log",
        target_size=500,
        max_size=1000
    )

    print(f"\nChunks created: {len(chunks)}")

    # 3. Initialize retriever
    retriever = IncidentRetriever(chunks)

    # 4. Investigation query
    query = "Why are order requests failing with HTTP 500 errors?"

    print("\nQUERY:")
    print(query)

    # 5. Hybrid retrieval
    retrieved_chunks = retriever.hybrid_search(
        query,
        top_k=3,
        candidate_k=5
    )

    print(f"\nRetrieved chunks: {len(retrieved_chunks)}")

    # 6. Context construction
    context_result = build_context(
        retrieved_chunks,
        max_tokens=2500
    )

    context = context_result["context"]
    evidence = context_result["evidence"]

    if not evidence:
        raise RuntimeError(
            "No evidence available for investigation."
        )

    print(f"Evidence items: {len(evidence)}")
    print(
        f"Estimated context tokens: "
        f"{context_result['estimated_tokens']}"
    )

    # 7. Generate investigation
    investigator = RootCauseInvestigator(
        model="qwen2.5:3b",
        temperature=0.2,
        max_tokens=800
    )

    print("\nGenerating investigation...")

    report = investigator.investigate(
        query=query,
        context=context
    )

    # 8. Display investigation
    print("\n" + "=" * 70)
    print("ROOTCAUSE AI INVESTIGATION REPORT")
    print("=" * 70)

    print("\nINCIDENT SUMMARY:")
    print(report["incident_summary"])

    print("\nPROBABLE ROOT CAUSE:")
    print(report["probable_root_cause"])

    print("\nCONFIDENCE:")
    print(report["confidence"])

    print("\nSUPPORTING EVIDENCE:")

    for item in report["supporting_evidence"]:
        print(
            f"\n[{item['evidence_id']}] "
            f"{item['observation']}"
        )

    print("\nALTERNATIVE HYPOTHESES:")

    for item in report["alternative_hypotheses"]:
        print(f"- {item['hypothesis']}")
        print(f"  Reason: {item['reason']}")

    print("\nRECOMMENDED ACTIONS:")

    for index, action in enumerate(
        report["recommended_actions"],
        start=1
    ):
        print(f"{index}. {action}")

    print("\nMISSING INFORMATION:")

    for item in report["missing_information"]:
        print(f"- {item}")

    # 9. AUTOMATIC EVALUATION
    print("\n" + "=" * 70)
    print("AUTOMATIC EVALUATION")
    print("=" * 70)

    evaluation = evaluate_report(
        report=report,
        evidence=evidence,
        context=context
    )

    citation = evaluation["citation_evaluation"]
    consistency = evaluation["consistency_evaluation"]

    print("\nCITATION VALIDATION:")

    print(
        "Citation accuracy:",
        citation["citation_accuracy"]
    )

    print(
        "Valid citations:",
        citation["valid_citations"]
    )

    print(
        "Invalid citations:",
        citation["invalid_citations"]
    )

    print(
        "All citations valid:",
        citation["all_citations_valid"]
    )

    print("\nEVIDENCE CONSISTENCY:")

    print(
        "Consistency passed:",
        consistency["consistency_passed"]
    )

    print(
        "Detected facts:",
        consistency["detected_facts"]
    )

    if consistency["inconsistencies"]:
        print("\nDetected inconsistencies:")

        for issue in consistency["inconsistencies"]:
            print("-", issue["issue"])
            print("  Model claim:", issue["model_claim"])

    print("\nOVERALL VALIDATION:")

    print(
        evaluation["overall_validation_passed"]
    )

    print("\n" + "=" * 70)
    print("INVESTIGATION AND EVALUATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
