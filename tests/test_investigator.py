
from app.chunking import chunk_log_text, load_text_file
from app.retrieval import IncidentRetriever
from app.context_builder import build_context
from app.investigator import RootCauseInvestigator


def main():

    print("=" * 70)
    print("ROOTCAUSE AI - END-TO-END INVESTIGATION")
    print("=" * 70)

    # 1. Load incident logs
    text = load_text_file(
        "data/samples/sample_incident.log"
    )

    # 2. Create structural-aware chunks
    chunks = chunk_log_text(
        text,
        source="sample_incident.log",
        target_size=500,
        max_size=1000
    )

    print(f"\nChunks created: {len(chunks)}")

    # 3. Initialize hybrid retriever
    retriever = IncidentRetriever(chunks)

    # 4. User investigation query
    query = "Why are order requests failing with HTTP 500 errors?"

    print("\nInvestigation query:")
    print(query)

    # 5. Retrieve relevant evidence
    retrieved_chunks = retriever.hybrid_search(
        query,
        top_k=3,
        candidate_k=5
    )

    print(f"\nRetrieved chunks: {len(retrieved_chunks)}")

    # 6. Construct evidence context
    context_result = build_context(
        retrieved_chunks,
        max_tokens=2500
    )

    print(
        f"Evidence items: {context_result['evidence_count']}"
    )

    print(
        f"Estimated context tokens: "
        f"{context_result['estimated_tokens']}"
    )

    if not context_result["context"]:
        raise RuntimeError("No evidence available for investigation.")

    # 7. Initialize LLM investigator
    investigator = RootCauseInvestigator(
        model="qwen2.5:3b",
        temperature=0.2,
        max_tokens=800
    )

    # 8. Generate investigation
    print("\nGenerating root-cause investigation...")
    print("Please wait for the local model...")

    report = investigator.investigate(
        query=query,
        context=context_result["context"]
    )

    # 9. Display report
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
        print(f"\nHypothesis: {item['hypothesis']}")
        print(f"Reason: {item['reason']}")

    print("\nRECOMMENDED ACTIONS:")

    for index, action in enumerate(
        report["recommended_actions"], start=1
    ):
        print(f"{index}. {action}")

    print("\nMISSING INFORMATION:")

    for item in report["missing_information"]:
        print(f"- {item}")

    print("\n" + "=" * 70)
    print("INVESTIGATION COMPLETED")
    print("=" * 70)


if __name__ == "__main__":
    main()
