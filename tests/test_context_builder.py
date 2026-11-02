
from app.context_builder import build_context


def main():

    # Sample retrieved chunks
    retrieved_chunks = [
        {
            "chunk_id": "chunk_0",
            "source": "sample_incident.log",
            "score": 0.01639,
            "text": (
                "Database connection pool exhausted. "
                "Maximum connections=10, active connections=10. "
                "Database query failed."
            )
        },
        {
            "chunk_id": "chunk_1",
            "source": "sample_incident.log",
            "score": 0.01613,
            "text": (
                "Database server is reachable, but active "
                "connections remain at maximum capacity."
            )
        },
        {
            "chunk_id": "chunk_2",
            "source": "sample_incident.log",
            "score": 0.01587,
            "text": (
                "Database connection pool exhausted. "
                "Maximum connections=10, active connections=10. "
                "Database query failed."
            )
        }
    ]

    print("=" * 60)
    print("ROOTCAUSE AI - CONTEXT BUILDER TEST")
    print("=" * 60)

    result = build_context(
        retrieved_chunks,
        max_tokens=2500
    )

    print("\n--- CONSTRUCTED CONTEXT ---")
    print(result["context"])

    print("\n--- CONTEXT STATISTICS ---")
    print("Evidence count:", result["evidence_count"])
    print("Estimated tokens:", result["estimated_tokens"])
    print("Maximum tokens:", result["max_tokens"])

    print("\n--- STRUCTURED EVIDENCE ---")

    for item in result["evidence"]:
        print(
            f"{item['evidence_id']} | "
            f"Source: {item['source']} | "
            f"Chunk: {item['chunk_id']}"
        )


if __name__ == "__main__":
    main()
