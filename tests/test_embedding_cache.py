
from app.storage import get_log_chunks
from app.retrieval import IncidentRetriever

DOCUMENT_ID = 1


def main():

    print("=" * 60)
    print("ROOTCAUSE AI - EMBEDDING CACHE INTEGRATION TEST")
    print("=" * 60)

    # Load chunks from MySQL.
    chunks = get_log_chunks(DOCUMENT_ID)

    assert chunks, "No chunks found in MySQL."

    print("\nChunks loaded:", len(chunks))

    # First initialization.
    print("\nTest 1: Initialize retriever")

    retriever1 = IncidentRetriever(chunks)

    assert len(retriever1.document_embeddings) == len(chunks)
    assert all(
        embedding is not None
        for embedding in retriever1.document_embeddings
    )

    print("First retriever initialized successfully.")

    # Second initialization should reuse cached embeddings.
    print("\nTest 2: Initialize retriever again")

    retriever2 = IncidentRetriever(chunks)

    assert len(retriever2.document_embeddings) == len(chunks)

    assert (
        retriever1.document_embeddings
        == retriever2.document_embeddings
    )

    print("Cached embeddings reused successfully.")

    # Verify semantic search.
    print("\nTest 3: Semantic search")

    results = retriever2.semantic_search(
        "Why did the database connection fail?",
        top_k=3
    )

    print("Semantic results:", len(results))

    assert results, "Semantic search returned no results."

    print("\nALL EMBEDDING CACHE TESTS PASSED!")


if __name__ == "__main__":
    main()
