
from app.chunking import chunk_log_text, load_text_file
from app.retrieval import IncidentRetriever


def print_results(title: str, results: list[dict], score_label: str):
    print(f"\n--- {title} ---")

    if not results:
        print("No matching results found.")
        return

    for rank, result in enumerate(results, start=1):

        print(f"\nRank: {rank}")
        print(f"{score_label}: {result['score']:.5f}")
        print(f"Chunk ID: {result['chunk_id']}")
        print(f"Source: {result['source']}")

        if "matched_tokens" in result:
            print(f"Matched tokens: {result['matched_tokens']}")

        print("Text:")
        print(result["text"][:300])


def main():

    # --------------------------------------------------
    # LOAD SAMPLE LOG
    # --------------------------------------------------

    text = load_text_file(
        "data/samples/sample_incident.log"
    )

    # --------------------------------------------------
    # STRUCTURAL-AWARE CHUNKING
    # --------------------------------------------------

    chunks = chunk_log_text(
        text,
        source="sample_incident.log",
        target_size=500,
        max_size=1000
    )

    print(f"Total chunks created: {len(chunks)}")

    for chunk in chunks:
        print(
            f"Chunk {chunk['chunk_id']}: "
            f"{len(chunk['text'])} characters"
        )

    # --------------------------------------------------
    # INITIALIZE RETRIEVER
    # --------------------------------------------------

    retriever = IncidentRetriever(chunks)

    # --------------------------------------------------
    # TEST QUERIES
    # --------------------------------------------------

    queries = [
        "TimeoutError",
        "connection pool exhausted",
        "Why are order requests failing?",
        "Is the database server down?",
        "database connection acquisition timeout",
        "Which service returned HTTP 500?"
    ]

    # --------------------------------------------------
    # RUN ALL RETRIEVAL METHODS
    # --------------------------------------------------

    for query in queries:

        print("\n" + "=" * 70)
        print("QUERY:", query)
        print("=" * 70)

        # BM25Plus keyword retrieval
        keyword_results = retriever.keyword_search(
            query,
            top_k=3
        )

        print_results(
            "BM25Plus Keyword Search",
            keyword_results,
            "BM25 Score"
        )

        # Semantic retrieval
        semantic_results = retriever.semantic_search(
            query,
            top_k=3
        )

        print_results(
            "Semantic Search",
            semantic_results,
            "Cosine Similarity"
        )

        # Hybrid retrieval
        hybrid_results = retriever.hybrid_search(
            query,
            top_k=3
        )

        print_results(
            "Hybrid Search (RRF)",
            hybrid_results,
            "RRF Score"
        )


if __name__ == "__main__":
    main()
