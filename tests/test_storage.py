from app.storage import (
    save_document,
    get_document_by_hash,
    save_log_chunks,
    get_log_chunks
)
from app.chunking import chunk_log_text
from app.storage import save_chunk_embedding, get_chunk_embedding
from app.retrieval import get_embeddings

def main():
    print("=" * 55)
    print("ROOTCAUSE AI - DOCUMENT AND CHUNK STORAGE TEST")
    print("=" * 55)

    sample_content = """
2026-11-03 10:00:00 INFO Application started
2026-11-03 10:00:01 ERROR Database connection failed
2026-11-03 10:00:02 ERROR Connection pool exhausted
"""

    # Test 1: Save document
    print("\nTest 1: Saving document")

    result1 = save_document(
        file_name="test_incident.log",
        content=sample_content,
        source_type="LOG"
    )

    print(result1)

    assert result1["document_id"] is not None

    # Test 2: Duplicate detection
    print("\nTest 2: Uploading duplicate content")

    result2 = save_document(
        file_name="duplicate_incident.log",
        content=sample_content,
        source_type="LOG"
    )

    print(result2)

    assert result1["document_id"] == result2["document_id"]
    assert result2["duplicate"] is True

    # Test 3: Generate and save chunks
    print("\nTest 3: Saving log chunks")

    chunks = chunk_log_text(
        sample_content,
        source="test_incident.log"
    )

    save_result = save_log_chunks(
        result1["document_id"],
        chunks
    )

    print(save_result)

    assert save_result["chunks_saved"] == len(chunks)

    # Test 4: Retrieve chunks
    print("\nTest 4: Retrieving log chunks")

    retrieved = get_log_chunks(result1["document_id"])

    print("Chunks retrieved:", len(retrieved))

    assert len(retrieved) == len(chunks)

    print("\nALL DOCUMENT AND CHUNK STORAGE TESTS PASSED!")


    print("\nTest 5: Generating embedding using Ollama")

    MODEL_NAME = "nomic-embed-text"

    # Generate an embedding for the first retrieved chunk.
    vectors = get_embeddings([retrieved[0]["text"]])

    embedding = vectors[0]

    print("Embedding dimensions:", len(embedding))

    assert len(embedding) > 0

    print("\nTest 6: Saving embedding to MySQL")

    embedding_result = save_chunk_embedding(
        retrieved[0]["db_chunk_id"],
        MODEL_NAME,
        embedding
    )

    print(embedding_result)

    print("\nTest 7: Retrieving cached embedding")

    cached_embedding = get_chunk_embedding(
        retrieved[0]["db_chunk_id"],
        MODEL_NAME
    )

    assert cached_embedding is not None
    assert len(cached_embedding) == len(embedding)

    print("Cached embedding dimensions:", len(cached_embedding))

    assert cached_embedding == embedding

    print("\nALL DOCUMENT, CHUNK AND EMBEDDING TESTS PASSED!")



if __name__ == "__main__":
    main()
