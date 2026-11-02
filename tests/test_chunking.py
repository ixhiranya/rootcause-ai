
from app.chunking import (
    chunk_log_text,
    load_text_file,
    split_into_log_events,
)


def main():
    path = "data/samples/sample_incident.log"
    text = load_text_file(path)

    events = split_into_log_events(text)
    chunks = chunk_log_text(
        text,
        source="sample_incident.log",
        target_size=500,
        max_size=1000,
    )

    print(f"Log events detected: {len(events)}")
    print(f"Chunks created: {len(chunks)}")

    for chunk in chunks:
        print(
            f"\nChunk {chunk['chunk_index']}: "
            f"{len(chunk['text'])} characters"
        )
        print(chunk["text"][:200])

    assert chunk_log_text("") == []
    assert chunk_log_text("  \n") == []

    # Verify stack-trace lines stay with their originating event.
    stack_log = (
        "2026-10-15 09:00:00 ERROR Request failed\n"
        "    at orders.py:42\n"
        "    at server.py:10\n"
        "2026-10-15 09:00:01 INFO Request completed\n"
    )

    stack_events = split_into_log_events(stack_log)

    assert len(stack_events) == 2
    assert "orders.py:42" in stack_events[0]["text"]
    assert "server.py:10" in stack_events[0]["text"]

    # Verify we preserve the full document text.
    reconstructed = "".join(chunk["text"] for chunk in chunks)
    assert reconstructed == text

    # Verify the hard maximum for ordinary chunks.
    assert all(len(c["text"]) <= 1000 for c in chunks)

    print("\nAll structural chunking checks passed!")


if __name__ == "__main__":
    main()
