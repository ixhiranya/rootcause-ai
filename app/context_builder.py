
import re


def estimate_tokens(text: str) -> int:
    """
    Approximate token count.

    This is a lightweight estimate, not an exact
    model-specific tokenizer.
    """
    return max(1, len(text) // 4)


def normalize_text(text: str) -> str:
    """
    Normalize text to identify duplicate chunks.
    """
    return re.sub(r"\s+", " ", text).strip().lower()


def build_context(
    retrieved_chunks: list[dict],
    max_tokens: int = 2500
) -> dict:
    """
    Construct an evidence-based context from retrieved chunks.

    Features:
    - Duplicate removal
    - Evidence ID assignment
    - Source tracking
    - Token budget management
    - Structured evidence output
    """

    if max_tokens <= 0:
        raise ValueError("max_tokens must be greater than zero.")

    evidence = []
    seen_chunks = set()

    context_parts = []
    total_tokens = 0

    for chunk in retrieved_chunks:

        text = chunk.get("text", "").strip()

        if not text:
            continue

        # Remove duplicate chunks
        normalized = normalize_text(text)

        if normalized in seen_chunks:
            continue

        seen_chunks.add(normalized)

        chunk_id = chunk.get("chunk_id", "unknown")
        source = chunk.get("source", "unknown")
        score = chunk.get("score")

        evidence_id = f"E{len(evidence) + 1}"

        evidence_header = (
            f"[{evidence_id}] "
            f"Source: {source} | "
            f"Chunk ID: {chunk_id}"
        )

        if score is not None:
            evidence_header += f" | Retrieval Score: {score:.5f}"

        evidence_block = (
            f"{evidence_header}\n"
            f"{text}"
        )

        required_tokens = estimate_tokens(evidence_block)

        # Respect the context token budget
        if total_tokens + required_tokens > max_tokens:
            continue

        evidence.append({
            "evidence_id": evidence_id,
            "source": source,
            "chunk_id": chunk_id,
            "score": score,
            "text": text
        })

        context_parts.append(evidence_block)
        total_tokens += required_tokens

    context = "\n\n".join(context_parts)

    return {
        "context": context,
        "evidence": evidence,
        "evidence_count": len(evidence),
        "estimated_tokens": estimate_tokens(context)
            if context else 0,
        "max_tokens": max_tokens
    }
