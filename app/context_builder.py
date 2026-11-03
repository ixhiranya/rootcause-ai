
import re


def estimate_tokens(text: str) -> int:
    """Estimate token count using approximately 4 characters per token."""
    return max(1, len(text) // 4)


def normalize_text(text: str) -> str:
    """Normalize text for duplicate detection."""
    return re.sub(r"\s+", " ", text).strip().lower()


def build_context(
    retrieved_chunks: list[dict],
    max_tokens: int = 2500,
    source_text: str = ""
) -> dict:
    """
    Build deduplicated evidence with original log line references.

    start_char and end_char come from the chunking module.
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

        normalized = normalize_text(text)

        if normalized in seen_chunks:
            continue

        seen_chunks.add(normalized)

        chunk_id = chunk.get("chunk_id", "unknown")
        source = chunk.get("source", "unknown")
        score = chunk.get("score")

        # Calculate original line numbers.
        start_char = chunk.get("start_char")
        end_char = chunk.get("end_char")

        if (
            source_text
            and isinstance(start_char, int)
            and isinstance(end_char, int)
        ):
            start_line = source_text.count(
                "\n", 0, start_char
            ) + 1

            end_line = source_text.count(
                "\n", 0, end_char
            )

            if end_line < start_line:
                end_line = start_line

        else:
            start_line = None
            end_line = None

        evidence_id = f"E{len(evidence) + 1}"

        if start_line is not None:
            line_reference = (
                f"Lines {start_line}-{end_line}"
            )
        else:
            line_reference = "Line numbers unavailable"

        header = (
            f"[{evidence_id}] "
            f"Source: {source} | "
            f"Chunk: {chunk_id} | "
            f"{line_reference}"
        )

        if score is not None:
            header += f" | Score: {score:.5f}"

        evidence_block = f"{header}\n{text}"

        required_tokens = estimate_tokens(evidence_block)

        if total_tokens + required_tokens > max_tokens:
            continue

        evidence.append({
            "evidence_id": evidence_id,
            "source": source,
            "chunk_id": chunk_id,
            "start_line": start_line,
            "end_line": end_line,
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
        "estimated_tokens": (
            estimate_tokens(context) if context else 0
        ),
        "max_tokens": max_tokens
    }
