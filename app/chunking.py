
import re
from pathlib import Path
from typing import Any


# Recognizes common timestamped log entries.
LOG_START_PATTERN = re.compile(
    r"^\s*(?:"
    r"\d{4}-\d{2}-\d{2}[ T]\d{2}:\d{2}:\d{2}"
    r"|"
    r"\d{2}:\d{2}:\d{2}"
    r")"
)


def load_text_file(file_path: str | Path) -> str:
    """Read a UTF-8 text file, replacing invalid byte sequences."""
    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")

    if not path.is_file():
        raise ValueError(f"Expected a file: {path}")

    return path.read_text(encoding="utf-8", errors="replace")


def split_into_log_events(text: str) -> list[dict[str, Any]]:
    """
    Group each timestamped log entry with its continuation lines.
    Stack traces and indented exception details remain attached
    to the log event that precedes them.
    """
    events = []
    current_lines = []
    current_start = None
    offset = 0

    for line in text.splitlines(keepends=True):
        line_start = offset
        offset += len(line)

        if LOG_START_PATTERN.match(line):
            if current_lines:
                events.append({
                    "text": "".join(current_lines),
                    "start_char": current_start,
                    "end_char": line_start,
                })

            current_lines = [line]
            current_start = line_start

        elif current_lines:
            # Continuation: stack frame, exception detail,
            # or other text belonging to the current log event.
            current_lines.append(line)

        elif line.strip():
            # Preserve preamble text instead of silently dropping it.
            events.append({
                "text": line,
                "start_char": line_start,
                "end_char": offset,
            })

    if current_lines:
        events.append({
            "text": "".join(current_lines),
            "start_char": current_start,
            "end_char": len(text),
        })

    return events


def chunk_log_text(
    text: str,
    source: str = "unknown",
    target_size: int = 800,
    max_size: int = 1500,
) -> list[dict[str, Any]]:
    """
    Group complete log events into bounded chunks.

    The target size is a grouping preference. The maximum size
    is a hard limit, except that a single indivisible line longer
    than max_size must be split as a fallback.
    """
    if target_size <= 0 or max_size <= 0:
        raise ValueError("Chunk sizes must be positive")

    if target_size > max_size:
        raise ValueError("target_size cannot exceed max_size")

    if not text or not text.strip():
        return []

    events = split_into_log_events(text)
    chunks = []
    group = []
    group_start = None
    group_end = None
    group_length = 0

    def save_group():
        nonlocal group, group_start, group_end, group_length

        if not group:
            return

        chunks.append({
            "chunk_id": f"{source}::chunk_{len(chunks)}",
            "source": source,
            "chunk_index": len(chunks),
            "start_char": group_start,
            "end_char": group_end,
            "text": "".join(event["text"] for event in group),
        })

        group = []
        group_start = None
        group_end = None
        group_length = 0

    for event in events:
        event_text = event["text"]
        event_length = len(event_text)

        # If a single event exceeds the maximum, split it at
        # line boundaries where possible. A very long individual
        # line is split as a final fallback.
        if event_length > max_size:
            save_group()

            lines = event_text.splitlines(keepends=True)
            piece = ""
            piece_start = event["start_char"]

            for line in lines:
                # Preserve each line unless it alone exceeds max_size.
                line_pieces = [
                    line[i:i + max_size]
                    for i in range(0, len(line), max_size)
                ]

                for line_piece in line_pieces:
                    if piece and len(piece) + len(line_piece) > max_size:
                        chunks.append({
                            "chunk_id": f"{source}::chunk_{len(chunks)}",
                            "source": source,
                            "chunk_index": len(chunks),
                            "start_char": piece_start,
                            "end_char": piece_start + len(piece),
                            "text": piece,
                        })
                        piece_start += len(piece)
                        piece = ""

                    piece += line_piece

            if piece:
                chunks.append({
                    "chunk_id": f"{source}::chunk_{len(chunks)}",
                    "source": source,
                    "chunk_index": len(chunks),
                    "start_char": piece_start,
                    "end_char": piece_start + len(piece),
                    "text": piece,
                })

            continue

        # Start a new chunk if adding this event would exceed
        # the hard maximum.
        if group and group_length + event_length > max_size:
            save_group()

        if not group:
            group_start = event["start_char"]

        group.append(event)
        group_end = event["end_char"]
        group_length += event_length

        # Target size is a preference, not a reason to split an event.
        if group_length >= target_size:
            save_group()

    save_group()
    return chunks
