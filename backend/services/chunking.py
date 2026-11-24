from typing import Any


def _split_long_line(line: str, max_length: int) -> list[str]:
    """Split an unusually long line at whitespace where possible."""
    words = line.split()
    if not words:
        return []

    parts: list[str] = []
    current = ""

    for word in words:
        if len(word) > max_length:
            if current:
                parts.append(current)
                current = ""
            parts.extend(
                word[index:index + max_length]
                for index in range(0, len(word), max_length)
            )
            continue

        candidate = f"{current} {word}".strip()
        if len(candidate) <= max_length:
            current = candidate
        else:
            if current:
                parts.append(current)
            current = word

    if current:
        parts.append(current)

    return parts


def _page_chunks(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Pack complete lines together, splitting only lines larger than a chunk."""
    lines: list[str] = []
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if len(line) > chunk_size:
            lines.extend(_split_long_line(line, chunk_size))
        else:
            lines.append(line)

    chunks: list[str] = []
    current = ""

    for line in lines:
        candidate = f"{current}\n{line}".strip()
        if len(candidate) <= chunk_size:
            current = candidate
            continue

        if current:
            chunks.append(current)

        prefix = current[-overlap:] if overlap and current else ""
        current = f"{prefix}\n{line}".strip() if prefix else line

        if len(current) > chunk_size:
            current = line

    if current:
        chunks.append(current)

    return chunks


def chunk_pages(
    pages: list[dict[str, Any]],
    filename: str,
    document_id: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Split PDF pages into line-aware chunks with stable document identity.

    Page boundaries are preserved so every indexed passage retains the
    correct PDF page reference. Stores both overlapped text (for vector search)
    and clean non-overlapping text / full_page_text (for exact page reconstruction).
    """
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks: list[dict[str, Any]] = []

    for page in pages:
        if "page" not in page or "text" not in page:
            raise ValueError("Each page must contain 'page' and 'text' fields.")

        page_number = page["page"]
        text = page["text"]
        if not isinstance(text, str):
            raise ValueError("Page text must be a string.")

        if not text.strip():
            continue

        clean_chunks = _page_chunks(text, chunk_size, overlap=0)
        overlapped_chunks = _page_chunks(text, chunk_size, overlap=overlap)

        for chunk_number, chunk_text in enumerate(overlapped_chunks, start=1):
            clean_text = (
                clean_chunks[chunk_number - 1]
                if chunk_number <= len(clean_chunks)
                else chunk_text
            )
            chunks.append({
                "chunk_id": (
                    f"{document_id}::page_{page_number}::chunk_{chunk_number}"
                ),
                "text": chunk_text,
                "metadata": {
                    "source": filename,
                    "document_id": document_id,
                    "page": page_number,
                    "chunk_number": chunk_number,
                    "clean_text": clean_text,
                    "full_page_text": text.strip(),
                },
            })

    return chunks