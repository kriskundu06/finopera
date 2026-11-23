
from typing import Any


def chunk_pages(
    pages: list[dict[str, Any]],
    filename: str,
    document_id: str,
    chunk_size: int = 1200,
    overlap: int = 200,
) -> list[dict[str, Any]]:
    """Split PDF pages into chunks with stable document identity."""

    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive.")

    if overlap < 0 or overlap >= chunk_size:
        raise ValueError(
            "overlap must be non-negative and smaller than chunk_size."
        )

    chunks = []
    step = chunk_size - overlap

    for page in pages:
        page_number = page["page"]
        text = page["text"].strip()

        if not text:
            continue

        chunk_number = 0

        for start in range(0, len(text), step):
            chunk_text = text[start:start + chunk_size].strip()

            if not chunk_text:
                continue

            chunk_number += 1

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
                },
            })

            if start + chunk_size >= len(text):
                break

    return chunks
