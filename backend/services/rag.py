import logging
import os
from typing import Any

from dotenv import load_dotenv
from google import genai

from backend.services.vector_store import search_chunks
from backend.services.semantic_cache import lookup_answer, store_answer

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

logger = logging.getLogger(__name__)

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def answer_question(
    question: str,
    top_k: int = 5,
    document_id: str | None = None,
) -> dict[str, Any]:
    """Answer a document-scoped question, using semantic cache when safe."""

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the project-root .env file."
        )

    question = question.strip()

    if not question:
        raise ValueError("Question cannot be empty.")

    if not document_id or not document_id.strip():
        raise ValueError("A document ID is required.")

    document_id = document_id.strip()

    # Cache failures must never make the normal RAG workflow unavailable.
    try:
        cached = lookup_answer(
            question=question,
            document_id=document_id,
            model=GEMINI_MODEL,
        )
    except Exception:
        logger.exception("Semantic cache lookup failed; falling back to normal RAG.")
        cached = None

    if cached is not None:
        logger.info(
            "Semantic cache HIT for document %s (similarity=%s)",
            document_id,
            cached.get("cache", {}).get("similarity"),
        )
        return cached

    logger.info("Semantic cache MISS for document %s", document_id)

    # Retrieve passages exclusively from the selected PDF.
    chunks = search_chunks(
        query=question,
        top_k=top_k,
        document_id=document_id,
    )

    if not chunks:
        return {
            "answer": (
                "I couldn't find relevant passages in the selected document. "
                "Try a more specific question or upload the relevant report."
            ),
            "sources": [],
            "model": GEMINI_MODEL,
            "cache": {"hit": False, "stored": False, "reason": "no_evidence"},
        }

    context_parts = []
    sources = []

    for index, chunk in enumerate(chunks, start=1):
        source_id = f"S{index}"
        text = str(chunk.get("text", ""))
        filename = chunk.get("source", "Unknown")
        page = chunk.get("page")

        context_parts.append(
            f"[{source_id}] File: {filename}; Page: {page}\n{text}"
        )

        sources.append({
            "source_id": source_id,
            "source": filename,
            "page": page,
            "chunk_number": chunk.get("chunk_number"),
            "text": text,
        })

    prompt = f"""
You are an AI due-diligence assistant analyzing company documents.

Answer the user's question using only the supplied document passages.

Rules:
- Cite supporting evidence using [S1], [S2], etc.
- Do not invent financial figures, facts, or page references.
- If the evidence does not contain the answer, say so explicitly.
- Distinguish facts stated in the document from your interpretation.
- Be concise, structured, and professional.

DOCUMENT PASSAGES:
{"".join(part + chr(10) + chr(10) for part in context_parts)}

USER QUESTION:
{question}
"""

    try:
        response = client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
            config={
                "temperature": 0.1,
                "max_output_tokens": 1200,
            },
        )
    except Exception as exc:
        logger.exception("Gemini request failed during document question answering")
        raise RuntimeError(
            "The AI answer provider could not complete the request."
        ) from exc

    answer = response.text

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    result = {
        "answer": answer,
        "sources": sources,
        "model": GEMINI_MODEL,
    }

    # Cache only successful answers grounded in retrieved document evidence.
    # A cache storage failure must not discard the valid Gemini response.
    try:
        cache_id = store_answer(
            question=question,
            document_id=document_id,
            answer=answer,
            sources=sources,
            model=GEMINI_MODEL,
        )
        result["cache"] = {
            "hit": False,
            "stored": True,
            "cache_id": cache_id,
        }
        logger.info("Stored semantic cache entry %s", cache_id)
    except Exception:
        logger.exception("Could not store semantic cache entry.")
        result["cache"] = {
            "hit": False,
            "stored": False,
            "reason": "cache_store_failed",
        }

    return result
