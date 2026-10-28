import os
from typing import Any

from dotenv import load_dotenv
from google import genai

from backend.services.vector_store import search_chunks

load_dotenv()

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=GEMINI_API_KEY) if GEMINI_API_KEY else None


def answer_question(question: str, top_k: int = 5) -> dict[str, Any]:
    """Answer a question using retrieved document passages and Gemini."""

    if client is None:
        raise RuntimeError(
            "GEMINI_API_KEY is missing. Add it to the project-root .env file."
        )

    question = question.strip()
    if not question:
        raise ValueError("Question cannot be empty.")

    chunks = search_chunks(query=question, top_k=top_k)

    if not chunks:
        return {
            "answer": (
                "I couldn't find relevant passages in the indexed documents. "
                "Try a more specific question or upload the relevant report."
            ),
            "sources": [],
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

        sources.append(
            {
                "source_id": source_id,
                "source": filename,
                "page": page,
                "chunk_number": chunk.get("chunk_number"),
                "text": text,
            }
        )

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

    response = client.models.generate_content(
        model=GEMINI_MODEL,
        contents=prompt,
        config={
            "temperature": 0.1,
            "max_output_tokens": 1200,
        },
    )

    answer = response.text

    if not answer:
        raise RuntimeError("Gemini returned an empty response.")

    return {
        "answer": answer,
        "sources": sources,
        "model": GEMINI_MODEL,
    }
