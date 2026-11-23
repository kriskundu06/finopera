"""
Document-scoped semantic cache for FinOpera RAG answers.

Storage:
- SQLite: authoritative cached answer, source evidence and metadata.
- ChromaDB: question embeddings in a dedicated collection.

The cache reuses the project's existing SentenceTransformer model and ChromaDB
persistence directory. It does not modify the document-chunk collection.
"""

from __future__ import annotations

import hashlib
import json
import re
import sqlite3
import threading
import time
from pathlib import Path
from typing import Any

from backend.services.vector_store import CHROMA_PATH, get_embedding_model

CACHE_DB_PATH = CHROMA_PATH.parent / "cache" / "semantic_cache.sqlite3"
CACHE_COLLECTION_NAME = "due_diligence_semantic_cache"

# Increment these when answer-generation behavior or cache record structure changes.
CACHE_VERSION = "1"
PROMPT_VERSION = "1"

# Seven-day default TTL. Override with FINOPERA_SEMANTIC_CACHE_TTL_SECONDS.
DEFAULT_TTL_SECONDS = 7 * 24 * 60 * 60

# Chroma cosine distance = 1 - cosine similarity for this collection.
DEFAULT_SIMILARITY_THRESHOLD = 0.93

_db_lock = threading.RLock()
_chroma_client = None
_cache_collection = None


def _get_ttl_seconds() -> int:
    import os

    raw = os.getenv("FINOPERA_SEMANTIC_CACHE_TTL_SECONDS", str(DEFAULT_TTL_SECONDS))
    try:
        return max(0, int(raw))
    except (TypeError, ValueError):
        return DEFAULT_TTL_SECONDS


def _get_threshold() -> float:
    import os

    raw = os.getenv("FINOPERA_SEMANTIC_CACHE_THRESHOLD", str(DEFAULT_SIMILARITY_THRESHOLD))
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return DEFAULT_SIMILARITY_THRESHOLD
    return min(1.0, max(0.0, value))


def _connect() -> sqlite3.Connection:
    CACHE_DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(
        str(CACHE_DB_PATH),
        timeout=10,
        check_same_thread=False,
    )
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute("PRAGMA busy_timeout=10000")
    return connection


def initialize_cache() -> None:
    """Create the SQLite cache table and indexes if they do not exist."""
    with _db_lock, _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS semantic_cache (
                cache_id TEXT PRIMARY KEY,
                document_id TEXT NOT NULL,
                normalized_question TEXT NOT NULL,
                original_question TEXT NOT NULL,
                answer TEXT NOT NULL,
                sources_json TEXT NOT NULL,
                model TEXT,
                cache_version TEXT NOT NULL,
                prompt_version TEXT NOT NULL,
                created_at REAL NOT NULL,
                expires_at REAL
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_semantic_cache_scope
            ON semantic_cache (
                document_id, cache_version, prompt_version, created_at
            )
            """
        )
        connection.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_semantic_cache_expiry
            ON semantic_cache (expires_at)
            """
        )


def _get_chroma_collection():
    """Open a dedicated persistent ChromaDB collection for question embeddings."""
    global _chroma_client, _cache_collection

    if _cache_collection is None:
        import chromadb

        CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _chroma_client = chromadb.PersistentClient(path=str(CHROMA_PATH))
        _cache_collection = _chroma_client.get_or_create_collection(
            name=CACHE_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    return _cache_collection


def _normalize_question(question: str) -> str:
    return re.sub(r"\s+", " ", question).strip().casefold()


def _numeric_signature(question: str) -> tuple[str, ...]:
    """
    Preserve numeric details (years, quarters, percentages, amounts, etc.).
    This deliberately rejects semantic reuse when numeric details differ.
    """
    pattern = r"(?<![A-Za-z])(?:₹|\$|€|£)?\s*\d[\d,]*(?:\.\d+)?\s*%?"
    values = re.findall(pattern, question)
    return tuple(re.sub(r"\s+", "", value).casefold() for value in values)


def _intent_signature(question: str) -> frozenset[str]:
    """Conservative intent guard for common question operations."""
    text = question.casefold()
    groups = {
        "calculate": ("calculate", "compute", "work out", "what is the"),
        "compare": ("compare", "versus", " vs ", "difference between"),
        "trend": ("trend", "over time", "year-on-year", "year over year", "qoq", "yoy"),
        "explain": ("explain", "why", "how did", "what caused"),
        "summarize": ("summarize", "summary", "overview", "key takeaways"),
        "list": ("list", "name all", "which items", "identify"),
    }
    return frozenset(
        label
        for label, phrases in groups.items()
        if any(phrase in text for phrase in phrases)
    )


def _period_signature(question: str) -> frozenset[str]:
    """Capture explicit period markers to avoid cross-period answer reuse."""
    text = question.casefold()
    patterns = (
        r"\bq[1-4]\b",
        r"\b(?:first|second|third|fourth) quarter\b",
        r"\bquarter(?:ly)?\b",
        r"\b(?:fy|fiscal year)\b",
        r"\b(?:annual|yearly|year-to-date|ytd)\b",
        r"\b(?:monthly|month-to-date|mtd)\b",
        r"\b(?:last|previous|prior|current|this|next) quarter\b",
        r"\b(?:last|previous|prior|current|this|next) year\b",
    )
    return frozenset(
        match.group(0)
        for pattern in patterns
        for match in re.finditer(pattern, text)
    )


def _safe_to_reuse(cached_question: str, new_question: str) -> bool:
    """
    Reject likely false positives before accepting an embedding match.

    Numeric values and explicit operation intent must agree. This is a
    conservative guard, not a substitute for evaluating cache quality.
    """
    if _numeric_signature(cached_question) != _numeric_signature(new_question):
        return False

    if _period_signature(cached_question) != _period_signature(new_question):
        return False

    if _intent_signature(cached_question) != _intent_signature(new_question):
        return False

    return True


def _validate_document_id(document_id: str) -> str:
    if not isinstance(document_id, str) or not re.fullmatch(r"[a-fA-F0-9]{64}", document_id):
        raise ValueError("document_id must be a 64-character SHA-256 hexadecimal hash.")
    return document_id.lower()


def _cache_id(
    document_id: str,
    normalized_question: str,
    model_version: str,
    prompt_version: str,
) -> str:
    raw = "|".join(
        (document_id, normalized_question, CACHE_VERSION, prompt_version, model_version)
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _embed(text: str) -> list[float]:
    model = get_embedding_model()
    vector = model.encode(
        text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()


def store_answer(
    *,
    question: str,
    document_id: str,
    answer: str,
    sources: list[dict[str, Any]],
    model: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    ttl_seconds: int | None = None,
) -> str:
    """
    Store a successful RAG answer and its evidence.

    Returns the cache entry ID. Empty answers and invalid document IDs are
    rejected; callers should not cache failed or no-evidence responses.
    """
    document_id = _validate_document_id(document_id)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question cannot be empty.")
    if not isinstance(answer, str) or not answer.strip():
        raise ValueError("answer cannot be empty.")
    if not isinstance(sources, list) or not sources:
        raise ValueError("sources must contain evidence before an answer is cached.")

    normalized = _normalize_question(question)
    entry_id = _cache_id(document_id, normalized, model or "", prompt_version)
    created_at = time.time()
    effective_ttl = _get_ttl_seconds() if ttl_seconds is None else max(0, ttl_seconds)
    expires_at = created_at + effective_ttl if effective_ttl else None
    sources_json = json.dumps(sources, ensure_ascii=False)

    # Embed before writing metadata. If embedding fails, no incomplete cache row is saved.
    embedding = _embed(normalized)
    collection = _get_chroma_collection()

    with _db_lock, _connect() as connection:
        connection.execute(
            """
            INSERT INTO semantic_cache (
                cache_id, document_id, normalized_question, original_question,
                answer, sources_json, model, cache_version, prompt_version,
                created_at, expires_at
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            ON CONFLICT(cache_id) DO UPDATE SET
                original_question = excluded.original_question,
                answer = excluded.answer,
                sources_json = excluded.sources_json,
                model = excluded.model,
                created_at = excluded.created_at,
                expires_at = excluded.expires_at
            """,
            (
                entry_id,
                document_id,
                normalized,
                question.strip(),
                answer,
                sources_json,
                model,
                CACHE_VERSION,
                prompt_version,
                created_at,
                expires_at,
            ),
        )

    try:
        collection.upsert(
            ids=[entry_id],
            embeddings=[embedding],
            documents=[normalized],
            metadatas=[
                {
                    "document_id": document_id,
                    "cache_version": CACHE_VERSION,
                    "prompt_version": prompt_version,
                    "model": model or "",
                }
            ],
        )
    except Exception:
        # Leave SQLite metadata in place; without a vector entry it cannot be
        # returned as a semantic hit. The next store for this ID can repair it.
        raise

    return entry_id


def lookup_answer(
    *,
    question: str,
    document_id: str,
    model: str | None = None,
    prompt_version: str = PROMPT_VERSION,
    similarity_threshold: float | None = None,
) -> dict[str, Any] | None:
    """
    Find a safe semantic match for a question within one document and version.

    Returns answer, sources, similarity and cache metadata, or None on a miss.
    """
    document_id = _validate_document_id(document_id)
    if not isinstance(question, str) or not question.strip():
        raise ValueError("question cannot be empty.")

    normalized = _normalize_question(question)
    threshold = _get_threshold() if similarity_threshold is None else similarity_threshold
    if not 0.0 <= threshold <= 1.0:
        raise ValueError("similarity_threshold must be between 0 and 1.")

    collection = _get_chroma_collection()
    if collection.count() == 0:
        return None

    query_embedding = _embed(normalized)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(5, collection.count()),
        where={
            "$and": [
                {"document_id": document_id},
                {"cache_version": CACHE_VERSION},
                {"prompt_version": prompt_version},
                {"model": model or ""},
            ]
        },
        include=["documents", "metadatas", "distances"],
    )

    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]

    for entry_id, cached_question, distance in zip(ids, documents, distances):
        similarity = 1.0 - float(distance)
        if similarity < threshold:
            continue
        if not _safe_to_reuse(str(cached_question), normalized):
            continue

        with _db_lock, _connect() as connection:
            row = connection.execute(
                """
                SELECT answer, sources_json, model, created_at, expires_at,
                       cache_version, prompt_version
                FROM semantic_cache
                WHERE cache_id = ? AND document_id = ?
                """,
                (entry_id, document_id),
            ).fetchone()

        if row is None:
            continue

        now = time.time()
        expires_at = row["expires_at"]
        if expires_at is not None and float(expires_at) <= now:
            delete_entry(entry_id)
            continue

        try:
            sources = json.loads(row["sources_json"])
        except (TypeError, json.JSONDecodeError):
            continue

        return {
            "answer": row["answer"],
            "sources": sources,
            "model": row["model"],
            "cache": {
                "hit": True,
                "similarity": round(similarity, 4),
                "cache_id": entry_id,
                "created_at": row["created_at"],
            },
        }

    return None


def delete_entry(cache_id: str) -> None:
    """Delete one entry from both SQLite and the semantic-cache collection."""
    with _db_lock, _connect() as connection:
        connection.execute("DELETE FROM semantic_cache WHERE cache_id = ?", (cache_id,))

    try:
        collection = _get_chroma_collection()
        collection.delete(ids=[cache_id])
    except Exception:
        # SQLite is authoritative; an orphaned vector has no answer to return.
        pass


def clear_document_cache(document_id: str) -> int:
    """Delete all semantic-cache entries belonging to one document."""
    document_id = _validate_document_id(document_id)

    with _db_lock, _connect() as connection:
        rows = connection.execute(
            "SELECT cache_id FROM semantic_cache WHERE document_id = ?",
            (document_id,),
        ).fetchall()
        ids = [row["cache_id"] for row in rows]
        connection.execute(
            "DELETE FROM semantic_cache WHERE document_id = ?",
            (document_id,),
        )

    if ids:
        try:
            _get_chroma_collection().delete(ids=ids)
        except Exception:
            pass

    return len(ids)


def get_cache_stats(document_id: str | None = None) -> dict[str, int]:
    """Return basic cache-entry counts; hit/miss telemetry is added at integration."""
    if document_id is not None:
        document_id = _validate_document_id(document_id)

    with _db_lock, _connect() as connection:
        if document_id:
            total = connection.execute(
                "SELECT COUNT(*) FROM semantic_cache WHERE document_id = ?",
                (document_id,),
            ).fetchone()[0]
            active = connection.execute(
                """
                SELECT COUNT(*) FROM semantic_cache
                WHERE document_id = ? AND (expires_at IS NULL OR expires_at > ?)
                """,
                (document_id, time.time()),
            ).fetchone()[0]
        else:
            total = connection.execute(
                "SELECT COUNT(*) FROM semantic_cache"
            ).fetchone()[0]
            active = connection.execute(
                """
                SELECT COUNT(*) FROM semantic_cache
                WHERE expires_at IS NULL OR expires_at > ?
                """,
                (time.time(),),
            ).fetchone()[0]

    return {"total_entries": int(total), "active_entries": int(active)}


# Initialize the lightweight SQLite schema on import. The embedding model and
# ChromaDB collection remain lazy-loaded until a cache operation is requested.
initialize_cache()
