
from pathlib import Path
from typing import Any

import chromadb
from sentence_transformers import SentenceTransformer


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CHROMA_PATH = PROJECT_ROOT / "data" / "chroma"

EMBEDDING_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
COLLECTION_NAME = "due_diligence_chunks"

_embedding_model = None
_chroma_collection = None


def get_embedding_model() -> SentenceTransformer:
    """Load the embedding model once and reuse it."""
    global _embedding_model

    if _embedding_model is None:
        _embedding_model = SentenceTransformer(
            EMBEDDING_MODEL_NAME
        )

    return _embedding_model


def get_collection():
    """Open or create the persistent ChromaDB collection."""
    global _chroma_collection

    if _chroma_collection is None:
        CHROMA_PATH.mkdir(parents=True, exist_ok=True)

        client = chromadb.PersistentClient(
            path=str(CHROMA_PATH)
        )

        _chroma_collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},
        )

    return _chroma_collection


def index_chunks(
    chunks: list[dict[str, Any]],
) -> int:
    """Embed chunks and save them in ChromaDB."""
    valid_chunks = [
        chunk
        for chunk in chunks
        if isinstance(chunk.get("text"), str)
        and chunk["text"].strip()
    ]

    if not valid_chunks:
        return 0

    model = get_embedding_model()
    collection = get_collection()

    texts = [chunk["text"] for chunk in valid_chunks]

    embeddings = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    ids = [chunk["chunk_id"] for chunk in valid_chunks]

    metadatas = [
        {
            "source": str(chunk["metadata"]["source"]),
            "page": int(chunk["metadata"]["page"]),
            "chunk_number": int(chunk["metadata"]["chunk_number"]),
        }
        for chunk in valid_chunks
    ]

    collection.upsert(
        ids=ids,
        documents=texts,
        embeddings=embeddings,
        metadatas=metadatas,
    )

    return len(valid_chunks)


def search_chunks(
    query: str,
    top_k: int = 5,
    source: str | None = None,
    page: int | None = None,
) -> list[dict[str, Any]]:
    """
    Retrieve relevant chunks by semantic similarity.

    Optional filters:
    - source: exact indexed filename
    - page: PDF page number stored in the chunk metadata

    Existing calls such as search_chunks(query, top_k=5)
    continue to work unchanged.
    """
    query = query.strip()

    if not query or top_k < 1:
        return []

    collection = get_collection()

    if collection.count() == 0:
        return []

    conditions = []

    if source is not None:
        conditions.append({"source": source})

    if page is not None:
        conditions.append({"page": page})

    if len(conditions) == 0:
        where = None
    elif len(conditions) == 1:
        where = conditions[0]
    else:
        where = {"$and": conditions}

    # Count chunks matching the metadata filter before querying.
    if where is None:
        matching_count = collection.count()
    else:
        matching_count = len(
            collection.get(
                where=where,
                include=[],
            )["ids"]
        )

    if matching_count == 0:
        return []

    model = get_embedding_model()

    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    query_args = {
        "query_embeddings": [query_embedding],
        "n_results": min(top_k, matching_count),
        "include": ["documents", "metadatas", "distances"],
    }

    if where is not None:
        query_args["where"] = where

    results = collection.query(**query_args)

    documents = results.get("documents") or [[]]
    metadatas = results.get("metadatas") or [[]]
    distances = results.get("distances") or [[]]

    if not documents or not documents[0]:
        return []

    matches = []

    for index, text in enumerate(documents[0]):
        metadata = metadatas[0][index] or {}

        matches.append(
            {
                "text": text,
                "source": metadata.get("source", "unknown"),
                "page": metadata.get("page"),
                "chunk_number": metadata.get("chunk_number"),
                "distance": distances[0][index],
            }
        )

    return matches
