
from backend.services.vector_store import search_chunks

queries = [
    "total stockholders equity March 31 2026",
    "total shareholders equity balance sheet",
    "total debt current debt long term debt",
    "long term debt debt obligations borrowings",
    "total liabilities and stockholders equity total assets",
    "financial statements amounts in millions currency USD",
]

for query in queries:
    print("\n" + "=" * 90)
    print(f"QUERY: {query}")
    print("=" * 90)

    try:
        results = search_chunks(query=query, top_k=5)
    except Exception as exc:
        print(f"Search failed: {type(exc).__name__}: {exc}")
        continue

    if not results:
        print("No matching passages.")
        continue

    for result in results:
        print(
            f"\nSource: {result.get('source')} | "
            f"Page: {result.get('page')} | "
            f"Distance: {result.get('distance')}"
        )
        print(result.get("text", ""))


if __name__ == "__main__":
    print("Inspecting indexed financial statement passages...")
