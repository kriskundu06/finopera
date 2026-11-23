from backend.services.vector_store import get_collection

collection = get_collection()

print("Total chunks:", collection.count())

data = collection.get(
    include=["metadatas", "documents"]
)

print("\nIndexed document IDs:")

document_ids = set()

for metadata in data["metadatas"]:
    document_ids.add(metadata.get("document_id"))

for document_id in document_ids:
    print(document_id)

print("\nFirst 5 indexed records:")

for i in range(min(5, len(data["ids"]))):
    print({
        "id": data["ids"][i],
        "metadata": data["metadatas"][i],
        "text": data["documents"][i][:200]
    })