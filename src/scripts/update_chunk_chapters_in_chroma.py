import json
import sys
sys.path.insert(0, ".")

from src.vectorstore.chroma_store import create_collection

CHUNKS_PATH = "datasets/mdcat_chunks.json"
COLLECTION_NAME = "mdcat_v2"
BATCH_SIZE = 5000

if __name__ == "__main__":
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    metadatas = [
        {"subject": chunk["subject"], "grade": chunk["grade"], "chapter": chunk["chapter"]}
        for chunk in chunks
    ]

    collection = create_collection(COLLECTION_NAME, persist_directory="chroma_db")
    print(f"Collection count before update: {collection.count()}")

    # No embeddings= / documents= passed -- vectors and text are untouched,
    # this only rewrites metadata for existing ids.
    for i in range(0, len(chunk_ids), BATCH_SIZE):
        collection.update(
            ids=chunk_ids[i:i + BATCH_SIZE],
            metadatas=metadatas[i:i + BATCH_SIZE],
        )
        print(f"Updated batch {i}:{i + BATCH_SIZE}")

    print(f"Collection count after update: {collection.count()}")

    sample = collection.get(ids=chunk_ids[:3], include=["metadatas"])
    print("Sample metadatas after update:", sample["metadatas"])
