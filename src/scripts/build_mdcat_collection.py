import json
import sys
sys.path.insert(0, ".")

from src.embeddings.huggingface_embeddings import get_embeddings_batch
from src.vectorstore.chroma_store import create_collection, add_documents

CHUNKS_PATH = "datasets/mdcat_chunks.json"
COLLECTION_NAME = "mdcat_v2"  # rebuild with stable hash-based chunk IDs + all 6 books (incl. Physics)

if __name__ == "__main__":
    # TODO: load chunks from CHUNKS_PATH
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    # TODO: build three parallel lists from the chunk dicts:
    #       texts     = [chunk["text"] for chunk in chunks]
    #       metadatas = [{"subject": ..., "grade": ...} for chunk in chunks]
    #       (embeddings come from get_embeddings_batch(texts) — 6,739 texts
    #       in one call; all-MiniLM-L6-v2 is small/fast, but watch memory —
    #       fine to try as one batch first, see if it's actually slow)
    texts = [chunk["text"] for chunk in chunks]
    metadatas = [{"subject": chunk["subject"], "grade": chunk["grade"]} for chunk in chunks]
    embeddings = get_embeddings_batch(texts)

    # TODO: create_collection(COLLECTION_NAME, persist_directory="chroma_db")
    collection = create_collection(COLLECTION_NAME, persist_directory="chroma_db")
    # TODO: add_documents(collection, texts, metadatas, embeddings)
    chunk_ids = [chunk["chunk_id"] for chunk in chunks]
    add_documents(collection, texts, metadatas, embeddings, chunk_ids)

    # TODO: print collection.count() to confirm everything landed
    print(f"Collection count: {collection.count()}")