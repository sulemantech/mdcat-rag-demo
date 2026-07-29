from sentence_transformers import CrossEncoder
model = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def rerank(query: str, chunks: list[dict], top_k: int) -> list[dict]:
    # Score on chunk text, but keep the full dict (chapter/subject/grade)
    # attached so it survives into generation.
    pairs = [(query, chunk["text"]) for chunk in chunks]
    scores = model.predict(pairs)
    ranked = sorted(zip(scores, chunks), key=lambda pair: pair[0], reverse=True)
    sorted_chunks = [chunk for _, chunk in ranked]
    return sorted_chunks[:top_k]
    
if __name__ == "__main__":
    import sys
    sys.path.append(".")
    from src.vectorstore.chroma_store import create_collection
    from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve


    # load chunks from cache
    import json
    with open("datasets/sample_text/mdcat_chunks.json", "r") as f:
        chunks = json.load(f)

    # store in ChromaDB
    collection = create_collection("mdcate", persist_directory="chroma_db")

    questions = [
        "what is the role of mitochondria?",
        "how does DNA replication work?",
        "what is the function of ribosomes?"
    ]

    for q in questions:
        candidates = hybrid_retrieve(q, chunks, collection, top_k=10)
        results = rerank(q, candidates, top_k=3)
        print(f"\nQ: {q}")
        for doc in results:
            print("---")
            print(doc["text"][:200])

