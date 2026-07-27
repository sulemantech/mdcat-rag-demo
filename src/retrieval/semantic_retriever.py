import sys
sys.path.insert(0, ".")
from src.embeddings.huggingface_embeddings import get_embedding, get_query_embedding

def retrieve(question: str, collection, metadata_filter: dict = None, top_k: int = 5):
    # Get the embedding of the question
    # question_embedding = get_embedding(question)
    question_embedding = get_query_embedding(question)
    # Query the collection
    results = collection.query(
        query_embeddings=[question_embedding],
        n_results=top_k,
        where=metadata_filter
    )

    return results


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.vectorstore.chroma_store import create_collection

    collection = create_collection("biology", persist_directory="chroma_db")

    questions = [
        "what is the role of mitochondria?",
        "how does DNA replication work?",
        "what is the function of ribosomes?"
    ]

    for q in questions:
        results = retrieve(q, collection, metadata_filter=None, top_k=3)
        print(f"\nQ: {q}")
        for doc in results["documents"][0]:
            print("---")
            print(doc[:200])
