import sys
sys.path.insert(0, ".")

from src.retrieval.semantic_retriever import retrieve as semantic_retrieve
from src.retrieval.bm25_retriever import retrieve as bm25_retrieve


def retrieve(
    question: str,
    chunks: list[dict],
    collection,
    metadata_filter: dict = None,
    top_k: int = 5
):
    """
    Hybrid retrieval using Reciprocal Rank Fusion (RRF).

    Returns:
        List of chunk texts (strings).
    """

    # ------------------------------------------------------------
    # Semantic Retrieval
    # ------------------------------------------------------------
    semantic_results = semantic_retrieve(
        question,
        collection,
        metadata_filter=metadata_filter,
        top_k=top_k * 2
    )

    semantic_ids = semantic_results["ids"][0]
    semantic_chunks = semantic_results["documents"][0]

    # Rank by ID
    semantic_ranks = {
        doc_id: rank + 1
        for rank, doc_id in enumerate(semantic_ids)
    }

    # ------------------------------------------------------------
    # BM25 Retrieval
    # ------------------------------------------------------------
    bm25_results = bm25_retrieve(
        question,
        chunks,
        metadata_filter,
        top_k=top_k * 2
    )

    # Rank by ID
    bm25_ranks = {
        doc_id: rank + 1
        for rank, (doc_id, _) in enumerate(bm25_results)
    }

    # ------------------------------------------------------------
    # Build lookup table (id -> chunk)
    # ------------------------------------------------------------

    lookup = {}

    # Semantic results only contain text
    for doc_id, text in zip(semantic_ids, semantic_chunks):
        lookup[doc_id] = {"text": text}

    # BM25 contains full chunk dictionaries
    for doc_id, chunk in bm25_results:
        lookup[doc_id] = chunk

    # ------------------------------------------------------------
    # Reciprocal Rank Fusion
    # ------------------------------------------------------------

    all_ids = set(semantic_ranks.keys()) | set(bm25_ranks.keys())

    k = 60

    scores = {}

    for doc_id in all_ids:

        score = 0.0

        if doc_id in semantic_ranks:
            score += 1 / (k + semantic_ranks[doc_id])

        if doc_id in bm25_ranks:
            score += 1 / (k + bm25_ranks[doc_id])

        scores[doc_id] = score

    # ------------------------------------------------------------
    # Sort by fused score
    # ------------------------------------------------------------

    ranked_ids = sorted(
        scores,
        key=lambda x: scores[x],
        reverse=True
    )

    # ------------------------------------------------------------
    # Return top-k chunk texts
    # ------------------------------------------------------------

    return [
        lookup[doc_id]["text"]
        for doc_id in ranked_ids[:top_k]
    ]


if __name__ == "__main__":

    import json

    from src.vectorstore.chroma_store import create_collection

    with open(
        "datasets/sample_text/mdcat_chunks.json",
        "r",
        encoding="utf-8"
    ) as f:
        chunks = json.load(f)

    
    collection = create_collection(
        "biology",
        persist_directory="chroma_db"
    )

    questions = [
        "What is the percentage of H₂O in bone cells?",
        "What is the role of mitochondria?",
        "How does DNA replication work?",
        "What is the function of ribosomes?"
    ]

    metadata = {
        "subject": "biology"
    }

    for q in questions:

        results = retrieve(
            q,
            chunks,
            collection,
            metadata_filter=metadata,
            top_k=3
        )

        print("=" * 80)
        print(f"Question: {q}")

        for i, chunk in enumerate(results, 1):

            print(f"\nResult {i}")

            if isinstance(chunk, dict):
                print(chunk["text"][:300])
            else:
                print(chunk[:300])

            print("-" * 80)