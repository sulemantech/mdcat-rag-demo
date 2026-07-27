import re
from typing import List
from rank_bm25 import BM25Okapi


STOP_WORDS = {
    "what",
    "is",
    "the",
    "a",
    "an",
    "and",
    "or",
    "but",
    "in",
    "on",
    "at",
    "to",
    "for",
    "of",
    "with",
    "by",
    "how",
    "does",
    "do",
    "it",
}

def tokenize(text: str) -> List[str]:
    """
    Simple tokenizer.
    Replace with a better tokenizer if needed.
    """
    # TODO: lowercase text, extract word tokens with re.findall(r'\b\w+\b', text)
    #       to strip punctuation, then filter out anything in STOP_WORDS
    tokens = re.findall(r'\b\w+\b', text.lower())
    return [token for token in tokens if token not in STOP_WORDS]

def retrieve(question: str, chunks: List[dict], metadata: dict = None, top_k: int = 5) -> list:
    """
    Retrieve the most relevant chunks using BM25.

    Returns:
        List of tuples:
            (chunk_id, chunk_dict)

        where chunk_id is the original index of the chunk
        converted to a string so it matches ChromaDB ids.
    """

    if metadata is None:
        metadata = {}

    subject = metadata.get("subject", "")

    # ------------------------------------------------------------------
    # Step 1: Preserve original indices BEFORE filtering
    # ------------------------------------------------------------------
    indexed_chunks = list(enumerate(chunks))
    # Example:
    # [
    #   (0, chunk0),
    #   (1, chunk1),
    #   (2, chunk2),
    #   ...
    # ]

    # ------------------------------------------------------------------
    # Step 2: Filter while keeping original indices
    # ------------------------------------------------------------------
    if subject:
        filtered_chunks = [
            (idx, chunk)
            for idx, chunk in indexed_chunks
            if chunk.get("subject") == subject
        ]
    else:
        filtered_chunks = indexed_chunks

    if not filtered_chunks:
        return []

    # ------------------------------------------------------------------
    # Step 3: Tokenize only the chunk text
    # ------------------------------------------------------------------
    tokenized_chunks = [
        tokenize(chunk["text"])
        for _, chunk in filtered_chunks
    ]

    # ------------------------------------------------------------------
    # Step 4: Build BM25 index
    # ------------------------------------------------------------------
    bm25 = BM25Okapi(tokenized_chunks)

    # ------------------------------------------------------------------
    # Step 5: Score the query
    # ------------------------------------------------------------------
    query_tokens = tokenize(question)
    scores = bm25.get_scores(query_tokens)

    # ------------------------------------------------------------------
    # Step 6: Get top-k positions in the FILTERED list
    # ------------------------------------------------------------------
    top_k_indices = sorted(
        range(len(scores)),
        key=lambda i: scores[i],
        reverse=True
    )[:top_k]

    # ------------------------------------------------------------------
    # Step 7: Map filtered positions back to original indices
    # ------------------------------------------------------------------
    top_k_results = [
        filtered_chunks[i]
        for i in top_k_indices
    ]

    # ------------------------------------------------------------------
    # Step 8: Return (chunk_id, chunk)
    # chunk_id must match ChromaDB ids
    # ------------------------------------------------------------------
    return [
        (str(chunk["chunk_id"]), chunk)
        for _, chunk in top_k_results
    ]

if __name__ == "__main__":

    import json
    import sys

    sys.path.append(".")

    # Load cached chunks
    with open("datasets/sample_text/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)

    questions = [
        "What is the role of mitochondria?",
        "How does DNA replication work?",
        "What is the function of ribosomes?"
    ]

    # Metadata filter (empty = search all)
    metadata = {
        "subject": "biology"
    }

    for q in questions:

        results = retrieve(
            question=q,
            chunks=chunks,
            metadata=metadata,
            top_k=3
        )

        print("=" * 80)
        print(f"Question: {q}")

        if not results:
            print("No matching chunks found.")
            continue

        for i, (chunk_id, chunk) in enumerate(results, 1):
            print(f"\nResult {i}")
            print(f"Chunk ID: {chunk_id}")

            if "subject" in chunk:
                print(f"Subject : {chunk['subject']}")

            if "grade" in chunk:
                print(f"Grade   : {chunk['grade']}")

            if "source" in chunk:
                print(f"Source  : {chunk['source']}")

            print("Text:")
            print(chunk["text"][:300])
            print("-" * 80)