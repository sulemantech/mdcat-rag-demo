from sentence_transformers import SentenceTransformer
model = SentenceTransformer("all-MiniLM-L6-v2")

def get_embedding(text: str) -> list:
    # encode the text
    embedding = model.encode(text)
    # return the result
    return embedding.tolist()

def get_query_embedding(text: str) -> list:
    # TODO: prepend "Represent this sentence for searching relevant
    #       passages: " to `text` before encoding — this is BGE's documented
    #       convention for query-side embeddings specifically
    text = f"Represent this sentence for searching relevant passages: {text}"
    return get_embedding(text)

def get_embeddings_batch(texts: list) -> list:
    embeddings = model.encode(texts)
    # return the result
    return embeddings.tolist()

# add main to use the above function to get the embedding of a text file and save it to a json file
if __name__ == "__main__":
    import os
    import json
    import sys
    sys.path.append(".")
    from src.chunking.text_splitters import recursive_split

    with open("datasets/sample_text/biology_clean.txt", "r", encoding="utf-8") as f:
        text = f.read()

    if os.path.exists("datasets/sample_text/mdcat_chunks.json"):
        with open("datasets/sample_text/mdcat_chunks.json", "r") as f:
            chunks = json.load(f)
        embeddings = chunks["embeddings"]
        print(f"Loaded {len(chunks)} chunks from cache")
    else:
        chunks = recursive_split(text, chunk_size=1000, overlap=200)
        embeddings = get_embeddings_batch(chunks)
        with open("datasets/sample_text/mdcat_chunks.json", "w") as f:
            json.dump({"chunks": chunks, "embeddings": embeddings}, f)
        print("Embedded and saved.")

    print(f"Chunks: {len(chunks)}, Dimensions: {len(embeddings[0])}")
