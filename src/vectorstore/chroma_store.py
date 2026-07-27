import chromadb

def create_collection(name: str, persist_directory: str):
    client = chromadb.PersistentClient(path=persist_directory)
    collection = client.get_or_create_collection(name=name)
    return collection


# def add_documents(collection, chunks: list, metadata: list, embeddings: list):
#     collection.add(
#         documents=chunks,
#         metadatas=metadata,
#         embeddings=embeddings,
#         ids=[str(i) for i in range(len(chunks))]
#     )

def add_documents(collection, chunks: list, metadata: list, embeddings: list, chunk_ids:list, batch_size: int = 5000) -> None:
    # TODO: loop over chunks/metadata/embeddings in slices of `batch_size`,
    #       calling collection.add() once per slice. Watch the ids= line —
    #       it currently does str(i) for i in range(len(chunks)), which
    #       assumes chunks starts at index 0. Each batch's ids need to stay
    #       globally unique and match their true position in the FULL list
    #       (not restart from 0 each batch), since hybrid_retriever relies
    #       on these ids lining up with BM25's original chunk indices.
    for i in range(0, len(chunks), batch_size):
        collection.add(
            documents=chunks[i:i + batch_size],
            metadatas=metadata[i:i + batch_size],
            embeddings=embeddings[i:i + batch_size],
            ids=chunk_ids[i:i + batch_size]
        )

def query(collection, query_embedding: list, metadata_filter: dict = None, n_results: int = 5):
    results = collection.query(query_embeddings=[query_embedding], where=metadata_filter, n_results=n_results)
    return results


