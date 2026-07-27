import re
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter

RECURSIVE_SEPARATORS = ["\n\n", "\n", " ", ""]

def filter_empty_chunks(chunks: list[dict], min_length: int = 10) -> list[dict]:
    """
    Filters out chunks that are empty or below a certain length.

    Args:
        chunks (list[dict]): List of chunk dictionaries.
        min_length (int): Minimum length of chunk text to keep.

    Returns:
        list[dict]: Filtered list of chunks.
    """
    return [chunk for chunk in chunks if len(chunk["text"].strip()) >= min_length]

def chunk_textbook(
    textbook: dict,
    chunk_size: int = 450,
    overlap: int = 100,
    method: str = "recursive"
) -> list[dict]:
    """
    Chunks a single textbook using the specified method.

    Args:
        textbook (dict): A textbook dictionary containing at least a 'text' field.
        chunk_size (int): Size of each chunk.
        overlap (int): Overlap between chunks.
        method (str): Chunking method. One of: "fixed", "sentence", "recursive".

    Returns:
        list[dict]: List of chunks for the textbook.
    """
    text = textbook["text"]
    subject = textbook.get("subject", "")
    grade = textbook.get("grade", "")

    if method == "fixed":
        return fixed_size_split(text, chunk_size, overlap, subject, grade)
    elif method == "sentence":
        return sentence_split(text, chunk_size, subject, grade)
    elif method == "recursive":
        return recursive_split(text, chunk_size, overlap, subject, grade)
    else:
        raise ValueError(f"Unknown chunking method: {method}")

import hashlib

def attach_chunk_ids(chunks: list[dict]) -> list[dict]:
    """
    Adds a stable chunk_id (hash of the chunk's text) to each chunk dict.
    Same text -> same ID, regardless of processing order or when it was
    added -- this is what makes incremental ingestion safe.
    """
    for chunk in chunks:
        chunk["chunk_id"] = hashlib.md5(
            chunk["text"].encode("utf-8")
        ).hexdigest()

    return chunks

def chunk_all_textbooks(
    textbooks: list[dict],
    chunk_size: int = 450,
    overlap: int = 100,
    method: str = "recursive"
) -> list[dict]:
    """
    Chunks all textbooks in the list using the specified method.

    Args:
        textbooks (list[dict]): List of textbook dictionaries.
        chunk_size (int): Size of each chunk.
        overlap (int): Overlap between chunks.
        method (str): Chunking method.

    Returns:
        list[dict]: Combined list of chunks from all textbooks.
    """
    all_chunks = []

    for textbook in textbooks:
        chunks = chunk_textbook(
            textbook,
            chunk_size=chunk_size,
            overlap=overlap,
            method=method,
        )
        all_chunks.extend(chunks)

    return all_chunks

def fixed_size_split(document, chunk_size=1000, overlap=200, subject:str="", grade:str="") -> List[dict]:
    chunks: List[dict] = []
    start = 0
    while start < len(document):
        end = start + chunk_size
        chunk = document[start:end]
        chunks.append({"text": chunk, "subject": subject, "grade": grade})
        start += chunk_size - overlap   
    return chunks

def sentence_split(document, chunk_size=1000, subject:str="", grade:str="") -> List[dict]:

    sentences = re.split(r'(?<=[.!?])\s+', document)

    chunks: List[dict] = []
    chunk = ""

    for sentence in sentences:

        # keep adding sentences until chunk is full
        if len(chunk) + len(sentence) <= chunk_size:
            chunk += sentence + " "

        else:
            chunks.append({"text": chunk.strip(), "subject": subject, "grade": grade})
            chunk = sentence + " "

    # add last chunk
    if chunk:
        chunks.append({"text": chunk.strip(), "subject": subject, "grade": grade})

    return chunks


def recursive_split(document, chunk_size=100, overlap=200, subject:str="", grade:str="") -> List[dict]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=RECURSIVE_SEPARATORS
    )

    chunks = splitter.split_text(document)

    print("Original Text length:", len(document))
    print("Number of Chunks:", len(chunks))
    print("Chunks Sizes:", [len(chunk) for chunk in chunks])
    
    chunk_dicts = []
    for chunk in chunks:
        chunk_dicts.append({
            "text": chunk,
            "subject": subject,
            "grade": grade
        })
    return chunk_dicts

if __name__ == "__main__":
    with open("datasets/sample_text/biology.txt", "r", encoding="utf-8") as f:
        text = f.read()

    fixed = fixed_size_split(text, chunk_size=1000, overlap=200, subject="Biology", grade="10th")
    recursive = recursive_split(text, chunk_size=1000, overlap=200, subject="Biology", grade="10th")
    sentence = sentence_split(text, chunk_size=1000, subject="Biology", grade="10th")

    print(f"Fixed:     {len(fixed)} chunks")
    print(f"Recursive: {len(recursive)} chunks")
    print(f"Sentence:  {len(sentence)} chunks")

    print("\n--- Fixed middle chunk ---")
    print(fixed[len(fixed)//2])
    print("\n--- Recursive middle chunk ---")
    print(recursive[len(recursive)//2])
    print("\n--- Sentence middle chunk ---")
    print(sentence[len(sentence)//2])
