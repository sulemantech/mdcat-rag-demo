import os
import time

from dotenv import load_dotenv
from groq import Groq, APIConnectionError

load_dotenv()
# TODO: create the Groq client using the GROQ_API_KEY from the environment
client = Groq(api_key=os.getenv("GROQ_API_KEY"))


# Groq model catalog changes over time — check https://console.groq.com/docs/models
# for the current instant/production model id before picking one.
DEFAULT_MODEL = "llama-3.1-8b-instant"


def _call_groq(model: str, messages: list, max_retries: int = 3, backoff_seconds: int = 5):
    """
    Wraps the Groq chat completion call with retry-on-connection-error.
    Long eval runs (200+ sequential calls) have hit transient DNS/network
    drops mid-run twice now — a single blip shouldn't kill 15+ minutes of work.
    """
    for attempt in range(1, max_retries + 1):
        try:
            return client.chat.completions.create(model=model, messages=messages, temperature=0)
        except APIConnectionError:
            if attempt == max_retries:
                raise
            time.sleep(backoff_seconds)

def generate(question: str, chunks: list[str], model: str = DEFAULT_MODEL) -> str:

    context = "\n\n".join(chunks)

    prompt = f"""
[1] Instruction

You are a helpful AI assistant. Answer the user's question using only the information provided in the Context section below.

If the answer is not contained in the context, reply:
"I don't have enough information in the provided context."

Do not use outside knowledge.
Do not make up facts.
Keep the answer clear and concise.

[2] Context

{context}

[3] Question

{question}

Answer:
"""

    response = _call_groq(model, [{"role": "user", "content": prompt}])
    return response.choices[0].message.content


def generate_mcq(question: str, options: dict, chunks: list[str], model: str = DEFAULT_MODEL) -> str:
    context = "\n\n".join(chunks)
    options_text = "\n".join([f"{letter}. {text}" for letter, text in options.items()])
    prompt = f"""You are an MDCAT exam assistant. Using ONLY the context below, select the correct answer.

Context:
{context}

Question: {question}

{options_text}

Reply with only the letter of the correct answer (A, B, C, or D). Nothing else.

Answer:"""
    response = _call_groq(model, [{"role": "user", "content": prompt}])
    return response.choices[0].message.content.strip()


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from src.vectorstore.chroma_store import create_collection
    from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
    from src.retrieval.reranker import rerank

    # load chunks from cache
    import json
    with open("datasets/sample_text/mdcat_chunks.json", "r") as f:
        chunks = json.load(f)
   
    # store in ChromaDB
    collection = create_collection("biology", persist_directory="chroma_db")

    questions = [
        "what is the role of mitochondria?",
        "how does DNA replication work?",
        "what is the function of ribosomes?"
    ]

    for q in questions:
        candidates = hybrid_retrieve(q, chunks, collection, top_k=10)
        results = rerank(q, candidates, top_k=3)
        answer = generate(q, results)
        print(f"\nQ: {q}")
        print(f"A: {answer}")