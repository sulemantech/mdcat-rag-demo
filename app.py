import sys
sys.path.insert(0, ".")
import json

import gradio as gr
from dotenv import load_dotenv

load_dotenv()

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate

with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
    CHUNKS = json.load(f)

COLLECTION = create_collection("mdcat_v2", persist_directory="chroma_db")


def answer_question(question: str, subject: str):
    if not question.strip():
        return "Enter a question above first.", ""

    metadata_filter = {"subject": subject} if subject != "Any" else None
    candidates = hybrid_retrieve(question, CHUNKS, COLLECTION, metadata_filter=metadata_filter, top_k=20)
    reranked = rerank(question, candidates, top_k=5)
    answer = generate(question, reranked)

    context_display = "\n\n---\n\n".join(
        f"[{i + 1}] {chunk[:500]}" for i, chunk in enumerate(reranked)
    )
    return answer, context_display


with gr.Blocks(title="MDCAT RAG Demo") as demo:
    gr.Markdown(
        """
        # MDCAT RAG Engine — Live Demo

        Ask a Biology, Chemistry, or Physics question (FSc / MDCAT level). This runs the real
        pipeline: hybrid retrieval (BM25 + semantic search, fused with Reciprocal Rank Fusion),
        cross-encoder reranking, then grounded generation — the same system behind the
        [full case study](https://sulemantech.github.io/production-rag-from-scratch/)
        ([source code](https://github.com/sulemantech/production-rag-from-scratch)).

        The "retrieved context" panel shows exactly what the model saw before answering —
        nothing here is hidden.
        """
    )
    with gr.Row():
        subject = gr.Dropdown(["Any", "Biology", "Chemistry", "Physics"], value="Any", label="Subject")
    question = gr.Textbox(
        label="Your question",
        placeholder="e.g. What is the role of mitochondria in a cell?",
        lines=2,
    )
    submit = gr.Button("Ask", variant="primary")
    answer = gr.Textbox(label="Answer", lines=6)
    with gr.Accordion("Retrieved context (what the model actually saw)", open=False):
        context = gr.Textbox(label="", lines=14, show_label=False)

    submit.click(answer_question, inputs=[question, subject], outputs=[answer, context])
    question.submit(answer_question, inputs=[question, subject], outputs=[answer, context])

if __name__ == "__main__":
    demo.launch()
