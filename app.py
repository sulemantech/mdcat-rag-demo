import sys
sys.path.insert(0, ".")
import json
import os

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

# generator.py reads GROQ_API_KEY via os.getenv() at import time. Fail
# loudly and specifically here instead of letting it crash later inside
# the Groq SDK with an opaque error -- shows exactly which secret keys
# Streamlit actually sees (names only, never values) so a missing/renamed
# secret is obvious instead of guessed at.
try:
    available_secret_keys = list(st.secrets.keys())
except Exception as e:
    available_secret_keys = None
    st.error(f"st.secrets could not be read at all: {e!r}")
    st.stop()

if "GROQ_API_KEY" in available_secret_keys:
    os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
elif not os.getenv("GROQ_API_KEY"):
    st.error(
        "GROQ_API_KEY is not set. Secret keys Streamlit currently sees for "
        f"this app: {available_secret_keys!r}. Add it under this app's "
        "Settings -> Secrets, exactly as: GROQ_API_KEY = \"your-key-here\" "
        "(top level, not nested under a [section])."
    )
    st.stop()

from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate

st.set_page_config(page_title="MDCAT Copilot", page_icon="🧬")


@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline():
    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    collection = create_collection("mdcat_v2", persist_directory="chroma_db")
    return chunks, collection


chunks, collection = load_pipeline()

st.title("MDCAT Copilot")
st.markdown(
    """
    Ask a Biology, Chemistry, or Physics question (FSc / MDCAT level). This runs the real pipeline:
    hybrid retrieval (BM25 + semantic search, RRF-fused), cross-encoder reranking, then grounded
    generation — the same system behind the
    [full case study](https://sulemantech.github.io/production-rag-from-scratch/)
    ([source code](https://github.com/sulemantech/production-rag-from-scratch)).

    The "retrieved context" panel below shows exactly what the model saw before answering.
    """
)

subject = st.selectbox("Subject", ["Any", "Biology", "Chemistry", "Physics"])
question = st.text_area("Your question", placeholder="e.g. What is the role of mitochondria in a cell?")

if st.button("Ask", type="primary") and question.strip():
    with st.spinner("Retrieving and generating..."):
        metadata_filter = {"subject": subject} if subject != "Any" else None
        candidates = hybrid_retrieve(question, chunks, collection, metadata_filter=metadata_filter, top_k=20)
        reranked = rerank(question, candidates, top_k=5)
        answer = generate(question, reranked)

    st.subheader("Answer")
    st.write(answer)

    with st.expander("Retrieved context (what the model actually saw)"):
        for i, chunk in enumerate(reranked, 1):
            st.markdown(f"**[{i}]** {chunk[:500]}")
            st.divider()
