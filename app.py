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
from src.generation.generator import generate, generate_mcq_explained

st.set_page_config(page_title="MDCAT Copilot", page_icon="🧬", layout="centered")

SUBJECT_COLORS = {"Biology": "#1f7a5c", "Chemistry": "#c97a1f", "Physics": "#2f5fa8"}

SAMPLE_QUESTIONS = {
    "Biology": [
        {
            "question": "Example of viruses having a polyhedral capsid that is with 252 capsomeres is:",
            "options": {"A": "Adenovirus", "B": "Tobacco Mosaic Virus", "C": "Influenza virus", "D": "Bacteriophage"},
            "correct": "A",
        },
        {
            "question": "The causative organism of measles is:",
            "options": {"A": "Poxvirus", "B": "Papovavirus", "C": "Picornovirus", "D": "Paramyxovirus"},
            "correct": "D",
        },
    ],
    "Physics": [
        {
            "question": "What is the shape of velocity-time graph for constant acceleration?",
            "options": {"A": "Parabola line", "B": "Straight line", "C": "Incline curve", "D": "Decline curve"},
            "correct": "B",
        },
        {
            "question": "A stone thrown horizontally from the top of a tall building follows a path that is:",
            "options": {"A": "Circular", "B": "Made of two straight line segments", "C": "Hyperbolic", "D": "Parabolic"},
            "correct": "D",
        },
    ],
    "Chemistry": [
        {
            "question": "According to which scientist, the probability of finding an electron at a certain position is possible?",
            "options": {"A": "Bohr's", "B": "De-Broglie", "C": "Hund's", "D": "Schrodinger"},
            "correct": "D",
        },
        {
            "question": "Which of the following factor does not affect the magnitude of vapor pressure?",
            "options": {"A": "amount of liquid", "B": "size of molecule", "C": "temperature of liquid", "D": "intermolecular forces"},
            "correct": "A",
        },
    ],
}
ALL_EXAMPLES = {ex["question"]: ex for lst in SAMPLE_QUESTIONS.values() for ex in lst}

st.markdown(
    """
    <style>
    .block-container { padding-top: 2.2rem; max-width: 760px; }
    .mc-header { margin-bottom: 0.2rem; }
    .mc-tagline { color: #6b7280; font-size: 0.98rem; margin-bottom: 1.4rem; }
    .mc-links a { text-decoration: none; margin-right: 14px; font-size: 0.9rem; }
    .stButton button { border-radius: 8px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown("## 🧬 MDCAT Copilot", help=None)
st.markdown(
    '<div class="mc-tagline">Ask a Biology, Chemistry, or Physics question — grounded, '
    "retrieval-based answers with a short explanation for why the correct option is right.</div>",
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="mc-links">'
    '<a href="https://sulemantech.github.io/production-rag-from-scratch/" target="_blank">📊 Full case study</a>'
    '<a href="https://github.com/sulemantech/production-rag-from-scratch" target="_blank">💻 Source code</a>'
    "</div>",
    unsafe_allow_html=True,
)
st.divider()


@st.cache_resource(show_spinner="Loading pipeline...")
def load_pipeline():
    with open("datasets/mdcat_chunks.json", "r", encoding="utf-8") as f:
        chunks = json.load(f)
    collection = create_collection("mdcat_v2", persist_directory="chroma_db")
    return chunks, collection


chunks, collection = load_pipeline()

if "question_text" not in st.session_state:
    st.session_state.question_text = ""

subject = st.radio("Subject", ["Biology", "Chemistry", "Physics"], horizontal=True)

st.caption("Try a question we know the answer to, or type your own below.")
cols = st.columns(2)
for col, example in zip(cols, SAMPLE_QUESTIONS[subject]):
    with col:
        if st.button(example["question"], key=f"ex_{subject}_{example['question'][:20]}", use_container_width=True):
            st.session_state.question_text = example["question"]

question = st.text_area("Your question", key="question_text", height=80)
ask = st.button("Ask", type="primary")

if ask and question.strip():
    matched = ALL_EXAMPLES.get(question.strip())

    with st.spinner("Retrieving context and generating..."):
        metadata_filter = {"subject": subject}
        candidates = hybrid_retrieve(question, chunks, collection, metadata_filter=metadata_filter, top_k=20)
        reranked = rerank(question, candidates, top_k=5)

        if matched:
            result = generate_mcq_explained(question, matched["options"], reranked)
        else:
            free_answer = generate(question, reranked)

    color = SUBJECT_COLORS.get(subject, "#6b7280")

    if matched:
        is_right = result["letter"] == matched["correct"]
        with st.container(border=True):
            for letter, text in matched["options"].items():
                if letter == matched["correct"]:
                    st.markdown(f"**{letter}. {text}** ✅ *(correct answer)*")
                elif letter == result["letter"] and not is_right:
                    st.markdown(f"~~{letter}. {text}~~ ⬅️ *(model picked this)*")
                else:
                    st.markdown(f"{letter}. {text}")
            st.divider()
            badge = "✅ Model answered correctly" if is_right else "⚠️ Model answered incorrectly"
            st.markdown(f"**{badge}**")
            st.write(result["explanation"])
    else:
        with st.container(border=True):
            st.markdown(f"**Answer** &nbsp;·&nbsp; :blue[{subject}]" if color else "**Answer**")
            st.write(free_answer)

    with st.expander("Retrieved context (what the model actually saw)"):
        for i, chunk in enumerate(reranked, 1):
            st.markdown(f"**[{i}]** {chunk[:500]}")
            st.divider()
elif ask:
    st.warning("Enter a question first, or pick one of the examples above.")
