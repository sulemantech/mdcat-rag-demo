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
        {"question": "Example of viruses having a polyhedral capsid that is with 252 capsomeres is:",
         "options": {"A": "Adenovirus", "B": "Tobacco Mosaic Virus", "C": "Influenza virus", "D": "Bacteriophage"}, "correct": "A"},
        {"question": "The causative organism of measles is:",
         "options": {"A": "Poxvirus", "B": "Papovavirus", "C": "Picornovirus", "D": "Paramyxovirus"}, "correct": "D"},
        {"question": "In the life cycle of a bacteriophage, the lysozymes are required in which of the following steps of infection process?",
         "options": {"A": "Genome injection", "B": "Penetration", "C": "Replication", "D": "Adsorption"}, "correct": "B"},
        {"question": "______ is transmitted through infected blood and hypodermic syringes.",
         "options": {"A": "HIV", "B": "Influenza Virus", "C": "Morbilli Virus (Measles)", "D": "Vibrio Cholerae (Cholera)"}, "correct": "A"},
        {"question": "In Calvin cycle CO₂ reacts with RuBP to produce:",
         "options": {"A": "3-PGA", "B": "G3P", "C": "6-Carbon unstable intermediate", "D": "1,3 bisphosphoglycerate"}, "correct": "C"},
        {"question": "Which option is correct about a chlorophyll molecule?",
         "options": {"A": "Chemical formula C₅₅H₇₀O₆N₄Mg", "B": "Porphyrin ring with nitrogen in center", "C": "(Methyl) group on second pyrrole ring", "D": "Aldehyde group on second pyrrole ring"}, "correct": "A"},
        {"question": "In the journey of electrons from photosystem II to photosystem I, plastocyanin is reduced by:",
         "options": {"A": "Plastoquinone", "B": "Cytochrome complex", "C": "Primary electron acceptor of PSI", "D": "Ferredoxin"}, "correct": "B"},
        {"question": "Enzyme NADP reductase is responsible for:",
         "options": {"A": "Reducing NADP⁺", "B": "Oxidizing NADP⁺", "C": "Reducing Ferredoxin", "D": "Reducing P₇₀₀"}, "correct": "A"},
        {"question": "Mono-saccharides have a general formula represented by:",
         "options": {"A": "Cₙ(H₂O)ₙ", "B": "C(H₂O)ₙ", "C": "C₂(H₂O)ₙ", "D": "Cⁿ(H₂O)ₙ"}, "correct": "A"},
        {"question": "Induced fit model of enzyme activity suggests that an enzyme:",
         "options": {"A": "Cannot modify its active sites", "B": "Can bind to a single substrate", "C": "Can catalyze related reaction", "D": "Usually belongs to non-regulatory enzyme"}, "correct": "C"},
    ],
    "Physics": [
        {"question": "What is the shape of velocity-time graph for constant acceleration?",
         "options": {"A": "Parabola line", "B": "Straight line", "C": "Incline curve", "D": "Decline curve"}, "correct": "B"},
        {"question": "A stone thrown horizontally from the top of a tall building follows a path that is:",
         "options": {"A": "Circular", "B": "Made of two straight line segments", "C": "Hyperbolic", "D": "Parabolic"}, "correct": "D"},
        {"question": "A fireman wants to slide down a rope. The breaking load of the rope is 3/4th of the weight of the man. With what acceleration should the fireman slide down?",
         "options": {"A": "g", "B": "g/4", "C": "3g/4", "D": "0"}, "correct": "B"},
        {"question": "The number of revolutions in 3π radians is:",
         "options": {"A": "1/60", "B": "3/2", "C": "2", "D": "6"}, "correct": "B"},
        {"question": "A fighter plane is moving in a vertical circle of radius r. Its minimum velocity at the highest point of the circle will be:",
         "options": {"A": "√3gr", "B": "√2gr", "C": "√gr", "D": "√(gr/2)"}, "correct": "C"},
        {"question": "Which of the following increases by increasing amplitude?",
         "options": {"A": "Wavelength", "B": "Frequency", "C": "Zero", "D": "Loudness"}, "correct": "D"},
        {"question": "The shortest distance between any two points in phase on a wave is called:",
         "options": {"A": "Displacement", "B": "Amplitude", "C": "Wavelength", "D": "Frequency"}, "correct": "C"},
        {"question": "What is the potential difference between two points in an electric field if it takes 600 J of energy to move a charge of 2 C between these two points?",
         "options": {"A": "0 J", "B": "1200 J", "C": "300 J", "D": "800 J"}, "correct": "C"},
        {"question": "The coulomb's constant k depends upon:",
         "options": {"A": "nature of medium", "B": "system of units", "C": "types of charge", "D": "nature of medium and system of units"}, "correct": "D"},
        {"question": "If a flywheel is rotating at 3.0 rad/s, the time it takes to complete one revolution is:",
         "options": {"A": "0.67 s", "B": "1.0 s", "C": "1.3 s", "D": "2.1 s"}, "correct": "D"},
    ],
    "Chemistry": [
        {"question": "According to which scientist, the probability of finding an electron at a certain position is possible?",
         "options": {"A": "Bohr's", "B": "De-Broglie", "C": "Hund's", "D": "Schrodinger"}, "correct": "D"},
        {"question": "Which gas in the discharge tube produces lightest canal ray particles?",
         "options": {"A": "Ar", "B": "He", "C": "H₂", "D": "Ne"}, "correct": "C"},
        {"question": "Which of the following factor does not affect the magnitude of vapor pressure?",
         "options": {"A": "amount of liquid", "B": "size of molecule", "C": "temperature of liquid", "D": "intermolecular forces"}, "correct": "A"},
        {"question": "A small building block which contains the whole information about a crystal structure is called:",
         "options": {"A": "Cell", "B": "Unit Cell", "C": "Crystal lattice", "D": "Crystal unit"}, "correct": "B"},
        {"question": "Precipitation occurs if the ionic concentration is:",
         "options": {"A": "Less than Ksp", "B": "More than Ksp", "C": "Equal to Ksp", "D": "Present in any amount"}, "correct": "B"},
        {"question": "One can estimate the direction in which equilibrium will shift with the help of:",
         "options": {"A": "Le Chatelier's principle", "B": "Law of mass action", "C": "Hess's law", "D": "Law of heat of formation"}, "correct": "A"},
        {"question": "The catalysis in which the catalyst and the reactants are in the same phase is known as:",
         "options": {"A": "Heterogeneous catalysis", "B": "Homogeneous catalysis", "C": "Slow catalysis", "D": "Fast catalysis"}, "correct": "B"},
        {"question": "Which of the following is a state function?",
         "options": {"A": "Freezing", "B": "Decomposition", "C": "Sublimation", "D": "Enthalpy"}, "correct": "D"},
        {"question": "What is the proton (atomic) number of an element that has four unpaired electrons in its ground state?",
         "options": {"A": "6", "B": "14", "C": "22", "D": "26"}, "correct": "D"},
        {"question": "Which type of solid is called an atomic solid?",
         "options": {"A": "Covalent solids", "B": "Ionic solids", "C": "Metallic solids", "D": "Molecular solids"}, "correct": "A"},
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
    .subject-chip {
        display: inline-block; padding: 2px 12px; border-radius: 20px;
        font-size: 0.82rem; font-weight: 600; color: white; margin-bottom: 0.6rem;
    }
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
color = SUBJECT_COLORS.get(subject, "#6b7280")
st.markdown(f'<span class="subject-chip" style="background:{color}">{subject}</span>', unsafe_allow_html=True)

_PLACEHOLDER = "— choose one of 10 real exam questions —"


def _apply_example():
    choice = st.session_state.get(f"example_picker_{subject}")
    if choice and choice != _PLACEHOLDER:
        st.session_state.question_text = choice


st.selectbox(
    "Try an example",
    [_PLACEHOLDER] + [ex["question"] for ex in SAMPLE_QUESTIONS[subject]],
    key=f"example_picker_{subject}",
    on_change=_apply_example,
    label_visibility="collapsed",
)
st.caption("Or type your own question below.")

question = st.text_area("Your question", key="question_text", height=80, label_visibility="collapsed",
                         placeholder="e.g. What is the role of mitochondria in a cell?")
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
            st.markdown(f'**Answer** &nbsp;·&nbsp; <span style="color:{color}">{subject}</span>', unsafe_allow_html=True)
            st.write(free_answer)

    with st.expander("Retrieved context (what the model actually saw)"):
        for i, chunk in enumerate(reranked, 1):
            st.markdown(f"**[{i}]** {chunk[:500]}")
            st.divider()
elif ask:
    st.warning("Enter a question first, or pick one of the examples above.")
