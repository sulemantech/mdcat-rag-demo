import sys
sys.path.insert(0, ".")
import json
import re
from src.vectorstore.chroma_store import create_collection
from src.retrieval.hybrid_retriever import retrieve as hybrid_retrieve
from src.retrieval.reranker import rerank
from src.generation.generator import generate_mcq



def load_questions(json_path: str) -> list:
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    questions = []
    for test_key, test_data in data.items():
        if isinstance(test_data, dict) and "questions" in test_data:
            for q in test_data["questions"]:
                q["source"] = test_data.get("name", test_key)
                q["subject"] = test_data.get("subject")
                questions.append(q)

    return questions

def parse_options(options: list) -> dict:
    """Convert ["A. Adenovirus", "B. TMV", ...] → {"A": "Adenovirus", "B": "TMV", ...}"""
    result = {}
    for opt in options:
        match = re.match(r'^([A-Za-z])[\.\)]\s*(.+)$', opt.strip())
        if match:
            result[match.group(1).upper()] = match.group(2).strip()
        else:
            letter = chr(65 + len(result))
            result[letter] = opt.strip()
    return result


def extract_letter(text: str) -> str:
    """Extract the answer letter (A/B/C/D) from a string."""
    if not text:
        return ""
    match = re.match(r'^([A-Za-z])[\.\)\s]?', text.strip())
    if match:
        return match.group(1).upper()
    return text.strip()[0].upper() if text.strip() else ""


def evaluate(questions: list, chunks: list, collection, sample_size: int = 20, top_k: int = 5) -> dict:
    correct = 0
    total = 0
    results = []

    for q in questions[:sample_size]:
        options_dict = parse_options(q["options"])

        # Run the full pipeline
        # Build subject filter for retrieval
        metadata_filter = None
        if q.get("subject"):
            metadata_filter = {"subject": q["subject"]}

        # Run the full pipeline
        candidates = hybrid_retrieve(
            q["question"],
            chunks,
            collection,
            metadata_filter=metadata_filter,
            top_k=top_k * 4,
        )
        reranked = rerank(q["question"], candidates, top_k=top_k)
        predicted = generate_mcq(q["question"], options_dict, reranked)  # returns a str

        predicted_letter = extract_letter(predicted)
        correct_letter = extract_letter(q["correct_answer"])

        is_correct = predicted_letter == correct_letter
        if is_correct:
            correct += 1
        total += 1

        status = "PASS" if is_correct else "FAIL"
        print(f"Q{q['id']}: predicted={predicted_letter} correct={correct_letter} {status}")
        if not is_correct:
            print(f"  -> Correct: {q['correct_answer']}")
            print(f"  -> Got:     {predicted}")

        results.append({
            "id": q["id"],
            "question": q["question"],
            "predicted": predicted_letter,
            "correct": correct_letter,
            "is_correct": is_correct,
        })

    accuracy = correct / total if total > 0 else 0
    return {"correct": correct, "total": total, "accuracy": accuracy, "results": results}


def print_summary(eval_results: dict):
    print("\n" + "=" * 50)
    print("EVALUATION SUMMARY")
    print("=" * 50)
    print(f"Correct : {eval_results['correct']} / {eval_results['total']}")
    print(f"Accuracy: {eval_results['accuracy']:.2%}")
    print("=" * 50)


if __name__ == "__main__":
    JSON_PATH = "datasets/evaluation/mdcat_mcqs.json"
    CHUNKS_PATH = "datasets/mdcat_chunks.json"
    SAMPLE_SIZE = 332
    TOP_K = 5

    print("Loading questions...")
    questions = load_questions(JSON_PATH)
    print(f"Loaded {len(questions)} questions")

    print("Loading chunks...")
    with open(CHUNKS_PATH, "r", encoding="utf-8") as f:
        chunks = json.load(f)
    print(f"Loaded {len(chunks)} chunks")

    print("Connecting to ChromaDB...")
    collection = create_collection("mdcat_v2", persist_directory="chroma_db")

    print(f"\nEvaluating {SAMPLE_SIZE} questions...\n")
    eval_results = evaluate(questions, chunks, collection, sample_size=SAMPLE_SIZE, top_k=TOP_K)

    print_summary(eval_results)
