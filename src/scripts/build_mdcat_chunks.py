import json
import sys
sys.path.insert(0, ".")

from src.ingestion.pdf_loader import load_all_textbooks
from src.chunking.text_splitters import attach_chunk_ids, chunk_all_textbooks, filter_empty_chunks

PDF_DIR = "datasets/pdfs/PunjabBoardTextbooks"
OUTPUT_PATH = "datasets/mdcat_chunks.json"

if __name__ == "__main__":
    # TODO: call load_all_textbooks(PDF_DIR) — expect ~5 min, this is the
    #       expensive step we're trying to avoid repeating

    books = load_all_textbooks(PDF_DIR)
    # TODO: call chunk_all_textbooks(books, chunk_size=450, overlap=100,
    #       method="recursive")
    chunks = chunk_all_textbooks(books, chunk_size=450, overlap=100, method="recursive")
    chunks = attach_chunk_ids (chunks)
    chunks = filter_empty_chunks(chunks)
    # TODO: print a summary — total chunk count, and a per-subject breakdown
    #       (e.g. count how many chunks have subject == "Biology" vs "Chemistry")
    #       so you can sanity-check the split before trusting it
    
    total_chunks = len(chunks)
    subject_counts = {}
    for chunk in chunks:
        subject = chunk["subject"]
        subject_counts[subject] = subject_counts.get(subject, 0) + 1

    print(f"Total chunks: {total_chunks}")
    for subject, count in subject_counts.items():
        print(f"Chunks for {subject}: {count}")

    # TODO: write the chunks list to OUTPUT_PATH as JSON — remember these
    #       chunks may still contain the unresolved mojibake replacement
    #       character (accepted noise) and other non-ASCII text (° from our
    #       fix), so make sure json.dump won't mangle it (check the
    #       ensure_ascii and encoding args)

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(chunks, f, ensure_ascii=False, indent=4)