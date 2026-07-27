import os

import pdfplumber

from src.ingestion.document_cleaner import clean_punjab_board_text

import os
import os

import os

def load_all_textbooks(directory: str) -> list[dict]:
    """
    Loads and cleans every real Punjab Board textbook PDF in `directory`.

    Returns:
        List of dicts: {"text": cleaned_full_text, "subject": ..., "grade": ...}
        one entry per book (chunking happens later, in a separate step).
    """
    all_textbooks = []

    # FIX: Use the `directory` parameter instead of the hardcoded absolute path
    if not os.path.exists(directory):
        raise FileNotFoundError(f"The system cannot find the path specified: '{directory}'")

    pdf_files = [f for f in os.listdir(directory) if f.endswith('.pdf')]

    for each_file in pdf_files:
        metadata = extract_metadata_from_filename(each_file)
        
        # This correctly joins the dynamic directory with the filename
        full_path = os.path.join(directory, each_file)
        
        raw_text = load_pdf(full_path)
        cleaned_text = clean_punjab_board_text(raw_text)
        
        result_dict = {
            "text": cleaned_text,
            "subject": metadata["subject"],
            "grade": metadata["grade"],
            "board": metadata["board"],
            "year": metadata["year"],
            "title": metadata["title"],
            "extension": metadata["extension"]
        }
        print(f"Loaded and cleaned textbook: {result_dict['title']}")
        
        all_textbooks.append(result_dict)
   
    return all_textbooks


def extract_metadata_from_filename(filename:str) -> dict:
    #Input: "11th Class Biology PunjabBoard Year 2026.pdf"
    #Output: {"subject": "Biology", "grade": "11th"", "Board": "PunjabBoard", "Year": "2026", "extension": "pdf"}
    parts = filename.split(" ")
    result = {
        "grade": parts[0],
        "class": parts[1],
        "subject": parts[2],
        "board": parts[3],
        "year": parts[4],
        "title": " ".join(parts[0:4]),
        "extension": parts[5]
    }
    print(f"Extracted metadata from filename: {result}")
    return result

def save_text(text:str, file_path:str) -> None:
    with open(file_path, "w", encoding="utf-8") as f:
        f.write(text)

def load_pdf(pdf_path):
    if not os.path.exists(pdf_path):
        raise ValueError("PDF path cannot be empty.")
    with pdfplumber.open(pdf_path) as pdf:
        text = ""
        for page in pdf.pages:  # Limit to the first 10 pages
            text += page.extract_text() or ""

    return text

if __name__ == "__main__":
    test_filenames = [
        "11th Class Biology PunjabBoard Year 2018.pdf",
        "12th Class Biology PunjabBoard Year 2026.pdf",
        "11th Class Chemistry PunjabBoard Year 2025.pdf",
        "12th Class Chemistry PunjabBoard Year 2026.pdf",
        "11th Class Physics PunjabBoard Year 2026.pdf",
        "12th Class Physics PunjabBoard Year 2026.pdf",
    ]
    for name in test_filenames:
        extract_metadata_from_filename(name)

    # content = load_pdf("datasets/pdfs/Biology-2e_-_WEB.pdf")
    # save_text(content, "datasets/biology.txt")
    # print(content)
