import json
import re
import sys

sys.path.insert(0, ".")

import pdfplumber

PDF_PATH = "datasets/sample_text/MDCAT-Solved-Past-Papers-2021-and-2022.pdf"
OUTPUT_PATH = "datasets/evaluation/mdcat_past_papers_2021_2022.json"

# Section headers as they literally appear in the document, in document
# order. "None" means: skip this section (don't extract questions from it).
# Position-based (not page-based) splitting avoids bleed when a section
# boundary falls in the middle of a page rather than at a page break.
SECTION_HEADERS = [
    ("Biology", "Biology"),
    ("Logical Reasoning", None),
    ("PHYSICS", "Physics"),
    ("CHEMISTRY", "Chemistry"),
    ("ENGLISH", None),
]

QUESTION_SPLIT = re.compile(r"\n(?=\d+\.\s)")
QUESTION_NUM = re.compile(r"^(\d+)\.\s*(.*)", re.DOTALL)
OPTION_SPLIT = re.compile(r"\n(?=[A-D]\.\s)")
CORRECT_MARKER = re.compile(r"\s*\(/c\)\s*$")


def extract_full_text(pdf) -> str:
    text = ""
    for page in pdf.pages:
        text += (page.extract_text() or "") + "\n"
    return text


def split_into_sections(full_text: str) -> dict:
    """
    Finds each section header's exact character position in the full text
    and slices between consecutive headers. Returns {subject: section_text}
    for sections with a real subject name (None-mapped ones are dropped).
    """
    positions = []
    for header, subject in SECTION_HEADERS:
        pattern = re.compile(r"(?m)^\s*" + re.escape(header) + r"\s*$")
        match = pattern.search(full_text)
        if not match:
            raise ValueError(f"Section header not found: {header!r}")
        positions.append((match.start(), match.end(), subject))

    sections = {}
    for i, (start, end, subject) in enumerate(positions):
        section_start = end
        section_end = positions[i + 1][0] if i + 1 < len(positions) else len(full_text)
        if subject is not None:
            sections[subject] = full_text[section_start:section_end]

    return sections


def extract_question_stem(preamble: str) -> tuple[str, str]:
    """
    Returns (question_stem, explanation). The question stem extends up to
    and including the first '?' anywhere in the preamble (may span multiple
    lines). Falls back to just the first line for fill-in-the-blank style
    questions that don't contain a '?' at all.
    """
    qmark_pos = preamble.find("?")
    if qmark_pos != -1:
        question_stem = preamble[: qmark_pos + 1].replace("\n", " ").strip()
        explanation = preamble[qmark_pos + 1:].strip()
    else:
        lines = preamble.split("\n", 1)
        question_stem = lines[0].strip()
        explanation = lines[1].strip() if len(lines) > 1 else ""

    question_stem = re.sub(r"\s+", " ", question_stem).strip()
    return question_stem, explanation


def parse_questions(section_text: str) -> list:
    questions = []
    blocks = QUESTION_SPLIT.split(section_text)

    for block in blocks:
        block = block.strip()
        match = QUESTION_NUM.match(block)
        if not match:
            continue

        qnum, rest = match.groups()

        option_start = re.search(r"\n[A-D]\.\s", rest)
        if not option_start:
            continue

        preamble = rest[: option_start.start()].strip()
        options_block = rest[option_start.start():].strip()

        question_stem, explanation = extract_question_stem(preamble)

        raw_options = OPTION_SPLIT.split(options_block)
        options = []
        correct_answer = None
        for opt in raw_options:
            opt = opt.strip().replace("\n", " ")
            is_correct = bool(CORRECT_MARKER.search(opt))
            clean_opt = CORRECT_MARKER.sub("", opt).strip()
            options.append(clean_opt)
            if is_correct:
                correct_answer = clean_opt

        if len(options) != 4 or not question_stem or correct_answer is None:
            continue

        # Drop anything that still doesn't look like a real standalone
        # question -- e.g. leaked "Statement I / II" reasoning items that
        # reference a preceding statement never captured here.
        if re.search(r"\bstatement\s+(i|ii|1|2)\b", question_stem, re.IGNORECASE):
            continue

        # Drop questions that still look cut off mid-sentence. A real,
        # complete question stem essentially always ends in '?', ':', or
        # '.' (':' means "options complete the sentence" -- a valid,
        # intentional ending, not truncation). Anything with no terminal
        # punctuation at all is almost certainly a stem the parser cut off
        # mid-line -- safer to drop it than guess at the missing part.
        if not question_stem or question_stem[-1] not in "?:.":
            continue

        questions.append({
            "id": int(qnum),
            "question": question_stem,
            "explanation": explanation,
            "options": options,
            "correct_answer": correct_answer,
        })

    return questions


if __name__ == "__main__":
    result = {}

    with pdfplumber.open(PDF_PATH) as pdf:
        full_text = extract_full_text(pdf)
        sections = split_into_sections(full_text)

        for subject, section_text in sections.items():
            questions = parse_questions(section_text)
            test_key = f"mdcat_2021_2022_{subject.lower()}"
            result[test_key] = {
                "name": "MDCAT Solved Past Papers 2021 and 2022",
                "subject": subject,
                "questions": questions,
            }
            print(f"{subject}: extracted {len(questions)} questions")

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)

    print(f"\nWrote {sum(len(v['questions']) for v in result.values())} total questions to {OUTPUT_PATH}")
