
import re

def clean_punjab_board_text(text: str) -> str:
    """
    Cleans text extracted from Punjab Board eLearn PDFs (pdfplumber).

    Handles:
    - removes "eLearn.Punjab" headers
    - removes standalone figure caption lines
      (e.g. "Fig. 2.1 ...", "Figure 3.4 ...", "Fig: 4.4 ...")
    - restores degree symbols in temperatures
      (e.g. "37�C" -> "37°C")
    - restores apostrophes in words
      (e.g. "scientist�s" -> "scientist's")
    - collapses excessive blank lines

    Known NOT fixed here (accepted as inherent source noise):
    - dropped 'f' in "fi" ligatures
      (e.g. "speciic" instead of "specific")
    - detached formula subscripts
      (e.g. "SnCl\\n4")
    - U+FFFD replacement characters outside the two known contexts
      above, since their original glyph cannot be inferred safely.
    """

    # ------------------------------------------------------------------
    # Remove eLearn.Punjab page headers/footers
    # ------------------------------------------------------------------
    text = re.sub(
        r"eLearn\.Punjab",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Remove standalone figure captions
    #
    # Matches:
    #   Fig. 2.1. : Unbranched and branched chains...
    #   Figure 3.4 Variation in resistance...
    #   Fig: 4.4. Unit membrane
    #   FIGURE 5.2 Energy band diagram
    #
    # Does NOT match:
    #   "As shown in Fig. 2.1 ..."
    # ------------------------------------------------------------------
    text = re.sub(
        r'^\s*Fig(?:ure)?[:.]?\s*\d+(?:\.\d+)*.*(?:\n|$)',
        '',
        text,
        flags=re.MULTILINE | re.IGNORECASE,
    )

    # ------------------------------------------------------------------
    # Restore degree symbol:
    #
    #   37�C -> 37°C
    #   98�F -> 98°F
    # ------------------------------------------------------------------
    text = re.sub(
        r'(\d+)\s*\uFFFD\s*([CF])\b',
        r'\1°\2',
        text,
    )

    # ------------------------------------------------------------------
    # Restore apostrophes between letters:
    #
    #   scientist�s -> scientist's
    #   Faraday�s -> Faraday's
    #   it�s -> it's
    # ------------------------------------------------------------------
    text = re.sub(
        r'([A-Za-z])\uFFFD([A-Za-z])',
        r"\1'\2",
        text,
    )

    # ------------------------------------------------------------------
    # Collapse excessive blank lines
    # ------------------------------------------------------------------
    text = re.sub(
        r'\n{3,}',
        '\n\n',
        text,
    )
        # TODO: remove (cid:N) glyph-code artifacts (unresolvable font glyphs,
    #       confirmed via font inspection: this PDF's fonts have no
    #       ToUnicode table, so these are unrecoverable — safe to strip
    #       unconditionally, accepting loss of exact equation notation)
    #       pattern: literal "(cid:" + digits + ")"
    text = re.sub(
        r'\(cid:\d+\)',
        '',
        text,
    )
    # TODO: collapse repeated spaces left behind after stripping (e.g.
    #       "(cid:3) (cid:3)" often represented actual spaces, so you'll
    #       get runs of multiple spaces — squeeze to one)
    text = re.sub(
        r' {2,}',
        ' ',
        text,
    )
    text = re.sub(r"[ \t]+", " ", text)

    # Remove the TopStudyWorld watermark/header
    text = re.sub(
        r"Go to www\.topstudyworld\.com/books to download all PDF books\s*for all classes\s*\n?",
        "",
        text,
        flags=re.IGNORECASE,
    )

    # Remove standalone 1-2 character lines (vertical watermark noise)
    text = re.sub(
        r"(?m)^\s*.{1,2}\s*$\n?",
        "",
        text,
    )

    # Optional: Collapse multiple blank lines into a single blank line
    text = re.sub(r"\n{3,}", "\n\n", text)

    return text.strip()

    
def clean_document(text: str) -> str:
    # remove known footers
    text = text.replace("Access for free at openstax.org", "")
    
    # remove figure captions
    text = re.sub(r'FIGURE \d+\.\d+.*?\n', '', text)
    
    # collapse 3+ blank lines into 2 (preserve paragraph breaks)
    text = re.sub(r'\n{3,}', '\n\n', text)

    # remove table of contents (pages with just titles and page numbers)
    text = re.sub(r'\n[\w\s,]+\d+\n', '\n', text)

    
    return text.strip()


if __name__ == "__main__":
    with open("datasets/sample_text/biology.txt", "r", encoding="utf-8") as f:
        text = f.read()

    cleaned = clean_document(text)
    #save the cleaned text to a new file
    with open("datasets/sample_text/biology_clean.txt", "w", encoding="utf-8") as f:
        f.write(cleaned)    
    print(f"Before: {len(text)} chars")
    print(f"After:  {len(cleaned)} chars")
    print(f"Removed: {len(text) - len(cleaned)} chars")
    print("\n--- Sample cleaned text ---")
    print(cleaned[5000:6000])
