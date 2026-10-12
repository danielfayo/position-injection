"""Text cleaning and sentence splitting shared by the document and trial scripts."""

import re

# Sections that hold references or lists rather than prose. Everything from the
# first of these headings onwards is dropped.
STOP_SECTIONS = {
    "references", "see also", "external links", "notes", "further reading",
    "bibliography", "sources", "citations", "footnotes", "works cited",
    "notes and references", "selected works", "works", "publications",
    "discography", "filmography", "gallery",
}

HEADING = re.compile(r"^\s*(=+)\s*(.*?)\s*\1\s*$")
SENTENCE_END = (".", "!", "?", '."', '!"', '?"', ".”", "!”", "?”", ".)")
MIN_PARAGRAPH_CHARS = 80

# Words that end in a full stop without ending a sentence.
ABBREVIATIONS = {
    "mr", "mrs", "ms", "dr", "prof", "st", "jr", "sr", "vs", "etc", "e.g", "i.e",
    "no", "nos", "vol", "fig", "approx", "c", "ca", "inc", "ltd", "co", "corp",
    "mt", "ft", "gen", "col", "lt", "sgt", "capt", "rev", "hon", "op", "pp", "p",
    "jan", "feb", "mar", "apr", "jun", "jul", "aug", "sep", "sept", "oct", "nov", "dec",
}

# A full stop, question or exclamation mark (plus any closing quotes or brackets),
# then whitespace, then something that can start a sentence.
_CANDIDATE = re.compile(r"[.!?][\"”’)\]]*(\s+)(?=[\"“‘(\[]?[A-Z0-9])")


def clean_extract(text: str) -> list[str]:
    """Turn a Wikipedia plain-text extract into a list of prose paragraphs."""
    paragraphs = []
    for line in text.splitlines():
        line = line.strip()
        heading = HEADING.match(line)
        if heading:
            if heading.group(2).lower() in STOP_SECTIONS:
                break
            continue
        line = re.sub(r"\s+", " ", line)
        # Lists, captions and table rows are short or lack a closing full stop;
        # formulas come through as LaTeX.
        if len(line) < MIN_PARAGRAPH_CHARS or not line.endswith(SENTENCE_END):
            continue
        if "{\\displaystyle" in line:
            continue
        paragraphs.append(line)
    return paragraphs


def _is_abbreviation(text_before: str) -> bool:
    """True if the word ending at a full stop is an abbreviation or an initial."""
    words = text_before.split()
    if not words:
        return False
    word = words[-1].lstrip("\"“‘([").rstrip(".").lower()
    return (
        word in ABBREVIATIONS
        or (len(word) == 1 and word.isalpha())          # initials: "J. Smith"
        or re.fullmatch(r"(?:[a-z]\.)+[a-z]", word) is not None   # "U.S.", "a.m."
    )


def split_sentences(paragraph: str) -> list[str]:
    """Split one paragraph into sentences, keeping closing quotes with their sentence."""
    sentences, start = [], 0
    for match in _CANDIDATE.finditer(paragraph):
        if paragraph[match.start()] == "." and _is_abbreviation(paragraph[start:match.start() + 1]):
            continue
        sentences.append(paragraph[start:match.start(1)].strip())
        start = match.end(1)
    rest = paragraph[start:].strip()
    if rest:
        sentences.append(rest)
    return sentences
