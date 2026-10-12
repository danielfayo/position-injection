from src.text import clean_extract, split_sentences


def test_splits_simple_sentences():
    assert split_sentences("The river is long. It flows north! Does it flood? Rarely.") == [
        "The river is long.", "It flows north!", "Does it flood?", "Rarely.",
    ]


def test_does_not_split_after_abbreviations_or_initials():
    text = "Dr. Smith met J. R. Brown in the U.S. in 1990. They wrote a book."
    assert split_sentences(text) == [
        "Dr. Smith met J. R. Brown in the U.S. in 1990.", "They wrote a book.",
    ]


def test_keeps_closing_quotes_with_their_sentence():
    assert split_sentences('He said "it is over." Then he left.') == [
        'He said "it is over."', "Then he left.",
    ]


def test_clean_extract_drops_headings_short_lines_and_reference_sections():
    prose = "This paragraph is long enough to count as prose because it has many words in it."
    extract = "\n".join([
        prose,
        "== History ==",
        "A caption",
        prose,
        "== References ==",
        prose,
    ])
    assert clean_extract(extract) == [prose, prose]
