import pytest

from src.attention import char_span, token_indices


def test_char_span_finds_the_one_occurrence():
    assert char_span("abc INJECT def", "INJECT") == (4, 10)


def test_char_span_rejects_missing_or_repeated_text():
    with pytest.raises(ValueError):
        char_span("abc", "INJECT")
    with pytest.raises(ValueError):
        char_span("INJECT INJECT", "INJECT")


def test_token_indices_selects_overlapping_tokens_only():
    # tokens: "abc", " IN", "JECT", " def"
    offsets = [(0, 3), (3, 6), (6, 10), (10, 14)]
    assert token_indices(offsets, (4, 10)) == [1, 2]
