"""Placing the injection in a document and building the messages sent to a model.

A document is a list of paragraphs, each a list of sentences. The injection can
go at any sentence boundary ("slot"): slot 0 is before the first sentence and
slot N is after the last of the document's N sentences. Positions 1–7 are mapped
to the slots whose token offsets are closest to 0, 1/6, ..., 6/6 of the document.

The runner rebuilds every prompt from a trial row with these functions, so the
trial list only needs to store the slot, not the full text.
"""

from pathlib import Path

from src.text import split_sentences

Document = list[list[str]]


def load_document(path: Path) -> Document:
    text = path.read_text(encoding="utf-8").strip()
    return [split_sentences(paragraph) for paragraph in text.split("\n\n")]


def render(document: Document) -> str:
    return "\n\n".join(" ".join(sentences) for sentences in document if sentences)


def _flat(document: Document) -> list[tuple[int, int]]:
    """(paragraph index, sentence index) for every sentence, in reading order."""
    return [(p, s) for p, sentences in enumerate(document) for s in range(len(sentences))]


def slot_token_offsets(document: Document, count_tokens) -> list[int]:
    """Number of document tokens before each slot (one entry per slot, N + 1 in all)."""
    sentences = [document[p][s] for p, s in _flat(document)]
    offsets = [0]
    for j in range(1, len(sentences) + 1):
        offsets.append(count_tokens(" ".join(sentences[:j])))
    return offsets


def position_slots(offsets: list[int], n_positions: int) -> list[int]:
    """The slot nearest to each of n evenly spaced fractions of the document."""
    total = offsets[-1]
    slots = []
    for k in range(n_positions):
        target = total * k / (n_positions - 1)
        slots.append(min(range(len(offsets)), key=lambda j: abs(offsets[j] - target)))
    return slots


def insert_injection(document: Document, slot: int, injection: str) -> str:
    """Return the document text with `injection` added as a sentence at `slot`."""
    flat = _flat(document)
    new = [list(sentences) for sentences in document]
    if slot == len(flat):                       # after the last sentence
        new[-1].append(injection)
    else:                                       # before sentence number `slot`
        p, s = flat[slot]
        new[p].insert(s, injection)
    return render(new)


def build_messages(prompt_config: dict, document_text: str) -> list[dict]:
    return [
        {"role": "system", "content": prompt_config["system"]},
        {"role": "user", "content": prompt_config["user_template"].format(document=document_text)},
    ]


def trial_messages(trial: dict, config: dict, documents_dir: Path, wordings: dict) -> list[dict]:
    """Rebuild the exact messages for one trial row."""
    document = load_document(documents_dir / f"{trial['doc_id']}.txt")
    injection = wordings[trial["wording"]].format(CODE=trial["code"])
    return build_messages(config["prompt"], insert_injection(document, trial["slot"], injection))
