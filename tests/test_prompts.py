from src.prompts import build_messages, insert_injection, position_slots, render, slot_token_offsets

DOCUMENT = [
    ["One a.", "One b.", "One c."],
    ["Two a.", "Two b."],
    ["Three a.", "Three b.", "Three c."],
]
INJECTION = "INJECTED."
count_words = lambda text: len(text.split())


def test_slot_zero_puts_the_injection_first():
    assert insert_injection(DOCUMENT, 0, INJECTION).startswith("INJECTED. One a.")


def test_last_slot_puts_the_injection_last():
    assert insert_injection(DOCUMENT, 8, INJECTION).endswith("Three c. INJECTED.")


def test_middle_slot_goes_between_sentences_and_keeps_everything_else():
    text = insert_injection(DOCUMENT, 4, INJECTION)
    assert "Two a. INJECTED. Two b." in text
    assert text.replace(" INJECTED.", "") == render(DOCUMENT)


def test_insertion_does_not_change_the_original_document():
    insert_injection(DOCUMENT, 4, INJECTION)
    assert DOCUMENT[1] == ["Two a.", "Two b."]


def test_positions_run_from_first_to_last_slot_in_order():
    offsets = slot_token_offsets(DOCUMENT, count_words)
    assert offsets[0] == 0 and offsets[-1] == 16 and len(offsets) == 9
    slots = position_slots(offsets, 7)
    assert slots[0] == 0 and slots[-1] == 8
    assert slots == sorted(slots)


def test_messages_contain_the_document():
    config = {"system": "Be helpful.", "user_template": "Summarise:\n{document}"}
    messages = build_messages(config, "Some text.")
    assert messages == [
        {"role": "system", "content": "Be helpful."},
        {"role": "user", "content": "Summarise:\nSome text."},
    ]
