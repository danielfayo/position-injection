"""Write every trial to a file before any are run (step 4 of the implementation plan).

Main run: each document × position × wording gets `codes_per_cell` trials, each
with its own random code, and every such item is run on every model. Pilot: a
separate, smaller set with its own codes, used only for tuning. Each model's
trials are shuffled, so position is not tied to when a trial runs.

Usage:
    uv run python -m src.build_trials
"""

import json
import random
import sys
from itertools import product
from pathlib import Path

import yaml
from transformers import AutoTokenizer

from src.prompts import load_document, position_slots, slot_token_offsets

ROOT = Path(__file__).resolve().parent.parent
CODE_ALPHABET = "ABCDEFGHJKLMNPQRSTUVWXYZ23456789"   # no 0/O or 1/I, which are easy to confuse


def new_code(rng: random.Random, used: set[str]) -> str:
    while True:
        chars = "".join(rng.choice(CODE_ALPHABET) for _ in range(7))
        code = f"{chars[:3]}-{chars[3:]}"
        if code not in used:
            used.add(code)
            return code


def expand(items: list[dict], models: list[dict], rng: random.Random, prefix: str) -> list[dict]:
    """One trial per item per model, shuffled within each model."""
    trials = []
    for model in models:
        rows = [{**item, "model": model["id"], "provider": model["provider"]} for item in items]
        rng.shuffle(rows)
        trials.extend(rows)
    for number, trial in enumerate(trials, start=1):
        trial["trial_id"] = f"{prefix}{number:05d}"
    return trials


def write_jsonl(path: Path, rows: list[dict]) -> None:
    keys = ["trial_id", "model", "provider", "item_id", "doc_id", "position", "slot",
            "rel_position", "wording", "rep", "code"]
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps({key: row[key] for key in keys}) + "\n")


def main() -> None:
    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    trial_config = config["trials"]
    out_path = ROOT / trial_config["output"]
    pilot_path = ROOT / trial_config["pilot_output"]
    if out_path.exists():
        sys.exit(f"{out_path.name} already exists. Delete it first if you really mean to rebuild "
                 "the trials (never after the main run has started).")

    rng = random.Random(config["seed"])
    tokenizer = AutoTokenizer.from_pretrained(config["documents"]["tokenizer"])
    count_tokens = lambda text: len(tokenizer.encode(text, add_special_tokens=False))
    wordings = yaml.safe_load((ROOT / trial_config["wordings_file"]).read_text())
    n_positions = trial_config["positions"]

    # Where each position falls in each document.
    documents_dir = ROOT / config["documents"]["output_dir"]
    placements = {}
    for path in sorted(documents_dir.glob("doc_*.txt")):
        offsets = slot_token_offsets(load_document(path), count_tokens)
        slots = position_slots(offsets, n_positions)
        placements[path.stem] = [(slot, round(offsets[slot] / offsets[-1], 4)) for slot in slots]
    print(f"{len(placements)} documents, {n_positions} positions, wordings {sorted(wordings)}")

    # Pilot files that have already been run are kept as they are; their codes
    # are never reused in the main run.
    used_codes: set[str] = set()
    for existing in ROOT.glob("trials_pilot*.jsonl"):
        with existing.open(encoding="utf-8") as f:
            used_codes |= {json.loads(line)["code"] for line in f}
    items = []
    for doc_id, position, wording, rep in product(
        sorted(placements), range(1, n_positions + 1), sorted(wordings),
        range(1, trial_config["codes_per_cell"] + 1),
    ):
        slot, rel_position = placements[doc_id][position - 1]
        items.append({
            "item_id": f"I{len(items) + 1:05d}", "doc_id": doc_id, "position": position,
            "slot": slot, "rel_position": rel_position, "wording": wording, "rep": rep,
            "code": new_code(rng, used_codes),
        })

    pilot_items = []
    for position in range(1, n_positions + 1):
        for rep in range(1, trial_config["pilot_per_position"] + 1):
            doc_id, wording = rng.choice(sorted(placements)), rng.choice(sorted(wordings))
            slot, rel_position = placements[doc_id][position - 1]
            pilot_items.append({
                "item_id": f"P{len(pilot_items) + 1:05d}", "doc_id": doc_id, "position": position,
                "slot": slot, "rel_position": rel_position, "wording": wording, "rep": rep,
                "code": new_code(rng, used_codes),
            })

    trials = expand(items, config["models"], rng, prefix="T")
    pilot = expand(pilot_items, config["models"], rng, prefix="P")
    write_jsonl(out_path, trials)
    print(f"Wrote {len(trials)} main trials to {out_path.name} "
          f"({len(items)} items × {len(config['models'])} models)")
    if pilot_path.exists():
        print(f"Kept the existing {pilot_path.name}")
    else:
        write_jsonl(pilot_path, pilot)
        print(f"Wrote {len(pilot)} pilot trials to {pilot_path.name}")


if __name__ == "__main__":
    main()
