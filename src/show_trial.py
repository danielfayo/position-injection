"""Print the exact prompt for one trial, with the injection marked, for spot checks.

Usage:
    uv run python -m src.show_trial T00001
    uv run python -m src.show_trial P00003 --file trials_pilot.jsonl
"""

import argparse
import json
from pathlib import Path

import yaml

from src.prompts import trial_messages

ROOT = Path(__file__).resolve().parent.parent


def main() -> None:
    parser = argparse.ArgumentParser(description="Show the prompt for one trial.")
    parser.add_argument("trial_id")
    parser.add_argument("--file", default="trials.jsonl")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    wordings = yaml.safe_load((ROOT / config["trials"]["wordings_file"]).read_text())
    with (ROOT / args.file).open(encoding="utf-8") as f:
        trial = next((t for t in map(json.loads, f) if t["trial_id"] == args.trial_id), None)
    if trial is None:
        raise SystemExit(f"No trial {args.trial_id} in {args.file}")

    messages = trial_messages(trial, config, ROOT / config["documents"]["output_dir"], wordings)
    injection = wordings[trial["wording"]].format(CODE=trial["code"])
    print(json.dumps(trial, indent=2))
    for message in messages:
        print(f"\n===== {message['role'].upper()} =====")
        # The markers are for display only; the model never sees them.
        print(message["content"].replace(injection, f">>>>> {injection} <<<<<"))


if __name__ == "__main__":
    main()
