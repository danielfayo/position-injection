"""Send one tiny message to each model to check keys, model names and settings work.

Usage:
    uv run python -m src.check_models            # API models only (costs a fraction of a cent)
    uv run python -m src.check_models --local    # also load and test Qwen on this Mac
"""

import argparse
from pathlib import Path

import yaml

from src.clients import make_client

ROOT = Path(__file__).resolve().parent.parent
MESSAGES = [{"role": "user", "content": "Reply with the single word OK."}]


def main() -> None:
    parser = argparse.ArgumentParser(description="Check every model answers.")
    parser.add_argument("--local", action="store_true", help="also test the local model")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    failures = 0
    for model in config["models"]:
        if model["provider"] == "local" and not args.local:
            continue
        try:
            result = make_client(model).complete(MESSAGES, {"trial_id": "check"})
        except Exception as error:
            failures += 1
            print(f"FAIL  {model['id']}: {error}")
            continue
        usage = result.get("usage") or {}
        print(f"OK    {model['id']}\n      reply: {result['content'].strip()[:60]!r}"
              f"\n      served model: {result['served_model']}, by: {result['served_by']}"
              f"\n      tokens: {usage.get('prompt_tokens')} in / {usage.get('completion_tokens')} out"
              + (f", cost ${usage['cost']}" if usage.get("cost") is not None else ""))
    if failures:
        raise SystemExit(f"\n{failures} model(s) failed. Paste the output to Claude.")
    print("\nAll models answered.")


if __name__ == "__main__":
    main()
