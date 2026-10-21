"""Send trials to the models and save every raw response (steps 7 and 10).

Responses are appended to results/raw/<trials file>/<model>.jsonl as they arrive
and are never edited. Rerunning the same command skips trials that already have
a response, so a crash, a rate limit or Ctrl+C costs nothing. Trials that fail
after retries are logged to errors.jsonl and tried again on the next run.

Usage:
    uv run python -m src.run --trials trials_pilot.jsonl              # pilot, all models
    uv run python -m src.run --provider openrouter                    # main run, API models
    uv run python -m src.run --provider local                         # main run, Qwen on this Mac
    uv run python -m src.run --trials trials_pilot.jsonl --fake       # free dry run with a fake model
"""

import argparse
import datetime as dt
import json
import time
import traceback
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import yaml
from tqdm import tqdm

from src.clients import make_client
from src.prompts import trial_messages

ROOT = Path(__file__).resolve().parent.parent


def model_slug(model_id: str) -> str:
    return model_id.replace("/", "__")


def read_done(path: Path) -> set[str]:
    if not path.exists():
        return set()
    with path.open(encoding="utf-8") as f:
        return {json.loads(line)["trial_id"] for line in f if line.strip()}


def now() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run trials against the models.")
    parser.add_argument("--trials", default="trials.jsonl", help="trial file (default: trials.jsonl)")
    parser.add_argument("--provider", choices=["openrouter", "local"], help="only run this kind of model")
    parser.add_argument("--model", help="only run models whose id contains this text")
    parser.add_argument("--workers", type=int, default=8, help="parallel API requests (default 8)")
    parser.add_argument("--limit", type=int, help="run at most this many new trials per model")
    parser.add_argument("--fake", action="store_true", help="use a free fake model; results go to results/fake")
    parser.add_argument("--results-dir", type=Path, help="override where results are written")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    wordings = yaml.safe_load((ROOT / config["trials"]["wordings_file"]).read_text())
    documents_dir = ROOT / config["documents"]["output_dir"]
    results_dir = args.results_dir or ROOT / ("results/fake" if args.fake else "results")
    out_dir = results_dir / "raw" / Path(args.trials).stem
    out_dir.mkdir(parents=True, exist_ok=True)

    with (ROOT / args.trials).open(encoding="utf-8") as f:
        trials = [json.loads(line) for line in f]

    models = [m for m in config["models"]
              if (not args.provider or m["provider"] == args.provider)
              and (not args.model or args.model in m["id"])]
    if not models:
        raise SystemExit("No models match --provider/--model.")

    for model in models:
        out_path = out_dir / f"{model_slug(model['id'])}.jsonl"
        done = read_done(out_path)
        todo = [t for t in trials if t["model"] == model["id"] and t["trial_id"] not in done]
        if args.limit:
            todo = todo[:args.limit]
        print(f"\n{model['id']}: {len(done)} already done, {len(todo)} to run")
        if not todo:
            continue

        client = make_client(model, fake=args.fake)
        workers = 1 if model["provider"] == "local" and not args.fake else args.workers

        def run_one(trial: dict) -> dict:
            messages = trial_messages(trial, config, documents_dir, wordings)
            started, t0 = now(), time.monotonic()
            result = client.complete(messages, trial)
            return {"trial_id": trial["trial_id"], "model": trial["model"], **result,
                    "requested_at": started, "latency_s": round(time.monotonic() - t0, 2)}

        errors = 0
        pool = ThreadPoolExecutor(max_workers=workers)
        futures = {pool.submit(run_one, trial): trial for trial in todo}
        try:
            with out_path.open("a", encoding="utf-8") as out:
                for future in tqdm(as_completed(futures), total=len(futures), desc=model["id"]):
                    trial = futures[future]
                    try:
                        record = future.result()
                    except Exception as error:
                        errors += 1
                        with (out_dir / "errors.jsonl").open("a", encoding="utf-8") as log:
                            log.write(json.dumps({"trial_id": trial["trial_id"], "model": model["id"],
                                                  "at": now(), "error": repr(error),
                                                  "traceback": traceback.format_exc()}) + "\n")
                        continue
                    out.write(json.dumps(record) + "\n")
                    out.flush()
        except KeyboardInterrupt:
            pool.shutdown(wait=False, cancel_futures=True)
            raise SystemExit("\nStopped. Run the same command again to carry on.")
        pool.shutdown()
        if errors:
            print(f"  {errors} trial(s) failed and were logged to {out_dir / 'errors.jsonl'}; "
                  "rerun to retry them.")


if __name__ == "__main__":
    main()
