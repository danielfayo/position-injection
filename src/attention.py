"""Measure how much attention Qwen gives the injected sentence (step 6, RQ3).

For each Qwen trial, runs the exact prompt through the model once, without
generating, and reads the attention from the last prompt token, the one that
produces the first word of the reply. For every layer it records, averaged over
heads, the attention on:

  - the injected sentence (summed over its tokens): the main measure
  - the user's task instruction ("Summarise the following document..."), since
    successful injections are known to pull attention away from it
  - the rest of the document, per token, to put the injection's share in context

Results are appended to results/attention/<trials file>.jsonl and the run
resumes like the main runner.

Usage:
    uv run python -m src.attention --limit 3       # quick check
    uv run python -m src.attention                 # every Qwen trial
"""

import argparse
import json
import time
from pathlib import Path

import torch
import yaml
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from src.prompts import trial_messages
from src.run import read_done

ROOT = Path(__file__).resolve().parent.parent


def char_span(text: str, part: str) -> tuple[int, int]:
    start = text.find(part)
    if start < 0 or text.find(part, start + 1) >= 0:
        raise ValueError(f"Expected exactly one occurrence of {part[:40]!r} in the prompt")
    return start, start + len(part)


def token_indices(offsets: list[tuple[int, int]], span: tuple[int, int]) -> list[int]:
    """Tokens whose characters overlap the character span."""
    return [i for i, (a, b) in enumerate(offsets) if a < span[1] and b > span[0] and b > a]


class AttentionReader:
    def __init__(self, model_id: str):
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        # "eager" attention computes the attention weights explicitly; faster kernels don't expose them.
        self.model = AutoModelForCausalLM.from_pretrained(
            model_id, dtype=torch.bfloat16, attn_implementation="eager",
        ).to(self.device).eval()
        self.last_rows: list[torch.Tensor] = []
        for layer in self.model.model.layers:
            layer.self_attn.register_forward_hook(self._keep_last_row)

    def _keep_last_row(self, module, inputs, output) -> None:
        # output[1] is (batch, heads, query, key); keep only the last query position.
        self.last_rows.append(output[1][0, :, -1, :].float().cpu())

    def measure(self, messages: list[dict], injection: str, task: str, document: str) -> dict:
        prompt = self.tokenizer.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
        encoded = self.tokenizer(prompt, add_special_tokens=False, return_offsets_mapping=True)
        offsets = encoded["offset_mapping"]
        injection_tokens = token_indices(offsets, char_span(prompt, injection))
        task_tokens = token_indices(offsets, char_span(prompt, task))
        document_tokens = token_indices(offsets, char_span(prompt, document))
        other_document_tokens = sorted(set(document_tokens) - set(injection_tokens))

        self.last_rows = []
        with torch.no_grad():
            self.model(input_ids=torch.tensor([encoded["input_ids"]], device=self.device))
        rows = torch.stack(self.last_rows).mean(dim=1)   # (layers, keys), averaged over heads

        per_layer_injection = rows[:, injection_tokens].sum(dim=1)
        per_layer_task = rows[:, task_tokens].sum(dim=1)
        per_layer_document = rows[:, other_document_tokens].mean(dim=1)
        injection_per_token = per_layer_injection.mean().item() / len(injection_tokens)
        return {
            "n_prompt_tokens": len(encoded["input_ids"]),
            "injection_token_span": [injection_tokens[0], injection_tokens[-1]],
            "n_injection_tokens": len(injection_tokens),
            "attn_injection": per_layer_injection.mean().item(),
            "attn_injection_per_token": injection_per_token,
            "attn_document_per_token": per_layer_document.mean().item(),
            "attn_injection_relative": injection_per_token / per_layer_document.mean().item(),
            "attn_task": per_layer_task.mean().item(),
            "row_sum_check": rows.sum(dim=1).mean().item(),   # should be ~1.0
            "per_layer_injection": [round(x, 6) for x in per_layer_injection.tolist()],
            "per_layer_task": [round(x, 6) for x in per_layer_task.tolist()],
        }


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure attention to the injection in Qwen.")
    parser.add_argument("--trials", help="trial file (default: from config)")
    parser.add_argument("--limit", type=int, help="measure at most this many new trials")
    parser.add_argument("--results-dir", type=Path, help="override where results are written")
    args = parser.parse_args()

    config = yaml.safe_load((ROOT / "config" / "experiment.yaml").read_text())
    attention_config = config["attention"]
    trials_file = args.trials or attention_config["trials"]
    wordings = yaml.safe_load((ROOT / config["trials"]["wordings_file"]).read_text())
    documents_dir = ROOT / config["documents"]["output_dir"]
    task = config["prompt"]["user_template"].split("{document}")[0].split("\n")[0]

    out_dir = (args.results_dir or ROOT / "results") / "attention"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{Path(trials_file).stem}.jsonl"
    done = read_done(out_path)

    with (ROOT / trials_file).open(encoding="utf-8") as f:
        trials = [json.loads(line) for line in f]
    todo = [t for t in trials if t["model"] == attention_config["model"] and t["trial_id"] not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"{len(done)} already measured, {len(todo)} to measure")
    if not todo:
        return

    reader = AttentionReader(attention_config["model"])
    with out_path.open("a", encoding="utf-8") as out:
        for trial in tqdm(todo):
            messages = trial_messages(trial, config, documents_dir, wordings)
            injection = wordings[trial["wording"]].format(CODE=trial["code"])
            user = messages[-1]["content"]
            document = user.split("<document>\n", 1)[1].rsplit("\n</document>", 1)[0]
            t0 = time.monotonic()
            result = reader.measure(messages, injection, task, document)
            out.write(json.dumps({"trial_id": trial["trial_id"], **result,
                                  "seconds": round(time.monotonic() - t0, 2)}) + "\n")
            out.flush()


if __name__ == "__main__":
    main()
