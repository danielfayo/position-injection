"""One small wrapper per kind of model. Each returns the response text plus the
details needed to report exactly what was run."""

import hashlib
import os
import random
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
MAX_TOKENS = 400   # a three-sentence summary needs far fewer; this only stops runaway output


class OpenRouterModel:
    def __init__(self, model_id: str):
        from openai import OpenAI

        load_dotenv(ROOT / ".env")
        self.model_id = model_id
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.environ["OPENROUTER_API_KEY"],
            max_retries=6,
            timeout=120,
        )

    def complete(self, messages: list[dict], trial: dict) -> dict:
        # No temperature: two of the models don't accept one, so all use their default.
        response = self.client.chat.completions.create(
            model=self.model_id,
            messages=messages,
            max_tokens=MAX_TOKENS,
            extra_body={"reasoning": {"effort": "none"}, "usage": {"include": True}},
        )
        if not response.choices:
            raise RuntimeError(f"No choices returned: {response.model_dump()}")
        choice = response.choices[0]
        return {
            "content": choice.message.content or "",
            "finish_reason": choice.finish_reason,
            "served_model": response.model,
            "served_by": (response.model_extra or {}).get("provider"),
            "response_id": response.id,
            "usage": response.usage.model_dump() if response.usage else None,
        }


class LocalModel:
    """Runs the open-weights model on this Mac's GPU with its default sampling settings."""

    def __init__(self, model_id: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.torch = torch
        self.model_id = model_id
        self.device = "mps" if torch.backends.mps.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, dtype=torch.bfloat16)
        self.model.to(self.device).eval()
        # The snapshot folder name is the exact model revision that was downloaded.
        from huggingface_hub import snapshot_download
        self.revision = Path(snapshot_download(model_id, local_files_only=True)).name

    def complete(self, messages: list[dict], trial: dict) -> dict:
        inputs = self.tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, return_tensors="pt", return_dict=True,
        ).to(self.device)
        # A fixed seed per trial makes each local response reproducible.
        self.torch.manual_seed(int(hashlib.sha256(trial["trial_id"].encode()).hexdigest()[:8], 16))
        with self.torch.no_grad():
            output = self.model.generate(**inputs, max_new_tokens=MAX_TOKENS)
        new_tokens = output[0, inputs["input_ids"].shape[1]:]
        return {
            "content": self.tokenizer.decode(new_tokens, skip_special_tokens=True),
            "finish_reason": "length" if len(new_tokens) >= MAX_TOKENS else "stop",
            "served_model": self.model_id,
            "served_by": f"local ({self.device})",
            "model_revision": self.revision,
            "usage": {"prompt_tokens": inputs["input_ids"].shape[1], "completion_tokens": len(new_tokens)},
        }


class FakeModel:
    """Costs nothing: answers instantly and follows the injection about 40% of the time."""

    def __init__(self, model_id: str):
        self.model_id = model_id

    def complete(self, messages: list[dict], trial: dict) -> dict:
        rng = random.Random(trial["trial_id"] + self.model_id)
        content = "This is a fake three-sentence summary."
        if rng.random() < 0.4:
            content += f" The code is {trial['code'].lower()}."
        return {"content": content, "finish_reason": "stop", "served_model": "fake",
                "served_by": "fake", "usage": None}


def make_client(model: dict, fake: bool = False):
    if fake:
        return FakeModel(model["id"])
    if model["provider"] == "openrouter":
        return OpenRouterModel(model["id"])
    if model["provider"] == "local":
        return LocalModel(model["id"])
    raise ValueError(f"Unknown provider {model['provider']!r}")
