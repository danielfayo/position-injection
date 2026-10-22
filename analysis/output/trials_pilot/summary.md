# Analysis summary: trials_pilot.jsonl

## RQ1: shape of the positional effect, per model
- **Qwen/Qwen2.5-3B-Instruct**: inconclusive. Success by position (%): 0, 0, 0, 0, 0, 0, 50. Omnibus p = 0.0775; straight-line change start to end +32.3 points (p = 0.0972); two-lines U test: slopes -65.8 / +42.1 points per document length (p = 0.103 / 0.0895, break at 2); within ±5 points everywhere: False. Excluded wordings: ['B']; n = 34.
- **anthropic/claude-haiku-5.5-20261007**: falling. Success by position (%): 50, 0, 20, 10, 0, 0, 0. Omnibus p = 0.0506; straight-line change start to end -36.8 points (p = 0.000462); two-lines U test: slopes -247.4 / -11.9 points per document length (p = 0.00443 / 0.0934, break at 2); within ±5 points everywhere: False. Excluded wordings: none; n = 70.
- **google/gemini-3.1-flash-lite-20260507**: inconclusive. Success by position (%): 33, 50, 89, 80, 50, 80, 100. Omnibus p = 0.017; straight-line change start to end +32.0 points (p = 0.254); two-lines U test: slopes +235.7 / +11.6 points per document length (p = 0.285 / 0.649, break at 2); within ±5 points everywhere: False. Excluded wordings: ['B']; n = 34.
- **openai/gpt-6-luna-20260922**: inconclusive. Success by position (%): 43, 62, 100, 40, 25, 40, 33. Omnibus p = 2.33e-09; straight-line change start to end -22.0 points (p = 0.455); two-lines U test: slopes -27.4 / -8.1 points per document length (p = 0.543 / 0.934, break at 5); within ±5 points everywhere: False. Excluded wordings: ['A']; n = 36.

## RQ2: do the models differ?
- Joint test of model × position terms: p = 0.00239 (differ).

## Robustness: verified refusals recoded as failures
- analysis/refusals_verified.csv not found (hand-check flagged responses first).

## Robustness: verdict per wording
- Qwen/Qwen2.5-3B-Instruct: A: inconclusive, B: not analysable
- anthropic/claude-haiku-5.5-20261007: A: falling, B: falling
- google/gemini-3.1-flash-lite-20260507: A: inconclusive, B: not analysable
- openai/gpt-6-luna-20260922: A: not analysable, B: inconclusive
