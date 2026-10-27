# Analysis summary: trials.jsonl

## RQ1: shape of the positional effect, per model
- **Qwen/Qwen2.5-3B-Instruct**: U-shaped. Success by position (%): 26, 9, 1, 9, 9, 15, 78. Omnibus p = 6.61e-158; straight-line change start to end +37.5 points (p = 5.22e-20); two-lines U test: slopes -108.2 / +95.8 points per document length (p = 5.15e-21 / 1.13e-127, break at 3); within ±10 points everywhere: False. Excluded wordings: ['B']; n = 1260.
- **anthropic/claude-haiku-5.5-20261007**: U-shaped. Success by position (%): 61, 28, 24, 23, 21, 17, 34. Omnibus p = 1.49e-36; straight-line change start to end -22.8 points (p = 4.17e-09); two-lines U test: slopes -41.8 / +136.1 points per document length (p = 9.67e-20 / 5.92e-09, break at 6); within ±10 points everywhere: False. Excluded wordings: none; n = 1890.
- **google/gemini-3.1-flash-lite-20260507**: inverted U. Success by position (%): 43, 88, 95, 89, 78, 82, 98. Omnibus p = 1.08e-31; straight-line change start to end +28.8 points (p = 4.43e-11); two-lines U test: slopes +36.8 / +8.8 points per document length (p = 8.66e-07 / 0.341, break at 5); within ±10 points everywhere: False. Excluded wordings: ['B']; n = 1260.
- **openai/gpt-6-luna-20260922**: falling. Success by position (%): 30, 14, 24, 20, 23, 16, 14. Omnibus p = 2.24e-06; straight-line change start to end -9.8 points (p = 0.0121); two-lines U test: slopes -60.2 / -3.7 points per document length (p = 0.00218 / 0.402, break at 2); within ±10 points everywhere: False. Excluded wordings: ['A']; n = 1260.

## RQ2: do the models differ?
- Joint test of model × position terms: p = 4.35e-121 (differ).

## RQ3: attention (Qwen)
- Attention predicts success: odds ratio 436.26 per SD of log attention (p = 7.13e-12); H3 supported.
- Shape of attention across positions: U-shaped.
- Position terms without / with attention: x +2.62 / -4.03, x² +16.11 / -6.75.
- Exploratory: attention to the user's task, odds ratio 0.16 per SD (p = 6.46e-11).

## Robustness: verified refusals recoded as failures
- Qwen/Qwen2.5-3B-Instruct: verdict: U-shaped, recoded: 0
- anthropic/claude-haiku-5.5-20261007: verdict: U-shaped, recoded: 98
- google/gemini-3.1-flash-lite-20260507: verdict: inverted U, recoded: 0
- openai/gpt-6-luna-20260922: verdict: falling, recoded: 0

## Robustness: verdict per wording
- Qwen/Qwen2.5-3B-Instruct: A: U-shaped, B: not analysable, C: U-shaped
- anthropic/claude-haiku-5.5-20261007: A: U-shaped, B: U-shaped, C: inconclusive
- google/gemini-3.1-flash-lite-20260507: A: rising, B: not analysable, C: rising
- openai/gpt-6-luna-20260922: A: not analysable, B: falling, C: rising
