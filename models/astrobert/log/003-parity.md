+++
hypothesis = "The kit runner reproduces the maker's transformers embeddings for adsabs/astroBERT at FP16."
change = "None (verification only)."
measurement = "transformers BertModel on MPS fp16 vs the MLX kit runner, four fixed texts, mean-pooled last hidden state, 2026-10-30, m5-max."
verdict = "kept"
numbers = { min_cosine = 0.999992, max_abs_diff = 0.00778 }
links = []
+++
# 003 Parity with the maker's code (2026-10-30)

Method: the maker's Tutorial 0 path in transformers (AutoModel = BertModel, tokenizer with do_lower_case=False), MPS at
float16, one unpadded sequence per text, last_hidden_state mean-pooled over all tokens. MLX: this folder's kit runner,
driven over its wire with `normalize: false`, same texts. Both run under the kit GPU lock on m5-max. Scripts and raw
vectors: lab/night1/astrobert/ (ref_torch.py, run_mlx.py, compare.py, inputs.json, ref.json, mlx.json, mlx-fp16.json).

Input: four texts. Two similar ("M67 is one of the most studied open clusters." / "M67 is a well-studied old open star
cluster in Cancer."), one unrelated (a baking sentence), and one awkward text with curly quotes, an em dash, accented
names and a newline, to exercise the tokenizer's accent stripping and whitespace cleaning.

Counts: 4 vectors of 768 each side; input tokens 88 on both sides.

| text | cosine | max abs diff (ref max abs value) |
|---|---|---|
| M67 #1 | 0.999992 | 0.0078 (6.85) |
| M67 #2 | 0.999995 | 0.0071 (7.19) |
| baking | 0.999997 | 0.0058 (7.24) |
| awkward | 0.999998 | 0.0039 (5.51) |

The same numbers hold after the FP16 cast of the stored weights (the runner already cast to FP16 at load).
Pairwise cosines agree to 0.0002 (M67 pair 0.876, M67 vs baking 0.677 and 0.713).

Verdict: pass (threshold cosine >= 0.999).

Port note: swift-transformers' BertTokenizer does not drop control characters in clean_text; Model.swift cleans text
first (control/format characters removed, all whitespace including newlines to spaces), so the token ids match.
