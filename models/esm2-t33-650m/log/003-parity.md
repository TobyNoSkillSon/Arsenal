+++
hypothesis = "The MLX kit runner matches transformers EsmModel (MPS, float16) on the same checkpoint."
change = "None: measurement of the Model.swift port before the FP16 cast."
measurement = "lab/night1/esm2/ref_parity.py vs mlx_dump.py on m5-max, 2026-10-30: 4 sequences, pooled and per-residue final hidden states."
verdict = "kept"
numbers = { pooled_cosine_min = 0.999979, last_hidden_max_abs_diff = 0.031 }
links = []
+++
# 003 Parity with transformers EsmModel (2026-10-30)

Method: transformers 5.22.0 `EsmModel` (no pooler, sdpa attention) on MPS at float16, PyTorch 2.15.0, against the kit
runner (MLX, FP16) on the same checkpoint (original FP32 files, before the FP16 cast), M5 Max. Both sides take the final
hidden state (after emb_layer_norm_after); the sequence embedding is its mean over residues, excluding <cls> and <eos>.
Token ids matched (EsmTokenizer on the reference side, the runner's vocab.txt tokenizer on the MLX side).
Scripts: lab/night1/esm2/ref_parity.py and mlx_dump.py; inputs: lab/night1/esm2/inputs.json. Same Model.swift as esm2-t6-8m.

| input | pooled cosine | pooled max abs diff | last hidden max abs diff | min per-residue cosine |
|---|---|---|---|---|
| human preproinsulin residues 1-60 | 0.999999 | 0.0060 | 0.031 | 0.999984 |
| same with D60E | 0.999999 | 0.0027 | 0.012 | 0.999990 |
| avGFP residues 1-60 | 1.000000 | 0.00052 | 0.0078 | 0.999998 |
| "M" (one residue) | 0.999979 | 0.012 | 0.012 | 0.999979 |

Hidden values reach about 8.5 in magnitude, where one float16 step is 0.0078; 33 layers of float16 on two different
backends put the largest difference at four steps.
Sense check (both sides agree): cosine insulin vs its D60E mutant 0.9998, insulin vs GFP 0.819.

Verdict: pass (pooled cosine >= 0.999 on every input).
