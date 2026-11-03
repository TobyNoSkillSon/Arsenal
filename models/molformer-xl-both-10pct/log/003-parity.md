+++
hypothesis = "The MLX kit runner matches the maker's MolformerModel on the same checkpoint."
change = "None: measurement of the Model.swift port on the original FP32 files (the runner casts to FP16 at load)."
measurement = "lab/night2/molformer-xl-both-10pct/ref_parity.py vs runjson.py replies on m5-max, 2026-10-31: 5 molecules, pooled embedding."
verdict = "kept"
numbers = { pooled_cosine_min = 0.999986, pooled_max_abs_diff = 0.0187 }
links = []
+++
# 003 Parity with the maker's MolformerModel (2026-10-31)

Method: the repository's own modeling_molformer.py and configuration_molformer.py (copied as text from the pinned
revision), `MolformerModel` with `deterministic_eval=True` as the card's example, transformers 5.22.0 and PyTorch 2.15.0
on MPS, against the kit runner (MLX, FP16 weights) on the same checkpoint, M5 Max. Both sides take `pooler_output`:
the final LayerNorm output averaged over every token, <bos> and <eos> included. Token counts matched on every input
(PreTrainedTokenizerFast from tokenizer.json on the reference side, the runner's regex + vocabulary on the MLX side).

Reference precision: float32. The maker's code cannot run float16: on aspirin, layer 0's relu random features reach
189 and their sum over tokens about 1,064, so the linear-attention normaliser q . sum(k) overflows float16's 65,504 and
the output is NaN from layer 1 on (probe: lab/night2/molformer-xl-both-10pct/probe.py and probe2.py). The MLX port
keeps weights and activations in float16 and computes only the linear-attention sums and normaliser in float32.

Scripts: lab/night2/molformer-xl-both-10pct/ref_parity.py; inputs: inputs.json there; MLX replies: mlx.json.

| input | tokens | cosine | max abs diff |
|---|---|---|---|
| caffeine Cn1c(=O)c2c(ncn2C)n(C)c1=O | 26 | 1.000000 | 0.0018 |
| aspirin CC(=O)Oc1ccccc1C(=O)O | 21 | 1.000000 | 0.0030 |
| theobromine Cn1cnc2c1c(=O)[nH]c(=O)n2C | 23 | 1.000000 | 0.0021 |
| ibuprofen CC(C)Cc1ccc(cc1)C(C)C(=O)O | 26 | 0.999999 | 0.0027 |
| methane C | 1 | 0.999986 | 0.0187 |

Embedding values reach about 6 to 9, so these differences are a few float16 steps.
Sense check (both sides agree to 0.0002): cosine caffeine vs theobromine 0.850, caffeine vs aspirin 0.508,
aspirin vs ibuprofen 0.681.

Verdict: pass (cosine >= 0.999 on every input).
