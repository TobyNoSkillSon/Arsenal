+++
hypothesis = "Whole-component voice tiers can be qualified with the approved BF16-source reference."
change = "Captured saved inputs on the default Load path, cast every floating source part and packed all eligible matrices."
measurement = "M5 Max busy-machine quick check, debug native worker, default streamed speech workload; internal precision qualification only."
verdict = "kept"
numbers = { qualified_tiers = 2, measured_tiers = 3, captured_components = 23, packed_matrices_per_quant_tier = 594 }
links = ["results/intake/intake-2026-10-15T20-42-05Z-599845a5-e07b-4000-8e2c-d103e0c6e5f0.json", "gate/evidence/intake/intake-2026-10-15T20-42-05Z-599845a5-e07b-4000-8e2c-d103e0c6e5f0/quality-binding.json"]
+++

# Whole-component voice intake — 15 Oct 2026

Run `intake-2026-10-15T20-42-05Z-599845a5-e07b-4000-8e2c-d103e0c6e5f0` used worker `c26487765fedf08ae488f56752e1b08f868e8c4613a8379ee507110ddb60badd`, source closure `f668d7250a76ed1bb9bc62c660b16be1ad3b29d0959eb7c68336e64eab2a3dfa`. The pipeline wrote all figures, recipes and exact quality bindings; none were hand-edited. The selected half is bf16; FP16 failed 16 of the frozen component bounds, retained in `format-trial.json`. The BF16 row is its A/A repeat, not independent fidelity proof. The approved FP32-source procedure is recorded once in `docs/ADDING-A-MODEL.md`.

| Tier | Verdict | Max rel L2 vs selected half | Min cosine vs selected half | Spectral distance (diagnostic) | RTF (compute s/audio s) | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 3.253179 | 8.382388 |
| int8 | PASS | 0.03580997 | 0.99936075 | 0.8200004 | 2.2898 | 5.186507 |
| int4 | FAIL | 0.2240733 | 0.97488015 | 0.9209195 | 2.800094 | 3.53821 |

All 3,624,819,713 source parameter/buffer values across declared archives were inspected. Each quantized tier packs 3,526,329,088 values in 594 eligible matrices; 682 tensors stay selected half or retain integer/bool buffer dtype. Skip counts: {"non-matrix layout (convolution/norm stays half)": 388, "not a Linear/Embedding weight (buffers, biases, norms stay half)": 294}. Full per-tensor reasons are in `int8-coverage.json` and `int4-coverage.json`. Eligible text/conditioning, generation and codec matrices are included. Qwen Base's compatible large ungrouped 1×1 speaker convolutions and Pocket's standalone speaker projection have registered packed adapters. Spatial/grouped/causal convolution layouts without such adapters, norms, biases, small tensors and incompatible group widths stay half where applicable. Breeze's batched heads are split into registered linears and its EOI vector is preserved.

The fixed saved-input component fixtures cover conditioning, early/middle/late generation inputs and codecs; all 15 Qwen predictor heads replay exact saved reference KV state. This is representative input coverage, not exhaustive coverage of every text, voice, token or latent. End-to-end common-prefix Hann256/hop128 spectral distance is diagnostic only: stochastic outputs can diverge and have unequal lengths. No independent matched vendor half-GPU capture was available; no vendor/ASR/listening fidelity claim is made. No download occurred.

Warm windows use balanced ABBA BAAB requests (four samples per arm for each quantized tier; eight baseline samples), ordinary production Load defaults and the shared default voice workload, English and seed 0. Timing and physical peak are measured with fresh serving workers, excluding capture-only tensors. They are debug-worker busy-Mac quick checks, not release/quiet-machine benchmark claims. Elapsed live intake: 291.487 s; hard physical budget: 48.675 GB. Existing swap was 1,023,279,104 bytes and did not grow. HF originals hash-match before/after; APFS source clones and derived lab weights are gone (`source-copy-receipt.json`).

Failed int8 is unavailable; int4 remains offered with its quality warning. BF16 and any passing quantized tier are offered by their bound recipe. No install, signing, merge or push was performed.
