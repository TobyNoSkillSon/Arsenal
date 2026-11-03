+++
hypothesis = "A shared source-rounding calibration selects half format; matrix-level retention repairs int8 without retaining whole components."
change = "Shared freezeFP32Source policy and intake-v3 matrix recipes with iterative ablation and complete stored-recipe replay."
measurement = "M5 Max, busy Mac, debug native worker; default Load and streamed speech, fresh workers for figures."
verdict = "kept"
numbers = { int8_extra_half_values = 50331648, int8_matrices = 386, int4_matrices = 393, components = 101, diagnosis_trials = 116 }
links = ["results/intake/intake-2026-10-15T21-53-15Z-93c9adfb-ce67-422b-b91a-898e49246579.json", "gate/evidence/intake/intake-2026-10-15T21-53-15Z-93c9adfb-ce67-422b-b91a-898e49246579/qualification-summary.json"]
+++

# Shared source-floor and matrix fallback — 15 Oct 2026 UTC

Selected **BF16**. The shared `fp32-source-bf16-nearest-truncation-fp16-ulp-v1` calibration was newly frozen before deriving/executing FP16. BF16 nearest and source-weight truncation run on identical immutable inputs; exact A/A stays separate. Historical A/A-floor outcomes remain in their original runs. Frozen bounds SHA256 `78eea1fe6c0ee7cfab07a2d866ef2071603497b7a77c3ed8d04557b159814cca`; final native worker `ec8d97def2e72b4b89b8b807f2658bcd2c3431c44e97402546547ce941ee740c`. No FP32 model executes and no learned FP32 weights remain in the derived snapshots.

| Component | Source perturbation floor (rel L2) | Floor cosine | Frozen FP16 bound (rel L2) |
|---|---:|---:|---:|
| codec-decoder | 0.0349174646 | 0.999649363 | 0.0705450376 |
| codec-downsample | 0.00469536773 | 0.999992572 | 0.0101149742 |
| codec-encoder | 0.179006313 | 0.983970623 | 0.358716045 |
| codec-encoder-transformer | 0.0230111453 | 0.999736539 | 0.0467056783 |

The other 97 components have zero measured nearest/truncation floor because their original stored weights were already BF16 or the perturbation did not change this output. Their bounds still include one FP16 ULP; see every component's floor, cosine and ULP in `bounds.json`. Every A/A relative L2 is zero. The rejected FP16 trial's maximum relative L2 is 0.494373926, minimum cosine 0.878977491, failing components 52/101. It reports 2 output zeros and 567 subnormal output values where BF16 was normal. These are output diagnostics, not a proof about all intermediate underflows.

| Tier | Verdict | Max rel L2 vs selected half | Min cosine vs selected half | Spectral distance (diagnostic) | RTF (compute s/audio s) | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 0.5469341 | 4.820275 |
| int8 | PASS | 0.04948302 | 0.99904251 | 0.8535501 | 0.4879679 | 3.003354 |
| int4 | FAIL | 1.078847 | 0.62152726 | 1.053623 | 0.4995909 | 1.966 |

## Matrix fallback

- `matrix@model.safetensors:talker.code_predictor.model.layers.3.mlp.gate_proj.weight` — 3,145,728 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.3.mlp.up_proj.weight` — 3,145,728 parameters.
- `matrix@model.safetensors:talker.code_predictor.small_to_mtp_projection.weight` — 2,097,152 parameters.
- `matrix@model.safetensors:talker.model.layers.2.mlp.down_proj.weight` — 12,582,912 parameters.
- `matrix@model.safetensors:talker.model.layers.27.mlp.gate_proj.weight` — 12,582,912 parameters.
- `matrix@model.safetensors:talker.model.layers.27.mlp.up_proj.weight` — 12,582,912 parameters.
- `matrix@model.safetensors:talker.model.layers.4.self_attn.o_proj.weight` — 4,194,304 parameters.

Extra 16-bit fallback is 50,331,648/2,099,234,881 source parameter/buffer values (2.39762%). Total 16-bit retention including required structural skips is 7.11101%. The denominator includes presets treated as weights and buffers; it is not a trainable-only parameter count. Int8 packs 1,949,958,144 values in 386 matrices; int4 packs 2,000,289,792 in 393 matrices. Every skip has a tensor-level reason in coverage. Measured int8 peak saving versus the selected half is 1.816921GB (37.693%). Per-owner retention of eligible parameters: {"predictor": {"eligible_parameters": 112197632, "eligible_share": 0.07476635514018691, "fallback_parameters": 8388608}, "talker": {"eligible_parameters": 1415577600, "eligible_share": 0.02962962962962963, "fallback_parameters": 41943040}}. Any high share is a limitation of this locally irreducible recipe; it does not prove every possible subset needs that share.

The initial fully packed int8 gate is preserved. Diagnosis starts with failing owners' eligible matrices retained and all other owners packed. It re-quantizes groups, then single matrices; a final single-matrix sweep continues until no removal passes. Each retained matrix's removal failed the fixed component gate with the other final fallbacks held fixed. `matrix-attribution.json` retains trial selections, worst metrics and exact failed component metrics; full raw captures and gates remain local. This proves local irreducibility on these fixtures, not a global optimum or general perceptual equivalence. Passing codec QKV packs need no source-to-runtime ablation mapping. The final newly derived stored recipe passes the complete 101-component replay under unchanged ≤0.05 relative L2 / ≥0.999 cosine, finite/range/no-new-clamp bounds. Scalar packing/dequantization audits are exact on the first four rows/all groups of every eligible matrix.

All figure workers are fresh, with zero component-capture requests; PIDs are in the result. RTF is compute seconds per generated audio second (lower is faster), and spectral distance is the unaligned common-prefix magnitude-spectrum diagnostic, with sample counts retained. It does not select format or quantization. Pocket's existing round-3 SEANet-input boundary is unchanged in this round.

Live intake 346.144s; hard aggregate worker limit 48.108GB, sampled aggregate worker peak 11.414GB. Preexisting swap 1,023,279,104bytes did not grow. Strict system-wide zero swap was not present before the run. No downloads; APFS clones hash-checked before/after and deleted, including derived and rounding copies. Source-copy receipt is retained. Figures are busy-Mac debug-worker quick checks. Independent vendor half-GPU comparison, listening and ASR checks remain unavailable. No main merge, push, signing or install.
