+++
hypothesis = "A shared source-rounding calibration selects half format; matrix-level retention repairs int8 without retaining whole components."
change = "Shared freezeFP32Source policy and intake-v3 matrix recipes with iterative ablation and complete stored-recipe replay."
measurement = "M5 Max, busy Mac, debug native worker; default Load and streamed speech, fresh workers for figures."
verdict = "kept"
numbers = { int8_extra_half_values = 16777216, int8_matrices = 62, int4_matrices = 66, components = 19, diagnosis_trials = 32 }
links = ["results/intake/intake-2026-10-15T22-22-05Z-5b6679bb-6ec6-47d6-90ef-01939a0c51b6.json", "gate/evidence/intake/intake-2026-10-15T22-22-05Z-5b6679bb-6ec6-47d6-90ef-01939a0c51b6/qualification-summary.json"]
+++

# Shared source-floor and matrix fallback — 15 Oct 2026 UTC

Selected **BF16**. The shared `fp32-source-bf16-nearest-truncation-fp16-ulp-v1` calibration was newly frozen before deriving/executing FP16. BF16 nearest and source-weight truncation run on identical immutable inputs; exact A/A stays separate. Historical A/A-floor outcomes remain in their original runs. Frozen bounds SHA256 `77a39ed825b8f562a8f6d2691db3f7471b4f0cbf55651971030204293c648d73`; final native worker `ec8d97def2e72b4b89b8b807f2658bcd2c3431c44e97402546547ce941ee740c`. No FP32 model executes and no learned FP32 weights remain in the derived snapshots.

| Component | Source perturbation floor (rel L2) | Floor cosine | Frozen FP16 bound (rel L2) |
|---|---:|---:|---:|
| preset-conditioning | 0.00405390487 | 0.999995699 | 0.008828203 |

The other 18 components have zero measured nearest/truncation floor because their original stored weights were already BF16 or the perturbation did not change this output. Their bounds still include one FP16 ULP; see every component's floor, cosine and ULP in `bounds.json`. Every A/A relative L2 is zero. The rejected FP16 trial's maximum relative L2 is 0.0262321544, minimum cosine 0.999660437, failing components 17/19. It reports 2 output zeros and 135 subnormal output values where BF16 was normal. These are output diagnostics, not a proof about all intermediate underflows.

| Tier | Verdict | Max rel L2 vs selected half | Min cosine vs selected half | Spectral distance (diagnostic) | RTF (compute s/audio s) | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 0.2340132 | 0.5086589 |
| int8 | PASS | 0.04441779 | 0.99902008 | 0.6605129 | 0.2345397 | 0.4237733 |
| int4 | FAIL | 0.4671014 | 0.89340712 | 1.109038 | 0.1888857 | 0.3531582 |

## Matrix fallback

- `matrix@model.safetensors:flow_lm.transformer.layers.1.linear2.weight` — 4,194,304 parameters.
- `matrix@model.safetensors:flow_lm.transformer.layers.2.linear2.weight` — 4,194,304 parameters.
- `matrix@model.safetensors:flow_lm.transformer.layers.3.linear2.weight` — 4,194,304 parameters.
- `matrix@model.safetensors:flow_lm.transformer.layers.5.linear2.weight` — 4,194,304 parameters.

Extra 16-bit fallback is 16,777,216/118,940,354 source parameter/buffer values (14.10557%). Total 16-bit retention including required structural skips is 28.01114%. The denominator includes presets treated as weights and buffers; it is not a trainable-only parameter count. Int8 packs 85,623,808 values in 62 matrices; int4 packs 102,401,024 in 66 matrices. Every skip has a tensor-level reason in coverage. Measured int8 peak saving versus the selected half is 0.084886GB (16.688%). Per-owner retention of eligible parameters: {"backbone": {"eligible_parameters": 75497472, "eligible_share": 0.2222222222222222, "fallback_parameters": 16777216}}. Any high share is a limitation of this locally irreducible recipe; it does not prove every possible subset needs that share.

The initial fully packed int8 gate is preserved. Diagnosis starts with failing owners' eligible matrices retained and all other owners packed. It re-quantizes groups, then single matrices; a final single-matrix sweep continues until no removal passes. Each retained matrix's removal failed the fixed component gate with the other final fallbacks held fixed. `matrix-attribution.json` retains trial selections, worst metrics and exact failed component metrics; full raw captures and gates remain local. This proves local irreducibility on these fixtures, not a global optimum or general perceptual equivalence. Passing codec QKV packs need no source-to-runtime ablation mapping. The final newly derived stored recipe passes the complete 19-component replay under unchanged ≤0.05 relative L2 / ≥0.999 cosine, finite/range/no-new-clamp bounds. Scalar packing/dequantization audits are exact on the first four rows/all groups of every eligible matrix.

All figure workers are fresh, with zero component-capture requests; PIDs are in the result. RTF is compute seconds per generated audio second (lower is faster), and spectral distance is the unaligned common-prefix magnitude-spectrum diagnostic, with sample counts retained. It does not select format or quantization. Pocket's existing round-3 SEANet-input boundary is unchanged in this round.

Live intake 41.985s; hard aggregate worker limit 48.822GB, sampled aggregate worker peak 0.974GB. Preexisting swap 1,023,279,104bytes did not grow. Strict system-wide zero swap was not present before the run. No downloads; APFS clones hash-checked before/after and deleted, including derived and rounding copies. Source-copy receipt is retained. Figures are busy-Mac debug-worker quick checks. Independent vendor half-GPU comparison, listening and ASR checks remain unavailable. No main merge, push, signing or install.
