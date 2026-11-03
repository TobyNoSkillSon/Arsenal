+++
hypothesis = "A shared source-rounding calibration selects half format; matrix-level retention repairs int8 without retaining whole components."
change = "Shared freezeFP32Source policy and intake-v3 matrix recipes with iterative ablation and complete stored-recipe replay."
measurement = "M5 Max, busy Mac, debug native worker; default Load and streamed speech, fresh workers for figures."
verdict = "kept"
numbers = { int8_extra_half_values = 72351744, int8_matrices = 356, int4_matrices = 377, components = 100, diagnosis_trials = 113 }
links = ["results/intake/intake-2026-10-15T22-05-48Z-04f44d82-31b9-4175-b39e-6b62c02c8188.json", "gate/evidence/intake/intake-2026-10-15T22-05-48Z-04f44d82-31b9-4175-b39e-6b62c02c8188/qualification-summary.json"]
+++

# Shared source-floor and matrix fallback — 15 Oct 2026 UTC

Selected **BF16**. The shared `fp32-source-bf16-nearest-truncation-fp16-ulp-v1` calibration was newly frozen before deriving/executing FP16. BF16 nearest and source-weight truncation run on identical immutable inputs; exact A/A stays separate. Historical A/A-floor outcomes remain in their original runs. Frozen bounds SHA256 `26b16b579a650254a58042d97e2326955db79168eac9214bdab6322aab45c727`; final native worker `ec8d97def2e72b4b89b8b807f2658bcd2c3431c44e97402546547ce941ee740c`. No FP32 model executes and no learned FP32 weights remain in the derived snapshots.

| Component | Source perturbation floor (rel L2) | Floor cosine | Frozen FP16 bound (rel L2) |
|---|---:|---:|---:|
| codec-decoder | 0.0349174646 | 0.999649363 | 0.0705450376 |
| codec-downsample | 0.00469536773 | 0.999992572 | 0.0101149742 |
| codec-encoder | 0.179006313 | 0.983970623 | 0.358716045 |
| codec-encoder-transformer | 0.0230111453 | 0.999736539 | 0.0467056783 |

The other 96 components have zero measured nearest/truncation floor because their original stored weights were already BF16 or the perturbation did not change this output. Their bounds still include one FP16 ULP; see every component's floor, cosine and ULP in `bounds.json`. Every A/A relative L2 is zero. The rejected FP16 trial's maximum relative L2 is 0.494373926, minimum cosine 0.878977491, failing components 51/100. It reports 2 output zeros and 565 subnormal output values where BF16 was normal. These are output diagnostics, not a proof about all intermediate underflows.

| Tier | Verdict | Max rel L2 vs selected half | Min cosine vs selected half | Spectral distance (diagnostic) | RTF (compute s/audio s) | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 0.6152788 | 4.732817 |
| int8 | PASS | 0.04903925 | 0.99907937 | 1.234811 | 0.5234227 | 2.948402 |
| int4 | FAIL | 1.445077 | 0.00069510485 | 1.095715 | 0.590128 | 1.875741 |

## Matrix fallback

- `matrix@model.safetensors:talker.code_predictor.model.layers.0.self_attn.k_proj.weight` — 1,048,576 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.0.self_attn.o_proj.weight` — 2,097,152 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.1.mlp.down_proj.weight` — 3,145,728 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.2.self_attn.k_proj.weight` — 1,048,576 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.2.self_attn.q_proj.weight` — 2,097,152 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.2.self_attn.v_proj.weight` — 1,048,576 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.3.mlp.down_proj.weight` — 3,145,728 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.3.mlp.gate_proj.weight` — 3,145,728 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.3.mlp.up_proj.weight` — 3,145,728 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.3.self_attn.k_proj.weight` — 1,048,576 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.3.self_attn.o_proj.weight` — 2,097,152 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.3.self_attn.q_proj.weight` — 2,097,152 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.3.self_attn.v_proj.weight` — 1,048,576 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.4.self_attn.k_proj.weight` — 1,048,576 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.4.self_attn.o_proj.weight` — 2,097,152 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.4.self_attn.q_proj.weight` — 2,097,152 parameters.
- `matrix@model.safetensors:talker.code_predictor.model.layers.4.self_attn.v_proj.weight` — 1,048,576 parameters.
- `matrix@model.safetensors:talker.code_predictor.small_to_mtp_projection.weight` — 2,097,152 parameters.
- `matrix@model.safetensors:talker.model.layers.2.mlp.down_proj.weight` — 12,582,912 parameters.
- `matrix@model.safetensors:talker.model.layers.27.mlp.gate_proj.weight` — 12,582,912 parameters.
- `matrix@model.safetensors:talker.model.layers.27.mlp.up_proj.weight` — 12,582,912 parameters.

Extra 16-bit fallback is 72,351,744/2,087,233,793 source parameter/buffer values (3.46639%). Total 16-bit retention including required structural skips is 8.17824%. The denominator includes presets treated as weights and buffers; it is not a trainable-only parameter count. Int8 packs 1,916,534,784 values in 356 matrices; int4 packs 1,988,886,528 in 377 matrices. Every skip has a tensor-level reason in coverage. Measured int8 peak saving versus the selected half is 1.784415GB (37.703%). Per-owner retention of eligible parameters: {"predictor": {"eligible_parameters": 112197632, "eligible_share": 0.308411214953271, "fallback_parameters": 34603008}, "talker": {"eligible_parameters": 1415577600, "eligible_share": 0.02666666666666667, "fallback_parameters": 37748736}}. Any high share is a limitation of this locally irreducible recipe; it does not prove every possible subset needs that share.

The initial fully packed int8 gate is preserved. Diagnosis starts with failing owners' eligible matrices retained and all other owners packed. It re-quantizes groups, then single matrices; a final single-matrix sweep continues until no removal passes. Each retained matrix's removal failed the fixed component gate with the other final fallbacks held fixed. `matrix-attribution.json` retains trial selections, worst metrics and exact failed component metrics; full raw captures and gates remain local. This proves local irreducibility on these fixtures, not a global optimum or general perceptual equivalence. Passing codec QKV packs need no source-to-runtime ablation mapping. The final newly derived stored recipe passes the complete 100-component replay under unchanged ≤0.05 relative L2 / ≥0.999 cosine, finite/range/no-new-clamp bounds. Scalar packing/dequantization audits are exact on the first four rows/all groups of every eligible matrix.

All figure workers are fresh, with zero component-capture requests; PIDs are in the result. RTF is compute seconds per generated audio second (lower is faster), and spectral distance is the unaligned common-prefix magnitude-spectrum diagnostic, with sample counts retained. It does not select format or quantization. Pocket's existing round-3 SEANet-input boundary is unchanged in this round.

Live intake 319.146s; hard aggregate worker limit 48.073GB, sampled aggregate worker peak 11.331GB. Preexisting swap 1,023,279,104bytes did not grow. Strict system-wide zero swap was not present before the run. No downloads; APFS clones hash-checked before/after and deleted, including derived and rounding copies. Source-copy receipt is retained. Figures are busy-Mac debug-worker quick checks. Independent vendor half-GPU comparison, listening and ASR checks remain unavailable. No main merge, push, signing or install.
