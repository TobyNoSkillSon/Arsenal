+++
hypothesis = "A shared source-rounding calibration selects half format; matrix-level retention repairs int8 without retaining whole components."
change = "Shared freezeFP32Source policy and intake-v3 matrix recipes with iterative ablation and complete stored-recipe replay."
measurement = "M5 Max, busy Mac, debug native worker; default Load and streamed speech, fresh workers for figures."
verdict = "kept"
numbers = { int8_extra_half_values = 0, int8_matrices = 594, int4_matrices = 594, components = 23, diagnosis_trials = 0 }
links = ["results/intake/intake-2026-10-15T22-13-24Z-ea9b5a7b-825a-4457-b14f-b792fd3355a4.json", "gate/evidence/intake/intake-2026-10-15T22-13-24Z-ea9b5a7b-825a-4457-b14f-b792fd3355a4/qualification-summary.json"]
+++

# Shared source-floor and matrix fallback — 15 Oct 2026 UTC

Selected **BF16**. The shared `fp32-source-bf16-nearest-truncation-fp16-ulp-v1` calibration was newly frozen before deriving/executing FP16. BF16 nearest and source-weight truncation run on identical immutable inputs; exact A/A stays separate. Historical A/A-floor outcomes remain in their original runs. Frozen bounds SHA256 `3162128447622dafa4c94b2ef88a5337b4966554aae6720bc53b8732403cf1a8`; final native worker `ec8d97def2e72b4b89b8b807f2658bcd2c3431c44e97402546547ce941ee740c`. No FP32 model executes and no learned FP32 weights remain in the derived snapshots.

| Component | Source perturbation floor (rel L2) | Floor cosine | Frozen FP16 bound (rel L2) |
|---|---:|---:|---:|
| codec-decoder | 0.0349174646 | 0.999649363 | 0.0705450376 |
| codec-downsample | 0.00469536773 | 0.999992572 | 0.0101149742 |
| codec-encoder | 0.179006313 | 0.983970623 | 0.358716045 |
| codec-encoder-transformer | 0.0230111453 | 0.999736539 | 0.0467056783 |

The other 19 components have zero measured nearest/truncation floor because their original stored weights were already BF16 or the perturbation did not change this output. Their bounds still include one FP16 ULP; see every component's floor, cosine and ULP in `bounds.json`. Every A/A relative L2 is zero. The rejected FP16 trial's maximum relative L2 is 0.494373926, minimum cosine 0.878977491, failing components 20/23. It reports 31 output zeros and 12,537 subnormal output values where BF16 was normal. These are output diagnostics, not a proof about all intermediate underflows.

| Tier | Verdict | Max rel L2 vs selected half | Min cosine vs selected half | Spectral distance (diagnostic) | RTF (compute s/audio s) | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 1.519452 | 8.477039 |
| int8 | PASS | 0.03580997 | 0.99936075 | 0.8200004 | 1.495642 | 5.360374 |
| int4 | FAIL | 0.2240733 | 0.97488015 | 0.9209195 | 1.483851 | 3.296791 |

## Matrix fallback

None: all eligible int8 matrices pass.

Extra 16-bit fallback is 0/3,624,819,713 source parameter/buffer values (0.00000%). Total 16-bit retention including required structural skips is 2.71712%. The denominator includes presets treated as weights and buffers; it is not a trainable-only parameter count. Int8 packs 3,526,329,088 values in 594 matrices; int4 packs 3,526,329,088 in 594 matrices. Every skip has a tensor-level reason in coverage. Measured int8 peak saving versus the selected half is 3.116665GB (36.766%). Per-owner retention of eligible parameters: {}. Any high share is a limitation of this locally irreducible recipe; it does not prove every possible subset needs that share.

The initial fully packed int8 gate is preserved. Diagnosis starts with failing owners' eligible matrices retained and all other owners packed. It re-quantizes groups, then single matrices; a final single-matrix sweep continues until no removal passes. Each retained matrix's removal failed the fixed component gate with the other final fallbacks held fixed. `matrix-attribution.json` retains trial selections, worst metrics and exact failed component metrics; full raw captures and gates remain local. This proves local irreducibility on these fixtures, not a global optimum or general perceptual equivalence. Passing codec QKV packs need no source-to-runtime ablation mapping. The final newly derived stored recipe passes the complete 23-component replay under unchanged ≤0.05 relative L2 / ≥0.999 cosine, finite/range/no-new-clamp bounds. Scalar packing/dequantization audits are exact on the first four rows/all groups of every eligible matrix.

All figure workers are fresh, with zero component-capture requests; PIDs are in the result. RTF is compute seconds per generated audio second (lower is faster), and spectral distance is the unaligned common-prefix magnitude-spectrum diagnostic, with sample counts retained. It does not select format or quantization. Pocket's existing round-3 SEANet-input boundary is unchanged in this round.

Live intake 228.823s; hard aggregate worker limit 48.034GB, sampled aggregate worker peak 15.287GB. Preexisting swap 1,023,279,104bytes did not grow. Strict system-wide zero swap was not present before the run. No downloads; APFS clones hash-checked before/after and deleted, including derived and rounding copies. Source-copy receipt is retained. Figures are busy-Mac debug-worker quick checks. Independent vendor half-GPU comparison, listening and ASR checks remain unavailable. No main merge, push, signing or install.
