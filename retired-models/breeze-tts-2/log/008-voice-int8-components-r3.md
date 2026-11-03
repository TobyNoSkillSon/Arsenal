+++
hypothesis = "Int8 can retain genuine sensitive components in half while quantizing every passing large component."
change = "Versioned mixed recipes, scalar packing diagnosis, full int8 component fallback and replay, maximum-compression int4."
measurement = "M5 Max, busy Mac, debug native worker, default Load and streamed speech; internal fixed-input precision checks."
verdict = "kept"
numbers = { int8_fallback_values = 0, packed_int8_matrices = 594, packed_int4_matrices = 594, captured_components = 23 }
links = ["results/intake/intake-2026-10-15T21-22-58Z-f9415e6f-bd77-4c95-bbd3-c4e0523fc8b5.json", "gate/evidence/intake/intake-2026-10-15T21-22-58Z-f9415e6f-bd77-4c95-bbd3-c4e0523fc8b5/packing-diagnosis.json"]
+++

# Component fallback intake — 15 Oct 2026

The pipeline emitted `intake-v2:breeze-tts-2:all` and `intake-v2:breeze-tts-2:all`. Worker SHA256 `8f3ee41bd53bc49d48360bb25cc9128e976e9a9a43f13996169dcae297617050`. BF16 remains selected after the frozen FP16 format trial. The all-packed int8 proof is retained separately, before fallback; only the freshly derived mixed int8 is qualified and measured. Int4 packs every eligible matrix and remains offered with its warning when it fails.

| Tier | Verdict | Max rel L2 vs BF16 | Min cosine vs BF16 | Spectral distance (diagnostic) | RTF (compute s/audio s) | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 1.590261 | 8.300403 |
| int8 | PASS | 0.03580997 | 0.99936075 | 0.8200004 | 1.495167 | 5.324935 |
| int4 | FAIL | 0.2240733 | 0.97488015 | 0.9209195 | 1.541514 | 3.248999 |

## Original int8 component diagnosis

No component failed the all-packed int8 probe.

The affine weight audit covers every eligible matrix (594 matrices, first four output rows and all input groups each): scalar unpack agrees exactly with GPU dequantization, and every scale/bias uses the selected half. Group/axis checks preserve `[out,kernel,in]`; only ungrouped stride-one, unpadded kernel-one speaker adapters map to NLC Linear. Spatial/grouped convolutions stay half. The independent deterministic well-conditioned fixed-tensor test gave weight rel L2 0.00424969 / 0.00379986 and output rel L2 0.00469564 / 0.00365617 in BF16 / FP16 respectively, with quantized matmul vs explicit dequantized matmul 0.00245493 / 0.000279894. No production packing/layout bug was found. One new diagnostic initially used the CPU dequantization default inside CPU unit fixtures; the diagnostic now explicitly uses the GPU like serving, and its regression tests pass. No production quantized weights or bound changed to mask the sensitivity.

Output RMS and maximum magnitudes for all failures are in the initial gate and diagnosis receipt. No boundary was moved: output RMS for the failed components is ordinary, so the failures are not almost-zero-output relative-error artifacts. No numeric bound was relaxed. This is representative fixed-input qualification, not a universal claim about all voices.

## Int8 coverage

No int8 fallback; every eligible component remains packed.

Total extra half retention: 0 / 3,624,819,713 declared source values (0.0000%). The denominator is the same source parameter/buffer inventory used by conversion, including presets treated as weights; it is not advertised as a trainable-only count. Int8 still packs 3,526,329,088 values; int4 packs 3,526,329,088. Structural skips (norms, biases, small tensors and incompatible convolution/group layouts) remain explicit per tensor in coverage. Passing prompt/text, speaker and codec components stay packed where eligible. Qwen predictor embeddings remain independent from its retained predictor transformer/heads; Pocket Mimi remains packed at the corrected boundary. All 23 final captures pass the unchanged int8 bounds.

Live intake elapsed 198.942s; hard worker budget 48.492GB; aggregate sampled worker peak 15.294GB. Preexisting swap 1,023,279,104 bytes did not grow (system-wide zero swap cannot be claimed). Source hashes match before/after; APFS donor and derived weight copies are deleted. The component fixtures are immutable; the pipeline binds exact recipes, build, bounds, captures and table stats. The all-packed probe and final mixed capture use identical saved inputs.

RTF/peak are default-path busy-machine quick checks, not controlled release benchmarks. Spectral distance is an unaligned common-prefix diagnostic with potentially unequal sample counts; it is not an audio-quality pass. No independent matched vendor half-GPU, ASR or listening comparison was available. No merge, push, signing or install.
