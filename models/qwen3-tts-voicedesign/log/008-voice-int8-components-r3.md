+++
hypothesis = "Int8 can retain genuine sensitive components in half while quantizing every passing large component."
change = "Versioned mixed recipes, scalar packing diagnosis, full int8 component fallback and replay, maximum-compression int4."
measurement = "M5 Max, busy Mac, debug native worker, default Load and streamed speech; internal fixed-input precision checks."
verdict = "kept"
numbers = { int8_fallback_values = 1527775232, packed_int8_matrices = 129, packed_int4_matrices = 377, captured_components = 100 }
links = ["results/intake/intake-2026-10-15T21-19-50Z-4fae7d64-da39-4366-abef-3c1bd8c88f2a.json", "gate/evidence/intake/intake-2026-10-15T21-19-50Z-4fae7d64-da39-4366-abef-3c1bd8c88f2a/packing-diagnosis.json"]
+++

# Component fallback intake — 15 Oct 2026

The pipeline emitted `intake-v2:qwen3-tts-voicedesign:predictor,talker` and `intake-v2:qwen3-tts-voicedesign:all`. Worker SHA256 `8f3ee41bd53bc49d48360bb25cc9128e976e9a9a43f13996169dcae297617050`. BF16 remains selected after the frozen FP16 format trial. The all-packed int8 proof is retained separately, before fallback; only the freshly derived mixed int8 is qualified and measured. Int4 packs every eligible matrix and remains offered with its warning when it fails.

| Tier | Verdict | Max rel L2 vs BF16 | Min cosine vs BF16 | Spectral distance (diagnostic) | RTF (compute s/audio s) | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 0.5187059 | 4.735668 |
| int8 | PASS | 0.03001209 | 0.99954954 | 1.108179 | 0.5234763 | 4.294873 |
| int4 | FAIL | 1.445077 | 0.00069510485 | 1.095715 | 0.4782839 | 1.872382 |

## Original int8 component diagnosis

- `talker-0`: rel L2 0.12375792, cosine 0.99284877, reference RMS 1.8014217, candidate RMS 1.7297594.
- `predictor-1-13`: rel L2 0.086962694, cosine 0.99791135, reference RMS 3.6970047, candidate RMS 3.9045226.
- `predictor-0-13`: rel L2 0.08616286, cosine 0.99636319, reference RMS 3.5930394, candidate RMS 3.5340085.
- `talker-1`: rel L2 0.079215049, cosine 0.9969578, reference RMS 2.4112338, candidate RMS 2.3698081.
- `predictor-0-0`: rel L2 0.074045847, cosine 0.99726524, reference RMS 2.2564519, candidate RMS 2.2605587.
- `talker-2`: rel L2 0.063819633, cosine 0.99799484, reference RMS 2.6135114, candidate RMS 2.5869359.

The affine weight audit covers every eligible matrix (377 matrices, first four output rows and all input groups each): scalar unpack agrees exactly with GPU dequantization, and every scale/bias uses the selected half. Group/axis checks preserve `[out,kernel,in]`; only ungrouped stride-one, unpadded kernel-one speaker adapters map to NLC Linear. Spatial/grouped convolutions stay half. The independent deterministic well-conditioned fixed-tensor test gave weight rel L2 0.00424969 / 0.00379986 and output rel L2 0.00469564 / 0.00365617 in BF16 / FP16 respectively, with quantized matmul vs explicit dequantized matmul 0.00245493 / 0.000279894. No production packing/layout bug was found. One new diagnostic initially used the CPU dequantization default inside CPU unit fixtures; the diagnostic now explicitly uses the GPU like serving, and its regression tests pass. No production quantized weights or bound changed to mask the sensitivity.

Output RMS and maximum magnitudes for all failures are in the initial gate and diagnosis receipt. No boundary was moved: output RMS for the failed components is ordinary, so the failures are not almost-zero-output relative-error artifacts. No numeric bound was relaxed. This is representative fixed-input qualification, not a universal claim about all voices.

## Int8 coverage

- `predictor`: 112,197,632 newly retained half values, 5.3754% of all declared source parameter/buffer values.
- `talker`: 1,415,577,600 newly retained half values, 67.8207% of all declared source parameter/buffer values.

Total extra half retention: 1,527,775,232 / 2,087,233,793 declared source values (73.1962%). The denominator is the same source parameter/buffer inventory used by conversion, including presets treated as weights; it is not advertised as a trainable-only count. Int8 still packs 461,111,296 values; int4 packs 1,988,886,528. Structural skips (norms, biases, small tensors and incompatible convolution/group layouts) remain explicit per tensor in coverage. Passing prompt/text, speaker and codec components stay packed where eligible. Qwen predictor embeddings remain independent from its retained predictor transformer/heads; Pocket Mimi remains packed at the corrected boundary. All 100 final captures pass the unchanged int8 bounds.

Live intake elapsed 108.629s; hard worker budget 48.458GB; aggregate sampled worker peak 9.179GB. Preexisting swap 1,023,279,104 bytes did not grow (system-wide zero swap cannot be claimed). Source hashes match before/after; APFS donor and derived weight copies are deleted. The component fixtures are immutable; the pipeline binds exact recipes, build, bounds, captures and table stats. The all-packed probe and final mixed capture use identical saved inputs.

RTF/peak are default-path busy-machine quick checks, not controlled release benchmarks. Spectral distance is an unaligned common-prefix diagnostic with potentially unequal sample counts; it is not an audio-quality pass. No independent matched vendor half-GPU, ASR or listening comparison was available. No merge, push, signing or install.
