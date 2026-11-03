+++
hypothesis = "Int8 can retain genuine sensitive components in half while quantizing every passing large component."
change = "Versioned mixed recipes, scalar packing diagnosis, full int8 component fallback and replay, maximum-compression int4."
measurement = "M5 Max, busy Mac, debug native worker, default Load and streamed speech; internal fixed-input precision checks."
verdict = "kept"
numbers = { int8_fallback_values = 75497472, packed_int8_matrices = 42, packed_int4_matrices = 66, captured_components = 19 }
links = ["results/intake/intake-2026-10-15T21-16-34Z-f675f9de-edf7-45b9-a637-42eeb175179f.json", "gate/evidence/intake/intake-2026-10-15T21-16-34Z-f675f9de-edf7-45b9-a637-42eeb175179f/packing-diagnosis.json"]
+++

# Component fallback intake — 15 Oct 2026

The pipeline emitted `intake-v2:pocket-tts:backbone` and `intake-v2:pocket-tts:all`. Worker SHA256 `8f3ee41bd53bc49d48360bb25cc9128e976e9a9a43f13996169dcae297617050`. BF16 remains selected after the frozen FP16 format trial. The all-packed int8 proof is retained separately, before fallback; only the freshly derived mixed int8 is qualified and measured. Int4 packs every eligible matrix and remains offered with its warning when it fails.

| Tier | Verdict | Max rel L2 vs BF16 | Min cosine vs BF16 | Spectral distance (diagnostic) | RTF (compute s/audio s) | Peak (GB) |
|---|---|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 0.1352586 | 0.5070041 |
| int8 | PASS | 0.0210022 | 0.99978115 | 0.6402702 | 0.1323504 | 0.4692717 |
| int4 | FAIL | 0.4671014 | 0.89340712 | 1.109038 | 0.1330736 | 0.355026 |

## Original int8 component diagnosis

- `backbone-2`: rel L2 0.089549438, cosine 0.99604178, reference RMS 0.37028291, candidate RMS 0.37284499.
- `backbone-1`: rel L2 0.078140379, cosine 0.99696893, reference RMS 0.37182503, candidate RMS 0.37340424.
- `backbone-0`: rel L2 0.052669116, cosine 0.99861595, reference RMS 0.40261676, candidate RMS 0.4031873.

The affine weight audit covers every eligible matrix (66 matrices, first four output rows and all input groups each): scalar unpack agrees exactly with GPU dequantization, and every scale/bias uses the selected half. Group/axis checks preserve `[out,kernel,in]`; only ungrouped stride-one, unpadded kernel-one speaker adapters map to NLC Linear. Spatial/grouped convolutions stay half. The independent deterministic well-conditioned fixed-tensor test gave weight rel L2 0.00424969 / 0.00379986 and output rel L2 0.00469564 / 0.00365617 in BF16 / FP16 respectively, with quantized matmul vs explicit dequantized matmul 0.00245493 / 0.000279894. No production packing/layout bug was found. One new diagnostic initially used the CPU dequantization default inside CPU unit fixtures; the diagnostic now explicitly uses the GPU like serving, and its regression tests pass. No production quantized weights or bound changed to mask the sensitivity.

Output RMS and maximum magnitudes for all failures are in the initial gate and diagnosis receipt. Pocket decoder: the original quiet waveform (RMS 0.0012083) had rel L2 0.1308619/cosine 0.9914147. Before rerunning, intake declared the real SEANet decoder input as the boundary. At that input, decoder-1 gives rel L2 0.01792415/cosine 0.99983937 and RMS 0.497733; all decoder inputs pass and remain packed. Both boundaries are retained in the diagnosis receipt. BF16 and int4 served PCM hashes are unchanged by the refactor, also checked with a tiny native fixture. No numeric bound was relaxed. This is representative fixed-input qualification, not a universal claim about all voices.

## Int8 coverage

- `backbone`: 75,497,472 newly retained half values, 63.4751% of all declared source parameter/buffer values.

Total extra half retention: 75,497,472 / 118,940,354 declared source values (63.4751%). The denominator is the same source parameter/buffer inventory used by conversion, including presets treated as weights; it is not advertised as a trainable-only count. Int8 still packs 26,903,552 values; int4 packs 102,401,024. Structural skips (norms, biases, small tensors and incompatible convolution/group layouts) remain explicit per tensor in coverage. Passing prompt/text, speaker and codec components stay packed where eligible. Qwen predictor embeddings remain independent from its retained predictor transformer/heads; Pocket Mimi remains packed at the corrected boundary. All 19 final captures pass the unchanged int8 bounds.

Live intake elapsed 31.940s; hard worker budget 46.690GB; aggregate sampled worker peak 0.970GB. Preexisting swap 1,023,279,104 bytes did not grow (system-wide zero swap cannot be claimed). Source hashes match before/after; APFS donor and derived weight copies are deleted. The component fixtures are immutable; the pipeline binds exact recipes, build, bounds, captures and table stats. The all-packed probe and final mixed capture use identical saved inputs.

RTF/peak are default-path busy-machine quick checks, not controlled release benchmarks. Spectral distance is an unaligned common-prefix diagnostic with potentially unequal sample counts; it is not an audio-quality pass. No independent matched vendor half-GPU, ASR or listening comparison was available. No merge, push, signing or install.
