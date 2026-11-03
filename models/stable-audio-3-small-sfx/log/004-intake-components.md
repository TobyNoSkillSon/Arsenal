+++
hypothesis = "Saved identical component inputs separate whole-part quantization loss from generated-audio differences."
change = "Keep SAME-S half/packed parameters half, capture its served overlapping windows, cast at Get and emit tier-bound intake stats."
measurement = "Busy M5 Max quick check, 15 October 2026; five-second/eight-step default Load/generate path."
numbers = { eligible_matrices = 280, int8_max_relative_l2 = 0.027093251370536964, int8_min_cosine = 0.9996393106435938, int4_max_relative_l2 = 0.41553861969186356, int4_min_cosine = 0.9134183835058503, added_swap_bytes = 0 }
verdict = "inconclusive"
links = ["../results/intake/intake-2026-10-15T19-53-13Z-63b6daef-d597-45e1-bf6d-935032b40028.json", "../gate/evidence/quality-binding-2026-10-15.json", "../gate/evidence/source-integrity-2026-10-15.json"]
+++

This historical `intake-v1` run is superseded by log 005, which adds the omitted duration projection and its independent component check.

Small SFX now enters the intake pipeline through ordinary Get/Load and the sound worker. The snapshot contains FP16 parameters (including SAME-S), or stored affine matrices with FP16 scales/biases. Load refuses the old FP32-only SAME-S file and asks for Get. Its decoder input follows the stored half dtype; the Small SFX DiT uses its half stream without executing the legacy FP32 self-test or creating promoted-weight fallback copies. Medium's served numerical branches and results are unchanged.

All three tiers have five independently replayed outputs: prompt, denoiser at steps 0/3/7, and chunked SAME-S decoder. Integer IDs/mask, reference conditioning, latents/times and decoder input are immutable saved tensors. The retry reused the first successful reference and A/A floor only after checking identical FP16 snapshot hashes, then replayed the current FP16 implementation on those inputs (relative L2 0, cosine 1). It did not regenerate the reference. Each tier recorded 119 finite FP16 activation/output traces, including every decoder window, with zero clamp events; maximum observed activation magnitude was 7,800.

| Tier | Saved-input verdict | Max relative L2 | Min cosine | Warm median s / 5-s clip | Lifetime peak GB |
|---|---|---:|---:|---:|---:|
| FP16 | A/A reproducibility passes; initial cast unchecked | 0 | 1 | 0.099985 | 2.537966 |
| int8 | Pass versus captured FP16 | 0.027093 | 0.999639 | 0.120200 | 1.576454 |
| int4 | Fail versus captured FP16; offered with warning | 0.415539 | 0.913418 | 0.111381 | 1.081853 |

These are busy-Mac quick checks, not controlled speed comparisons. Warm requests use balanced ABBA BAAB (four samples per arm, baseline pooled across both pairs). Compare a quant with its own saved paired baseline; all samples, spreads, baseline drift, exact requested default Load configuration, worker identity and cold latency are in the result. Peaks are per-worker lifetime physical footprint covering Load and component probes, rather than packed bytes or the sum of resident workers.

Both quantized tiers pack 280 matrices: DiT 170, T5Gemma 85 (embedding included), SAME-S 25. In DiT, 249 non-matrix tensors, 20 input widths incompatible with group 64 and two non-module-weight matrices remain half. SAME-S leaves 89 non-matrix tensors half, including its mapping convolution, norms, biases and learned tokens; T5Gemma leaves 52 non-matrix tensors (including tokenizer bytes). The original eligibility predicate missed the custom duration projection; log 005 corrects that omission. Per-key reasons and dtype histograms are retained in each tier's `*-component-coverage.json`; source shapes/dtypes are in `source-inventory-2026-10-15.json`. Conversion streams tensor by tensor and rebuilds the new tier's receipt instead of copying the read-only source receipt over it.

Five-second/eight-step end-to-end sanity artifacts are finite and unclipped. Log-mel mean absolute differences versus the FP16 artifact are 0 / 0.541262 / 14.875630 dB (FP16/int8/int4), diagnostic only. They do not select the 16-bit format or replace the component gate.

The initial FP32-only SAME-S cast remains a precise qualification gap. No original identical-input decoder component capture exists, and no FP32 reference was executed. The retained original two-step PCM is already 30.88% clipped; the new half output is 34.77% clipped and differs by waveform relative L2 1.297196 / cosine 0.185138. The nominal FP16 source archives also contain three FP32 conditioning tensors and T5Gemma `rope_inv_freq`, all cast by Get; the legacy PCM comparison cannot isolate SAME-S. This is unsuitable as a cast reference and does not show that BF16 would improve it. The table therefore leaves FP16's quality outcome unchecked, rather than inheriting an A/A pass as cast qualification. Int8/int4 verdicts are explicitly relative to the captured FP16 implementation; independent vendor qualification remains open.

The guarded run completed in 24.093 seconds under a 29.070-GB worker cap. Existing system swap was 1,023,279,104 bytes and did not grow; the guard did not stop the run. Six original pinned files match before/after hashes, with zero network bytes. Lab weight/cache/donor copies and owned worker PIDs were removed after saving evidence. Raw tensors and WAVs remain local and are excluded from Git.

An earlier attempt reached the original FP16 capture but failed during int8 conversion when a read-only FP16 coverage receipt collided with the newly generated tier receipt. That failure and its inputs are retained under `intake-2026-10-15T19-47-42Z-54ff4580-1e88-4f25-957a-1ba23662407e`; a regression now exercises the immutable-source conversion. The earlier missing-metallib lab-layout failure is retained separately. The final run alone supplies table stats and exact run/bounds/build/load identity bindings. No merge, push, signing or installation was performed.
