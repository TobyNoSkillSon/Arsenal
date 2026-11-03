+++
hypothesis = "An exponent-preserving BF16 cast provides a no-FP32-execution reference for qualifying FP32-source SAME-S and conditioning."
change = "Qualify FP16 within the measured BF16 rounding floor, tie the exact served replay to the source-cast capture, and bind all three outcomes through final intake."
measurement = "Busy M5 Max quick checks, 15 October 2026; default Load, five seconds/eight steps, balanced warm ABBA BAAB."
numbers = { format_decoder_relative_l2 = 0.10813467428123456, format_decoder_bound = 0.24989752973148727, int8_max_relative_l2 = 0.026991469224513766, int4_max_relative_l2 = 0.41682300937710437, added_swap_bytes = 0 }
verdict = "kept"
links = ["../results/intake/intake-2026-10-15T20-56-44Z-74176d13-dd38-456e-8a7d-5096e3b83efe.json", "../gate/evidence/quality-binding-2026-10-15.json"]
+++

FP16 is selected. This supersedes log 005's initial-cast gap; its captures and narrower outcomes remain history. The shared FP32-source method belongs to `docs/ADDING-A-MODEL.md` (voices owns that section). This implementation follows the coordinator's voices rollout decisions (round 2), including GPU BF16 references and FP16 activation range rather than a BF16-extrema equality requirement. No change to the shared documentation section was needed or made.

The reference casts original FP32 SAME-S, duration weight/bias and learned padding to BF16. A second BF16 capture truncates mantissas toward zero instead of rounding to nearest; this deliberately conservative alternate rounding supplies each component's floor. Identical saved decoder latents and duration scalar are reused from the existing v2 fixture. Before FP16 execution, each output bound was frozen at twice the measured rounding relative L2 plus one BF16 output ULP; cosine uses the corresponding floor. No FP32 model/reference execution occurs; original FP32 arrays are read for conversion only. Both BF16 captures and the FP16 candidate use the production SAME-S module and chunked decode on the GPU.

| FP32-source component | FP16 vs BF16 relative L2 | Cosine | Frozen relative L2 bound |
|---|---:|---:|---:|
| SAME-S decoder | 0.108135 | 0.994176 | 0.249898 |
| Duration projection | 0.004498 | 0.999990 | 0.018353 |
| Learned padding | 0.001647 | 0.999999 | 0.013601 |

All outputs and traced activations are finite, within FP16 range, and have zero new clamps. The decoder's BF16-nearest versus BF16-truncated rounding floor itself is 0.122081 relative L2; this sensitivity makes the format bound conservative. It is not evidence of independent vendor fidelity or perceptual equivalence.

An initial use of the generic gate additionally required intermediate maxima to remain inside a BF16-extrema envelope. FP16 passed every frozen output error/cosine, finiteness and clamp check but one decoder maximum exceeded that envelope by 0.4845%. That extra condition is absent from the coordinator's source-format decision. The original bounds and failed gate are retained as `format-qualification/*-original-extrema-envelope.json`. The implementation was corrected to the stated finite/no-new-clamps/FP16-range policy and rescored the same saved tensors; every output L2 and cosine limit stayed unchanged. No capture was regenerated and no live rerun was used to obtain the format pass. A regression covers the changed-extremum case while retaining failures for overflow and new clamps.

The final intake requires these saved source-format results, verifies pinned source and fixture hashes, recomputes the format gate, and requires the served FP16 decoder and duration outputs to equal the qualified candidate exactly before writing stats. FP16's final outcome combines that format qualification with its six-component A/A replay. All three results bind the exact final worker, source closure, load identity, recipe, result digest and bounds; FP16 additionally binds the format-gate digest. Int8/int4 are evaluated against the chosen FP16 tier.

| Tier | Verdict | Max component relative L2 | Min cosine | Log-mel diagnostic dB vs FP16 | Warm median s / 5-s clip | Lifetime peak GB |
|---|---|---:|---:|---:|---:|---:|
| FP16 | Pass; selected 16-bit format | 0.108135 | 0.994176 | 0 | 0.100176 | 2.524138 |
| int8 | Pass | 0.026991 | 0.999646 | 0.552379 | 0.112401 | 1.569556 |
| int4 | Fail; offered with warning | 0.416823 | 0.913101 | 14.901255 | 0.109287 | 1.080199 |

FP16's error/cosine in this table include source-format comparisons against BF16; its production A/A replay is exactly equal. Integer errors compare with FP16. The log-mel diagnostic compares each saved five-second served output with FP16, not BF16, and does not decide precision or perceptual quality. Warm values are busy-Mac quick checks, not a controlled speed ranking; per-pair samples and baseline drift remain in the result. Peaks are per-worker lifetime physical footprint covering Load and probes. The served run completed in 23.131 seconds with a 47.637-GB hard worker cap; existing swap 1,023,279,104 bytes did not grow. The format probes had a separate 4-GB cap and 590-second ceiling. The first reference guard used the displayed sysctl swap value (unchanged); candidate and served guards use exact kernel bytes, also unchanged.

Coverage remains 281 packed matrices per integer tier: DiT/conditioning 171, T5Gemma 85, SAME-S 25. Existing per-key reasons justify vectors, norms, convolution, learned concatenated tokens and incompatible widths remaining half. No FP32 learned weights remain. The bookkeeping-only Medium receipt fix changes the shared Sound identity, so this final run uses that rebuilt worker instead of inheriting round 1 bindings. Independent vendor half-reference qualification remains unavailable and is explicitly separate. Source integrity, cleanup and verification receipts accompany this run; no installed app/store, merge, push, signing or installation is part of this work.
