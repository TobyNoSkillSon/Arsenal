+++
hypothesis = "Shared Small SFX changes can preserve Medium arithmetic while requiring fresh quality-identity bindings."
change = "Re-run Medium through ordinary lab Get/Load, rebuild coverage receipts per tier, and write exact-run table stats and outcome bindings."
measurement = "Busy M5 Max quick check, 15 October 2026; default Load, five seconds/eight steps, balanced warm ABBA BAAB."
numbers = { int8_max_relative_l2 = 0.01513868040624071, int4_max_relative_l2 = 0.22856979873298003, added_swap_bytes = 0 }
verdict = "kept"
links = ["../results/intake/intake-2026-10-15T20-52-52Z-7f928893-6229-42f0-9201-516c8e5cf12f.json", "../gate/evidence/quality-binding-2026-10-15.json"]
+++

This intake binds Medium's FP16/int8/int4 outcomes and figures to the current Sound implementation. Log 005's acceptance basis and old results remain history; they are not relabelled as the new implementation. The existing accepted FP16 SAME-L basis remains unchanged (retained original PCM plus finite/no-clamp evidence); the new SAME-S BF16 source-format decision belongs to Small SFX, not a retroactive SAME-L qualification claim.

Source weights came from an APFS clone of the pinned local HF cache. No downloads or installed-store mutations occurred. The initial run (`intake-2026-10-15T20-48-14Z-9eb41eb5-198b-4d1b-b478-d13516be5f87`) failed when int8 conversion copied the immutable FP16 coverage receipt onto its newly generated receipt. The prior Small SFX fix was model-specific. The converter now discards source coverage receipts for both variants and rebuilds each tier's receipt. The regression exercises both IDs, preserving the original read-only source receipt and Medium's original duration key. This affects conversion bookkeeping only; no served arithmetic or stored learned tensor formula changed. All nine learned archive SHA256s (three components × three precisions) exactly match the historical snapshots; `medium-weight-parity.json` records the comparison. Completion also writes `intake-latest.json`; the older pointer was caught by the contract suite and refreshed from the existing verified run binding without repeating models.

Medium's fixed-tensor regression compares the original `decoder(input)` output with `decodeWindow` and production `sa3DecodeLatents`, and independently reconstructs the original duration Fourier-feature/matmul expression. Decoder and conditioner outputs must be exactly equal. The separately retained native verification records the executed checks; the first concurrent attempt trapped before assertions completed, and its result is not counted as a pass.

| Tier | Verdict vs accepted FP16 | Max component relative L2 | Min cosine | Log-mel diagnostic dB vs FP16 | Warm median s / 5-s clip | Lifetime peak GB |
|---|---|---:|---:|---:|---:|---:|
| FP16 | Pass | 0 | 1 | 0 | 0.205531 | 6.197138 |
| int8 | Pass | 0.015139 | 0.999886 | 0.609363 | 0.226130 | 3.947433 |
| int4 | Fail; offered with warning | 0.228570 | 0.973705 | 4.736615 | 0.217145 | 2.735278 |

The five independently teacher-forced outputs are prompt, denoiser at steps 0/3/7 and SAME-L decoder; exact keys, shapes and counts precede value checks. Quantized bounds remain relative L2 ≤0.05, cosine ≥0.999, finite, no new clamps and FP16 activation range. The new table writer handles both Sound variants and binds each emitted stat to the exact saved run/result, source closure, worker SHA, default Load identity, recipe and bounds. Int4 remains offered with the standing orange warning rather than inheriting the older unavailable state.

Warm results are busy-Mac quick checks with four requests per arm in balanced ABBA BAAB windows; compare each candidate to its paired baseline and retained drift, not to an unrelated historical timing. No speed improvement is claimed. Log-mel is a saved-output diagnostic only. Peak GB is each worker's lifetime physical footprint including Load and component probes. This run completed in 51.119 seconds under a 48.306-GB worker cap; aggregate sampled worker peak was 8.359 GB. Existing swap 1,023,279,104 bytes did not grow. All live runs have a 590-second ceiling, and source-hash, cleanup and closing verification receipts accompany the result. Independent vendor qualification is not newly established. No merge, push, signing or installation was performed.
