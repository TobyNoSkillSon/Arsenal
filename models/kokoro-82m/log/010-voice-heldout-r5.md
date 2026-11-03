+++
hypothesis = "Distinct held-out cases qualify the final stored voice recipe without adaptive selection leakage."
status = "complete"
change = "Separate frozen selection/heldout fixtures, ordinary stored-pack qualification, and lazy capture diagnostics."
measurement = "M5 Max busy-Mac debug-worker component gates and fresh serving-worker quick checks."
verdict = "kept"
numbers = "Per-tier values and retained half shares are in the table below; int8 passes both sets and int4 fails."
links = ["../results/intake/intake-2026-10-16T01-49-45Z-ba782bdc-ce08-4873-9f5d-cef39bce1835.json", "../gate/evidence/intake/intake-2026-10-16T01-49-45Z-ba782bdc-ce08-4873-9f5d-cef39bce1835/qualification-summary.json"]
+++

# Round 5: selection and held-out qualification

Run `intake-2026-10-16T01-49-45Z-ba782bdc-ce08-4873-9f5d-cef39bce1835` on M5 Max, busy Mac, debug worker. Selected FP16; source-format trial uses the shared BF16 nearest/truncation floor, separate exact A/A repeat and FP16 ULP policy. Selection 33 outputs; held-out 22 outputs. Both fixture plans/inputs and bounds were frozen before matrix search. Shared tier qualification remains relative L2 <=0.05 and cosine >=0.999, finite, no new clamps. Selection-only safety bounds are saved separately; held-out outcomes never drove removals. The final stored recipe was reloaded by ordinary Load for the complete held-out gate. Historical schema-2 voice completion bindings are selection-set passes only.

| Tier | Verdict | Selection max rel L2 | Selection min cosine | Held-out max rel L2 | Held-out min cosine | Spectral (diagnostic) | RTF (compute s/audio s) | Peak (GB) | Retained 16-bit (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fp16 | PASS | 0 | 1 | 0 | 1 | 0 | 0.02889388 | 1.386104 | 100 |
| int8 | PASS | 0.01977892 | 0.99980534 | 0.02073692 | 0.99978635 | 0.08393894 | 0.02907008 | 1.071744 | 85.1 |
| int4 | FAIL | 0.3296547 | 0.94541153 | 0.3078881 | 0.95200122 | 0.6546086 | 0.02939616 | 1.036863 | 85.1 |

Int8 selection strategy: adaptive projection-group removal.

All deltas use the selected 16-bit baseline. Cold WAV common-prefix spectra are diagnostic, not listening, ASR or perceptual fidelity. Fresh serving workers have zero capture requests. Int8 packs 64 matrices; int4 packs 64. Extra int8 half fallback is 0% of stored source values. All structural, recurrent-interface, cutoff and layout skips retain tensor-level reasons.

FP16 trial: selection L2 0.092641443, cosine 0.9957; held-out L2 0.086117295, cosine 0.99629683; selected True.

Kokoro includes the pinned BART phonemizer: each source tensor is rounded before module construction, eligible projections packed tensor-wise at ordinary Load, original FP32 arrays discarded, all persistent learned floating parameters checked against the selected half. Its sparse packs use the same independent scalar-unpack audit as main weights. Main model packs are prederived by Get; the small pinned phonemizer asset derives at Load.

Guarded live transaction 156.857s; aggregate worker limit 57.032GB; no increase over initial system swap 1023279104 bytes. APFS source copies hash-checked before/after and deleted. No installed app/support changes, push, signing or main merge. Independent matched vendor half-GPU reference, listening/ASR and long/context-diverse perceptual quality remain open.
