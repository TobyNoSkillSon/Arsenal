+++
hypothesis = "Distinct held-out cases qualify the final stored voice recipe without adaptive selection leakage."
status = "complete"
change = "Separate frozen selection/heldout fixtures, ordinary stored-pack qualification, and lazy capture diagnostics."
measurement = "M5 Max busy-Mac debug-worker component gates and fresh serving-worker quick checks."
verdict = "kept"
numbers = "Per-tier values and retained half shares are in the table below; int8 passes both sets and int4 fails."
links = ["../results/intake/intake-2026-10-16T01-49-02Z-4eccf9ac-9bbd-4ac1-b6b6-cc51567c3627.json", "../gate/evidence/intake/intake-2026-10-16T01-49-02Z-4eccf9ac-9bbd-4ac1-b6b6-cc51567c3627/qualification-summary.json"]
+++

# Round 5: selection and held-out qualification

Run `intake-2026-10-16T01-49-02Z-4eccf9ac-9bbd-4ac1-b6b6-cc51567c3627` on M5 Max, busy Mac, debug worker. Selected BF16; source-format trial uses the shared BF16 nearest/truncation floor, separate exact A/A repeat and FP16 ULP policy. Selection 66 outputs; held-out 44 outputs. Both fixture plans/inputs and bounds were frozen before matrix search. Shared tier qualification remains relative L2 <=0.05 and cosine >=0.999, finite, no new clamps. Selection-only safety bounds are saved separately; held-out outcomes never drove removals. The final stored recipe was reloaded by ordinary Load for the complete held-out gate. Historical schema-2 voice completion bindings are selection-set passes only.

| Tier | Verdict | Selection max rel L2 | Selection min cosine | Held-out max rel L2 | Held-out min cosine | Spectral (diagnostic) | RTF (compute s/audio s) | Peak (GB) | Retained 16-bit (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 1 | 0 | 0.1218199 | 0.5030555 | 100 |
| int8 | PASS | 0.03696105 | 0.99931789 | 0.03782645 | 0.99929692 | 0.6644291 | 0.1239481 | 0.426198 | 31.54 |
| int4 | FAIL | 0.4821436 | 0.89023095 | 0.50865 | 0.87704971 | 1.109038 | 0.1119351 | 0.3576965 | 13.91 |

Int8 selection strategy: adaptive projection-group removal.

All deltas use the selected 16-bit baseline. Cold WAV common-prefix spectra are diagnostic, not listening, ASR or perceptual fidelity. Fresh serving workers have zero capture requests. Int8 packs 61 matrices; int4 packs 66. Extra int8 half fallback is 17.63% of stored source values. All structural, recurrent-interface, cutoff and layout skips retain tensor-level reasons.

FP16 trial: selection L2 0.13250426, cosine 0.99121809; held-out L2 0.054620761, cosine 0.99850744; selected False.

Pocket separately checks transformer/upsampling output and SEANet waveform output on a saved normalized input (RMS floor .01), under predeclared shared bounds.

Guarded live transaction 42.354s; aggregate worker limit 57.328GB; no increase over initial system swap 1023279104 bytes. APFS source copies hash-checked before/after and deleted. No installed app/support changes, push, signing or main merge. Independent matched vendor half-GPU reference, listening/ASR and long/context-diverse perceptual quality remain open.
