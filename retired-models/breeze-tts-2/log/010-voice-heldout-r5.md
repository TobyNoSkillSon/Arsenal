+++
hypothesis = "Distinct held-out cases qualify the final stored voice recipe without adaptive selection leakage."
status = "complete"
change = "Separate frozen selection/heldout fixtures, ordinary stored-pack qualification, and lazy capture diagnostics."
measurement = "M5 Max busy-Mac debug-worker component gates and fresh serving-worker quick checks."
verdict = "kept"
numbers = "Per-tier values and retained half shares are in the table below; int8 passes both sets and int4 fails."
links = ["../results/intake/intake-2026-10-16T00-10-54Z-b463e56c-9fcc-4b95-8fd8-576f20c23a7c.json", "../gate/evidence/intake/intake-2026-10-16T00-10-54Z-b463e56c-9fcc-4b95-8fd8-576f20c23a7c/qualification-summary.json"]
+++

# Round 5: selection and held-out qualification

Run `intake-2026-10-16T00-10-54Z-b463e56c-9fcc-4b95-8fd8-576f20c23a7c` on M5 Max, busy Mac, debug worker. Selected BF16; source-format trial uses the shared BF16 nearest/truncation floor, separate exact A/A repeat and FP16 ULP policy. Selection 69 outputs; held-out 46 outputs. Both fixture plans/inputs and bounds were frozen before matrix search. Shared tier qualification remains relative L2 <=0.05 and cosine >=0.999, finite, no new clamps. Selection-only safety bounds are saved separately; held-out outcomes never drove removals. The final stored recipe was reloaded by ordinary Load for the complete held-out gate. Historical schema-2 voice completion bindings are selection-set passes only.

| Tier | Verdict | Selection max rel L2 | Selection min cosine | Held-out max rel L2 | Held-out min cosine | Spectral (diagnostic) | RTF (compute s/audio s) | Peak (GB) | Retained 16-bit (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 1 | 0 | 1.50025 | 8.504761 | 100 |
| int8 | PASS | 0.04373285 | 0.99912295 | 0.03936104 | 0.99923266 | 0.9053963 | 1.493463 | 4.953297 | 2.753 |
| int4 | FAIL | 0.7349715 | 0.73506007 | 0.8085983 | 0.66692316 | 0.9209195 | 1.475121 | 3.285634 | 2.717 |

Int8 selection strategy: adaptive projection-group removal.

All deltas use the selected 16-bit baseline. Cold WAV common-prefix spectra are diagnostic, not listening, ASR or perceptual fidelity. Fresh serving workers have zero capture requests. Int8 packs 591 matrices; int4 packs 594. Extra int8 half fallback is 0.03616% of stored source values. All structural, recurrent-interface, cutoff and layout skips retain tensor-level reasons.

FP16 trial: selection L2 0.49437393, cosine 0.87897749; held-out L2 0.38610063, cosine 0.92669916; selected False.



Guarded live transaction 353.173s; aggregate worker limit 56.561GB; no increase over initial system swap 1023279104 bytes. APFS source copies hash-checked before/after and deleted. No installed app/support changes, push, signing or main merge. Independent matched vendor half-GPU reference, listening/ASR and long/context-diverse perceptual quality remain open.
