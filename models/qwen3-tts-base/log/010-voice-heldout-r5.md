+++
hypothesis = "Distinct held-out cases qualify the final stored voice recipe without adaptive selection leakage."
status = "complete"
change = "Separate frozen selection/heldout fixtures, ordinary stored-pack qualification, and lazy capture diagnostics."
measurement = "M5 Max busy-Mac debug-worker component gates and fresh serving-worker quick checks."
verdict = "kept"
numbers = "Per-tier values and retained half shares are in the table below; int8 passes both sets and int4 fails."
links = ["../results/intake/intake-2026-10-16T01-52-22Z-267e7abd-0dc9-4a20-b3f6-8bcddc8dd9d0.json", "../gate/evidence/intake/intake-2026-10-16T01-52-22Z-267e7abd-0dc9-4a20-b3f6-8bcddc8dd9d0/qualification-summary.json"]
+++

# Round 5: selection and held-out qualification

Run `intake-2026-10-16T01-52-22Z-267e7abd-0dc9-4a20-b3f6-8bcddc8dd9d0` on M5 Max, busy Mac, debug worker. Selected BF16; source-format trial uses the shared BF16 nearest/truncation floor, separate exact A/A repeat and FP16 ULP policy. Selection 303 outputs; held-out 202 outputs. Both fixture plans/inputs and bounds were frozen before matrix search. Shared tier qualification remains relative L2 <=0.05 and cosine >=0.999, finite, no new clamps. Selection-only safety bounds are saved separately; held-out outcomes never drove removals. The final stored recipe was reloaded by ordinary Load for the complete held-out gate. Historical schema-2 voice completion bindings are selection-set passes only.

| Tier | Verdict | Selection max rel L2 | Selection min cosine | Held-out max rel L2 | Held-out min cosine | Spectral (diagnostic) | RTF (compute s/audio s) | Peak (GB) | Retained 16-bit (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 1 | 0 | 0.4937418 | 4.819866 | 100 |
| int8 | PASS | 0.03667861 | 0.99936009 | 0.03880428 | 0.9992498 | 0.9906702 | 0.510431 | 4.394537 | 79.74 |
| int4 | FAIL | 1.078847 | 0.62152726 | 0.6622083 | 0.79573071 | 1.053623 | 0.4535459 | 1.963969 | 4.713 |

Int8 selection strategy: fixed prior selection candidate; no minimality claim.

All deltas use the selected 16-bit baseline. Cold WAV common-prefix spectra are diagnostic, not listening, ASR or perceptual fidelity. Fresh serving workers have zero capture requests. Int8 packs 83 matrices; int4 packs 393. Extra int8 half fallback is 75.03% of stored source values. All structural, recurrent-interface, cutoff and layout skips retain tensor-level reasons.

FP16 trial: selection L2 0.49437393, cosine 0.87897749; held-out L2 0.38610063, cosine 0.92669916; selected False.



Guarded live transaction 183.281s; aggregate worker limit 57.405GB; no increase over initial system swap 1023279104 bytes. APFS source copies hash-checked before/after and deleted. No installed app/support changes, push, signing or main merge. Independent matched vendor half-GPU reference, listening/ASR and long/context-diverse perceptual quality remain open.
