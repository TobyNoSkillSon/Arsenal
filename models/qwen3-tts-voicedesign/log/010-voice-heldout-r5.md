+++
hypothesis = "Distinct held-out cases qualify the final stored voice recipe without adaptive selection leakage."
status = "complete"
change = "Separate frozen selection/heldout fixtures, ordinary stored-pack qualification, and lazy capture diagnostics."
measurement = "M5 Max busy-Mac debug-worker component gates and fresh serving-worker quick checks."
verdict = "kept"
numbers = "Per-tier values and retained half shares are in the table below; int8 passes both sets and int4 fails."
links = ["../results/intake/intake-2026-10-16T01-55-29Z-e2b19fea-30e4-4a6f-8ded-443c9defa410.json", "../gate/evidence/intake/intake-2026-10-16T01-55-29Z-e2b19fea-30e4-4a6f-8ded-443c9defa410/qualification-summary.json"]
+++

# Round 5: selection and held-out qualification

Run `intake-2026-10-16T01-55-29Z-e2b19fea-30e4-4a6f-8ded-443c9defa410` on M5 Max, busy Mac, debug worker. Selected BF16; source-format trial uses the shared BF16 nearest/truncation floor, separate exact A/A repeat and FP16 ULP policy. Selection 300 outputs; held-out 200 outputs. Both fixture plans/inputs and bounds were frozen before matrix search. Shared tier qualification remains relative L2 <=0.05 and cosine >=0.999, finite, no new clamps. Selection-only safety bounds are saved separately; held-out outcomes never drove removals. The final stored recipe was reloaded by ordinary Load for the complete held-out gate. Historical schema-2 voice completion bindings are selection-set passes only.

| Tier | Verdict | Selection max rel L2 | Selection min cosine | Held-out max rel L2 | Held-out min cosine | Spectral (diagnostic) | RTF (compute s/audio s) | Peak (GB) | Retained 16-bit (%) |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| bf16 | PASS | 0 | 1 | 0 | 1 | 0 | 0.5054157 | 4.754919 | 100 |
| int8 | PASS | 0.03001209 | 0.99954954 | 0.02099625 | 0.99977961 | 1.10909 | 0.5153213 | 4.382216 | 80.17 |
| int4 | FAIL | 1.697369 | 0.00069510485 | 0.9110336 | 0.4219871 | 1.095715 | 0.4564121 | 1.876838 | 4.712 |

Int8 selection strategy: fixed prior selection candidate; no minimality claim.

All deltas use the selected 16-bit baseline. Cold WAV common-prefix spectra are diagnostic, not listening, ASR or perceptual fidelity. Fresh serving workers have zero capture requests. Int8 packs 67 matrices; int4 packs 377. Extra int8 half fallback is 75.46% of stored source values. All structural, recurrent-interface, cutoff and layout skips retain tensor-level reasons.

FP16 trial: selection L2 0.49437393, cosine 0.87897749; held-out L2 0.42145307, cosine 0.91143007; selected False.



Guarded live transaction 154.817s; aggregate worker limit 57.117GB; no increase over initial system swap 1023279104 bytes. APFS source copies hash-checked before/after and deleted. No installed app/support changes, push, signing or main merge. Independent matched vendor half-GPU reference, listening/ASR and long/context-diverse perceptual quality remain open.
