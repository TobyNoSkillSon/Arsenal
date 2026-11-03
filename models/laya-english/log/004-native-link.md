+++
hypothesis = "The decisions family can join the single Native worker binary (one MLX copy) without changing a single Exact answer."
change = "DecisionsWorker library + family dispatch in Workers/Native; SmallMGEMM module renamed VerdictSmallMGEMM (Voices vendors a newer revision under the old name; same source); swift-transformers moved from 1.3.4 to the family pin 1.1.9 (150169bf) so Native resolves one copy; swift-collections pin dropped (unused)."
measurement = "Verdict's gates re-run through the Native binary's arsenal-worker-decisions link under gpulock: Laya 202-item fixture gate 9/9 Exact with 0 mismatches and 9/9 Fast within noise (max |Δp| 0.0066 / 0.0053 / 0.0033 at fp16, 0 changed answers), Von SDK gate 8/8 (max |Δp| unchanged: 0.0 / 0.0069 fp32/fp16 for 1.2), smoke suite 15/15. Library vs fast tokenizer on 1,324 fixture texts per model: identical except the empty string on Laya Multilingual."
verdict = "kept"
numbers = { exact_mismatches = 0, fast_changed_answers = 0, von_violations = 0, suite_pass = 15, native_stripped_bytes_before = 25277608, native_stripped_bytes_after = 26023896, native_delta_bytes = 746288 }
links = ["gate/parity-fp16-exact.json", "gate/parity-fp16.json", "../../docs/evidence/v2-decisions/native-gate.txt", "../../Workers/Decisions/Tests/DecisionsTests/TokenizerAgreementTests.swift"]
+++

# Decisions inside the Native worker

v1 left decisions out of the app to keep the bundle small. Linked into the one Native binary it shares MLX, ArsenalWire,
swift-transformers and the Swift runtime with voices and chat, so it adds 0.71 MiB to the stripped binary (25.28 →
26.02 MB with voices + chat; measured before sound joined).

**swift-transformers 1.3.4 → 1.1.9.** SwiftPM resolves one version per package, and Voices/Chat pin 1.1.9 by revision.
Exact and Fast never touch the library tokenizer (Verdict's fast BPE tokenizers serve them, validated at load), so the
pin only affects the Standard recipe (the `stock_mlx` setting) and the fallback after an optimized-path failure. The new
`TokenizerAgreementTests` compares the library and the fast tokenizer on every text of Verdict's three Laya fixture gates
(rendered inputs, instructions, criteria) for all five tokenizers, with and without special tokens: identical, except
that 1.1.9 encodes the empty string as `[235248]` (▁) with Laya Multilingual's Metaspace pre-tokenizer where the fast
tokenizer and HF tokenizers 0.22.1 give `[]`. Kept: only an empty text on the diagnosis path is affected.

**Module rename.** Two packages may not both define a `SmallMGEMM` module in one graph. Decisions keeps the revision
Verdict's gate qualified (tile kernel only); Voices has a newer one with a GEMV path that would route single-row calls
differently (inexact), so sharing it would have changed Fast's numbers. The kernel names coincide but each process runs
one family, so the two never meet in one MLX kernel cache.
