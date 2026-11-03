+++
hypothesis = "The fresh review's findings are fixed without changing the parity result."
change = "Blocked attention queries (1,024 rows) and per-layer evaluation; non-finite slot logits refused; NaN written as json.dumps does; keys deduplicated by UTF-8 bytes; schemas allow null instructions and extra noul criteria keys; PORTED-FROM.md and upstream licences; trained-length note; tests compare bytes, check token and slot ids and the maker's decide mapping without weights."
measurement = "Weight-free suite (11 tests) and heavy parity rerun; long-input worker runs at 16,039 and 32,391 tokens with whole-Mac memory sampled every 0.5 s."
verdict = "kept"
numbers = { max_abs_dp_vs_maker_fp16 = 0.000974, max_abs_dp_vs_maker_bf16 = 0.011567, worst_layer_cosine = 0.999992, peak_footprint_gb_16k = 19.6, peak_footprint_gb_32k = 22.5, mac_peak_gb_32k = 49.2 }
links = ["README.md", "results/real-check-2026-11-02.md", "runners/swift/PORTED-FROM.md"]
+++

# 002 Review fixes (2 Nov 2026)

A fresh Opus review found no answer-changing bug on ordinary requests. Fixed, each with a test:

1. Long inputs: attention queries run in blocks of 1,024 rows against all keys (exact, the mask is none; a heavy test
   checks blocked against whole on fixed tensors), every layer is evaluated before the next, and the prefix state is
   evaluated after prefill. Measured: 16,039 tokens peak 19.6 GB, 32,391 tokens peak 22.5 GB (worker footprint), no
   swap, so the makers' 32,768-token limit stays.
2. Provenance: PORTED-FROM.md with the OneJev (Apache-2.0), mlx-swift-lm (MIT) and Verdict (Apache-2.0) licence files.
3. A non-finite slot logit is refused (`non_finite`) instead of becoming an answer (a NaN P(yes) was clamped to 0.0).
4. PyJSON writes NaN, Infinity and -Infinity as json.dumps does (it wrote "nan").
5. Question names and option names are deduplicated by UTF-8 bytes, as Python's dict keys are, not by Swift's canonical
   String equality.
6. Schemas: `instructions` may be null; noul criteria accept extra keys, which the makers' schema ignores.
7. README and model.toml: trained to 16,384 tokens, longer requests accepted but unverified beyond one check.
8. Tests: prompts and answers compared as UTF-8 bytes; token ids and slot ids against the maker's from the tokenizer
   alone; the maker's own `decide()` with stubbed logits (including a choice split into chunks) against
   `ElephantEngine.assemble`; the fixture's generator name corrected.
9. The M-RoPE comments now say why text needs only ordinary RoPE.
