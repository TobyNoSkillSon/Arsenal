+++
hypothesis = "A native Swift MLX port of mlx-audio fish_drama3 (dad2087f) reproduces the reference on the vendor 4/8-bit checkpoint."
change = "New FishDrama3 (Qwen3.5-MoE backbone on frame embeddings, codebook tower, sampler, prompt, chunking) and FishS1Codec (Fish S1 DAC with streaming decode) targets in the voices family."
measurement = "CPU unit tests plus fixed-tensor parity against the reference on the real 4/8-bit weights and end-to-end sentences through the native worker (quick checks, M5 Max, Mac in use)."
verdict = "kept"
numbers = { checkpoint_tensors = 1841, layer_relative_l2_max = 0.014, decode_steps_checked = 104, codes_agreement_float32 = 1.0, codes_agreement_fp16 = 0.933, peak_footprint_mb = 25548 }
links = ["README.md", "gate/gate.toml", "Workers/Voices/PORTED-FROM.md"]
+++

Reference: mlx-audio `mlx_audio/tts/models/fish_drama3` and `mlx_audio/codec/models/fish_s1_dac` at dad2087f (the merge of PR #1018). Weights: the 4-bit snapshot above, used as stored. The text rules and tokenizer are checked against outputs captured once from the reference; every backbone and tower tensor maps to a module parameter with the stored shape. Fixed-tensor parity on the real weights and one end-to-end sentence are pending a GPU window.

Parity, captured once from the reference and replayed natively (`Workers/Voices/Tests/VoicesTests/FishDrama3ParityTests.swift`):
- Embeddings equal exactly. Each of the 40 backbone layers on the reference's own inputs: gated-delta rel-L2 ≤ 0.0045, attention ≤ 0.0065, MoE ≤ 0.014 (cosine ≥ 0.9999). Whole prompt prefill: rel-L2 0.039, cosine 0.9992, the accumulated bf16 rounding of 40 layers with expert routing.
- Semantic head logits rel-L2 0.028, same argmax; tower logits rel-L2 0.008, same argmax for all 9 codebooks.
- Teacher-forced decode over 16 + 32 + 56 steps: per-step semantic rel-L2 0.005 to 0.044 without growth; argmax differs only where the reference's top-two margin is 0 to 0.25 (bf16 ties). P(EOS) along a run that ended by itself: 0.768 against 0.826 at the end, mean 0.0167 against 0.0174.
- Sampler on saved logits: empirical draws match the reference distribution (mean total variation 0.014 over 32 rows, at the sampling-noise floor); the two larger rows differ only in which of two tied tokens is kept.
- Codec: float32 encode equals the reference's codes exactly; the FP16 runtime codec agrees on 93% (near-tie flips in the VQ argmax). FP16 streamed decode rel-L2 0.0088 against the float32 reference.

End to end through the native worker (load int4, speak, clone_voice, speak with the saved voice, voices, unload): a cloned sentence transcribes exactly ("The tests are green. And the voice was cloned.", Parakeet v3). Unprompted voices (no reference) finished "Done. The build passed." in 3 of 11 seeds natively and 7 of 11 in the reference; the rest stop after "Done." and fill the 96-frame cap with silence in both implementations. With P(EOS) and the sampler matching, this reads as sampling variance near bf16 ties, but the sample is small.


Catalog declaration (integration, 22 Oct): the int4 tier is the mlx-community 4-bit release pinned in `[vendor.int4]` and served as stored, not derived, so the repository's int8-with-int4 rule for derived tiers does not apply and no int8 is declared (quantized tiers are optional; the 16-bit release stays listed with no runner). The folder stays schema 2 because the voices family's streamed speak protocol has no schema-3 output schemas. Params read "35B" in the table's one format; the backbone activates about 3B per token.
