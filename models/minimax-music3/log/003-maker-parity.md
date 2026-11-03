+++
hypothesis = "The Swift port reproduces the maker's MiniMax Music 3 components within the maker's own half-precision rounding floor, and makes intelligible songs."
change = "Captured the maker's diffusers modules once on PyTorch MPS at BF16 and FP16 (prefill, six teacher-forced frames, fusion encoder, flow transformer, a full window, the vocoder, and two spliced windows through the maker's own blocks with their noise recorded); compared the port per component; one real Get, Load and generate_music through the isolated service; Parakeet transcription of the song."
measurement = "Relative L2 against the maker's BF16 outputs, bound frozen before the first comparison at twice the maker's FP16-vs-BF16 difference plus 0.001; vocoder probe against the maker's FP16 vocoder with a float32 snake, bound 0.05; Get's vocoder kernels against a float32 fold of the FP32 source."
verdict = "kept"
numbers = { prefill_hidden = 0.0105, audio_logits = 0.0073, top50_overlap = 49, frames = 0.0100, depth_logits = 0.0146, condition = 0.0038, velocity_max = 0.0328, window = 0.0200, spliced_windows = 0.0121, vocoder_probe = 0.0131, fold_max = 0.00038, song_seconds = 20.016, lyric_words_heard = 0.5 }
links = ["README.md", "gate/golden/golden.json", "results/real-check-2026-11-02.md", "log/002-bf16-global-lm.md"]
+++

Every component sits at or under the maker's own FP16-vs-BF16 difference, and against the maker's FP16 run the flow transformer and windows agree to about 0.001. The vocoder's BF16 floor is wide (BF16 nearest against truncation of the same weights differ by 0.49), so it was settled with an op-level probe: the maker's vocoder at FP16 with only its snake in float32, as Arsenal computes it, agrees to 0.013. Get folds the vocoder's weight normalization after the FP16 source normalization; each stored kernel is within twice one FP16 rounding of a float32 fold from the raw source (worst 3.8e-4 against 2.1e-4), so no raw-source exception is needed.

The real check made a 20-second song at full length; Parakeet heard both verse lines (every word but "the") and none of the chorus within those 20 seconds. Musical quality has not been judged by ear.

A second session the same day made a 60-second song with the same four lines and a 15-second instrumental. Parakeet heard every given line in the 60-second song, the chorus twice, followed by lines the model wrote itself once the lyrics ran out; the 13 window joins rank at the song's median for level, spectral and sample-step change (none systematically higher); the instrumental had no words.
