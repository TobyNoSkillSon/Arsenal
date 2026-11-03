+++
hypothesis = "The kit runner's FP16 MLX port matches transformers' ClapModel for laion/clap-htsat-fused."
change = "None: a parity check of the Model.swift port."
measurement = "transformers ClapModel + ClapProcessor on MPS at float16 vs the kit runner on the FP32 originals loaded as FP16, three 48 kHz clips, m5-max, 2026-10-30."
verdict = "kept"
numbers = { audio_embedding_cosine_min = 0.999999, audio_embedding_max_abs_diff = 0.00022, similarity_max_abs_diff = 0.00067, probability_max_abs_diff = 0.0039, argmax_match = "3/3" }
links = ["log/003-parity.md"]
+++
# 003 Parity with the maker's code (2026-10-30)

Method: transformers' `ClapModel.from_pretrained(..., torch_dtype=float16)` and `ClapProcessor` (the maker's
feature extractor: 48 kHz, fusion truncation, repeatpad, 64 HTK mel bins) on MPS at float16, loaded from the pinned
snapshot. MLX side: this runner, driven directly with the same framing `kit try` uses, on the original FP32 checkpoint
loaded as FP16, before the cast, with `return_embedding: true`. Both read the same 48 kHz mono float WAVs, so no
resampling is involved. Labels: "a bell", "a dog barking", "speech", "music". The processor marks a lone short clip
as `is_longer` (it flags one random clip per batch when none is long), so the fusion branch ran in both.
For the 12 s clip the processor picks a random crop in each third; the reference pinned `np.random.choice` to the
middle of each third, which is what the runner always does. Scripts: `lab/night1/clap-htsat-fused/{ref.py,drive.py,compare.py}`.

| clip | audio embedding cosine | max abs diff (embedding) | max abs diff (cosine similarity) | max abs diff (probability) | top label, both |
|---|---|---|---|---|---|
| glass-48k.wav (Glass.aiff, 1.65 s, mono) | 0.999999 | 0.00020 | 0.00050 | 0.0039 | a bell |
| glass-0.3s-48k.wav (first 0.3 s) | 0.999999 | 0.00022 | 0.00035 | 0.0025 | music |
| speech-48k.wav (12.2 s of `say` speech) | 0.999999 | 0.00022 | 0.00067 | 0.00001 | speech |

The label order is identical in all three. Token ids from the reference tokenizer were [0, 102, 10342, 2] for "a bell"
etc.; the text similarities agreeing to 0.0007 shows the Swift tokenizer and text tower match.

Verdict: pass (cosine >= 0.999, identical argmax and ranking).

Not covered: resampling for files not at 48 kHz (AVAudioConverter in the runner; the card leaves resampling to the user)
and the random crop choice for clips over 10 s (the runner is deterministic, the maker's processor is not).
