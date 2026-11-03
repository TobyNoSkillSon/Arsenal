+++
hypothesis = "The MLX kit runner matches the maker's TIGER (look2hear/models/tiger.py) on the same checkpoint."
change = "None: measurement of the Tiger.swift / Model.swift port on the original FP32 file (the runner casts it to FP16 at load)."
measurement = "lab/night2/tiger-speech/ref.py vs the runner's raw float32 tracks (ARSENAL_PARITY_DIR), compare.py, m5-max, 2026-10-31: two 16 kHz mixtures."
verdict = "kept"
numbers = { cosine_min_vs_fp32 = 0.9999993, cosine_min_vs_fp16_autocast = 0.999461, max_abs_diff_vs_fp32 = 0.000606 }
links = []
+++
# 003 Parity with the maker's TIGER (2026-10-31)

Method: the maker's `look2hear/models/tiger.py` (github.com/JusperLee/TIGER @ 9f18d4a), imported unchanged, built from the
pinned config.json and loaded strictly from the pinned `model.safetensors`, run on MPS (PyTorch 2.15.0, M5 Max) on the same
mono 16 kHz file as the runner. Compared per output track over every sample: cosine and max abs difference.

Reference precision: `model.half()` cannot run the maker's code (UConvBlock builds its global sum with `torch.zeros()` in
float32, so the next half Conv1d fails on mixed dtypes), so the float16 reference is `torch.autocast("mps", float16)` over the
FP32 weights; a float32 run is reported too. Two reference-side shims, no maths changed: `Tensor.type("torch.mps.HalfTensor")`
(rejected by torch 2.15) is mapped to a dtype, and `F.adaptive_avg_pool1d`, which MPS lacks for non-divisible sizes, is the
same floor/ceil window average as a matrix on MPS.

Inputs: `mix.wav` (the makers' example mixture, 6.0 s) and `say-mix.wav` (two macOS voices, Samantha and Daniel, mixed at
0.5 and 0.4 peak, 6.2 s, so the clean sources are known).

| input | track | cosine vs fp32 | max abs diff vs fp32 | cosine vs fp16 autocast |
|---|---|---|---|---|
| mix.wav | speaker1 | 0.9999998 | 0.00041 | 0.999934 |
| mix.wav | speaker2 | 0.9999995 | 0.00041 | 0.999823 |
| say-mix.wav | speaker1 | 0.9999999 | 0.00061 | 0.999889 |
| say-mix.wav | speaker2 | 0.9999993 | 0.00061 | 0.999461 |

The torch autocast run is itself 0.99946 to 0.99993 from torch float32 (same numbers as the right column), so the runner's
FP16 is closer to the maker's float32 than torch's own half path. The makers' `test/s1.wav` and `s2.wav` (their published
outputs for mix.wav) match the torch float32 run at 113 dB and 109 dB SI-SDR, which confirms the reference set-up.

Separation quality on say-mix.wav (SI-SDR against the clean voices): the mix scores 6.8 dB and -7.1 dB; the runner's
speaker1 and speaker2 score 21.2 dB and 14.3 dB, the same as the reference to 0.01 dB.

The runner slices UConvBlock over its batch to bound memory (its GroupNorms are per sequence, so the result is the same) and
evaluates one attention head at a time; these numbers are from that final code.

Not checked: inputs longer than 6.2 s against the reference (a 60-second file runs, checked for completion only); resampled
or stereo input (the runner uses AVFoundation's resampler, the maker torchaudio's).

Verdict: pass (cosine >= 0.999 on every track).
