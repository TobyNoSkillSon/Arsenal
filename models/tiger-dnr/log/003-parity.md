+++
hypothesis = "The MLX kit runner matches the maker's TIGERDNR (look2hear/models/tiger_dnr.py) on the same checkpoint."
change = "None: measurement of the Tiger.swift / Model.swift port on the original FP32 file (the runner casts it to FP16 at load)."
measurement = "lab/night2/tiger-speech/ref.py (dnr) vs the runner's raw float32 stems (ARSENAL_PARITY_DIR), compare.py, m5-max, 2026-10-31: 15 s of the makers' example clip, plus the full 60 s dialogue against the makers' published output."
verdict = "kept"
numbers = { cosine_min_vs_fp32 = 0.9999998, cosine_min_vs_fp16_autocast = 0.9999998, max_abs_diff_vs_fp32 = 0.000166, cosine_dialog_60s_vs_published = 0.999999 }
links = []
+++
# 003 Parity with the maker's TIGERDNR (2026-10-31)

Method: the maker's `look2hear/models/tiger_dnr.py` (github.com/JusperLee/TIGER @ 9f18d4a), imported unchanged, built from
the pinned config.json and loaded strictly from the pinned `model.safetensors`, run with its own `forward` (three TIGER
networks through `wav_chunk_inference`, 12 s windows every 4 s) on MPS (PyTorch 2.15.0, M5 Max). The reference set-up, its
float16 autocast and its two shims are the same as tiger-speech's (`model.half()` cannot run the maker's code; see that log).

Input: `mix466-15s.wav`, seconds 20 to 35 of the makers' example `test/test_mixture_466.wav` (44.1 kHz mono, 16-bit), so six
windows are separated and overlap-added per stem.

| stem | cosine vs fp32 | max abs diff vs fp32 | cosine vs fp16 autocast | max abs diff vs fp16 |
|---|---|---|---|---|
| dialog | 1.0000000 | 0.00010 | 1.0000000 | 0.00009 |
| effects | 1.0000000 | 0.00011 | 1.0000000 | 0.00010 |
| music | 0.9999998 | 0.00017 | 0.9999998 | 0.00015 |

Full clip: the runner's dialog stem for the whole 60 s `test_mixture_466.wav` (written as its 16-bit WAV) matches the makers'
published `test/test_target_dialog_466.wav` at cosine 0.999999, max abs diff 0.0009.

Mono: the maker's chunker effectively uses the first channel's estimate for every channel; the runner mixes the input to mono
before separation. The test clip is mono, so this does not affect the numbers above.

Not checked: the effects and music stems over the full 60 s against the published files (only dialog was compared there);
resampled input (AVFoundation vs torchaudio).

Verdict: pass (cosine >= 0.999 on every stem).
