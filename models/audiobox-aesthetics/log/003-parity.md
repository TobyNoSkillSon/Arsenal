+++
hypothesis = "The kit runner's FP16 MLX port matches the maker's PyTorch AesMultiOutput."
change = "None: a parity check of the Model.swift port."
measurement = "Maker's aes.py/wavlm.py on MPS at float16 vs kit try on the FP32 originals loaded as FP16, three 16 kHz clips, m5-max, 2026-10-30."
verdict = "kept"
numbers = { max_abs_diff = 0.0022, cosine_12_scores = 0.99999998 }
links = ["log/003-parity.md"]
+++
# 003 Parity with the maker's code (2026-10-30)

Method: the maker's own `AesMultiOutput` (github.com/facebookresearch/audiobox-aesthetics at 2618e9d, `model/aes.py` and
`model/wavlm.py`) loaded from the pinned `model.safetensors` (strict), run on MPS at float16 (`model.half()`), with
`infer.py`'s windowing copied as-is (10 s windows, hop 10 s, zero padding with a mask, length-weighted mean) and the
card's `target_transform` inverse. Script: `lab/night1/audiobox-aesthetics/ref.py` in the Arsenal repo. MLX side: this
runner through `kit try`, original FP32 checkpoint loaded as FP16, before the cast. Both read the same 16 kHz mono
float WAVs, so no resampling is involved.

Inputs (made from macOS system sounds and `say`):
- glass-16k.wav: /System/Library/Sounds/Glass.aiff, 1.65 s, converted to 16 kHz mono with afconvert
- speech-16k.wav: 12.2 s of `say` speech (two windows, the second padded)
- glass-0.3s-16k.wav: the first 0.3 s of glass-16k.wav (awkward: 97% of the window is padding)

Decisive output: the four axis scores (CE, CU, PC, PQ, about 1-10).

| clip | max abs diff over the 4 axes |
|---|---|
| glass-16k.wav | 0.0000004 |
| speech-16k.wav | 0.0012 |
| glass-0.3s-16k.wav | 0.0022 |

Cosine over all 12 scores: 0.99999998. Every diff is within one or two float16 steps of the head's output
(about 0.001 on this scale after the inverse transform).

Verdict: pass.

Not covered: resampling. Arsenal resamples non-16 kHz files with AVAudioConverter (max quality); the maker uses
torchaudio's sinc resampler, which is not in the reference env. On Glass.aiff (48 kHz stereo) the runner's own
resampled path gives CE 3.85 against 3.94 for the afconvert 16 kHz copy, so resampler choice moves scores by up to
about 0.1 on short transient clips. Unmeasured against torchaudio.
