# TIGER Speech

Splits a recording of two people talking over each other into one clean track per speaker. TIGER is a 0.8M-parameter separation network from Tsinghua University (ICLR 2025) that works on 67 frequency bands of the spectrum and alternates band-wise and frame-wise modelling; the makers' recipe for this size trains on EchoSet, two overlapping speakers with noise and realistic room reverberation.

Input: `audio` (any file AVFoundation opens), and optionally `stems` (`speaker1`, `speaker2`) and `mode` (`extract`, the default, returns the listed stems; `remove` returns the ones not listed). `prompt`, `seed` and `steps` are accepted by the audio kind but ignored. Output: `files`, one 16 kHz mono 16-bit WAV per speaker (`id` and name `speaker1` / `speaker1.wav`, label "Speaker 1"), with `path`, `mime` `audio/wav`, `bytes`, `sha256` and `order`.

Example request:

```json
{"audio": {"path": "/absolute/path/to/two-people.wav"}}
```

How it runs: the audio is resampled to 16 kHz mono and goes through the network in one pass, as the makers' `inference_speech.py` does (STFT window 640, hop 160; the separator block repeats 8 times). Samples outside -1...1 are clipped when the WAV is written.

Limits: always two tracks, and which speaker is `speaker1` is arbitrary; one speaker leaves a residue track, three or more are not separated. Input is 40 ms to 60 s; peak GPU memory was about 2.5 GB for a 60-second file. Trained on speech only, mostly English; other languages, Polish included, were not tested.

Parity: matches the maker's PyTorch TIGER (MPS) at cosine 0.99999 or better against float32 and 0.9994 or better against float16 autocast on two mixtures; on a synthetic two-voice mix the speakers come back at 21.2 dB and 14.3 dB SI-SDR, up from 6.8 dB and -7.1 dB in the mix (`log/003-parity.md`).

Source: [JusperLee/TIGER-speech](https://huggingface.co/JusperLee/TIGER-speech/tree/f0340340b2d9bbf72074edf8c076dcab59a10ba2), Apache-2.0; code [JusperLee/TIGER](https://github.com/JusperLee/TIGER), paper [arXiv 2410.01469](https://arxiv.org/abs/2410.01469). By Mohan Xu, Kai Li, Guo Chen and Xiaolin Hu. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: No rate, channel or length limits given. [model card, 2025-01, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `POST /arsenal/v1/models/tiger-speech/ops/separate_audio`, or the same operation through MCP. Vendor example: `python inference_speech.py --audio_path test/mix.wav` [model card, 2025-01, first-hand]

**Good and bad**
- Good at: 0.82M parameters; 94.3% fewer parameters and 95.3% fewer MACs than TF-GridNet, about 5% better on EchoSet. [arXiv 2410.01469, 2024-10, first-hand]
- Bad at: Synthetic-to-real gap: DAVE's TIGER probe shows 1.31 points for additive-only training and 3.56 for additive fine-tuning. [arXiv 2608.09288, 2026-08-10, first-hand]
- Bad at: Two speakers only. Issue #37 asked about three or more; maintainer: "adjust the config" (untested; no results posted). [JusperLee/TIGER issue #37, maintainer reply, opinion]
- Against the obvious alternatives: Only the paper's TF-GridNet comparison. [arXiv 2410.01469, 2024-10, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Install fails with `No module named 'pkg_resources'` (issue #40, 2026-06-10, no replies). Fix or workaround: none shown. [JusperLee/TIGER issue #40, first-hand]
- Complaint or known bug: CUDA 12.4 requirement (issue #6, open). Maintainer fixed requirements errors in commit b23f8f2 (2025-03-06); CUDA pin removal not confirmed. [JusperLee/TIGER issue #6, maintainer reply]
- Complaint or known bug: Poor separation on non-English speech (issues #11, #16). Maintainer suggested fine-tuning on that language; no user reports it working. [JusperLee/TIGER issues #11, #16, mixed]

**Evidence**
- Strength: thin. Card, paper, one independent paper, five issue threads. No controlled user tests.
- Sources:
  - JusperLee/TIGER-speech model card, https://huggingface.co/JusperLee/TIGER-speech, 2025-01, first-hand.
  - TIGER paper, https://arxiv.org/abs/2410.01469, 2024-10, first-hand.
  - DAVE paper, https://arxiv.org/abs/2608.09288, 2026-08-10, first-hand.
  - JusperLee/TIGER issues #6, #11, #16, #37, #40, read 2026-10-31, mixed.
