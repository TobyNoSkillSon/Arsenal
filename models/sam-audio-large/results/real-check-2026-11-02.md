# Real check, 2 Nov 2026

One request through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). M5 Max, Mac in use, on
AC. No speed or memory figures are claimed.

- Get: the pinned `mlx-community/sam-audio-large-fp16` files and the T5-base text encoder imported from the local
  Hugging Face cache, FP16.
- `separate_audio`, prompt "speech", seed 7, on a 3.25 s speech-over-music clip: HTTP 200, target and residual WAVs at
  48 kHz, 156,000 frames each; target RMS 0.0096, residual 0.0028. The target correlates 0.94 with Arsenal's 22 Oct
  target for the same clip (that run had no seed). Not listened to.
- Not checked: a maker reference, time anchors, `remove` mode, clips over ten seconds.
- Re-pin to `facebook/sam-audio-large` is parked: Meta's repository is gated and the Hugging Face account on this Mac
  has not accepted its terms. It ships one 14.9 GB PyTorch pickle (`checkpoint.pt`), so a re-pin also needs a Get
  recipe that reads it and casts to FP16.
