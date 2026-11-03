# Real check, 2 Nov 2026

One request through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). M5 Max, Mac in use, on
AC. No speed or memory figures are claimed.

- Get: the pinned `stabilityai/stable-audio-3-optimized` MLX files imported from the local Hugging Face cache and cast
  to FP16.
- `generate_sound`, "Rain on a tin roof", 2 s, 8 steps, seed 1: HTTP 200, 44.1 kHz stereo, 88,200 frames, RMS 0.044,
  peak 0.64, all samples finite. Not listened to.
- Not checked: a maker reference (Stability's own code was not run), `generate_music`, longer clips, int8.
- Re-pin to `stabilityai/stable-audio-3-medium` is parked: that repository is gated and the Hugging Face account on this
  Mac has not accepted its terms, so its files could not be compared. Its LICENSE.md is byte-identical to the pinned
  one (SHA-256 d6f6b1a4…). It ships one 9.2 GB PyTorch `model.safetensors` instead of the MLX `.npz` parts, so a
  re-pin needs a new Get recipe.
