# Real check, 2 Nov 2026

One request through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). M5 Max, Mac in use, on
AC. No speed or memory figures are claimed.

- Get: the pinned `mlx-community/Nemotron-3-Diarization` files imported from the local Hugging Face cache, FP16.
- `diarize` on a 3.25 s speech-over-music clip: HTTP 200, one speaker, 0.22 to 2.83 s. Identical to Arsenal's
  22 Oct 2026 reply for the same file.
- Not checked: a maker reference (NVIDIA's NeMo code was not run), several speakers, overlapping speech, the streaming
  `diarize_session` operation. NVIDIA's own repository uses a different tensor layout (417 tensors against 363), so
  the pin stays on the MLX conversion.
