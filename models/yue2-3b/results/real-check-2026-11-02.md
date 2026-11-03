# Real check, 2 Nov 2026

One request through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from
the committed folder, Hugging Face client on loopback so Get could only import the local copy). M5 Max, Mac in use, on
AC. No speed or memory figures are claimed.

- Get: the pinned `m-a-p/YuE2-3B` model and `m-a-p/YuE2-Vae` imported from the local Hugging Face cache and cast to
  FP16 (7.8 GB stored).
- `generate_music`, "calm acoustic folk, female vocal", two lyric lines, 10 s, seed 7: HTTP 200, one 48 kHz stereo
  WAV, 9.999 s, RMS 0.099, peak 0.89, all samples finite. Not listened to, so whether the vocal sings the lyrics is
  unchecked.
- Not checked: a maker reference (M-A-P's PyTorch pipeline was not run), score (ABC) planning, instrumental mode,
  longer songs, int8 and int4.
