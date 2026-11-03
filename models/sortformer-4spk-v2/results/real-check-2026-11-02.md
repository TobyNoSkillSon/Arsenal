# Real check, 2 Nov 2026

One run through Arsenal's native audio worker (the debug build of this branch) on an isolated profile
(ARSENAL_SUPPORT_DIR / ARSENAL_CACHE_DIR in a temporary folder), under the GPU lock. M5 Max, Mac in use, on battery.
No speed or memory figures are claimed.

- Get's conversion `cast:audio-sortformer-v2-v1` on an APFS clone of the pinned `.nemo`: FP16 snapshot of 235.5 MB.
- Load at fp16, then `diarize` on the 27.5 s three-voice fixture with default settings (30.4 s latency): two speakers,
  five segments, identical to NeMo's own `diarize` segments at the same geometry.
- `latency: "ultra_low"` with probabilities: 344 frames, three speakers.
- `diarize_session` at `low` latency fed in 2 s pieces: results grew as audio arrived; the finished result equals
  `diarize` on the whole file at the same latency.
- A 0.3 s clip answers; `num_speakers` (a pyannote option) is refused as invalid_request.
- Not checked: the app-level Get/Load through arsenal-testserver (its real mode lists fixed model ids), real
  recordings, four speakers, non-English speech.
