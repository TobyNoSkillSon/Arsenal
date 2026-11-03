# Real check, 2 Nov 2026 (rerun after review)

Runs through Arsenal's shipped native audio worker binary (the debug build of this branch, rebuilt after the review fixes) over ArsenalWire, headless, on an isolated profile (ARSENAL_SUPPORT_DIR / ARSENAL_CACHE_DIR in a temporary folder), under the GPU lock. M5 Max, Mac in use, on battery. No speed or memory figures are claimed as measurements of the model.

- Get's conversion `cast:audio-diarizen-v1` on APFS clones of the pinned files from the Hugging Face cache (no download): FP16 snapshot of 152 MB, with the WeSpeaker asset in `embedding/`.
- Load at fp16, then `diarize` on the maker's 30 s AMI example (EN2002a): four speakers, 14 segments, the heavy parity test's segments with the last one ended at the recording's 30.000 s (the maker's ends at 30.392 s). Unclipped, the port is 4.3% DER from the maker's FP32-embedding run and 21.0% from its FP16 run; the maker's two runs differ by 16.7%.
- `diarize` on the 27.5 s three-voice fixture: six segments, exactly the maker's (two of the three macOS voices share a label in both).
- A 0.3 s clip (one segment ending at 0.300 s), 5 s of silence (no segments) and a stereo file (first channel) all answer; `num_speakers` is refused as invalid_request.
- A ten-minute recording (the two clips alternated) finished in about a minute (62.9 s wall clock, one run, Mac in use), well inside the 590 s request limit, so the ten-minute limit stays. Seven speakers were found: the repeated clips are not linked across repeats, which this synthetic file cannot judge.
- Swap unchanged across every run; the worker exited cleanly.
- Not checked: the app-level Get/Load through arsenal-testserver (its QA catalog reads committed folders and an allow-list), a real ten-minute meeting, non-English speech, files with more than two channels end to end (covered by a unit test).
