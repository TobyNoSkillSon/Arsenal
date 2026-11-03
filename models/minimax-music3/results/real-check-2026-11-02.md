# Real check, 2 Nov 2026

One request through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from the committed folder at 312ceee6, Hugging Face client on loopback so Get could only import the local copy). M5 Max, Mac in use, on AC. Headless: no window opened. No speed figures are claimed.

- Get: the 23 pinned files (28.5 GB) imported from the local Hugging Face cache by APFS clone, normalised to FP16 and laid out by `cast:minimax-music3-v1` (22 GB stored). The Models table row reads "MiniMax Music 3", non-commercial, 11.7B, fp16.
- Load: HTTP 200; the loaded worker's footprint was 22.4 GB (global language model cast back to BF16, head trimmed to its 16,385 sampled rows). Whole Mac peaked at 56 GB; swap stayed at its prior 2,093 MB.
- `generate_music`, an acoustic-pop description with a clear female vocal, four lyric lines (two verse, two chorus), 20 s, seed 7: HTTP 200, one 44.1 kHz stereo 16-bit WAV of 20.016 s and its JSON receipt (500 frames, not ended early by the model, 93 prompt tokens). RMS 0.130 / 0.121, peak 0.96, all samples finite, left/right correlation 0.60, level varying second to second. Not played.
- Lyrics: Parakeet v3 Ultra, run in the audio worker from an APFS clone of the installed snapshot, transcribed "Morning light upon river. Golden waves are rolling slow." The verse is sung intelligibly (every word but "the"); the two chorus lines were not recognised within the 20 s, so 11 of the 22 lyric words were heard (WER 0.59).

## Second session: a 60 s song and an instrumental

Same isolated path (fresh profile, folder at 279cb510, Get imported and converted again), under the GPU lock. Whole Mac peaked at 52 GB; swap stayed at 2,093 MB. Files saved, never played.

- `generate_music`, the same description and four lyric lines, 60 s, seed 7: HTTP 200, 44.1 kHz stereo WAV of 60.07 s (1,500 frames, not ended early), RMS 0.146 / 0.137, quietest second 0.047 and loudest 0.246, all samples finite; 13 of 5.3 million samples at full scale (the output is clamped to ±1, as the maker's is).
- Lyrics: Parakeet heard every line: "Morning light upon river. Golden waves are rolling slow. Sing with me the morning song, hold the light and carry on." The chorus came twice, word for word. After the given lyrics ran out the model sang lines of its own ("Let the dearest songs come as set… dance with me…"), so the word error rate against the four given lines is 1.77 while every given word was heard.
- Window joins: the song is 14 flow windows stitched at 13 joins. At each join the level change, spectral change and sample step were ranked against the same measures at all 5,166 latent boundaries of the song: median ranks 48, 46 and 50 percent, spectral change never above 71 percent; the highest level change (12 dB at 5.0 s, 94 percent) and the highest sample step (98.6 percent, at 21.0 s) are single joins, no higher than 13 random points would reach. No systematic discontinuity at joins.
- `generate_music` with `instrumental: true`, an ambient description, 15 s, seed 11: HTTP 200, 15.000 s (375 frames), RMS 0.082 / 0.102, peak 0.83, all samples finite. Parakeet heard no words (empty transcript).
- Not checked: batches, formats other than WAV and the 32 kHz resample, songs over 60 s, Polish or Chinese lyrics, and musical quality by ear.

