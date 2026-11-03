# Real check, 2 Nov 2026

One run through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from the
committed folder at 68718223, Hugging Face client on loopback so Get could only import the local copy), under the GPU
lock, headless. M5 Max, Mac in use. No speed or memory figures are claimed.

- Get: the BF16 safetensors cloned from the local Hugging Face cache, cast to FP16 (1.83 GB snapshot); Load at fp16.
- `transcribe` on the 29 s two-speaker fixture: the raw transcript is identical to the maker's (six segments, both voices, every word and timestamp).
- `diarize` on the 58 s recording (two 30 s chunks): twelve segments; speakers and words identical to the maker's, and three end timestamps differ by 0.01 to 0.03 s (FP16 near-ties in the timestamp tokens).
- `transcribe` with `stream` and two hotwords on a 6.4 s Polish sentence (macOS voice Zosia): "Dzień dobry, nazywam się Zosia. Jutro rano jedziemy pociągiem do Poznania na spotkanie z klientem." word for word, one speaker. A probe, not a language claim: the maker does not name Polish.
- `max_new_tokens: 0` is refused as invalid_request.
- No swap added; no leftover processes.
- Not checked: real recordings, recordings over a minute, the app's own Get and Load (headless rule).
