# Real check, 2 Nov 2026

One run through Arsenal's real service on an isolated profile (`arsenal-testserver --real`, private QA catalog from the
committed folder at 2a4df6b6, Hugging Face client on loopback so Get could only import the local copy), under the GPU
lock, headless. M5 Max, Mac in use. No speed or memory figures are claimed.

- Get: `MelBandRoformer.ckpt` (913 MB, FP32) cloned from the local Hugging Face cache, nothing downloaded; `cast:audio-mel-band-roformer-v1` wrote a 456 MB FP16 snapshot.
- Load at fp16, then `separate_audio` on the 12 s fixture with default settings: HTTP 200, two files (vocals, instrumental), 529,200 frames each, stereo 44.1 kHz. Against the maker's own `demix_track` output: vocals relative error 0.086%, instrumental 0.10%.
- `stems: ["vocals"]` with `overlap: 4`: HTTP 200, one file; 0.44% from the maker's overlap-2 output (a different setting, so not a parity figure).
- No swap added; no leftover processes.
- Not checked: real commercial recordings, clips over 12 s through the service, the app's own tool window (headless only).
