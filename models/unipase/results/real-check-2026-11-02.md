# Real check, 2 Nov 2026

Arsenal's own Get, Load, enhance and Unload on an isolated profile (`RealCycleTests.realGetLoadRunUnload` with the
native sound worker; `ARSENAL_SUPPORT_DIR`, `ARSENAL_CACHE_DIR` and the Get store under a temporary
`profile-unipase`; Hugging Face client on loopback, so Get could only import the local cache). M5 Max, Mac in use, on
AC, through `scripts/mlock gpulock`. No speed or memory figures are claimed.

- Get: the four pinned checkpoints imported from the local Hugging Face cache, 0 network bytes; `cast:unipase` made one
  FP16 file of 1,093,066,562 bytes (1,033 tensors, shapes and FP16 range checked). Load, one `enhance`, Unload; the
  worker exited.
- The cycle's request: the maker's noisy sample `fileid_49.flac` (12.36 s, 48 kHz, mono) with
  `one_time_decode_length` 8, so the long-form path ran (two segments). Back: 48 kHz, mono, the input's exact length
  (593,402 samples), all finite, streamed and written.
- Further requests on the same converted weights, one loaded worker:
  - The maker's three samples with defaults (48, 44.1 and 48 kHz): each came back at its own rate and length. Against
    the maker's own published enhanced files for the same samples (`audio/enh/`), STFT magnitude differs by 0.023,
    0.046 and 0.007 relative L2. Log-spectral distance to the clean references: noisy 3.01, 2.76, 2.00 dB; the maker's
    output 1.23, 1.93, 1.60 dB; Arsenal's 1.23, 1.95, 1.68 dB (measured, single run).
  - `fileid_210` (44.1 kHz) with `mode` upscale and `enable_plc` false: 48 kHz, 701,280 samples (14.61 s).
  - A 48 kHz clip with `rate` 8000: 8 kHz, 24,000 samples. A two-channel 48 kHz file: two channels back.
- No-weights contract check (`scripts/model-check.py smoke`): three `enhance` requests on random FP16 weights, all
  passed, 1,033 parameters, nothing downloaded.
- Not checked: the HTTP/MCP service path (the cycle drives the native worker, as the Sound family's real check does);
  inputs over 15 s in one pass; listening by a person. The maker's published outputs were made on their hardware at a
  precision they do not state.
