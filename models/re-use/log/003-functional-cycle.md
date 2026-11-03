+++
hypothesis = "The original pinned checkpoint can be cast by Get and enhanced through the gated native Fast path."
change = "Configured MLX library before CPU derivation; ran Get, load, enhance, PCM stream, WAV validation and unload/exit."
measurement = "M5 Max short gpulock functional smoke on a synthetic 0.5-second 16 kHz tone, not a benchmark or full speech-quality suite."
verdict = "kept"
numbers = { retry_network_bytes = 0, load_s = 4.930959939956665, run_s = 0.07852399349212646, output_s = 0.5, footprint_mb_after_run = 380.798344, progress_events = 2 }
links = ["gate/smoke-evidence.json", "docs/evidence/mb-v2-sound/re-use-derived.json", "docs/evidence/mb-v2-sound/re-use-cycle.json"]
+++

The ~39 MB original FP32 source was retrieved on the first Get attempt (coordinator-approved; absent from the local HF cache), but that cast worker terminated because its MLX library was not configured before first stream use. The corrected CPU derivation used those already-verified source blobs, so the successful retry transferred zero network bytes. The source was cast to FP16, then widened losslessly for model arithmetic at load. Native conversion/worker logs retain the failure rather than hiding it.

Load selected `optimized`: Metal selective scan + causal conv (self-test passed: kernels scan 2.2e-07 conv 0.0e+00, model 48 kHz 78 dB, model 16 kHz 81 dB). Enhancement preserved 8000 mono frames at 16 kHz, all samples finite. Both PCM16 stream and hashed WAV were verified; unload exited the worker. No int8/int4 cycle or full speech suite was run.
