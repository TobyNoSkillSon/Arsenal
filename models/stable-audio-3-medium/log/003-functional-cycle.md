+++
hypothesis = "Verified local Get and the native default Fast path can produce finite audio through Arsenal."
change = "Ran pinned local import, load, generate_music, PCM stream, WAV hash check, status and unload/exit."
measurement = "M5 Max short gpulock functional smoke; uncontrolled, not a benchmark or Measured result."
verdict = "kept"
numbers = { network_bytes = 0, load_s = 5.935935020446777, run_s = 0.12197601795196533, output_s = 1.5, footprint_mb_after_load = 5564.158632, progress_events = 4 }
links = ["gate/smoke-evidence.json", "docs/evidence/mb-v2-sound/stable-audio-3-medium-cycle.json"]
+++

1.5 seconds, two sampler steps, seed 20260927; stereo 44.1 kHz. Both file and PCM16 stream were produced, all decoded samples finite, file SHA256 matched the returned hash, and the worker exited after unload. The source came from the local HF cache; remote HTTP fallback was disabled.

The load selected `optimized`: fp16 residual stream (self-test passed: log-mel distance 0.04 dB to stock) + SWA views in the decoder (self-test passed: 3 decodes bit-identical to stock). These are load self-tests, not a full task-quality qualification. Quantized cycles were not run.
