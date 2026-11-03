+++
hypothesis = "Verified local Get and the native default Fast path can produce finite audio through Arsenal."
change = "Ran pinned local import, load, generate_sound, PCM stream, WAV hash check, status and unload/exit."
measurement = "M5 Max short gpulock functional smoke; uncontrolled, not a benchmark or Measured result."
verdict = "kept"
numbers = { network_bytes = 0, load_s = 7.473864912986755, run_s = 0.04094803333282471, output_s = 1.5, footprint_mb_after_load = 2084.997232, progress_events = 4 }
links = ["gate/smoke-evidence.json", "docs/evidence/mb-v2-sound/stable-audio-3-small-sfx-cycle.json"]
+++

1.5 seconds, two sampler steps, seed 20260927; stereo 44.1 kHz. Both file and PCM16 stream were produced, all decoded samples finite, file SHA256 matched the returned hash, and the worker exited after unload. The source came from the local HF cache; remote HTTP fallback was disabled.

The load selected `optimized`: fp16 residual stream (self-test passed: log-mel distance 0.07 dB to stock). These are load self-tests, not a full task-quality qualification. Quantized cycles were not run.
