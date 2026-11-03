+++
hypothesis = "The copied engine can load and emit identical file/stream PCM under the Arsenal wire."
change = "Built the headless worker with CLT Swift and the pinned no-fast-math Metal library."
measurement = "mlock gpulock smoke: both offered dtypes in Fast and Exact; WAV/stream match, finite PCM, nonzero RMS, unload exits."
verdict = "kept"
numbers = { configurations = 4, failures = 0, first_clip_seconds = 3.25 }
links = ["gate/smoke-evidence.json", "Workers/Voices/scripts/verify-wire.py"]
+++

FP16 and int8 loaded and spoke the fixed short sentence with seed 24301. Streamed PCM matches the file bytes and SHA-256. Engine timings are diagnostic smoke readings under gpulock, not benchmark measurements. No ASR, listening-quality gate or Measured badge is claimed. Unload exits the process (memory reclaimed by the OS).
