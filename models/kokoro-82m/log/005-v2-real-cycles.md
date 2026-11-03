+++
hypothesis = "Every offered dtype and both modes can run through the real Arsenal service and return worker residency to zero."
change = "V2 settings/ops, signed-folder Get, standalone derived snapshots, saved voice conditioning and the combined Native worker."
measurement = "Short gpulock functional service cycles; local import except the explicitly authorized initial Kokoro HF Get; no benchmark."
verdict = "kept"
numbers = { functional_cells = 4, failures = 0, worker_mb_after_unload = 0, max_cycle_s = 119.834 }
links = ["docs/evidence/mb-v2-voices/kokoro-fp16-fast.json", "docs/evidence/mb-v2-voices/kokoro-82m-fp16-exact.json", "docs/evidence/mb-v2-voices/kokoro-82m-int8-fast.json", "docs/evidence/mb-v2-voices/kokoro-82m-int8-exact.json"]
+++

# V2 real service coverage

| Dtype | Mode | Get (s) | Cycle (s) | Audio (s) | Generation (s) |
|---|---|---:|---:|---:|---:|
| fp16 | Fast | 108.804 | 119.834 | 1.600 | 0.356 |
| fp16 | Exact | 0.595 | 3.765 | 1.600 | 0.048 |
| int8 | Fast | 0.416 | 2.453 | 1.600 | 0.113 |
| int8 | Exact | 0.106 | 1.368 | 1.600 | 0.055 |

Each cell verifies signed catalog/folder installation, Get, Load, preset speech, WAV SHA256, voices listing and Unload. All four dtype/mode cells returned worker residency to zero. The first fp16 Fast cell used the explicitly authorized fresh 342 MB HF Get; later cells reused verified snapshots.

Cold/warm cache states and machine contention differ. These numbers are functional evidence, not comparisons, energy results, a ranking or a Measured badge. Existing Vireo gates still select/fallback components; no dtype or optimization was withdrawn.
