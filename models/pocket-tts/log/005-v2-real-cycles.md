+++
hypothesis = "Every offered dtype and both modes can run through the real Arsenal service and return worker residency to zero."
change = "V2 settings/ops, signed-folder Get, standalone derived snapshots, saved voice conditioning and the combined Native worker."
measurement = "Short gpulock functional service cycles; local import except the explicitly authorized initial Kokoro HF Get; no benchmark."
verdict = "kept"
numbers = { functional_cells = 4, failures = 0, worker_mb_after_unload = 0, max_cycle_s = 1.532 }
links = ["docs/evidence/mb-v2-voices/pocket-tts-bf16-fast.json", "docs/evidence/mb-v2-voices/pocket-tts-bf16-exact.json", "docs/evidence/mb-v2-voices/pocket-tts-int8-fast.json", "docs/evidence/mb-v2-voices/pocket-tts-int8-exact.json"]
+++

# V2 real service coverage

| Dtype | Mode | Get (s) | Cycle (s) | Audio (s) | Generation (s) |
|---|---|---:|---:|---:|---:|
| bf16 | Fast | 0.595 | 1.532 | 1.200 | 0.739 |
| bf16 | Exact | 0.026 | 0.402 | 1.200 | 0.182 |
| int8 | Fast | 0.385 | 0.851 | 1.200 | 0.223 |
| int8 | Exact | 0.029 | 0.348 | 1.200 | 0.108 |

Each cell verifies signed catalog/folder installation, local Get, Load, saved cloning from the generated reference, speech with the saved ID, WAV SHA256, voices listing and Unload. Four saved voices survived worker exit/reload and dtype changes and stayed Pocket-scoped. Network fallback is disabled, proving zero network weight bytes.

Cold/warm cache states and machine contention differ. These numbers are functional evidence, not comparisons, energy results, a ranking or a Measured badge. Existing Vireo gates still select/fallback components; no dtype or optimization was withdrawn.
