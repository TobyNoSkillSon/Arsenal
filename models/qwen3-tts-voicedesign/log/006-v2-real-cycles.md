+++
hypothesis = "Every offered dtype and both modes can run through the real Arsenal service and return worker residency to zero."
change = "V2 settings/ops, signed-folder Get, standalone derived snapshots, saved voice conditioning and the combined Native worker."
measurement = "Short gpulock functional service cycles; local import except the explicitly authorized initial Kokoro HF Get; no benchmark."
verdict = "kept"
numbers = { functional_cells = 6, failures = 0, worker_mb_after_unload = 0, max_cycle_s = 15.041 }
links = ["docs/evidence/mb-v2-voices/qwen3-tts-voicedesign-bf16-fast.json", "docs/evidence/mb-v2-voices/qwen3-tts-voicedesign-bf16-exact.json", "docs/evidence/mb-v2-voices/qwen3-tts-voicedesign-int8-fast.json", "docs/evidence/mb-v2-voices/qwen3-tts-voicedesign-int8-exact.json", "docs/evidence/mb-v2-voices/qwen3-tts-voicedesign-int4-fast.json", "docs/evidence/mb-v2-voices/qwen3-tts-voicedesign-int4-exact.json"]
+++

# V2 real service coverage

| Dtype | Mode | Get (s) | Cycle (s) | Audio (s) | Generation (s) |
|---|---|---:|---:|---:|---:|
| bf16 | Fast | 2.754 | 15.041 | 0.960 | 0.380 |
| bf16 | Exact | 0.037 | 2.847 | 0.960 | 0.349 |
| int8 | Fast | 5.640 | 8.246 | 0.800 | 0.213 |
| int8 | Exact | 0.034 | 2.130 | 0.800 | 0.184 |
| int4 | Fast | 3.944 | 6.204 | 0.880 | 0.168 |
| int4 | Exact | 0.035 | 1.935 | 0.880 | 0.167 |

Each cell verifies signed catalog/folder installation, local Get, Load, voice design/preview, saved-description speech, WAV SHA256, voices listing and Unload. Six saved voices survived worker exit/reload and dtype changes and stayed VoiceDesign-scoped. The M5 folder RoPE is active in all six cells; network fallback is disabled, proving zero network weight bytes.

Cold/warm cache states and machine contention differ. These numbers are functional evidence, not comparisons, energy results, a ranking or a Measured badge. Existing Vireo gates still select/fallback components; no dtype or optimization was withdrawn.
