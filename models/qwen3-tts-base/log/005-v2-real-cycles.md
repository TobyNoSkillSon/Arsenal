+++
hypothesis = "All offered dtypes and both modes can complete the real Arsenal service with saved voice conditioning."
change = "V2 schema/settings, signed folder Get, native runner, saved reference clips and qualified optimization defaults."
measurement = "Short gpulock functional cycles from pinned local HF sources; network fallback disabled; no benchmark."
verdict = "kept"
numbers = { functional_cells = 6, failures = 0, network_weight_bytes = 0, worker_mb_after_unload = 0, max_cycle_s = 7.101 }
links = ["docs/evidence/mb-v2-voices/qwen3-tts-base-bf16-fast.json", "docs/evidence/mb-v2-voices/qwen3-tts-base-bf16-exact.json", "docs/evidence/mb-v2-voices/qwen3-tts-base-int8-fast.json", "docs/evidence/mb-v2-voices/qwen3-tts-base-int8-exact.json", "docs/evidence/mb-v2-voices/qwen3-tts-base-int4-fast.json", "docs/evidence/mb-v2-voices/qwen3-tts-base-int4-exact.json"]
+++

# Real v2 service cycles

| Dtype | Mode | Get (s) | Cycle (s) | Audio (s) | Generation (s) |
|---|---|---:|---:|---:|---:|
| bf16 | Fast | 2.006 | 4.701 | 1.040 | 0.860 |
| bf16 | Exact | 0.033 | 2.363 | 1.040 | 0.336 |
| int8 | Fast | 4.841 | 7.101 | 1.120 | 0.523 |
| int8 | Exact | 0.036 | 2.141 | 1.120 | 0.278 |
| int4 | Fast | 4.132 | 6.216 | 1.120 | 0.418 |
| int4 | Exact | 0.026 | 1.917 | 1.120 | 0.242 |

Each cell verifies signed catalog/folder installation, local Get, Load, save copied reference/transcript, speak with the saved ID, WAV SHA256, voices listing and Unload. Breeze additionally exercises design_voice and its generated preview. Saved voices persist per model across worker exits and dtype changes. All cells use the combined Native binary.

Fast and Exact retain Vireo's accepted components and component-level gates/fallback. Qwen's active M5 folder RoPE reports 24 bit-identical check outputs (27,648 elements); Fast inherits the exact kernel. Breeze int8 is a local affine-g64 derive; int4 is the pinned vendor MXFP4 g32 snapshot, not an affine conversion.

Cold/warm cache states and contention differ. Figures are functional evidence, not a comparison, energy measurement, ranking or Measured badge. No subjective listening or full ASR word-quality suite was run in Arsenal; the existing Vireo presence-gated dtype list was retained.
