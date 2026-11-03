+++
hypothesis = "All offered dtypes and both modes can complete the real Arsenal service with saved voice conditioning."
change = "V2 schema/settings, signed folder Get, native runner, saved reference clips and qualified optimization defaults."
measurement = "Short gpulock functional cycles from pinned local HF sources; network fallback disabled; no benchmark."
verdict = "kept"
numbers = { functional_cells = 6, failures = 0, network_weight_bytes = 0, worker_mb_after_unload = 0, max_cycle_s = 16.144 }
links = ["docs/evidence/mb-v2-voices/breeze-tts-2-bf16-fast.json", "docs/evidence/mb-v2-voices/breeze-tts-2-bf16-exact.json", "docs/evidence/mb-v2-voices/breeze-tts-2-int8-fast.json", "docs/evidence/mb-v2-voices/breeze-tts-2-int8-exact.json", "docs/evidence/mb-v2-voices/breeze-tts-2-int4-fast.json", "docs/evidence/mb-v2-voices/breeze-tts-2-int4-exact.json"]
+++

# Real v2 service cycles

| Dtype | Mode | Get (s) | Cycle (s) | Audio (s) | Generation (s) |
|---|---|---:|---:|---:|---:|
| bf16 | Fast | 3.070 | 14.207 | 1.600 | 1.303 |
| bf16 | Exact | 0.042 | 7.074 | 1.520 | 1.610 |
| int8 | Fast | 8.219 | 16.144 | 1.120 | 0.774 |
| int8 | Exact | 0.040 | 6.315 | 1.520 | 1.484 |
| int4 | Fast | 0.902 | 9.486 | 1.120 | 0.807 |
| int4 | Exact | 0.038 | 5.809 | 1.120 | 1.138 |

Each cell verifies signed catalog/folder installation, local Get, Load, save copied reference/transcript, speak with the saved ID, WAV SHA256, voices listing and Unload. Breeze additionally exercises design_voice and its generated preview. Saved voices persist per model across worker exits and dtype changes. All cells use the combined Native binary.

Fast and Exact retain Vireo's accepted components and component-level gates/fallback. Qwen's active M5 folder RoPE reports 24 bit-identical check outputs (27,648 elements); Fast inherits the exact kernel. Breeze int8 is a local affine-g64 derive; int4 is the pinned vendor MXFP4 g32 snapshot, not an affine conversion.

Cold/warm cache states and contention differ. Figures are functional evidence, not a comparison, energy measurement, ranking or Measured badge. No subjective listening or full ASR word-quality suite was run in Arsenal; the existing Vireo presence-gated dtype list was retained.
