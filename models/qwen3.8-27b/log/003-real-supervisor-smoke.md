+++
hypothesis = "The unchanged native family runtime can load the real vendor Qwen3.8-27B and its local affine int4 derivative, stream an answer and stop promptly."
change = "Used verified APFS-cloned vendor bf16, a guarded real-supervisor smoke, local affine-g64 int4 conversion and another bounded smoke."
measurement = "Both precisions answered 17 plus 25 correctly with 11 incremental fragments; cancellation settled and both worker processes exited."
verdict = "kept"
numbers = { bf16_load_s = 7.037020, bf16_ttft_s = 8.524951, bf16_tokens_per_s = 7.959960, bf16_peak_gib = 50.653021, int4_convert_s = 25.092966, int4_load_s = 0.598320, int4_ttft_s = 0.338019, int4_tokens_per_s = 31.478044, int4_peak_gib = 14.748577, int4_cancel_s = 0.034176 }
links = ["runners/native/smoke-bf16.json", "runners/native/convert-int4.json", "runners/native/smoke-int4.json", "runners/native/memory-bf16.json", "../../docs/notes/chat.md"]
+++

Before bf16, vm_stat plus memory_pressure showed **83.9 GiB free plus estimated reclaimable pages**
(free + inactive + speculative), above our conservative 72 GiB threshold. Vella's hot models were not evicted
or altered. A new worker held the real model, answered "17 plus 25 equals 42.", then received task cancellation
mid-stream on a second counting request. Its status reply showed the generation loop had settled; unload
waited for exit and verified the PID no longer existed. No Python runner or dependency pin change was needed.

The bf16 run was cold: its first-token time includes kernel preparation and prompt processing. The int4
run is a later separate process after conversion; these single-shot diagnostics are **not a controlled
benchmark comparison**, and no ranking/Measured badge is created. Peak RAM uses the kernel's lifetime peak
physical-footprint ledger (MiB in the raw memory objects), not merely the MLX allocator's current usage.

Int4 derivation took 25.093 seconds, within the authorized three-minute budget. It used the original bf16,
MLX affine group 64, no calibration, no community quant and no re-quantization. The 14-file output is
15,156,145,069 bytes and loads independently; hashes/size records are in convert-int4.json.
Int8 machinery and recipe exist, but int8 was **not run or qualified** in this task.
The multi-item quality gate, longer contexts, vision, tool decoding and controlled benchmarks were not run.

Commands used the real ArsenalCore.Supervisor from Workers/Chat/ModelProof. GPU jobs were serialized through
`scripts/mlock gpulock chat`; convert/int4 jobs also had a 180-second process alarm. The native runtime has
no polling timer: stdin blocks when there is no request, and status is pushed only after a change or on request.
