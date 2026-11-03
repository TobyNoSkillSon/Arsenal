+++
hypothesis = "Request timing can observe MLX token boundaries without changing generated answers."
change = "Reset and sample MLX peak memory, observe prefill/first/last/token gaps, emit progress and optional final timing. Mark the existing receive helper nonisolated for Swift 6.4 closure isolation and report resident footprint_mb in decimal MB for activity; generation algorithm and parameters remain unchanged."
measurement = "2026-10-06 Apple M5 Max, debug combined native worker, pre-existing Qwen3.8-27B int4 APFS clone; scripts/mlock gpulock bounded to 180 s. Three fixtures from gate/fixtures-v1.json run at temperature 0 and seed 24301, max_tokens 64. No downloads or catalog measure session."
verdict = "kept"
numbers = { passed = 3, total = 3, downloaded_bytes = 0 }
links = ["../gate/fixtures-v1.json", "../../../Workers/Chat/Sources/ChatEngine/ChatEngine.swift", "../../../docs/notes/v3-telemetry.md"]
+++

<!-- SPDX-License-Identifier: MIT -->

The quick gate produced exactly `42`, `blueberry`, and `no`. A separate short native chat emitted 20 generated tokens and live timing; a Python mlx-lm reference from the same local int4 snapshot confirmed that its final generation count includes EOS, while Arsenal decode timing uses emitted generated tokens after the first. These checks establish instrumentation and answer preservation, not a release throughput claim. Local raw evidence (private, not distributed): `qwen-gate.json`, `chat-smoke.json`, `mlx-reference.json`.

The gate was repeated after the decimal-footprint correction on the rebuilt native worker: 3/3 passed again (`qwen-gate-round2.json`, same parameters and gpulock bound). The final raw-wire completion check also asserts that the done frame and reply agree on 20 prompt, 0 cached and 20 generated tokens (`chat-final-fields.json`).
