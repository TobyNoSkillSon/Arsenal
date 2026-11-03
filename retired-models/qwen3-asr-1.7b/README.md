# Qwen3 ASR 1.7B

Speech transcription. 30 languages plus 22 Chinese dialects; language detection and optional context prompt. Word timestamps require the separate Qwen ForcedAligner checkpoint and are not declared. Realtime uses the existing commit-based transcription session; incremental encoder state is not carried by the host contract.

## Source and precision

Checkpoint: `mlx-community/Qwen3-ASR-1.7B-bf16@e1f6c266914abc5a46e8756e02580f834a6cf8a7`. Native port: `Blaizzy/mlx-audio-swift@244458d7b61e54ff135073ddc9ed76782bb20758`. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

FP16 is the serving default. int8 and int4 pack every matrix whose input width is divisible by 64 (group size 64); spatial convolutions, norms and narrower matrices stay FP16. Casting and quantized output quality remain unqualified without trained weights.

## Verification

The no-weight fixture constructs a shrunk version of the actual graph and uses the ordinary worker operation. This verifies wiring and shapes, not trained output, speed, memory, or numerical fidelity. Native source closure and shared-host prerequisites are recorded in `Workers/Audio/PORTED-FROM.md` in the source repository. Trained-weight qualification is still required.
