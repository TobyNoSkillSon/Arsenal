# Confucius4 R2T2

Live speech transcription. 30 languages plus 22 Chinese dialects; language detection and optional context prompt. Word timestamps require the separate Qwen ForcedAligner checkpoint and are not declared. `transcribe_session` implements the maker's rolling-audio/prefix/rollback algorithm (`streaming_transcribe_no_reset`) with 80–2000 ms chunks, stable committed text, partial text and append-only deltas. As in the maker's `example.py`, the first chunk carries a 160 ms lookahead (320 ms at the default 160 ms chunk), each chunk generates an adaptive 1–32 tokens (growing while the transcript stalls), and one token is rolled back by default (`rollback_tokens`). Rollback counts only transcript tokens after `<asr_text>`, and without a forced language nothing is committed before the model has written its language header, so the header never reaches the text. Create, push, finish and close are supported; two sessions and ten minutes per recording are allowed. Chunk-aligned final input still flushes the last rollback tokens. Whole audio/text chunks are evicted together when trimming the rolling window. The Realtime WebSocket streams through this operation: each roughly 200 ms of appended audio is pushed and its stable delta forwarded while the speaker talks (docs/API.md, Live transcription sessions).

## Source and precision

Checkpoint: `netease-youdao/Confucius4-R2T2@185ce639118ad1362d049ca0d8ed04b6ec5cd6c9`. Native graph port: `Blaizzy/mlx-audio-swift@244458d7b61e54ff135073ddc9ed76782bb20758`. Streaming state/prefix algorithm: `netease-youdao/Confucius4-R2T2@26d55a54ce5670cff9947a167d8ed95d569fd4d9`. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

FP16 is the serving default. int8 and int4 pack every matrix whose input width is divisible by 64 (group size 64); spatial convolutions, norms and narrower matrices stay FP16. Casting and quantized output quality remain unqualified without trained weights.

## Verification

The no-weight fixture constructs a shrunk version of the actual graph and uses the ordinary worker operation. This verifies wiring and shapes, not trained output, speed, memory, or numerical fidelity. Native source closure and shared-host prerequisites are recorded in `Workers/Audio/PORTED-FROM.md` in the source repository. Trained-weight qualification is still required.
