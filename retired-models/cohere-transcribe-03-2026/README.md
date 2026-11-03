# Cohere Transcribe

Multilingual transcription. 14 languages. This is the open FP16 conversion of the gated Cohere checkpoint. The current adapter performs greedy transcription. Punctuation and capitalisation are on by default; `punctuation: false` selects the maker's no-punctuation prompt. Audio longer than 30 s is cut into pieces of at most 35 s at the quietest 0.1 s window within the last 5 s before each limit, as the maker's feature extractor does, and the pieces' text is joined (without spaces for Japanese and Chinese). Word alignment and speaker identity are not declared.

## Source and precision

Checkpoint: `beshkenadze/cohere-transcribe-03-2026-mlx-fp16@06bbe0d853a93cbbf7c1f40cffccd4c5481459ef`. Native port: `Blaizzy/mlx-audio-swift@244458d7b61e54ff135073ddc9ed76782bb20758`. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

FP16 is the serving default. int8 and int4 pack every matrix whose input width is divisible by 64 (group size 64); spatial convolutions, norms and narrower matrices stay FP16. Casting and quantized output quality remain unqualified without trained weights.

## Verification

The no-weight fixture constructs a shrunk version of the actual graph and uses the ordinary worker operation. This verifies wiring and shapes, not trained output, speed, memory, or numerical fidelity. Native source closure and shared-host prerequisites are recorded in `Workers/Audio/PORTED-FROM.md` in the source repository. Trained-weight qualification is still required.
