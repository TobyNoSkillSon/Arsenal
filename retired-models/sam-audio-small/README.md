# SAM Audio Small

Isolate a sound by description. Text-prompted sound separation returns both target and residual WAV files. Extract lists the target first; remove lists the residual first. The required T5-base asset is pinned separately; Get copies its encoder into the selected snapshot (`text_encoder/`) at FP16 or with the tier's packed matrices, so Load never reads the raw asset. Temporal anchors (positive/negative time spans) are supported on clips up to ten seconds, matching the source port's single-chunk limit. Visual prompts require a vision encoder absent from this Swift port and are not declared. Long recordings use the source port's ten-second windows and three-second cosine overlaps; the combined two-WAV result is bounded by the host's 64 MiB output limit.

## Source and precision

Checkpoint: `mlx-community/sam-audio-small@4ca84acc5c47f64bb2acee448f835dfac65ed318`. Native port: `Blaizzy/mlx-audio-swift@244458d7b61e54ff135073ddc9ed76782bb20758`. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

FP16 is the only offered precision. Casting quality remains unqualified without trained weights. The recurrent LSTM projections are packed with the same rule.

## Verification

The no-weight fixture constructs a shrunk version of the actual graph and uses the ordinary worker operation. This verifies wiring and shapes, not trained output, speed, memory, or numerical fidelity. Native source closure and shared-host prerequisites are recorded in `Workers/Audio/PORTED-FROM.md` in the source repository. Trained-weight qualification is still required.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
