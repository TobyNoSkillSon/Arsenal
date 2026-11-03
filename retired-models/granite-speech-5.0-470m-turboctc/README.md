# Granite Speech 5 TurboCTC

Fast English transcription. English CTC transcription with word timings derived from CTC frame emissions. Timings are approximate at 12.5 output frames per second. Audio longer than 122.88 s is encoded in 122.88 s pieces, each with 10.24 s of surrounding audio as context, and decoded as one CTC sequence, as upstream Granite-MLX does. No translation, diarization, or separate punctuation model is implied.

## Source and precision

Checkpoint: `iky1e/granite-speech-5.0-470m-turboctc-mlx-fp16@319abff7072204bc6cd30485aac58da6c1216214`. Native port: `kylehowells/Granite-MLX@7ad6b19c698aac5aa8328c911e4a6eead456c5a1`. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

FP16 is the only offered precision. Casting quality remains unqualified without trained weights.

## Verification

The no-weight fixture constructs a shrunk version of the actual graph and uses the ordinary worker operation. This verifies wiring and shapes, not trained output, speed, memory, or numerical fidelity. Native source closure and shared-host prerequisites are recorded in `Workers/Audio/PORTED-FROM.md` in the source repository. Trained-weight qualification is still required.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
