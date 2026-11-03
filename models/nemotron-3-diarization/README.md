# Nemotron 3 Diarization

Identify who spoke when. Up to eight speakers, including overlap. The port retains the learned silence embedding and streaming speaker cache during a recording. `diarize` starts a fresh recording. `diarize_session` creates a persistent stream, accepts audio chunks with push, flushes with finish, and releases with close. Each response is the current recording-wide segment snapshot. Up to two sessions, ten minutes per recording and thirty seconds per pushed chunk are allowed; a session untouched for ten minutes is dropped when a new one needs its slot, and unload invalidates every ID. Chunk geometry is configured at create (encoder frames are 80 ms each). The Realtime WebSocket carries this operation when this model is selected: each changed snapshot replaces the previous one as an `arsenal.diarization.snapshot` event, and at commit the finished timeline also arrives as standard segment events (docs/API.md, Live transcription sessions).

## Source and precision

NVIDIA source: `nvidia/Nemotron-3-Diarization@723e19c601d99b7e58fba6a14e32153e0afe48d9`, the revision recorded in the no-weights gate; speech-swift's port cites `a435e98`. Checkpoint: `mlx-community/Nemotron-3-Diarization@59ed2dbfc1346dcea9d423c71306a3a2499c568f`. Native port: `soniqo/speech-swift@3f58a58a8f867a4c87b97dbf4d6050d6a88ef665`. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

FP16 is the only offered precision. Casting quality remains unqualified without trained weights.

## Verification

The no-weight fixture constructs a shrunk version of the actual graph and uses the ordinary worker operation. This verifies wiring and shapes, not trained output, speed, memory, or numerical fidelity. Native source closure and shared-host prerequisites are recorded in `Workers/Audio/PORTED-FROM.md` in the source repository. Trained-weight qualification is still required.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.

## Usage notes (2 Nov 2026)

Nemotron 3 Diarization says who spoke when: `diarize` takes an audio file (`audio: {path}`) and returns speaker
segments (`speaker`, `start`, `end` in seconds) for up to eight speakers; speakers are numbered in the order they first
speak. `diarize_session` streams: `create`, `push` audio chunks, `finish`, `close`. Up to ten minutes per recording,
two sessions at a time. NVIDIA's terms are OpenMDW-1.1 (open, commercial use allowed).

Parity: no maker reference yet (NVIDIA's NeMo code was not run). On a one-speaker clip the reply is unchanged from
22 Oct; `results/real-check-2026-11-02.md`, golden in `gate/golden/`.

