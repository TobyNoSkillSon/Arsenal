# Streaming Sortformer Diarizer 4spk v2

NVIDIA's Streaming Sortformer finds who spoke when in a live stream or a recording, for up to four speakers. A 17-layer Fast-Conformer (NEST) encoder and an 18-layer Transformer (117 million parameters) output each speaker's activity every 80 ms, and speakers are numbered in the order they first speak, so there is no clustering step. An arrival-order speaker cache carries the voices heard so far from chunk to chunk, which keeps the numbering stable over a live session or a recording hours long. Arsenal runs a Swift and MLX port of the makers' NeMo model in FP16.

## What it is for

- Meetings, calls and interviews with up to four people, live or recorded
- Live captions that need speaker labels, with latency chosen from 0.32 s to 30 s
- Long recordings: the speaker cache keeps memory bounded
- Overlapping speech: each speaker has an independent activity track
- Raw speaker activity probabilities every 80 ms, for your own post-processing

The card reports a diarization error rate of 13.24% on DIHARD III (one to four speakers) and 6.57% on two-speaker CALLHOME at 1.04 s latency, both with tuned post-processing.

## Inputs and outputs

### `diarize` (default)

Diarizes one audio file, streamed chunk by chunk exactly as NeMo's `diarize()` does.

| Field | Type | Notes |
|---|---|---|
| `audio` | file `{"path": string}` | Required. Any audio macOS decodes; resampled to 16 kHz mono. Up to one hour. |
| `latency` | string | The card's presets: `very_high` (30.4 s, default, the card's Quick Start), `high` (10.0 s), `low` (1.04 s), `ultra_low` (0.32 s). Lower latency costs some accuracy. |
| `chunk_frames`, `right_context_frames`, `fifo_frames`, `update_period_frames`, `speaker_cache_frames` | integer | Override one value of the preset, in 80 ms frames (the card's CHUNK_SIZE, RIGHT_CONTEXT, FIFO_SIZE, UPDATE_PERIOD, SPEAKER_CACHE_SIZE). The speaker cache must be at least 16 frames. |
| `onset`, `offset` | number 0–1 | Activity thresholds that start and end a segment (default 0.5 each, NeMo's defaults). |
| `pad_onset`, `pad_offset`, `min_duration_on`, `min_duration_off` | seconds | NeMo's post-processing: padding around segments, dropping short segments, filling short gaps. All 0 by default. NVIDIA publishes tuned values per dataset in NeMo's `conf/post_processing` folder. |
| `include_probabilities` | boolean | Also return every 80 ms frame's four speaker probabilities. |

Returns the audio kind's diarization result, the same shape as `pyannote-community-1`'s `segments`, `speakers` and `rttm`, so callers can swap models:

- `segments`: `{"start", "end", "speaker"}` in seconds, sorted by start; overlapping speech gives overlapping segments. Speakers are `SPEAKER_00` … `SPEAKER_03` in order of arrival.
- `speakers`: the labels that occur.
- `rttm`: the segments as RTTM lines.
- With `include_probabilities`: `frame_seconds` (0.08) and `probabilities`, one `[p0, p1, p2, p3]` per frame.

There are no `exclusive_segments` or speaker embeddings, and no requested speaker count: the model decides, up to four.

### `diarize_session`

Live diarization. `create` takes the same streaming and post-processing settings and returns a `session_id`; `push` sends the next piece of audio (a file of up to 30 s) and returns everything confirmed so far; `finish` flushes the last chunk and closes the session; `close` discards it. A chunk is confirmed once its right context has arrived, so the first result comes after chunk + right context (1.04 s at `low`). After `finish` the result is identical to `diarize` on the whole recording with the same settings. Up to two open sessions and ten minutes per session; a session left untouched for ten minutes gives up its slot to a new one. A push that fails or is cancelled keeps the session and the audio received so far, so continue with the next piece or `finish` rather than resending it.

## Run it

MCP: `arsenal_run(model:"sortformer-4spk-v2", input:{"audio":{"path":"/absolute/path/meeting.m4a"}})`

HTTP: `curl -s "$ARSENAL_URL/arsenal/v1/models/sortformer-4spk-v2/ops/diarize" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/meeting.m4a"},"latency":"low"}'`

## Limits

- Four speakers at most: a fifth person gets no label of their own, and the card's error rate roughly triples with five or more speakers (DIHARD III: 13.2% up to four, 42.6% for five to nine).
- Trained mostly on English. Other languages, Polish included, are untested here.
- Similar voices can be merged: on Arsenal's synthetic test clip both NeMo and this port give two macOS voices of similar timbre the same label.
- Speakers are numbered per recording or session, not identified.
- Noisy, far-field or very long recordings degrade accuracy (the card's own warning).
- Each request has a 590 s limit. An hour of audio at `low` or `ultra_low` latency means 7,500 to 15,000 network steps and may not finish in time (speed not yet measured); use `very_high` or `high` for long files.

## Usage notes (2 Nov 2026)

What it does: speaker diarization for up to four speakers, streaming, with the model card's four latency presets and NeMo's post-processing options. Inputs: one audio file (`diarize`) or a live sequence of pieces (`diarize_session`); 16 kHz mono after decoding. Limits: one hour per file, ten minutes per session, four speakers.

Source: `nvidia/diar_streaming_sortformer_4spk-v2` at `84edd514`, the FP32 `.nemo` archive (471 MB). Get reads the checkpoint inside the archive without unpacking it, checks the mel filterbank and window against the Swift frontend, and writes an FP16 snapshot (`cast:audio-sortformer-v2-v1`, about 235 MB). The network runs in FP16; the mel frontend and the speaker-cache bookkeeping run in float32 on the CPU, as NeMo runs its frontend.

Port: the module structure follows mlx-audio-swift's Sortformer (MIT), corrected against NeMo 3.0 (NeMo's Transformer has no positional embedding); the speaker-cache update reuses the audio family's `SortformerStateUpdater` (from speech-swift, a port of NeMo's `streaming_update`), which this model was checked against step by step.

Parity: checked against NVIDIA's own NeMo 3.0 code, run once on the Mac's GPU in half precision on a fixed 27.5 s three-voice clip. On NeMo's own inputs: mel features within 0.0004 (largest difference, log scale), pre-encoder relative error 0.15%, Fast-Conformer 0.6%, speaker probabilities within 0.002, and one speaker-cache update step reproduces NeMo's cache exactly. Whole streams: the card's default and low-latency settings stay within 0.01 of NeMo's probabilities, and a stress setting with a tiny 32-frame cache and 33 cache compressions within 0.072 (average 0.0035); all three give exactly NeMo's segments. Bounds were set before the comparison (probabilities 0.08, component relative error 3%). Not checked: accuracy on a benchmark, real four-speaker recordings, other languages.

## Licence and credit

CC-BY-4.0. Commercial use, changes and redistribution are allowed with credit to NVIDIA, a link to the licence and a note of changes; Arsenal converts the weights to FP16. Cite Medennikov et al., "Streaming Sortformer: Speaker Cache-Based Online Speaker Diarization with Arrival-Time Ordering" (2025) and Park et al., "Sortformer: Seamless Integration of Speaker Diarization and ASR by Bridging Timestamps and Tokens" (2024).
