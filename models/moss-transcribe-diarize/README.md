# MOSS-Transcribe-Diarize 0.9B

OpenMOSS's transcription model writes who said what, and when, in one pass. A Whisper-Medium audio encoder listens to the recording and a Qwen3-0.6B decoder (908.5 million parameters in all) writes a compact transcript: every segment opens with its start time and an anonymous speaker label ([S01], [S02], …) and closes with its end time. No separate diarization model is needed. Arsenal runs a Swift and MLX port of the maker's code in FP16.

## What it is for

- Meetings, calls, interviews and podcasts: who said what, with start and end times per segment
- Long recordings in one pass (the maker says up to 90 minutes)
- Names and terms spelled right, through hotwords
- Custom instructions, such as speaker labels without timestamps

On the maker's benchmarks it reports cpCER 15.83 on AISHELL-4, 22.17 on AliMeeting, 7.37 on podcasts and 12.76 on movies, ahead of Doubao, ElevenLabs and Gemini 2.5 Pro there, and it won the 2nd MLC-SLM challenge at INTERSPEECH 2026. These are the maker's figures, not reproduced here.

## Inputs and outputs

### `transcribe` (default)

| Field | Type | Notes |
|---|---|---|
| `audio` | file `{"path": string}` | Required. Any audio macOS decodes; resampled to 16 kHz mono. Up to 90 minutes. |
| `prompt` | string | Your own instruction instead of the maker's default (timestamped segments with speaker labels). The maker's speaker-only recipe: "Transcribe the audio as text using speaker labels such as [S01], [S02], and [S03]." |
| `hotwords` | array of strings | Names and terms to spell right; appended as the maker's hotword hint (in Chinese after the default instruction, as "Hotwords:" after your own). |
| `max_new_tokens` | integer 1–65,536 | Longest transcript, 5,120 by default (the maker's), which covers about 13 minutes of dense speech. Raise it for long recordings. |
| `stream` | boolean | Send each finished segment as it is written (`event: token`, with `text` and `speaker`). |

Returns `text` (the segments' text, one per line), `duration` in seconds, `segments` (`start`, `end`, `speaker`, `text`), `speakers` in order of first appearance, `transcript`, the model's raw `[start][Sxx]text[end]` output, and `truncated`, true when `max_new_tokens` ran out before the model finished, so the transcript stops early. A custom prompt that yields no segment markup comes back in `text` as written.

### `diarize`

The same model and inputs (without `stream`), answering in the audio kind's diarization shape: `segments` with text, `speakers`, `rttm`, the raw `transcript` and `truncated`. It does not take a speaker count: the model decides. Under the default instruction, a recording the model writes no timestamps for (music, noise) returns no segments. A custom instruction whose transcript has no timestamped segments (such as the speaker-only recipe) is refused here; use `transcribe` for it.

## Usage notes (2 Nov 2026)

What it does: one-pass transcription with speaker labels and segment timestamps, exactly as the maker's `generate_transcription` with greedy decoding: 30 s Whisper chunks, four encoder frames merged per audio token (12.5 tokens per second), the seconds written into the audio span every 5 s as the maker's processor does, the maker's chat prompt, and the maker's transcript parser (code point for code point, its quirks included). Streaming hands over text as it is decoded and sends the last segment when the transcript ends; generation stops only at the maker's end token. Inputs: one audio file, an optional instruction and hotwords. Limits: anonymous speaker labels renumbered per recording, a request must finish within the service's 590 s limit (not measured for long recordings; by estimate a full hour of speech needs about 34,000 decode steps, more than fits in 590 s), the default 5,120-token budget covers about 13 minutes of dense speech (derived from the 29 s check, 184 tokens) and a reply it cuts short says `truncated: true`, timestamps per segment rather than per word, 90 minutes per recording (checked here only up to one minute), and languages beyond Chinese and English are the maker's unnamed "50+" claim.

Source: `OpenMOSS-Team/MOSS-Transcribe-Diarize` at `10304d92`, one BF16 safetensors file (1.82 GB). Get casts it to FP16 (`cast:audio-moss-transcribe-diarize-v1`); the output head is the tied token embedding. The encoder, adaptor and decoder run in FP16; the log-mel frontend is computed in float32.

Port: the decoder is the audio family's Qwen3 text model (shared with Qwen3-ASR) and the log-mel frontend is the family's Whisper filterbank; the Whisper-Medium encoder, the adaptor, the processor (chunking, token counts, time markers, prompt) and the parser are new here. Long prompts are prefilled 1,024 tokens at a time, and logits are taken for the last position only.

Parity: checked against the maker's own code (the repository's remote code with its GitHub helpers), run once on the Mac's GPU in FP16, on a 29 s two-speaker recording made with two macOS voices. Log-mel features within 0.014% relative error, Whisper encoder 0.76%, adaptor 0.30%, and the greedy transcript identical to the maker's token for token (184 tokens: both voices separated, every word and timestamp the same). The parser and the time-marker layout match the maker's functions on fixed inputs, including awkward transcripts. Bounds were set before the comparison (components 2%, transcript token-identical). Through the service, a 58 s two-chunk recording gives the maker's speakers and words in all twelve segments, with three end timestamps 0.01 to 0.03 s apart, and a Polish sentence from a macOS voice came back word for word (a probe; the maker does not name Polish). Not checked: real recordings, the maker's benchmarks, recordings longer than a minute, speed and memory figures.

## Licence and credit

Apache-2.0. Credit OpenMOSS; cite "MOSS Transcribe Diarize: Accurate Transcription with Speaker Diarization" (arXiv 2601.01554). The model builds on Whisper (OpenAI, MIT) and Qwen3 (Alibaba, Apache-2.0).
