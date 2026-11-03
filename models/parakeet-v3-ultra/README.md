# Parakeet v3 Ultra

Parakeet v3 Ultra is Moondream's post-training of NVIDIA's Parakeet TDT 0.6B v3, a 600-million-parameter speech recognizer built from a FastConformer encoder and a Token-and-Duration Transducer (TDT) decoder. It keeps the original architecture, tokenizer and 25 European languages, and Moondream reports lower word error rates than the original on every benchmark group it published. Arsenal runs an MLX conversion of those weights. Its strength is fast, punctuated transcription in those 25 languages with sentence and word timings.

## What it is for

NVIDIA built the base model for high-throughput multilingual transcription with automatic punctuation, capitalization and word and segment timestamps; Moondream's post-training targets accuracy, with the largest gains on the 25-language FLEURS set, in background noise and on long-form audio.

- Dictation and voice notes in any of the 25 languages, without saying which language in advance
- Transcripts of meetings, interviews, lectures and podcasts up to ten minutes per file
- Subtitles and captions from sentence segments, or word-level timing for highlighting, search and cutting
- Transcripts to line up with a diarization from `pyannote-community-1`

## Inputs and outputs

### `transcribe` (default)

Transcribes one audio file in its spoken language and returns the text with sentence timings and, on request, word timings.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file object `{"path": string}` | yes | Path 1–4096 characters. Accepts `audio/*`. Arsenal types the file by extension: `.wav`, `.flac`, `.mp3`, `.m4a`, `.aiff`, `.caf`; others (`.aif`, `.ogg`, `.opus`, video files) are refused. macOS AVFoundation decodes it. |
| `word_timestamps` | boolean | no | `true` adds the `words` array. Default off. |
| `task` | string | no | Only `"transcribe"`. The model does not translate. |

Returns JSON:

- `text` (string): the whole transcript, punctuated and capitalized.
- `duration` (number, 0–600): length of the decoded audio in seconds.
- `segments` (array, at most 1024): one entry per sentence, `{"start": seconds, "end": seconds, "text": string}`.
- `words` (array, at most 16384; only with `word_timestamps: true`): `{"word": string, "start": seconds, "end": seconds}`. Words are built from the model's sub-word tokens: a piece that begins with a space starts a new word.

Times are seconds from the start of the file, clamped to `duration`.

Audio: the model expects 16 kHz mono (NVIDIA's card: 16 kHz, monochannel, 1-D signal). Arsenal converts any file it accepts to 16 kHz mono Float32 itself, averaging the channels of stereo or multichannel files, so the caller sends the file as it is. A file longer than 600 seconds (ten minutes) is refused, not truncated; split it first. Audio over 30 seconds is decoded in 30-second windows with 2 seconds of overlap, and the windows are merged on their overlapping tokens. There is no language field: the model detects the language itself and cannot be told which one to expect.

## Run it

MCP: `arsenal_run(model:"parakeet-v3-ultra", input:{"audio":{"path":"/absolute/path/meeting.m4a"},"word_timestamps":true})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/parakeet-v3-ultra/ops/transcribe" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/meeting.m4a"},"word_timestamps":true}'`

A path may name any file on this Mac; Arsenal clones it into its inbox before the model reads it. `arsenal_upload` is only for a file that is not on this Mac. The operation does not stream; the whole result arrives at the end. In a quick check on an M5 Max (22 Oct 2026, Mac in use, short fixed test clip), a warm request took 0.04 s and the worker peaked at 2.09 GB.

## Limits

- Only the 25 languages it was trained on: bg, hr, cs, da, nl, en, et, fi, fr, de, el, hu, it, lv, lt, mt, pl, pt, ro, sk, sl, es, sv, ru, uk.
- Accuracy varies widely by language. On FLEURS, Moondream reports 2.00% word error rate for Italian and 3.55% for English, but 32.25% for Greek and 14–17% for Latvian, Lithuanian, Maltese and Slovene.
- NVIDIA's training data uses European Portuguese, and NVIDIA notes that this partly explains weaker scores on Brazilian Portuguese benchmarks.
- Ten minutes per file in Arsenal. The base model handles up to 24 minutes with full attention on an A100 80 GB, or 3 hours with local attention, but Arsenal does not expose that.
- Moondream's own long-audio segmentation cuts at pauses using a voice-activity head inside its weights. The MLX conversion drops that head, so Arsenal cuts fixed 30-second windows instead, and Moondream's long-form figures may not carry over.
- The MLX conversion's author could not reproduce Moondream's accuracy gains on synthesized test speech, where both models were saturated, and suggests measuring on real audio.
- No speaker labels; use `pyannote-community-1` for who spoke when.

## Licence and credit

CC-BY-4.0 (Creative Commons Attribution 4.0), inherited from NVIDIA's base model. Commercial use, modification and redistribution are allowed if you credit the source, link the licence and say what you changed. Credit Moondream for parakeet-ultra, NVIDIA for parakeet-tdt-0.6b-v3, and selcukkubur for the MLX conversion. Neither card asks for a specific citation; NVIDIA's technical report is "Canary-1B-v2 & Parakeet-TDT-0.6B-v3: Efficient and High-Performance Models for Multilingual ASR and AST" (arXiv:2509.14128).

## Links

- Model card (pinned MLX conversion): https://huggingface.co/selcukkubur/parakeet-ultra-mlx/tree/b554592c50b2a48471add2daa3d46fa9f00fef5e
- Moondream's original: https://huggingface.co/moondream/parakeet-ultra
- NVIDIA base model: https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3
- Code (layout the conversion targets): https://github.com/Blaizzy/mlx-audio-swift
- Paper: https://arxiv.org/abs/2509.14128

## How Arsenal runs it

Checkpoint: `selcukkubur/parakeet-ultra-mlx@b554592c50b2a48471add2daa3d46fa9f00fef5e`, an MLX conversion that Vella records as tensor-identical to `moondream/parakeet-ultra@73175eb`. The conversion renames and re-lays out the upstream tensors, drops Moondream's 6-tensor voice-activity head and 24 BatchNorm counters, and leaves 697 tensors that match `mlx-community/parakeet-tdt-0.6b-v3` name for name. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

The release is BF16; Get casts it to FP16, the serving base. Arsenal offers FP16 only: at 627M parameters an 8-bit copy would save under 1 GB. The log-mel frontend runs in Float32 because the power spectrum of loud audio can exceed FP16's range; Vella computes it in BF16.

The worker runs Vella's Parakeet runtime (`Workers/Audio/Sources/VellaSTT`, copied from Vella's dictation worker): the FastConformer encoder and TDT decoder, plus Vella's exact optimized components on Apple GPUs of Metal family 9 (M3 and later): the fused Conformer encoder, the TDT decoder compiled into 32-step blocks with five custom kernels per step, and tail block sizing, which Vella enables for Ultra only. Each is token-exact against stock MLX; an optimized run that throws, produces non-finite values, or returns an empty transcript stock does not confirm reruns on stock MLX, which then stays selected until the model is reloaded. Vella's inexact small-M GEMM kernel is BF16-only in Vella and stays off. Other Macs run stock MLX.

Verification: the no-weight fixture builds a shrunk version of the actual graph and runs the ordinary transcribe operation; the full-size graph inventory matches all 697 tensors of the pinned checkpoint header. Worker tests check that the optimized components give stock's token IDs at FP16 on a random graph. Checked live at FP16 on 20 Oct: an English test sentence transcribed exactly. The 22 Oct quick check (load 1.31 s, warm request 0.044 s, peak 1,994 MiB summed over Arsenal's worker processes) is in `results/`. Trained-weight transcription quality across languages, FP16 against the BF16 original, and controlled speed and memory are not yet measured in Arsenal. Research log: `log/001-integration.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The card's only example is `speech.wav`; no other formats or limits are listed. [model card, moondream/parakeet-ultra, undated, first-hand]
- Parameters people recommend: Not found in sources checked. The card's `timestamps` setting is for its Photon runtime, which Arsenal does not run. [model card, undated, first-hand]
- Arsenal call: `arsenal_run(model:"parakeet-v3-ultra", input:{"audio":{"path":"/absolute/path/talk.wav"}})` [ops/transcribe.input.json, checked 2026-10-31, first-hand]

**Good and bad**
- Good at: Lower word error than v3 on every card benchmark group, e.g. FLEURS 11.62 to 9.55 and background noise 6.72 to 5.82. [model card, undated, first-hand]
- Good at: A transcription app's own FLEURS test (Core ML on a Mac, not Arsenal's MLX path): 13.99% to 10.63%, better in 24 languages and worse in none. [Whisper Notes, undated, first-hand]
- Bad at: Greek is the weakest language on the card (32.25% on FLEURS). The app's clips were under 30 seconds, so long audio is untested there. [model card, undated; Whisper Notes, undated, first-hand]
- Against the obvious alternatives: The card compares only with v3. The app says Ultra beats Whisper Large V3 Turbo in six languages, with no full table. [Whisper Notes, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Not found in sources checked. The three Hugging Face threads report no bug; one asks for training-data stats and a language-ID head. Fix or workaround: Not found in sources checked. [moondream/parakeet-ultra discussions, checked 2026-10-31, opinion]

**Evidence**
- Strength: thin. The card is the maker's. The one other test is a competing app's Core ML run, which agrees on direction. No neutral test found. Reddit not checked.
- Sources:
  - Model card, https://huggingface.co/moondream/parakeet-ultra, undated, first-hand (maker).
  - Whisper Notes, https://whispernotes.app/blog/parakeet-ultra-vs-parakeet-v3, undated, first-hand (competing app).
  - Discussions, https://huggingface.co/moondream/parakeet-ultra/discussions, opinion.
