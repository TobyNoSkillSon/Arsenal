# Qwen3 ASR 1.7B

Qwen3 ASR 1.7B is the larger speech recognizer of Alibaba Qwen's Qwen3-ASR family: an audio encoder feeding a Qwen3 language model that writes the transcript, built on the audio understanding of Qwen3-Omni and about 2 billion parameters in Arsenal's count. It identifies the language and transcribes 30 languages and 22 Chinese dialects, and the makers report state-of-the-art accuracy among open-source recognizers. Its strength is accurate transcription well beyond European languages, including Chinese, Japanese, Korean, Arabic and Southeast Asian languages, with an optional context prompt for names and terms.

## What it is for

Qwen optimised it for accuracy and robustness in difficult acoustic conditions and text patterns across many languages, accents and dialects, including singing and songs with background music; the 0.6B sibling trades some accuracy for throughput.

- Transcribing Chinese, Japanese, Korean and the other 27 supported languages, or audio whose language is unknown
- Meetings, interviews and recordings with domain vocabulary, guided by a prompt listing names, products or jargon
- Lyrics and sung audio
- Detecting which language a recording is in

## Inputs and outputs

### `transcribe` (default)

Transcribes one audio file, detecting the language or using the one given, and returns the text with the detected language and per-segment timings.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | file object `{"path": string}` | yes | Path 1–4096 characters. Accepts `audio/*`. Arsenal types the file by extension: `.wav`, `.flac`, `.mp3`, `.m4a`, `.aiff`, `.caf`; others (`.aif`, `.ogg`, `.opus`, video files) are refused. macOS AVFoundation decodes it. |
| `language` | string | no | 2–32 characters, one of `zh`, `en`, `yue`, `ar`, `de`, `fr`, `es`, `pt`, `id`, `it`, `ko`, `ru`, `th`, `vi`, `ja`, `tr`, `hi`, `ms`, `nl`, `sv`, `da`, `fi`, `pl`, `cs`, `fil`, `fa`, `el`, `hu`, `mk`, `ro`. Forces that language; omit it to let the model detect the language. |
| `prompt` | string | no | Up to 32768 characters. Context text (names, terms, topic) given to the model as its system message, as the makers' `context` argument is. |
| `stream` | boolean | no | `true` emits each segment's text as a `token` event as soon as that segment is done; the final JSON is unchanged. |
| `task` | string | no | Only `"transcribe"`. The model does not translate. |

Returns JSON:

- `text` (string): the transcript, segment texts joined with spaces.
- `language` (string, present when detected or given): the language most segments were in, as a code from the list above; a language name outside that table is returned as the model wrote it.
- `duration` (number, 0–600): length of the decoded audio in seconds.
- `segments` (array, at most 1024): `{"start": seconds, "end": seconds, "text": string}`, one per decoding segment of up to 30 seconds. These are cut points, not sentence or word timings.

Audio: the model works on 16 kHz mono (the makers' toolkit converts every input to mono 16 kHz Float32; the feature extractor is a 128-bin Whisper-style log-mel at a 160-sample hop). Arsenal converts any file it accepts to 16 kHz mono Float32 itself, averaging the channels of stereo or multichannel files, so the caller sends the file as it is. A file longer than 600 seconds (ten minutes) is refused, not truncated. Audio over 30 seconds is split into segments of up to 30 seconds at the quietest point near each boundary, and each segment is transcribed separately with the same `language` and `prompt`; audio under 1 second is padded with silence to 1 second. Each segment may produce up to 1024 tokens.

## Run it

MCP: `arsenal_run(model:"qwen3-asr-1.7b", input:{"audio":{"path":"/absolute/path/interview.m4a"},"language":"ja","prompt":"Speakers discuss Arsenal and MLX."})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/qwen3-asr-1.7b/ops/transcribe" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/interview.m4a"},"language":"ja"}'`

A path may name any file on this Mac; Arsenal clones it into its inbox before the model reads it. `arsenal_upload` is only for a file that is not on this Mac. Decoding is greedy (temperature 0). In a quick check on an M5 Max (22 Oct 2026, Mac in use, short fixed test clip), a warm request took 0.18 s and the worker peaked at 5.91 GB.

## Limits

- No word or sentence timestamps. The makers provide them through a separate model, Qwen3-ForcedAligner-0.6B, which Arsenal does not include; segment times here are only the 30-second cut points.
- The `language` field accepts the 30 languages only. The 22 Chinese dialects (for example Sichuan, Dongbei, Wu, Minnan, Hong Kong and Guangdong Cantonese) are recognized only through automatic detection.
- Ten minutes per file in Arsenal. The makers' toolkit transcribes up to 1200 seconds per pass; Arsenal decodes 30-second segments, so a sentence that crosses a cut point is split between two segments, and context from one segment does not carry into the next.
- No speaker labels; use `pyannote-community-1` for who spoke when.
- Streaming here means per-segment results from a finished file, not live microphone input. The makers' streaming mode needs vLLM and is not used.

## Licence and credit

Apache-2.0. Commercial use, modification and redistribution are allowed; keep the licence and any NOTICE text, and mark changed files. The licence also grants a patent licence from contributors. Made by the Qwen team at Alibaba; the MLX conversion is by mlx-community with mlx-audio 0.3.1. The makers ask for this citation:

```bibtex
@article{Qwen3-ASR,
  title={Qwen3-ASR Technical Report},
  author={Xian Shi, Xiong Wang, Zhifang Guo, Yongqi Wang, Pei Zhang, Xinyu Zhang, Zishan Guo, Hongkun Hao, Yu Xi, Baosong Yang, Jin Xu, Jingren Zhou, Junyang Lin},
  journal={arXiv preprint arXiv:2601.21337},
  year={2026}
}
```

## Links

- Model card (pinned MLX conversion): https://huggingface.co/mlx-community/Qwen3-ASR-1.7B-bf16/tree/e1f6c266914abc5a46e8756e02580f834a6cf8a7
- Original model card: https://huggingface.co/Qwen/Qwen3-ASR-1.7B
- Code: https://github.com/QwenLM/Qwen3-ASR
- Paper: https://arxiv.org/abs/2601.21337

## How Arsenal runs it

Checkpoint: `mlx-community/Qwen3-ASR-1.7B-bf16@e1f6c266914abc5a46e8756e02580f834a6cf8a7`, the release Vella uses. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

The release is BF16; Get casts it to FP16, the serving base. Arsenal offers it at FP16 only: in Vella's measurements (BF16 base) its int8 and int4 tiers each lose a clip the 16-bit tier transcribes, and a tier that fails its quality check is removed.

The worker runs Vella's Qwen3-ASR runtime (`Workers/Audio/Sources/VellaSTT`, copied from Vella's dictation worker) with Vella's optimized decoder on Apple GPUs of Metal family 9 (M3 and later): greedy decoding pipelined with asynchronous evaluation, and a prefill that skips the vocabulary projection whose logits are discarded. Both are token-exact against stock MLX. Vella's optimized encoder holds the audio tower in FP32 weights; Arsenal serves no FP32 weights, so the tower runs as stock MLX does. An optimized run that throws, produces non-finite logits, or returns an empty transcript stock does not confirm reruns on stock MLX, which then stays selected until the model is reloaded. Other Macs run stock MLX.

Verification: the no-weight fixture builds a shrunk version of the actual graph and runs the ordinary transcribe operation at FP16; the full-size graph inventory matches all 707 tensors of the pinned checkpoint header. Worker tests check that the optimized decoder gives stock's token IDs and that the tiers leave the audio tower half. Checked live at FP16 on 20 Oct: an English test sentence transcribed exactly. The 22 Oct quick check (load 1.63 s, warm request 0.182 s, peak 5,637 MiB summed over Arsenal's worker processes) is in `results/`. Trained-weight transcription quality across languages, FP16 against the BF16 original, and controlled speed and memory are not yet measured in Arsenal. Research log: `log/001-integration.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Pass `language` to force one of 30 languages; leave it out to detect. [this README input table, first-hand]
- Prompts, reference audio or voice settings, if relevant: The card has no prompt input. Issue #186 reports prompt text leaking into transcripts. [QwenLM/Qwen3-ASR #186, 2026-06-17, first-hand]
- Parameters people recommend: The maker's evaluation used `max_new_tokens` 1024 with vLLM. [model card, first-hand (maker)]
- Arsenal call: `arsenal_run(model:"qwen3-asr-1.7b", input:{"audio":{"path":"/absolute/path/clip.m4a"},"language":"en"})` [this README, 2026-10-31, first-hand]

**Good and bad**
- Good at: Maker offline WER: LibriSpeech clean/other 1.63/3.38, Fleurs-en 3.35, Fleurs-zh 2.41; language ID 97.9%. [model card, first-hand (maker)]
- Bad at: The card's streaming mode is worse: LibriSpeech 1.95/4.51, Fleurs-en 4.02, Fleurs-zh 2.84. [model card, first-hand (maker)]
- Against the obvious alternatives: Whisper Notes (vendor, FLEURS, 30 languages): 1.7B ahead in 14, Whisper Large V3 Turbo in 15, one tie. Soniqo (vendor): 1.7B 5-bit 1.32% against WhisperKit 1.71% on LibriSpeech test-clean, on different subsets. [whispernotes.app blog, 2026-08-25, updated 2026-10-17, first-hand (vendor); soniqo.audio benchmark, undated, first-hand (vendor)]

**Complaints, bugs and fixes**
- Complaint or known bug: Issue #211 reports Fleurs-zh CER of 7.19% against the card's 2.41, with the reporter's own normalization. Fix or workaround: none; no maintainer reply. [QwenLM/Qwen3-ASR #211, 2026-09-11, first-hand]
- Complaint or known bug: Prompt text leaked into transcripts, and near-silent audio returned hotwords (issue #186, size not named). Fix or workaround: Reported by one user, not confirmed: a "Technical terms:" prefix helped somewhat but still leaked sometimes. No maintainer reply. [QwenLM/Qwen3-ASR #186, 2026-06-17, first-hand]

**Evidence**
- Strength: thin. Maker and vendor figures; no independent 1.7B test.
- Sources:
  - Qwen3-ASR-1.7B model card, huggingface.co/Qwen/Qwen3-ASR-1.7B, first-hand (maker).
  - Whisper Notes blog, whispernotes.app/blog/qwen3-asr-vs-whisper, 2026-08-25, first-hand (vendor).
  - Issues #211 and #186, github.com/QwenLM/Qwen3-ASR, first-hand.
