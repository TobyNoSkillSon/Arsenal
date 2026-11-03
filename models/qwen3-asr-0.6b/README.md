# Qwen3 ASR 0.6B

Qwen3 ASR 0.6B is the smaller speech recognizer of Alibaba Qwen's Qwen3-ASR family: an audio encoder feeding a Qwen3 language model that writes the transcript, built on the audio understanding of Qwen3-Omni and 782 million parameters in Arsenal's count. It identifies the language and transcribes the same 30 languages and 22 Chinese dialects as the 1.7B, in about half the memory (measured in Arsenal: 2.77 GB against 5.91 GB peak). Its strength is broad-language transcription, including Chinese, Japanese, Korean, Arabic and Southeast Asian languages, on Macs with less memory to spare.

## What it is for

Qwen positions the 0.6B as the accuracy-efficiency trade-off of the family, reporting "2000 times throughput at a concurrency of 128"; it shares the family's robustness to noise, accents, dialects, singing and songs with background music.

- Transcribing Chinese, Japanese, Korean and the other 27 supported languages when memory is short or a larger model is already loaded
- Audio whose language is unknown
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

MCP: `arsenal_run(model:"qwen3-asr-0.6b", input:{"audio":{"path":"/absolute/path/interview.m4a"},"language":"ja","prompt":"Speakers discuss Arsenal and MLX."})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/qwen3-asr-0.6b/ops/transcribe" -H 'Content-Type: application/json' -d '{"audio":{"path":"/absolute/path/interview.m4a"},"language":"ja"}'`

A path may name any file on this Mac; Arsenal clones it into its inbox before the model reads it. `arsenal_upload` is only for a file that is not on this Mac. Decoding is greedy (temperature 0). In a quick check on an M5 Max (22 Oct 2026, Mac in use, short fixed test clip), a warm request took 0.09 s and the worker peaked at 2.77 GB.

## Limits

- No word or sentence timestamps. The makers provide them through a separate model, Qwen3-ForcedAligner-0.6B, which Arsenal does not include; segment times here are only the 30-second cut points.
- The `language` field accepts the 30 languages only. The 22 Chinese dialects (for example Sichuan, Dongbei, Wu, Minnan, Hong Kong and Guangdong Cantonese) are recognized only through automatic detection.
- Ten minutes per file in Arsenal. The makers' toolkit transcribes up to 1200 seconds per pass; Arsenal decodes 30-second segments, so a sentence that crosses a cut point is split between two segments, and context from one segment does not carry into the next.
- A little less accurate than `qwen3-asr-1.7b` in Vella's measurements; use the 1.7B when memory allows.
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

- Model card (pinned MLX conversion): https://huggingface.co/mlx-community/Qwen3-ASR-0.6B-bf16/tree/eae2b51f96265328f1e7beced788adb0e4536f92
- Original model card: https://huggingface.co/Qwen/Qwen3-ASR-0.6B
- Code: https://github.com/QwenLM/Qwen3-ASR
- Paper: https://arxiv.org/abs/2601.21337

## How Arsenal runs it

Checkpoint: `mlx-community/Qwen3-ASR-0.6B-bf16@eae2b51f96265328f1e7beced788adb0e4536f92`, the release Vella uses. Exact files, SHA-256 digests and byte counts are in model.toml; only metadata and safetensors headers were fetched during integration.

The release is BF16; Get casts it to FP16, the serving base. Arsenal offers FP16 only: at 782M parameters an 8-bit copy would save under 1 GB.

The worker runs Vella's Qwen3-ASR runtime (`Workers/Audio/Sources/VellaSTT`, copied from Vella's dictation worker) with Vella's optimized decoder on Apple GPUs of Metal family 9 (M3 and later): greedy decoding pipelined with asynchronous evaluation, and a prefill that skips the vocabulary projection whose logits are discarded. Both are token-exact against stock MLX. Vella's optimized encoder holds the audio tower in FP32 weights; Arsenal serves no FP32 weights, so the tower runs as stock MLX does. An optimized run that throws, produces non-finite logits, or returns an empty transcript stock does not confirm reruns on stock MLX, which then stays selected until the model is reloaded. Other Macs run stock MLX.

Verification: the no-weight fixture builds a shrunk version of the actual graph and runs the ordinary transcribe operation; the full-size graph inventory matches all 611 tensors of the pinned checkpoint header. Worker tests check that the optimized decoder gives stock's token IDs. Checked live at FP16 on 20 Oct: an English test sentence transcribed exactly. The 22 Oct quick check (load 1.14 s, warm request 0.089 s, peak 2,645 MiB summed over Arsenal's worker processes) is in `results/`. Trained-weight transcription quality across languages, FP16 against the BF16 original, and controlled speed and memory are not yet measured in Arsenal. Research log: `log/001-integration.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Pass `language` to force one of 30 languages; leave it out to detect. [model card, paper Jan 29 2026, first-hand (maker)]
- Prompts, reference audio or voice settings, if relevant: The card has no prompt input. Issue #186 (size not named) reports prompt text leaking into transcripts. [QwenLM/Qwen3-ASR #186, 2026-06-17, first-hand]
- Parameters people recommend: Card's code comment: set a larger `max_new_tokens` for long audio. [model card, first-hand (maker)]
- Arsenal call: `arsenal_run(model:"qwen3-asr-0.6b", input:{"audio":{"path":"/absolute/path/clip.m4a"},"language":"en"})` [this README, 2026-10-31, first-hand]

**Good and bad**
- Good at: Maker-reported language ID 96.8% and LibriSpeech clean/other WER 2.11/4.55. [model card, first-hand (maker)]
- Bad at: Higher WER than the 1.7B: LibriSpeech 2.11/4.55 against 1.63/3.38; Fleurs-en 4.39 against 3.35. [model card, first-hand (maker)]
- Against the obvious alternatives: Vendor Soniqo: 0.6B 5-bit 1.74% against WhisperKit Large-v3 Turbo 1.71% on LibriSpeech test-clean, on different utterance subsets. [soniqo.audio benchmark, undated, first-hand (vendor)]

**Complaints, bugs and fixes**
- Complaint or known bug: Chinese mixed with English came out all English on vLLM. Fix or workaround: Reported by one user, not confirmed: leaving `language` unset gave "pretty good" results; another said setting Chinese worked. No maintainer reply. [QwenLM/Qwen3-ASR #163, 2026-04-27, first-hand]
- Complaint or known bug: The bundled transformers path skips the encoder's attention window unless FlashAttention 2 runs, so clips over 8 s encode differently (size not named). Fix or workaround: A second user reports FA2 gives identical outputs and open PR #103 matches production; no maintainer reply. [QwenLM/Qwen3-ASR #213, 2026-09-19, first-hand]

**Evidence**
- Strength: thin. Maker figures and one vendor benchmark.
- Sources:
  - Qwen3-ASR-0.6B model card, huggingface.co/Qwen/Qwen3-ASR-0.6B, first-hand (maker).
  - Soniqo benchmark, soniqo.audio/benchmarks/qwen3-asr-vs-whisper, undated, first-hand (vendor).
  - Issues #163, #186 and #213, github.com/QwenLM/Qwen3-ASR, first-hand.
