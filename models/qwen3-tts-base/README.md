# Qwen3 TTS 12Hz 1.7B Base

Qwen3-TTS-12Hz-1.7B-Base is the Qwen team's 1.7-billion-parameter voice-cloning text-to-speech model. A multi-codebook language model generates speech codes, and the Qwen3-TTS-Tokenizer-12Hz codec (16 codebooks at 12.5 frames per second) turns them into 24 kHz audio. It covers ten languages. What it does best is clone a voice from a short reference clip and its transcript. Arsenal runs the MLX conversion `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16`.

## What it is for

The makers built Base for fast voice cloning ("3-second rapid voice clone from user audio input") and as a starting point for fine-tuning. Its Seed-TTS zero-shot word error rate is 0.77 on test-zh and 1.24 on test-en. It has no preset speakers and takes no voice description.

- Speaking new text in a voice cloned from a short recording
- Speech in Chinese, English, Japanese, Korean, German, French, Russian, Portuguese, Spanish or Italian
- Reusing a voice designed by Qwen3 TTS VoiceDesign: save that model's preview, then clone it here

## Inputs and outputs

### `speak` (default)

Speaks text in a cloned voice. Every request needs either `reference` or a saved `voice`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | 1 to 100000 characters. This is Arsenal's request cap, not a vendor context length; output length is bounded by `max_tokens`. |
| `voice` | string | one of `voice`/`reference` | A saved voice ID from `voices`, matching `^voice-[0-9a-f-]+$`. These checkpoints have no built-in speaker presets. |
| `reference` | object `{"path": string, "text": string}` | one of `voice`/`reference` | Both members are required and at least 1 character. `path` is the clip on this Mac and `text` is exactly what it says. The Reference clip and Reference transcript settings fill these when a request leaves them out. |
| `language` | string | no | One of `auto`, `chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`. Default `english`; when left out, the Language setting applies. |
| `seed` | integer | no | 0 to 4294967295. |
| `max_tokens` | integer | no | 1 to 8192. The maximum number of codec frames; the runner default is 8192, about 655 s of audio at 12.5 frames/s (derived). |
| `temperature` | number | no | 0 to 2. Runner default 0.9. |
| `top_p` | number | no | 0.000001 to 1. Runner default 1.0. |
| `top_k` | integer | no | 0 to 4096. Runner default 50. |
| `repetition_penalty` | number | no | 0.01 to 2. Runner default 1.05. |
| `stream` | boolean | no | Over HTTP, `true` streams raw audio as it is made (see Run it). MCP always returns a file. |
| `out_dir` | string | no | Managed by Arsenal. MCP and HTTP refuse requests that set it. |

Returns one WAV file: mono, 16-bit PCM, 24000 Hz. The result also reports `duration_s`, `gen_s`, `first_audio_s`, `rtf`, `sample_rate`, `channels`, `bytes` and `sha256`. A streamed HTTP response is headerless signed 16-bit little-endian PCM (`audio/pcm`), with the rate in the `X-Arsenal-Sample-Rate` header.

Reference clips may be any audio file macOS can decode (WAV, AIFF, CAF, MP3, M4A, FLAC and so on). Each must hold more than zero and at most 30 seconds of audio. Arsenal mixes it to mono and resamples it to 24000 Hz, the codec's input rate; longer clips are refused, not trimmed. The transcript must match the clip's words. The upstream card describes 3-second cloning but gives no recommended clip length. Upstream also offers a speaker-embedding-only mode without a transcript, which Arsenal does not expose. The worker caches the encoded reference, so repeated requests with the same clip, transcript and language skip re-encoding. The whole text is generated in one pass, with audio streamed out every 0.25 s of generation.

### `clone_voice`

Saves a reference clip and its transcript as a reusable voice.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | object `{"path": string}` | yes | The reference clip, with the rules above. Arsenal checks it and copies it, so later changes to the original do not affect the saved voice. |
| `transcript` | string | yes | 1 to 100000 characters. Exactly what the clip says. |
| `name` | string | yes | 1 to 200 characters. A display name. |
| `language` | string | no | Same choices as in `speak`. Stored with the voice. |

Returns `{"voice_id": "voice-…"}`. Pass that ID as `voice` in `speak`. Saved voices belong to this model only.

### `voices`

Lists saved voices; this model has no presets. Takes no fields (`{}`).

Returns `{"voices":[...]}`. Each entry has `id`, `name`, `language` (the stored language, or `auto`), `description` (`Saved reference voice`) and `saved: true`.

### `delete_voice`

Arsenal adds this operation; it is not declared in the folder. It deletes one saved voice. Its one field is `voice_id` (string, required, at least 1 character).

## Run it

MCP: `arsenal_run(model:"qwen3-tts-base", operation:"speak", input:{"text":"…","reference":{"path":"/absolute/path/clip.wav","text":"…"},"language":"english"})`. Speech plays aloud by default; pass `play:false` to only save the file. Cloning uses `operation:"clone_voice"` with `input:{"audio":{"path":"…"},"transcript":"…","name":"…"}`.

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/qwen3-tts-base/ops/speak" -H 'Content-Type: application/json' -d '{"text":"…","voice":"voice-…","language":"english"}' -o speech.wav`. Other operations use `/ops/clone_voice`, `/ops/voices` and `/ops/delete_voice` and return JSON.

The Language, Reference clip and Reference transcript settings fill those fields when a request leaves them out. The model has no Speak Clipboard menu action, because it needs a reference. In quick checks on the M5 Max, first audio came after 0.18 to 0.20 s. The real-time factor was 0.49 at BF16, 0.51 at int8 and 0.45 at int4, with peak memory of 4.82, 4.39 and 1.96 GB (measured; Mac in use).

## Limits

- Ten languages only (listed above); dialect voice profiles belong to the CustomVoice models, not this one.
- No voice design, no preset speakers, no delivery instructions and no `speed`.
- Reference clips over 30 seconds are refused; a wrong transcript degrades the clone.
- Int4 failed Arsenal's quality check (see below) and is offered with a warning that quality may be lower.
- One request can produce at most `max_tokens` codec frames (8192, about 655 s, derived).

## Licence and credit

Apache-2.0, which allows commercial use, modification and redistribution if the licence and notices are kept. Cloning a real person's voice needs their consent and the rights to the recording; the licence does not grant these. Made by the Qwen team. Citation requested by the makers:

```bibtex
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

## Links

- Model card (pinned conversion): https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16/tree/a6eb4f68e4b056f1215157bb696209bc82a6db48
- Original model card: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-Base
- Code: https://github.com/QwenLM/Qwen3-TTS
- Paper: https://huggingface.co/papers/2601.15621

## How Arsenal runs it

Source: `mlx-community/Qwen3-TTS-12Hz-1.7B-Base-bf16` at `a6eb4f68e4b056f1215157bb696209bc82a6db48`. The source dtype is `bf16`, and BF16 is the runtime base. The 16-bit tier casts every learned source tensor, including the codec. FP16 must pass the BF16-reference format gate to be used; it failed (held-out relative L2 0.386, cosine 0.927), so BF16 stays. Port provenance is in `Workers/Voices/PORTED-FROM.md`.

Runner: the bundled native voices worker (`arsenal-worker-voices`; see `runners/native/README.md`). Fast and Exact keep Vireo's accepted components with their self-tests and fallbacks. On M5, the folder kernel `kernels/qwen-fused-rope.metal` replaces `qwen.rotate-half-rope` in Exact mode, and Fast inherits it. It reports 24 bit-identical check outputs (27,648 elements) and falls back to the stock expression if it fails to compile or to match. Saved voices persist per model in Arsenal's support folder, under `voices/qwen3-tts-base`.

Precisions: BF16 (4.5 GB download), int8 (2.4 GB) and int4 (1.3 GB), all derived from the 16-bit talker as affine group-64, not community quants. Int8 keeps the matrices listed in `model.toml` `[derive]` at 16-bit.

Quality gate: Round 5 checked BF16, int8 and int4 on separate selection and held-out sets (303 and 202 component outputs), with bounds of relative L2 at most 0.05 and cosine at least 0.999. BF16 passed, with RTF 0.494 and a peak of 4.820 GB. Int8 passed, with held-out relative L2 0.0388, cosine 0.99925, RTF 0.510, a peak of 4.395 GB and 79.74% kept 16-bit. Int4 failed, with held-out relative L2 0.662, cosine 0.796, RTF 0.454 and a peak of 1.964 GB; it stays offered with its quality warning (`log/010-voice-heldout-r5.md`). Earlier intake rounds are in `log/006` to `log/009`. Figures are debug-worker quick checks on a busy Mac; spectral distance is diagnostic only. Independent vendor and perceptual parity remain unqualified.

Service cycles: a Get, Load, save a copied reference and transcript, speak with the saved ID, voices and Unload cycle passed in all six dtype and mode cells. Cycles took 1.9 to 7.1 s for about 1.1 s of audio, and no weight bytes came from the network (`log/005-v2-real-cycles.md`). `log/002-scaled-architecture.md` records the reduced-width random-weight stand-in used for tests that read no vendor weights.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The card gives no reference-length advice; it advertises a 3-second rapid clone. [model card, paper Jan 22 2026, first-hand (maker)]
- Prompts, reference audio or voice settings, if relevant: Set `language` explicitly when known. [model card, paper Jan 22 2026, first-hand (maker)]
- Parameters people recommend: The card's evaluation used bf16 and `max_new_tokens` 2048. [model card, paper Jan 22 2026, first-hand (maker)]
- Arsenal call: `arsenal_run(model:"qwen3-tts-base", operation:"speak", input:{"text":"Hi.","voice":"voice-<id>","language":"english"}, play:false)` [this README, 2026-10-31, first-hand]

**Good and bad**
- Good at: Maker's Seed-TTS WER for this Base: 0.77 zh, 1.24 en (card's Seed-TTS baseline 1.12, 2.25; CosyVoice 3 zh 0.71). [model card, first-hand (maker)]
- Bad at: Not found in sources checked beyond the bug reports below.
- Against the obvious alternatives: Not found in sources checked beyond the CosyVoice 3 row in the card.

**Complaints, bugs and fixes**
- Complaint or known bug: Periodic ~40 ms echo: the reporter measured a 40 ms cepstral peak in every 12 Hz output tried, including Base via an MLX port; the controls (macOS `say`, Chatterbox) show little or none. Fix or workaround: none; open, no maintainer reply. [QwenLM/Qwen3-TTS #386, 2026-10-22, first-hand (reporter's measurement)]
- Complaint or known bug: Voice-clone output repeats the reference transcript's last words in the first 1 to 2 seconds. Fix or workaround: Reported by one user, not confirmed: trimming the reference's final phrase stopped it for one of six voices. No maintainer reply; open. [QwenLM/Qwen3-TTS #341, 2026-07-09, first-hand]

**Evidence**
- Strength: thin. Maker WER figures and two open bug reports, no maintainer replies.
- Sources:
  - Qwen3-TTS-12Hz-1.7B-Base model card, paper Jan 22 2026, first-hand (maker).
  - Issue #341, github.com/QwenLM/Qwen3-TTS/issues/341, 2026-07-09, first-hand.
  - Issue #386, github.com/QwenLM/Qwen3-TTS/issues/386, 2026-10-22, first-hand.
