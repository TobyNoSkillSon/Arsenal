# Qwen3 TTS 12Hz 1.7B VoiceDesign

Qwen3-TTS-12Hz-1.7B-VoiceDesign is the Qwen team's 1.7-billion-parameter text-to-speech model that creates a voice from a written description. A multi-codebook language model generates speech codes, and the Qwen3-TTS-Tokenizer-12Hz codec (16 codebooks at 12.5 frames per second) turns them into 24 kHz audio. It covers ten languages. What it does best is invent a new voice, with its timbre, emotion and prosody, from plain-language instructions; it needs no recording. Arsenal runs the MLX conversion `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16`.

## What it is for

The makers built it for voice design: speech whose timbre, emotion and prosody follow a natural-language description. They evaluate it on InstructTTSEval. It has no preset speakers and does not clone.

- Making a voice for a character, narrator or assistant from a description such as age, gender, pitch, accent and mood
- Delivery that follows an instruction, such as "speak in an incredulous tone, with a hint of panic"
- Saving a designed voice and reusing it, or cloning its preview with Qwen3 TTS 12Hz 1.7B Base

## Inputs and outputs

### `speak` (default)

Speaks text in a voice described in the request or saved earlier. Every request needs `instructions`, `description` or a saved `voice`.

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | 1 to 100000 characters. This is Arsenal's request cap, not a vendor context length; output length is bounded by `max_tokens`. |
| `instructions` | string | one of three | At least 1 character. A description of the voice and its delivery. When a request leaves out `instructions`, `description` and `voice`, the Voice description setting fills it (default "A warm adult narrator with a clear, relaxed conversational voice."). |
| `description` | string | one of three | At least 1 character. The same role as `instructions`; if both are sent, `instructions` wins. |
| `voice` | string | one of three | A saved voice ID from `voices`, matching `^voice-[0-9a-f-]+$`. These checkpoints have no built-in speaker presets. A saved voice replays its stored description, seed and sampling settings. |
| `language` | string | no | One of `auto`, `chinese`, `english`, `japanese`, `korean`, `german`, `french`, `russian`, `portuguese`, `spanish`, `italian`. Default `english`; when left out, the Language setting applies. |
| `seed` | integer | no | 0 to 4294967295. Ignored for a saved voice, which uses its own seed. |
| `max_tokens` | integer | no | 1 to 8192. The maximum number of codec frames; the runner default is 8192, about 655 s of audio at 12.5 frames/s (derived). |
| `temperature` | number | no | 0 to 2. Runner default 0.9. |
| `top_p` | number | no | 0.000001 to 1. Runner default 1.0. |
| `top_k` | integer | no | 0 to 4096. Runner default 50. |
| `repetition_penalty` | number | no | 0.01 to 2. Runner default 1.05. |
| `stream` | boolean | no | Over HTTP, `true` streams raw audio as it is made (see Run it). MCP always returns a file. |
| `out_dir` | string | no | Managed by Arsenal. MCP and HTTP refuse requests that set it. |

Returns one WAV file: mono, 16-bit PCM, 24000 Hz. The result also reports `duration_s`, `gen_s`, `first_audio_s`, `rtf`, `sample_rate`, `channels`, `bytes` and `sha256`. A streamed HTTP response is headerless signed 16-bit little-endian PCM (`audio/pcm`), with the rate in the `X-Arsenal-Sample-Rate` header.

The description is free text in any of the supported languages. The makers' examples describe timbre, age, pitch range, emotion and pacing ("Male, 17 years old, tenor range, gaining confidence…") or give a delivery instruction for the line. Without a seed, each request samples afresh, so the same description can give a different voice each time. To keep one voice, save it with `design_voice` and a `name`, then speak with its ID. The whole text is generated in one pass, with audio streamed out every 0.25 s of generation.

### `design_voice`

Designs a voice from a description and renders a preview. With a `name`, it also saves the voice.

| Field | Type | Required | Notes |
|---|---|---|---|
| `description` | string | yes | 1 to 10000 characters. The voice description. |
| `preview_text` | string | no | 1 to 100000 characters. What the preview says. When left out it is "Hello. This is a preview of your new voice." |
| `language` | string | no | Same choices as in `speak`; stored with the design (default `english`). |
| `seed` | integer | no | 0 to 4294967295. When left out, a random seed is drawn and returned, so the design can be reproduced. |
| `name` | string | no | 1 to 200 characters. If given, the design is saved and `voice_id` is a durable `voice-…` ID. Without it, the result is an unsaved preview (`preview-…`). |
| `max_tokens`, `temperature`, `top_p`, `top_k`, `repetition_penalty` | as in `speak` | no | Same bounds. The sampling settings (all except `max_tokens`) are saved with a named design and replayed by later speech. |
| `out_dir` | string | no | Managed by Arsenal. MCP and HTTP refuse requests that set it. |

Returns JSON with `voice_id`, `saved` (true or false), `preview` (the preview WAV: `path`, 24000 Hz mono 16-bit, with the same timing fields as `speak`) and `design` (`seed`, `model_revision`, `dtype`, `fast`, `preview_text`, `language` and any sampling settings). A saved voice is tied to the model revision it was designed on; after a model update Arsenal refuses it and asks for a new design.

### `voices`

Lists saved voices; this model has no presets. Takes no fields (`{}`).

Returns `{"voices":[...]}`. Each entry has `id`, `name`, `language`, `description` (the saved description) and `saved: true`.

### `delete_voice`

Arsenal adds this operation; it is not declared in the folder. It deletes one saved voice. Its one field is `voice_id` (string, required, at least 1 character).

## Run it

MCP: `arsenal_run(model:"qwen3-tts-voicedesign", operation:"speak", input:{"text":"…","instructions":"…","language":"english"})`. Speech plays aloud by default; pass `play:false` to only save the file. Designing uses `operation:"design_voice"` with `input:{"description":"…","name":"…"}`.

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/qwen3-tts-voicedesign/ops/speak" -H 'Content-Type: application/json' -d '{"text":"…","instructions":"…","language":"english"}' -o speech.wav`. Other operations use `/ops/design_voice`, `/ops/voices` and `/ops/delete_voice` and return JSON.

The Voice, Language and Voice description settings fill those fields when a request leaves them out. In quick checks on the M5 Max, first audio came after 0.15 to 0.17 s. The real-time factor was 0.51 at BF16, 0.52 at int8 and 0.46 at int4, with peak memory of 4.75, 4.38 and 1.88 GB (measured; Mac in use).

## Limits

- Ten languages only (listed above).
- No cloning from a recording and no `speed` field. To clone a designed voice, use the Base model with this model's preview.
- An unsaved description is not a stable voice: without a fixed seed it changes from request to request. A saved design replays exactly only for the same text, language, model revision, precision and mode (`log/007-reproducible-saved-design.md`). Its consistency on new text has not been measured.
- Int4 failed Arsenal's quality check badly (see below) and is offered with a warning that quality may be lower.
- One request can produce at most `max_tokens` codec frames (8192, about 655 s, derived).

## Licence and credit

Apache-2.0, which allows commercial use, modification and redistribution if the licence and notices are kept. Made by the Qwen team. Citation requested by the makers:

```bibtex
@article{Qwen3-TTS,
  title={Qwen3-TTS Technical Report},
  author={Hangrui Hu and Xinfa Zhu and Ting He and Dake Guo and Bin Zhang and Xiong Wang and Zhifang Guo and Ziyue Jiang and Hongkun Hao and Zishan Guo and Xinyu Zhang and Pei Zhang and Baosong Yang and Jin Xu and Jingren Zhou and Junyang Lin},
  journal={arXiv preprint arXiv:2601.15621},
  year={2026}
}
```

## Links

- Model card (pinned conversion): https://huggingface.co/mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16/tree/7d3824abff87e49756bb0f83fb5411de75d160c4
- Original model card: https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign
- Code: https://github.com/QwenLM/Qwen3-TTS
- Paper: https://huggingface.co/papers/2601.15621

## How Arsenal runs it

Source: `mlx-community/Qwen3-TTS-12Hz-1.7B-VoiceDesign-bf16` at `7d3824abff87e49756bb0f83fb5411de75d160c4`. The source dtype is `bf16`, and BF16 is the runtime base. The 16-bit tier casts every learned source tensor, including the codec. FP16 must pass the BF16-reference format gate to be used; it failed (held-out relative L2 0.421, cosine 0.911), so BF16 stays. Port provenance is in `Workers/Voices/PORTED-FROM.md`.

Runner: the bundled native voices worker (`arsenal-worker-voices`; see `runners/native/README.md`). Fast and Exact keep Vireo's accepted components with their self-tests and fallbacks. On M5, the folder kernel `kernels/qwen-fused-rope.metal` replaces `qwen.rotate-half-rope` in Exact mode, and Fast inherits it. It gave 24 bit-identical check outputs (27,648 elements), and the Fast and Exact WAVs were byte-identical. It falls back to the stock expression if it fails to compile or to match (`log/005-folder-rope.md`). Saved designs persist per model in Arsenal's support folder, under `voices/qwen3-tts-voicedesign`, with the description, seed, revision, dtype, mode, preview text, language, sampling settings and preview audio.

Precisions: BF16 (4.5 GB download), int8 (2.4 GB) and int4 (1.3 GB), all derived from the 16-bit talker as affine group-64, not community quants. Int8 keeps the matrices listed in `model.toml` `[derive]` at 16-bit.

Quality gate: Round 5 checked BF16, int8 and int4 on separate selection and held-out sets (300 and 200 component outputs), with bounds of relative L2 at most 0.05 and cosine at least 0.999. BF16 passed, with RTF 0.505 and a peak of 4.755 GB. Int8 passed, with held-out relative L2 0.0210, cosine 0.99978, RTF 0.515, a peak of 4.382 GB and 80.17% kept 16-bit. Int4 failed, with selection relative L2 1.70 and cosine 0.0007, held-out relative L2 0.911 and cosine 0.422, RTF 0.456 and a peak of 1.877 GB; it stays offered with its quality warning (`log/010-voice-heldout-r5.md`). Earlier intake rounds are in `log/006` to `log/009`. Figures are debug-worker quick checks on a busy Mac; spectral distance is diagnostic only. Independent vendor and perceptual parity remain unqualified.

Service cycles: a Get, Load, design with preview, speech from the saved description, voices and Unload cycle passed in all six dtype and mode cells. Six saved voices survived worker restarts and dtype changes, and no weight bytes came from the network (`log/006-v2-real-cycles.md`). A saved preview replayed byte-for-byte after unload and reload (`log/007-reproducible-saved-design.md`).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: text plus a natural-language instruction describing the voice; the card's overview says speech is driven by natural language instructions. [VoiceDesign model card, Hugging Face, published 2026-01-22, first-hand]
- Prompts, reference audio or voice settings, if relevant: One Mac write-up's example description: "A young female. The voice is not too high, with a slight breathy quality. Modest and nervous." [dev.to, tumf, originally 2026-01-24, posted 2026-01-29, first-hand]
- Parameters people recommend: Not found in sources checked. The card's quickstart sets dtype=torch.bfloat16 in its CustomVoice example. [VoiceDesign model card, 2026-01-22, first-hand]
- Arsenal call: `arsenal_run(model:"qwen3-tts-voicedesign", operation:"design_voice", input:{"description":"Male, 17 years old, tenor range, gaining confidence","name":"young-tenor"})`

**Good and bad**
- Good at: The maker says it beats MiniMax-Voice-Design on instruction following; the launch post gives no figures. [Alibaba Cloud launch post, 2026-01-26, first-hand]
- Bad at: The same instructions can give a slightly different voice. [dev.to, tumf, originally 2026-01-24, posted 2026-01-29, first-hand]
- Against the obvious alternatives: Not found in sources checked beyond the maker's MiniMax claim.

**Complaints, bugs and fixes**
- Complaint or known bug: In mlx-audio, split_pattern is ignored for "the other two models" besides voice cloning (Base). The sentence does not name VoiceDesign. [myByways, 2026-02-02, 6 Feb update, first-hand]
- Fix or workaround: Reported by one user, not confirmed: the author wrote a separate manual-splitting path in their own code. [myByways, 6 Feb update, first-hand]

**Evidence**
- Strength: thin. Maker claims and two Mac write-ups; no independent consistency test.
- Sources:
  - Qwen3-TTS-12Hz-1.7B-VoiceDesign model card, https://huggingface.co/Qwen/Qwen3-TTS-12Hz-1.7B-VoiceDesign, published 2026-01-22, first-hand (maker)
  - Qwen3-TTS launch post, https://www.alibabacloud.com/blog/602826, 2026-01-26, first-hand (maker)
  - dev.to, tumf, https://dev.to/tumf/qwen3-tts-surprised-by-the-quality-of-japanese-on-apple-silicon-m3-creating-rights-free-voices-k1d, originally 2026-01-24, posted 2026-01-29, first-hand
  - myByways, https://mybyways.com/blog/qwen3-tts-with-mlx-audio-on-macos, 2026-02-02 with a 6 Feb update, first-hand
