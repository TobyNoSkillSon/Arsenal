# Pocket TTS

Pocket TTS is a small English text-to-speech model from Kyutai, built on their Continuous Audio Language Model (CALM) design. A Transformer produces a context embedding at each step, and a small MLP head generates the next continuous audio frame through consistency modelling; Kyutai's Mimi codec decodes the frames to audio. Kyutai gives it about 100M parameters, and Arsenal's count of the pinned files, including preset voice conditioning, is 118M. It streams speech early and can clone a voice from a short clip with no transcript. Arsenal runs the MLX conversion `mlx-community/pocket-tts`.

## What it is for

Kyutai built it to run fast on a laptop CPU: about 200 ms to first audio and about 6× real time on a MacBook Air M4 CPU, using two cores, with streaming and text of any length. Arsenal runs it on the GPU through MLX.

- Short spoken replies where first audio must come quickly
- Reading English text in one of six preset voices
- Speaking in a voice cloned from a clip of up to 30 seconds, without writing out what the clip says

## Inputs and outputs

### `speak` (default)

Speaks text in a preset voice, a saved cloned voice, or a voice taken from a reference clip on the spot.

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | 1 to 100000 characters. This is Arsenal's request cap, not a vendor context length; the model splits the text into sentences itself. |
| `voice` | string | no | A preset ID (`alba`, `azelma`, `eponine`, `fantine`, `javert`, `marius`) or a saved voice ID (`voice-…`) from `voices`. When left out, the Voice setting applies (default `alba`). An unknown preset is refused. |
| `reference` | object `{"path": string}` | no | `path` is required inside it (at least 1 character). It points to a clip on this Mac to clone for this request only. A reference overrides `voice`. When left out, the Reference clip setting applies (default empty). |
| `seed` | integer | no | 0 to 4294967295. |
| `stream` | boolean | no | Over HTTP, `true` streams raw audio as it is made (see Run it). MCP always returns a file. |
| `out_dir` | string | no | Managed by Arsenal. MCP and HTTP refuse requests that set it. |

Returns one WAV file: mono, 16-bit PCM, 24000 Hz. The result also reports `duration_s`, `gen_s`, `first_audio_s`, `rtf`, `sample_rate`, `channels`, `bytes` and `sha256`. A streamed HTTP response is headerless signed 16-bit little-endian PCM (`audio/pcm`), with the rate and channel count in the `X-Arsenal-Sample-Rate`, `X-Arsenal-Channels` and `X-Arsenal-Encoding` response headers.

Reference clips (here and in `clone_voice`) may be any audio file macOS can decode, such as WAV, AIFF, CAF, MP3, M4A or FLAC. Each must hold more than zero and at most 30 seconds of audio. Arsenal mixes it down to mono and resamples it to 24000 Hz; longer clips are refused, not trimmed. No transcript is needed. Kyutai notes that the clip's audio quality carries into the output, so a clean, dry recording works best.

### `clone_voice`

Saves a voice from a reference clip so later `speak` requests can use it by ID.

| Field | Type | Required | Notes |
|---|---|---|---|
| `audio` | object `{"path": string}` | yes | The reference clip, with the format and length rules above. Arsenal copies it, so later changes to the original do not affect the saved voice. |
| `name` | string | yes | 1 to 200 characters. A display name. |

Returns `{"voice_id": "voice-…"}`. Pass that ID as `voice` in `speak`. Saved voices belong to this model only.

### `voices`

Lists preset and saved voices. Takes no fields (`{}`).

Returns `{"voices":[...]}`. A preset entry has `id`, `name`, `language` (`en`), `model` and `gender` (always `unspecified` for Pocket presets). A saved entry has `id`, `name`, `language` (`auto`), `description` (`Saved reference voice`) and `saved: true`.

### `delete_voice`

Arsenal adds this operation; it is not declared in the folder. It deletes one saved voice. Its one field is `voice_id` (string, required, at least 1 character).

## Run it

MCP: `arsenal_run(model:"pocket-tts", operation:"speak", input:{"text":"…","voice":"alba"})`. Speech plays aloud by default; pass `play:false` to only save the file. Cloning uses `operation:"clone_voice"` with `input:{"audio":{"path":"/absolute/path/clip.wav"},"name":"…"}`.

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/pocket-tts/ops/speak" -H 'Content-Type: application/json' -d '{"text":"…","voice":"alba"}' -o speech.wav`. Other operations use `/ops/clone_voice`, `/ops/voices` and `/ops/delete_voice` and return JSON.

The Voice and Reference clip settings fill those fields when a request leaves them out. In quick checks on the M5 Max at BF16, first audio came after 0.04 s, the real-time factor was 0.12 and peak memory was 0.50 GB (measured; Mac in use).

## Limits

- English only in this conversion. Kyutai's current repository lists more languages and 26 presets; Arsenal pins six of its eight English presets (two have non-commercial terms; see Licence and credit).
- No voice design, no delivery or emotion instructions, no `speed` and no `language` field.
- Kyutai notes that pauses cannot be added by inserting silence in the text.
- Reference clips over 30 seconds are refused.

## Licence and credit

The weights are CC-BY-4.0. This allows commercial use and adaptation if you credit Kyutai and indicate changes. The six preset voices Arsenal downloads come from sources with open terms listed in `kyutai/tts-voices` (VCTK, Common Voice and donated recordings). Kyutai also publishes `cosette` (Expresso) and `jean` (EARS), whose datasets are CC BY-NC 4.0, non-commercial use only; Arsenal leaves those two out. Kyutai's terms forbid cloning or impersonating a voice without explicit, lawful consent, and presenting generated speech as a genuine recording of a real person. The upstream `kyutai/pocket-tts` repository requires accepting its conditions before download; Arsenal downloads the `mlx-community` conversion.

Makers: Kyutai (Manu Orsini, Simon Rouard, Gabriel De Marmiesse, Václav Volhejn, Neil Zeghidour, Alexandre Défossez). The paper is "Continuous Audio Language Models" by Simon Rouard, Manu Orsini, Axel Roebel, Neil Zeghidour and Alexandre Défossez, arXiv:2509.06926 (2025).

## Links

- Model card (pinned conversion): https://huggingface.co/mlx-community/pocket-tts/tree/cbf71d5f6657bbc3f4bc02f85ee408261225bec7
- Original model card: https://huggingface.co/kyutai/pocket-tts
- Voices and their licences: https://huggingface.co/kyutai/tts-voices
- Code: https://github.com/kyutai-labs/pocket-tts
- Paper: https://arxiv.org/abs/2509.06926

## How Arsenal runs it

Source: `mlx-community/pocket-tts` at `cbf71d5f6657bbc3f4bc02f85ee408261225bec7`. The source dtype is `bf16`, and BF16 is the runtime base. The 16-bit tier casts every learned source tensor, including the codec and the preset conditioning. FP16 must pass the BF16-reference format gate to be used; it failed, so BF16 stays (FP16 trial: held-out relative L2 0.0546, cosine 0.9985). Port provenance is in `Workers/Voices/PORTED-FROM.md`.

Runner: the bundled native voices worker (`arsenal-worker-voices`; see `runners/native/README.md`), going through Get, Load and then speak. Pocket has no accepted optimisation components, so Fast and Exact use the same path (`log/004-v2-folder.md`). Saved voices persist per model in Arsenal's support folder, under `voices/pocket-tts`. Cloning copies the clip there.

Precisions: 16-bit (BF16) only since 26 Oct 2026. The 16-bit weights are 0.240 GB, under the 1 GB line, so int8 and int4 are no longer offered (`log/011-sixteen-bit-only.md`).

Quality gate: Round 5 checked BF16, int8 and int4 on separate selection and held-out sets (66 and 44 component outputs), with bounds of relative L2 at most 0.05 and cosine at least 0.999. BF16 passed, with RTF 0.122 and a peak of 0.503 GB. Int8 passed, with held-out relative L2 0.0378, cosine 0.99930 and a peak of 0.426 GB. Int4 failed, with held-out relative L2 0.509 and cosine 0.877 (`log/010-voice-heldout-r5.md`). Earlier rounds are recorded in `log/006` to `log/009`: the blocked first intake, the whole-component intake, the SEANet decoder-input boundary and the matrix fallback. Figures are debug-worker quick checks on a busy Mac; spectral distance is diagnostic only. Independent vendor and perceptual parity remain unqualified.

Service cycles: a Get, Load, clone from a generated reference, speak with the saved ID, voices and Unload cycle completed for BF16 and int8 in Fast and Exact. Saved voices survived worker restarts and dtype changes, and no weight bytes came from the network (`log/005-v2-real-cycles.md`).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Six languages on the card; this conversion is English only. Clean reference clips first. [model card, kyutai/pocket-tts, undated, first-hand]
- Parameters people recommend: Temperature 0.3 to 0.5, for one user on the multilingual model; not in Arsenal's `speak` schema. [GitHub issue #221, opinion]
- Arsenal call: `arsenal_run(model:"pocket-tts", input:{"text":"Your order has shipped.","voice":"alba"})` [ops/speak.input.json, first-hand]

**Good and bad**
- Good at: The card claims about 200 ms to first audio and 6x real time on a MacBook Air M4 CPU. [model card, undated, first-hand]
- Bad at: Phrase-end repeats, e.g. "an hour an hour", on stock voices (#347); this conversion's checkpoint is unidentified. [GitHub issue #347, opinion]
- Bad at: Repeated and inserted words with exported voice profiles (#221). [GitHub issue #221, opinion]
- Against the obvious alternatives: A review of community comparisons ranks cloning above Kokoro, calling it subjective. [andrew.ooo review, 2026-07-12, opinion]

**Complaints, bugs and fixes**
- Complaint or known bug: Phrase-end repeats (#347, open). Fix or workaround: The maintainer's new model, english_2026-09_2 (default with #353), cut "an hour." repeats from 22 of 90 to 0 of 90 on their runs. A later reporter (17 Oct) says it still repeats. [#347 comments, first-hand for maintainer's runs]
- Complaint or known bug: Repeated and inserted words (#221). Fix or workaround: Reported by one user, not confirmed: rolling back to v1 was clean; an ASR round-trip check regenerates bad takes; temp 0.3 to 0.5 helped partly. [GitHub issue #221, opinion]

**Evidence**
- Strength: thin. Speed is the maker's claim; quality rests on two issues and one review.
- Sources:
  - Model card, https://huggingface.co/kyutai/pocket-tts, undated, first-hand (maker).
  - Issues #347 and #221, https://github.com/kyutai-labs/pocket-tts/issues, opinion.
  - Review, https://andrew.ooo/posts/pocket-tts-kyutai-cpu-voice-cloning-review/, 2026-07-12, opinion.
