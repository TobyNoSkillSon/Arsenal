# Kokoro 82M

Kokoro is an open-weight text-to-speech model from Hexgrad with 82 million parameters. It is built on StyleTTS 2 and an ISTFTNet vocoder and is decoder-only: no diffusion and no encoder. It turns text into speech in one of 28 fixed English preset voices (American and British), and what it does best is speak quickly with very little memory. Arsenal runs the MLX conversion `mlx-community/Kokoro-82M-bf16`.

## What it is for

The makers aimed for quality comparable to much larger models while running faster and cheaper. They trained it on a few hundred hours of permissively licensed, public-domain and synthetic audio. Arsenal uses it as the default voice for spoken replies.

- Reading replies, notifications and documents aloud in English
- Speaking clipboard text from the Arsenal menu (Speak Clipboard)
- Short, low-latency speech where a fixed preset voice is enough

## Inputs and outputs

### `speak` (default)

Speaks text in a preset voice.

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | yes | 1 to 100000 characters. This is Arsenal's request cap, not a vendor context length; the worker splits the text itself (see below). |
| `voice` | string | no | A preset ID from the `voices` operation, for example `af_heart`. When left out, the Voice setting applies (default `af_heart`). An ID that is not one of the 28 presets is refused. |
| `speed` | number | no | 0.25 to 4. 1 is the voice's own pace. When left out, the Speed setting applies (default 1.0). |
| `seed` | integer | no | 0 to 4294967295. |
| `stream` | boolean | no | Over HTTP, `true` streams raw audio as it is made (see Run it). MCP always returns a file. |
| `out_dir` | string | no | Managed by Arsenal. MCP and HTTP refuse requests that set it; the result's path is printed instead. |

Returns one WAV file: mono, 16-bit PCM, 24000 Hz. The result also reports `duration_s`, `gen_s`, `first_audio_s`, `rtf`, `sample_rate`, `channels`, `bytes` and `sha256`. A streamed HTTP response is headerless signed 16-bit little-endian PCM (`audio/pcm`), with the rate and channel count in the `X-Arsenal-Sample-Rate`, `X-Arsenal-Channels` and `X-Arsenal-Encoding` response headers.

The language comes from the voice: IDs starting `a` are American English and IDs starting `b` are British English. The second letter gives the gender (`f` female, `m` male). Kokoro takes no language field. Text is converted to phonemes by the bundled English G2P model and the British lexicon. The worker splits the text into sentences, and splits any sentence longer than the model's 510-token context at the nearest clause boundary or space. It then streams one audio chunk per sentence, so first audio arrives after the first sentence, not after the whole text.

### `voices`

Lists the preset voices. Takes no fields (`{}`).

Returns `{"voices":[...]}`. Each entry has `id`, `name`, `language` (`en-US` or `en-GB`), `model` and `gender`. `gender` is `female` for `*f_*` voices and `unspecified` for the male voices.

The presets are `af_alloy`, `af_aoede`, `af_bella`, `af_heart`, `af_jessica`, `af_kore`, `af_nicole`, `af_nova`, `af_river`, `af_sarah`, `af_sky`, `am_adam`, `am_echo`, `am_eric`, `am_fenrir`, `am_liam`, `am_michael`, `am_onyx`, `am_puck`, `am_santa`, `bf_alice`, `bf_emma`, `bf_isabella`, `bf_lily`, `bm_daniel`, `bm_fable`, `bm_george` and `bm_lewis`. The makers grade each voice by the quality and amount of its training data. `af_heart` is graded A and `af_bella` A-. `af_nicole` and `bf_emma` are B-. Several voices are graded D or lower, for example `am_adam` (F+) and `am_santa` (D-).

## Run it

MCP: `arsenal_run(model:"kokoro-82m", input:{"text":"…","voice":"af_heart"})`. Speech plays aloud by default; pass `play:false` to only save the file. Use `operation:"voices"` with `input:{}` to list voices.

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/kokoro-82m/ops/speak" -H 'Content-Type: application/json' -d '{"text":"…","voice":"af_heart"}' -o speech.wav`. The `voices` route is the same with `/ops/voices` and `-d '{}'`.

The Voice and Speed settings in the model's settings card fill those fields when a request leaves them out. In quick checks on the M5 Max at FP16, first audio came after 0.17 s, the real-time factor was 0.03 (compute seconds per audio second) and peak memory was 1.39 GB (measured; Mac in use).

## Limits

- English only. The upstream v1.0 release covers eight languages, but this package ships only the American and British English voices and English G2P.
- It has no voice cloning, no voice design and no instructions about delivery or emotion.
- The makers say voices work best on chunks of about 100 to 200 tokens. Quality may drop below about 10 to 20 tokens, and voices may rush past about 400 tokens. Arsenal's sentence splitting keeps chunks within the 510-token context, but very short or very long sentences can still be affected.
- Voice quality varies widely between presets (see the grades above).

## Licence and credit

The weights are Apache-2.0, which allows commercial use, modification and redistribution if the licence and notices are kept. The bundled G2P assets are `beshkenadze/kitten-tts-g2p` (MIT) and the British lexicon from `hexgrad/misaki` (Apache-2.0). Kokoro is by Hexgrad. It builds on StyleTTS 2 and ISTFTNet (papers below), and the base model in the conversion metadata is `yl4579/StyleTTS2-LJSpeech`. The model card asks for no citation. The training data credits include the Koniwa (CC BY 3.0) and SIWIS (CC BY 4.0) datasets.

## Links

- Model card (pinned conversion): https://huggingface.co/mlx-community/Kokoro-82M-bf16/tree/a71e4d38b236d968966a2002c4c895dbd12b1c3c
- Original model card: https://huggingface.co/hexgrad/Kokoro-82M
- Voice list and grades: https://huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md
- Code: https://github.com/hexgrad/kokoro
- Paper (StyleTTS 2): https://arxiv.org/abs/2306.07691
- Paper (ISTFTNet): https://arxiv.org/abs/2203.02395

## How Arsenal runs it

Source: `mlx-community/Kokoro-82M-bf16` at `a71e4d38b236d968966a2002c4c895dbd12b1c3c`. The source dtype is `fp32`; it is cast once to FP16 at Get, and FP16 is the runtime base. The English G2P and British lexicon assets are required and pinned in `model.toml`. Port provenance is in `Workers/Voices/PORTED-FROM.md`.

Runner: the bundled native voices worker (`arsenal-worker-voices`, framed stdio v1; see `runners/native/README.md`), going through Get, Load and then speak. Fast and Exact keep per-component self-tests and fallbacks. Exact components are `kokoro-host-alignment`, `kokoro-conv-weight-cache` and `kokoro-fused-lstm`. Fast adds `kokoro-fp16-decoder`. Dtypes follow the pinned Vireo catalog and are never cast between BF16 and FP16. `optimizations.json` records the accepted Exact and Fast components; lab-only candidates are not defaults (`log/004-v2-folder.md`).

Precisions: 16-bit (FP16) only since 26 Oct 2026. The 16-bit weights are 0.171 GB, under the 1 GB line below which models ship 16-bit only (`log/012-sixteen-bit-only.md`). Before that, int8 was withdrawn after a quick check on the M5 Max (22 Oct 2026, Mac in use) found it no faster and no smaller than FP16. A short speak plus voices took 0.059 s against 0.062 s, with a peak of 1.43 GB against 1.40 GB; int4 took 0.045 s at 1.45 GB (`log/011-int8-withdrawn.md`). Copies of old precisions already on a Mac show in the settings card for removal.

Quality gate: Round 5 checked FP16, int8 and int4 on separate selection and held-out sets (33 and 22 component outputs). The shared bound was relative L2 at most 0.05 and cosine at least 0.999. FP16 passed at 0 and 1, with RTF 0.0289 and a peak of 1.386 GB. Int8 passed, with held-out max relative L2 0.0207 and min cosine 0.99979. Int4 failed, with held-out relative L2 0.308 and cosine 0.952 (`log/010-voice-heldout-r5.md`). These figures are debug-worker quick checks on a busy Mac; common-prefix spectral distance is diagnostic only. Historical schema-2 voice completion bindings qualify only their selection set. Independent vendor parity and perceptual parity (listening or ASR) remain unqualified.

Service cycles: a signed-catalog Get, Load, speak, voices and Unload cycle completed in all four dtype and mode cells, and worker residency returned to zero each time (`log/003-catalog-cycle.md`, `log/005-v2-real-cycles.md`). The first FP16 Fast cell included a fresh 342 MB Get: 108.8 s to Get and 119.8 s for the full cycle, for 1.6 s of audio. Later cells reused the verified snapshots. These are functional figures, not benchmarks.

Research history: `log/001-native-port.md` (port), `log/002-functional-smoke.md` (first streamed and file PCM match), and the logs above. Codec and quantization exclusions are kept there.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Plain English text. VOICES.md: best at 100 to 200 tokens; weak under about 10 to 20; rushed over about 400. [VOICES.md, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Graded voices: af_bella A-, af_nicole B-, bf_emma B-, am_adam F+. VOICES.md suggests lowering `speed` for rushing; #344 reports a phantom vowel below speed 1. [VOICES.md, undated; GitHub hexgrad/kokoro #344, 2026-07-02, first-hand]
- Parameters people recommend: Not found in sources checked beyond the card's example call. [model card, first-hand]
- Arsenal call: `arsenal_run(model:"kokoro-82m", operation:"speak", input:{"text":"Your build finished without errors.","voice":"af_bella"})`

**Good and bad**
- Good at: Speed and small size. The maker says it gives comparable quality to larger models. No listening test found. [model card, 2025-01-27 (v1.0), first-hand]
- Bad at: "ge" or "ch" words mispronounced when last or alone in a short string (#380); voiced noise after final punctuation (#365); a vowel at clip start below speed 1 (#344). [GitHub hexgrad/kokoro issues, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: The three issues above. Fix or workaround: No hexgrad reply in any thread. #380: one user appends " ." to such strings, unconfirmed. #344: a patch scaling only interior token durations, applied in a Hangry Labs KokoroTTS fork (not hexgrad). [#380, 2026-10-21; #365, 2026-08-28; #344, 2026-07-02; first-hand]

**Evidence**
- Strength: thin. Maker card and voice guide, plus three bug threads with no hexgrad reply.
- Sources:
  - Model card, huggingface.co/hexgrad/Kokoro-82M, v1.0 published 2025-01-27, first-hand (maker)
  - VOICES.md, huggingface.co/hexgrad/Kokoro-82M/blob/main/VOICES.md, undated, first-hand (maker)
  - MLX copy, huggingface.co/mlx-community/Kokoro-82M-bf16, repo created 2025-02-28 (inferred), no settings given
  - Issues #380, #365, #344, github.com/hexgrad/kokoro
