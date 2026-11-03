# ACE-Step 1.5 Turbo

ACE-Step 1.5 is an open music foundation model from ACE-Step, co-led by ACE Studio and StepFun. It pairs a 1.7B-parameter language model (`acestep-5Hz-lm-1.7B`, trained from Qwen3-1.7B) that plans a song with a diffusion transformer that renders it: the planner turns a short request into a song blueprint (metadata, a rewritten caption and 5 Hz audio codes), Qwen3-Embedding-0.6B encodes the caption and lyrics, the turbo DiT (`acestep-v15-turbo`, distilled to 8 steps) denoises latents, and an Oobleck VAE decodes them to 48 kHz stereo. The whole set is 5.01B parameters. What it does best is a complete song with vocals, from a style description and lyrics, in seconds.

## What it is for

The makers built it to bring commercial-grade music generation to consumer hardware: songs from short loops to 10-minute compositions, lyrics in 50+ languages, and editing (covers, repainting) in the same model. The turbo DiT is their 8-step distilled variant, rated "Very High" quality and "Medium" diversity in their model zoo.

- Songs with sung lyrics from a genre, mood and instrument description
- Instrumental music, loops and beds
- Covers: a supplied song re-rendered in another style
- Repainting: regenerating one time span of a song, or extending it before the start or past the end
- Taking the timbre of a reference recording
- Variations of one song through seeds, retake seeds and batches

## Inputs and outputs

### `generate_music` (default)

Makes one or more songs from a style prompt and optional lyrics, or covers or repaints a source song.

Most fields have no default in the schema; where the runner has one it is given below as "runner default" (from the runner's request type, which copies upstream's `GenerationParams` defaults).

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | 1–512 characters. Upstream's `caption`: what the music is (genre, mood, instruments, production). |
| `lyrics` | string | no | Up to 100000 characters. Lyrics with section tags in square brackets, such as `[Verse]` and `[Chorus]`. Empty lyrics are treated as instrumental. |
| `instrumental` | boolean | no | Default `false`. `true` replaces the lyrics with upstream's `[Instrumental]` marker. |
| `language` | string | no | Default `unknown`. Upstream's `vocal_language`. One of `ar`, `az`, `bg`, `bn`, `ca`, `cs`, `da`, `de`, `el`, `en`, `es`, `fa`, `fi`, `fr`, `he`, `hi`, `hr`, `ht`, `hu`, `id`, `is`, `it`, `ja`, `ko`, `la`, `lt`, `ms`, `ne`, `nl`, `no`, `pa`, `pl`, `pt`, `ro`, `ru`, `sa`, `sk`, `sr`, `sv`, `sw`, `ta`, `te`, `th`, `tl`, `tr`, `uk`, `ur`, `vi`, `yue`, `zh`, or `unknown` to let the planner detect it when `detect_language` is on. |
| `duration` | number | no | Seconds, 10–600. Omitted: the planner chooses it when `plan_metadata` is on, otherwise 120. Ignored for `cover` and `repaint`, which take the source's length. |
| `bpm` | integer | no | 30–300. Omitted: the planner fills it when `plan_metadata` is on. |
| `key` | string | no | Up to 30 characters, upstream's `keyscale`, such as `C major`, `F# minor`, `Bb major`. Omitted: planner. |
| `time_signature` | integer | no | One of `2`, `3`, `4`, `6` (2/4, 3/4, 4/4, 6/8). Omitted: planner. |
| `seed` | integer | no | 0–4294967295. Omitted: a random seed, recorded in the metadata file. A batch uses seed, seed+1, … |
| `batch_size` | integer | no | 1–16, default 1. Songs made from one request. |
| `task` | string | no | `text2music` (default), `cover` or `repaint`. |
| `source_audio` | file | for `cover`, `repaint` | `{"path":"/absolute/path"}`, `audio/*`. The song to cover or repaint, up to 600 s. |
| `reference_audio` | file | no | `{"path":"/absolute/path"}`, `audio/*`. A recording whose timbre the song takes. |
| `repaint_start` | number | no | −600 to 600 s, runner default 0. Negative extends the song before its start. |
| `repaint_end` | number | no | −1 to 1200 s. Omitted or −1: the end of the source; past the end extends it. |
| `repaint_mode` | string | no | `conservative`, `balanced` (runner default) or `aggressive`. |
| `repaint_strength` | number | no | 0–1, runner default 0.5. In balanced mode, 0 is aggressive and 1 conservative. |
| `repaint_crossfade_frames` | integer | no | 0–15000. 25 Hz latent frames blended at the repaint edges; omitted: set by `repaint_mode`. |
| `repaint_wave_crossfade` | number | no | 0–30 s. Waveform crossfade when splicing the repaint back; omitted: set by `repaint_mode`. |
| `chunk_mask_mode` | string | no | `auto` or `explicit`. Omitted: `explicit` for a repaint, `auto` otherwise. |
| `cover_strength` | number | no | 0–1, runner default 1. How long the cover keeps to the source's codes; upstream suggests about 0.2 for style transfer. |
| `cover_noise_strength` | number | no | 0–1, runner default 0. 0 starts from noise, 1 from the source itself. |
| `audio_codes` | array of integer | no | 1–3000 items, each 0–63999. 5 Hz semantic codes to start from instead of the planner's. Advanced. |
| `instruction` | string | no | Up to 10000 characters. The DiT's task line; omitted: the task's own. |
| `thinking` | boolean | no | Default `true`. Let the planner write the audio codes the DiT starts from. |
| `plan_metadata` | boolean | no | Default `true`. Let the planner fill `bpm`, `key`, `time_signature` and `duration` left open. |
| `rewrite_caption` | boolean | no | Default `true`. Let the planner rewrite the prompt into the detailed form the DiT was trained on. |
| `detect_language` | boolean | no | Default `true`. Let the planner name the vocal language. |
| `constrained_decoding` | boolean | no | Default `true`. Hold the planner to the metadata format and the code count. |
| `planner_temperature` | number | no | 0–5, runner default 0.85. |
| `planner_guidance` | number | no | 0–20, runner default 2. Guidance on the planner's codes. |
| `planner_top_k` | integer | no | 0–217088, runner default 0 (off). |
| `planner_top_p` | number | no | 0–1, runner default 0.9; 1 is off. |
| `planner_repetition_penalty` | number | no | 0.01–10, runner default 1 (off). |
| `planner_negative_prompt` | string | no | Up to 10000 characters, runner default `NO USER INPUT`. |
| `steps` | integer | no | 1–200, runner default 8 (the turbo model's distilled count). |
| `shift` | number | no | 1–5, runner default 1. Timestep shift `t' = s·t / (1 + (s−1)·t)`. |
| `timesteps` | array of number | no | 1–20 values, each 0–1, from 1 towards 0. Overrides `steps` and `shift`; each value is snapped to the nearest the turbo model was distilled on. |
| `method` | string | no | `ode` (runner default, deterministic) or `sde` (re-noises each step). |
| `sampler` | string | no | `euler` (runner default) or `heun`. |
| `guidance` | number | no | 0–20, runner default 1. Above 1 applies upstream's APG guidance. |
| `guidance_start` | number | no | 0–1, runner default 0. Fraction of the schedule where guidance starts. |
| `guidance_end` | number | no | 0–1, runner default 1. Where it stops. |
| `velocity_norm_threshold` | number | no | 0–100, runner default 0 (off). Clamps each velocity's norm to this multiple of the latent's. |
| `velocity_ema` | number | no | 0–1, runner default 0 (off). Blends each velocity with the previous one. |
| `dcw` | boolean | no | Default `true`. Differential correction in the Haar wavelet domain. |
| `dcw_mode` | string | no | `low`, `high`, `double` (runner default) or `pix`. |
| `dcw_scaler` | number | no | 0–1, runner default 0.05. Low-band (or single) strength. |
| `dcw_high_scaler` | number | no | 0–1, runner default 0.02. High-band strength in `double` mode. |
| `retake_seed` | integer | no | 0–4294967295. A second noise for variations. |
| `retake_variance` | number | no | 0–1, runner default 0. 0 keeps the seed's noise, 1 is the retake seed's. |
| `latent_shift` | number | no | −10 to 10, runner default 0. Added to the latents before decoding. |
| `latent_rescale` | number | no | 0–10, runner default 1. Multiplied into the latents before decoding. |
| `normalize` | boolean | no | Default `true`. Peak-normalise the result. |
| `normalization_db` | number | no | −40 to 0 dBFS, runner default −1. |
| `fade_in` | number | no | 0–30 s, runner default 0. |
| `fade_out` | number | no | 0–30 s, runner default 0. |
| `format` | string | no | `wav` (default), `wav32`, `flac`, `m4a`, `alac`. |
| `include_metadata` | boolean | no | Default `false`. Adds a JSON receipt per song. |

Returns `files`: per song, one 48 kHz stereo audio file, `audio-0`, `audio-1`, … in batch order. `wav` is 16-bit PCM WAV (`audio/wav`), `wav32` 32-bit float WAV (`audio/wav`), `flac` FLAC (`audio/flac`), `m4a` AAC at 256 kbps in M4A (`audio/mp4`), `alac` Apple Lossless in M4A (`audio/mp4`). With `include_metadata`, each song also gets `metadata-<n>`, a JSON file (`application/json`) holding the resolved request (the seed and the planner's bpm, key, time signature and duration filled in), the planner's plan, the audio codes, and whether the output was truncated. Each file entry carries `id`, `path`, `name`, `mime`, `bytes` and `sha256`.

Source and reference audio can be any file macOS's AVFoundation can decode; Arsenal converts it to 48 kHz stereo, clamps it to ±1 and refuses files over 600 s. A reference shorter than 30 s is repeated; Arsenal then takes three 10 s pieces from its front, middle and back (offsets drawn from the seed), as upstream's `process_reference_audio` does. A cover keeps the source's length; a repaint keeps the source and regenerates `repaint_start`–`repaint_end`, padding with silence when the span reaches outside it. The planner does not run for `cover` and `repaint`, so `thinking`, `plan_metadata`, `rewrite_caption` and `detect_language` have no effect there, matching upstream. A song's length in 5 Hz codes is `duration × 5`, so 3000 `audio_codes` is 600 s.

## Run it

MCP: `arsenal_run(model:"ace-step-1.5", input:{"prompt":"melancholic indie folk with acoustic guitar and soft vocals","lyrics":"[Verse]\n…","language":"en","duration":120})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/ace-step-1.5/ops/generate_music" -H 'Content-Type: application/json' -H 'Prefer: respond-async' -d '{"prompt":"melancholic indie folk with acoustic guitar and soft vocals","lyrics":"[Verse]\n…","language":"en","duration":120}'`

`Prefer: respond-async` returns a job to poll at `GET /arsenal/v1/jobs/{id}` and cancel at `POST /arsenal/v1/jobs/{id}/cancel`; the operation reports progress and allows up to 3600 s. Over MCP a long run returns a job for `arsenal_wait`. Pass `seed` to repeat a song. The model declares no load settings; precision (FP16, int8, int4) is picked in Arsenal's model settings. On an M5 Max with the Mac in use, one warm request on a short fixed test input took 15.7 s at FP16, 12.9 s at int8 and 11.2 s at int4 (quick check, 22 Oct 2026, measured; the test input's duration is not recorded in the folder).

## Limits

- Turbo DiT only: no stem extraction, Lego (adding a track) or completion, which upstream offers on the base model. Adaptive dual guidance (base model) is not implemented.
- The turbo model is CFG-distilled; upstream forces guidance to 1 for it. Arsenal honours `guidance` above 1 with APG, which the makers do not use for turbo.
- Upstream recommends `shift` 3.0 for turbo; Arsenal keeps upstream's parameter default of 1.
- Upstream caps lyrics at 4096 characters and documents turbo steps as 1–20 and batches as 1–8; Arsenal accepts up to 100000 characters, 200 steps and 16 songs. Results past upstream's ranges are not checked.
- Upstream states the model was trained on licensed, royalty-free/public-domain and synthetic MIDI-to-audio music; styles far from that data may come out weaker *(unverified)*.
- Song files count against Arsenal's 4 GiB per-job output limit; a request whose songs could exceed it is refused before generation. Every schema-allowed request fits (16 songs of 600 s, even as `wav32`).
- Output is complete files with progress events, not live PCM.
- Trained-output fidelity of the FP16 conversion and of int8/int4 has not been qualified (see below).

## Licence and credit

MIT (SPDX `MIT`), per the pinned repository and model card. It allows commercial use, modification and redistribution provided the copyright and licence notice are kept. The makers state that generated music may be used commercially. Made by ACE-Step (ACE Studio and StepFun). Citation requested by the makers:

```bibtex
@misc{gong2026acestep,
  title={ACE-Step 1.5: Pushing the Boundaries of Open-Source Music Generation},
  author={Junmin Gong, Yulin Song, Wenxiao Zhao, Sen Wang, Shengyuan Xu, Jing Guo},
  howpublished={\url{https://github.com/ace-step/ACE-Step-1.5}},
  year={2026},
  note={GitHub repository}
}
```

## Links

- Model card: https://huggingface.co/ACE-Step/Ace-Step1.5/tree/19671f406d603126926c1b7e2adc169acbcade22
- Code: https://github.com/ace-step/ACE-Step-1.5
- Parameter reference: https://github.com/ace-step/ACE-Step-1.5/blob/main/docs/en/INFERENCE.md
- Paper: https://arxiv.org/abs/2602.00744
- Project page: https://ace-step.github.io/ace-step-v1.5.github.io/

## How Arsenal runs it

Source: `ACE-Step/Ace-Step1.5@19671f406d603126926c1b7e2adc169acbcade22`, MIT. The manifest pins all 24 selected nested component/config/tokenizer files, including the planner and silence latent. Swift port: `arraypress/swift-music-generation@51034b9874a046d4fe6e092524f8daf0940309ef`, whose reference is `ace-step/ACE-Step-1.5@ca1e85fe9430179831e6bc6be790c332190a3866`. It used the tensor-identical `arraypress/song-ace-step-1.5@7cdff8bafa5893c4b067d396d684993339eee878` mirror. Arsenal uses the original pinned source. The incomplete FP32 `mlx-community/ACE-Step1.5-MLX` snapshot was rejected during source selection. Runner: the native Swift MLX music worker, sequential only, macOS 26 or later.

Pipeline, as upstream's `generate_music` makes it: the planner fills what the caller left open and, when `thinking`, writes duration × 5 audio codes; codes (or a cover's source) become 25 Hz hints through the FSQ detokenizer, and text-to-music with codes runs as a cover of them, as upstream switches it; the prompt goes through Qwen3-Embedding, the lyrics through its embedding table and the timbre reference (or 30 s of the silence latent) through the timbre encoder; the DiT denoises from seeded noise; the VAE encodes source audio in 30 s windows with 2 s overlap and decodes in 512-frame windows with 64 frames discarded each side; a peak above 1 is scaled down, then the result is normalised to `normalization_db` and faded. Source and reference latents are sampled with a seeded key where upstream samples unseeded.

Get's model-owned `cast:ace-step-v1` prepares all nested components into a standalone FP16 snapshot: DiT projection key/layout transforms, Qwen module prefixes, VAE weight-normalization folding, convolution axes and Snake-vector squeezing, plus bounded Torch-ZIP silence conversion to half safetensors. Tokenizers/configs remain alongside their components. No runtime component is loaded from an unconverted external asset. Tiers derive recursively from that canonical snapshot, without applying layouts twice. Download sizes: FP16 10.1 GB, int8 5.4 GB, int4 2.8 GB.

Header-only exact Range reads inspected 1,662 BF16 source tensors across four safetensors files. After normalization folding and layout transforms, all 1,588 retained parameters match the native names/shapes; the serialized silence tensor adds the 1,589th native entry. `gate/component-map.json` records every transform. Silence's source shape is checked at Get by the bounded reader. Source-to-FP16 dtype conversion is explicitly unqualified on trained data.

All 785 eligible group-64 matrices across the DiT, lyric/text encoders and planner are packed for int8/int4: 4,842,038,272 of 5,013,513,478 values (96.58%). Spatial kernels/small tensors/silence remain FP16. Random text-to-music with actual planner codes, cover plus reference audio, repaint, instrumental explicit-code SDE/Heun/DCW, and constrained metadata planning all pass production result validation at FP16/int8/int4. Synthetic original-layout source snapshots test Get conversion, ordinary load/generation and direct stored-tier loading, with an independent nontrivial weight-normalization/layout check. Lifecycle evidence covers invalid input, cancel and unload. `include_score` was removed because ACE-Step plans never fill it (`log/003-documented-controls.md`).

Quick check with the downloaded weights (22 Oct 2026, M5 Max 40-core GPU, Mac in use, one Load then a warm-up and a timed request on a short fixed input, `results/quick-checks/night-2026-10-22.json`): peak memory 15.0 GB at FP16, 9.70 GB at int8, 7.25 GB at int4; load 3.4, 4.1 and 4.0 s; warm request 15.7, 12.9 and 11.2 s. The int8 and int4 songs differ from the FP16 song (audio relative L2 1.26 and 1.31), which is not a quality score. This is not the controlled interleaved benchmark, which is deferred (`bench/bench.toml`).

Gate (`gate/gate.toml`): exact checkpoint names and shapes and finite component outputs pass; trained-output and precision qualification is pending. Trained planner/tokenizer agreement, musical fidelity, FP16 VAE activation range, source-preserving repaint quality, long generation and int8/int4 quality remain open. Live-test risk: **uncertain**, because the pinned Swift closure is adapted to Arsenal's older dependency pins and FP16 execution. Source and third-party notices are retained in `Workers/Music`. Research log: `log/001-blocked.md`, `log/002-integrated.md`, `log/003-documented-controls.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Style prompt, optional lyrics, language, duration, BPM, key, time signature. [ops/generate_music.input.json, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: Turbo runs 8 steps with no guidance (CFG). [model card, undated, first-hand (maker)] Base and SFT with DCW on gave garbled audio; DCW off fixed it for two users. [GitHub issue 1259, first-hand]
- Arsenal call: `arsenal_run(model:"ace-step-1.5", operation:"generate_music", input:{"prompt":"warm acoustic pop","language":"en","duration":30,"bpm":120,"time_signature":4})` [ops/generate_music.input.json, first-hand]

**Good and bad**
- Good at: The card rates turbo "Very High" quality, without saying how it was scored. [model card, undated, first-hand (maker)]
- Bad at: Not found in sources checked beyond the complaint below.
- Against the obvious alternatives: The paper says quality is "beyond most commercial music models" on "commonly used evaluation metrics", naming no model or metric. [arXiv 2602.00744, v1 2026-01-31, first-hand (maker claim)]

**Complaints, bugs and fixes**
- Complaint or known bug: With DCW on, base and SFT give garbled audio on Apple Silicon and other hardware. Turbo was fine. Closed as stale by a bot on 23 Jul; users kept commenting. [GitHub issue 1259, first-hand]
- Fix or workaround: Set dcw_enabled=False explicitly for CLI and API runs. Two users confirmed it; one version also set shift 1.0 and 50 steps. No maintainer comment; PR #1282 by a user would default it off for non-turbo models. [GitHub issue 1259, first-hand (users)]

**Evidence**
- Strength: thin. Maker card and paper, plus one GitHub thread with user A/B tests. Reddit claim dropped: mirror 404, original unreadable.
- Sources:
  - Model card, https://huggingface.co/ACE-Step/Ace-Step1.5, undated, first-hand (maker).
  - arXiv 2602.00744, https://arxiv.org/abs/2602.00744, v1 2026-01-31, first-hand (maker).
  - GitHub issue 1259, https://github.com/ace-step/ACE-Step-1.5/issues/1259, opened 2026-07-01, first-hand (users).
