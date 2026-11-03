<!-- SPDX-License-Identifier: MIT -->
# Stable Audio 3 Medium

Stable Audio 3 Medium is Stability AI's 1.45-billion-parameter text-to-audio model: a diffusion transformer conditioned by a T5Gemma text encoder, decoded to stereo audio by its SAME-L codec. It makes sound effects and instrumental music from an English description.

## What it is for

- Sound effects and ambiences for video, games and prototypes
- Instrumental music beds and loops from a style description

## Inputs and outputs

### `generate_sound` (default)

A sound effect or ambience from a description.

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | What to make, in English; at least one character. |
| `duration` | number | no | Seconds, more than 0 and at most 380; default 5. |
| `steps` | integer | no | Denoising steps, 1 to 100; default 8. |
| `guidance` | number | no | Prompt guidance, 0 to 20; default 1. |
| `seed` | integer | no | 0 to 4294967295; random when left out. |
| `negative_prompt` | string | no | What to steer away from. |
| `out_dir` | string | no | Folder for the result. |
| `stream` | boolean | no | Stream progress; default false. |

Returns a stereo audio file in `files`.

### `generate_music`

Instrumental music from a description.

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | What to make, in English; at least one character. |
| `duration` | number | no | Seconds, more than 0 and at most 380; default 30. |
| `steps` | integer | no | Denoising steps, 1 to 100; default 8. |
| `guidance` | number | no | Prompt guidance, 0 to 20; default 1. |
| `seed` | integer | no | 0 to 4294967295; random when left out. |
| `negative_prompt` | string | no | What to steer away from. |
| `out_dir` | string | no | Folder for the result. |
| `stream` | boolean | no | Stream progress; default false. |

Returns a stereo audio file in `files`.

## Run it

MCP: `arsenal_run(model:"stable-audio-3-medium", input:{"prompt":"rain on a tin roof","duration":5})` (add `operation:"generate_music"` for music).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/stable-audio-3-medium/ops/generate_sound" -H 'Content-Type: application/json' -d '{"prompt":"rain on a tin roof","duration":5}'`

Duration, steps and guidance can also be set as the model's own settings; a request's values win. The Make a Sound and Make Music windows offer the same fields.

## Limits

- Instrumental only: no singing or lyrics.
- English prompts.
- Long durations take proportionally longer and use more memory; the window shows the memory need before loading.

## Licence and credit

Stability AI Community License (`LicenseRef-Stability-AI-Community`), which covers the weights and what you make with them. Research and non-commercial use are free. Commercial use is allowed only after registering with Stability AI and while you and your affiliates earn under US$1M a year; above that an enterprise licence is needed. The T5Gemma text encoder also falls under Google's Gemma Terms of Use. Made by Stability AI.

## Links

- Model card: https://huggingface.co/stabilityai/stable-audio-3-optimized
- Licence: https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/da6edc54ddba10bfd79a077102ded687f80e882b/LICENSE.md

## How Arsenal runs it

The native sound worker (`Workers/Sound`, Swift and MLX) runs the diffusion transformer and the codec at FP16 after Get casts the codec's FP32 source; int8 covers every eligible large component and passed its quality check; int4 did not and is not offered. Port provenance and gate details: `Workers/Sound/PORTED-FROM.md`; the precision history is in `log/`.
