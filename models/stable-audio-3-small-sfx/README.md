<!-- SPDX-License-Identifier: MIT -->
# Stable Audio 3 Small SFX

Stable Audio 3 Small SFX is Stability AI's 459-million-parameter text-to-audio model tuned for sound effects: a diffusion transformer with a T5Gemma text encoder and the SAME-S codec, making stereo effects from an English description, faster and smaller than Medium.

## What it is for

- Short sound effects and foley for video, games and interfaces
- Quick iterations on an effect before a longer render with Medium

## Inputs and outputs

### `generate_sound` (default)

A sound effect or ambience from a description.

| Field | Type | Required | Notes |
|---|---|---|---|
| `prompt` | string | yes | What to make, in English; at least one character. |
| `duration` | number | no | Seconds, more than 0 and at most 120; default 5. |
| `steps` | integer | no | Denoising steps, 1 to 100; default 8. |
| `guidance` | number | no | Prompt guidance, 0 to 20; default 1. |
| `seed` | integer | no | 0 to 4294967295; random when left out. |
| `negative_prompt` | string | no | What to steer away from. |
| `out_dir` | string | no | Folder for the result. |
| `stream` | boolean | no | Stream progress; default false. |

Returns a stereo audio file in `files`.

## Run it

MCP: `arsenal_run(model:"stable-audio-3-small-sfx", input:{"prompt":"rain on a tin roof","duration":5})`.

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/stable-audio-3-small-sfx/ops/generate_sound" -H 'Content-Type: application/json' -d '{"prompt":"rain on a tin roof","duration":5}'`

Duration, steps and guidance can also be set as the model's own settings; a request's values win. The Make a Sound window offers the same fields.

## Limits

- Sound effects only; use Stable Audio 3 Medium or ACE-Step for music.
- English prompts.
- At most 120 seconds per request.

## Licence and credit

Stability AI Community License (`LicenseRef-Stability-AI-Community`), which covers the weights and what you make with them. Research and non-commercial use are free. Commercial use is allowed only after registering with Stability AI and while you and your affiliates earn under US$1M a year; above that an enterprise licence is needed. The T5Gemma text encoder also falls under Google's Gemma Terms of Use. Made by Stability AI.

## Links

- Model card: https://huggingface.co/stabilityai/stable-audio-3-optimized
- Licence: https://huggingface.co/stabilityai/stable-audio-3-optimized/blob/da6edc54ddba10bfd79a077102ded687f80e882b/LICENSE.md

## How Arsenal runs it

The native sound worker (`Workers/Sound`, Swift and MLX) runs every component at FP16 after Get casts the FP32 sources; int8 passed its component quality check; int4 did not and is not offered. Port provenance and gate details: `Workers/Sound/PORTED-FROM.md`; the precision history is in `log/`.
