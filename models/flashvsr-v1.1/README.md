# FlashVSR v1.1

FlashVSR is OpenImagingLab's one-step streaming diffusion model for video super-resolution (arXiv 2510.12747). It has 1.75 billion parameters: a Wan2.1-1.3B-shaped diffusion transformer distilled to a single step, a causal projector that feeds it the low-resolution frames, and a small decoder conditioned on those frames. Its attention is locality-constrained and block-sparse, and it carries a key/value cache from chunk to chunk, so a whole clip runs as one stream. Arsenal runs the MLX-Swift port by xocialize on the mlx-community conversion of the v1.1 weights, FP16 by default.

## What it is for

The makers built it to make diffusion video super-resolution fast enough for long, high-resolution footage: one step instead of many, and streaming instead of fixed windows.

- Enlarging live-action footage two to four times with photographic detail
- Long clips: memory depends on the output frame size, not on the clip's length
- A sharper version of compressed or soft live-action video at a larger size

## Inputs and outputs

### `upscale_video` (default)

Enlarge a video clip, keeping its audio.

| Field | Type | Required | Notes |
|---|---|---|---|
| `video` | file `{"path":"/absolute/path"}` | yes | `video/*`, any file AVFoundation can decode; path 1 to 4096 characters |
| `scale` | number | no | 1 to 4, default 2; output height is source height times `scale` (4 is the model's native factor) |
| `target_height` | integer | no | 2 to 8192; replaces `scale` when given, and must be one to four times the source height |
| `seed` | integer | no | 0 to 4294967295, default 0; one seed serves the whole clip, so a run is repeatable |
| `color_correction` | boolean | no | default true; matches each output frame's colour statistics to the enlarged source frame (the makers' AdaIN fix) |

Returns `files`: one MP4 (`id` `video`, `mime` `video/mp4`, name `upscaled.mp4`) with `path`, `bytes` and `sha256`. Video is H.264, one output frame per source frame, at the source presentation timestamps shifted so the first frame is at zero, with the source's preferred orientation. Every source audio track that overlaps the video is kept and aligned to the first video frame.

Output height is `target_height`, or source height times `scale` rounded; width follows the source aspect ratio; both are floored to even pixels. Arsenal reads each frame as 8-bit RGB, enlarges it with PIL-style bicubic resampling (the input the model was trained on), pads right and bottom by repeating edge pixels to a multiple of 128, runs the stream and crops the padding away. The makers' example centre-crops to a multiple of 128 instead; padding keeps every pixel, and the colour correction's statistics include the padded edge. The padded output frame may be at most 2048 × 2048 pixels' worth (about 4.2 megapixels); larger requests fail rather than shrink. The first output frames arrive after 25 source frames; after that every 8 source frames yield 8 output frames. The final chunk repeats the last frame and drops the repeats. There is no text prompt: the checkpoint's fixed prompt embedding is always used.

## Run it

MCP: `arsenal_run(model:"flashvsr-v1.1", input:{"video":{"path":"/absolute/clip.mov"},"scale":2})`.

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/flashvsr-v1.1/ops/upscale_video" -H 'Content-Type: application/json' -H 'Prefer: respond-async' -d '{"video":{"path":"/absolute/clip.mov"},"target_height":1080}'`, then poll `GET $ARSENAL_URL/arsenal/v1/jobs/<id>`.

The operation has a one-hour timeout. Speed and memory are in `results/m5-max.md`.

## Limits

- It is generative: it invents plausible detail at the output size rather than recovering it. Do not use it where exact detail matters.
- Recommended for live action only. On anime, cartoons, motion graphics, interface captures and text it paints flat colour and line art as photographic texture.
- It turns soft focus and bokeh into crisp invented texture.
- It enlarges; it does not remove scratches, dust or flicker from old film.
- At most about 4.2 megapixels per padded output frame (1080p doubled to 4K is refused); output files at most 4 GiB.
- FP16 is checked against the BF16 release on fixed inputs only; the makers' PyTorch code has not been run as a reference here.

## Licence and credit

Apache-2.0 for the code and the weights: commercial use, modification and redistribution are allowed with the licence and notices kept and changes stated. The weights are OpenImagingLab's FlashVSR v1.1 release (`JunhaoZhuang/FlashVSR-v1.1`), converted to MLX layout by xocialize and published by mlx-community. The model builds on Wan2.1 (Apache-2.0) and DiffSynth-Studio (Apache-2.0); its decoder derives from TAEHV (MIT, Ollin Boer Bohan); its prompt embedding comes from umT5 (Apache-2.0). Credit OpenImagingLab and cite the paper:

```bibtex
@article{zhuang2025flashvsr,
  title={FlashVSR: Towards Real-Time Diffusion-Based Streaming Video Super-Resolution},
  author={Zhuang, Junhao and Guo, Shi and Cai, Xin and Li, Xiaohui and Liu, Yihao and Yuan, Chun and Xue, Tianfan},
  journal={arXiv preprint arXiv:2510.12747},
  year={2025}
}
```

## Links

- Model card: https://huggingface.co/mlx-community/FlashVSR-v1.1-bf16/tree/1466bf43ff8010ba4184d21084f121d1671854dc
- Original weights: https://huggingface.co/JunhaoZhuang/FlashVSR-v1.1
- Code: https://github.com/OpenImagingLab/FlashVSR
- Swift port: https://github.com/xocialize/mlx-flashvsr-swift
- Paper: https://arxiv.org/abs/2510.12747

## How Arsenal runs it

Checkpoint: `mlx-community/FlashVSR-v1.1-bf16@1466bf43ff8010ba4184d21084f121d1671854dc`, converted from `JunhaoZhuang/FlashVSR-v1.1@27561b18`. All five files, SHA256s and exact byte counts are in `model.toml`. Get casts the four BF16 tensor files to FP16 in place (`cast:restoration-flashvsr`); nothing is quantized and no int8 or int4 tier is offered. Port: `xocialize/mlx-flashvsr-swift@ec67024b`, its engine-agnostic `FlashVSRMLX` core only; the source closure and changes are in `Workers/Restoration/PORTED-FROM.md`, and its Apache-2.0 licence and notice in `Workers/Restoration/LICENSE-upstream-FlashVSRNative`. The block-sparse attention runs as the port's Metal kernel in place of the makers' CUDA kernel. The native worker uses MLX Swift for all model execution, with AVFoundation video I/O and the source soundtrack kept by `MediaBridge.keepAudio`, shared with SeedVR2 and RIFE.

The whole clip is one stream with one seed: the transformer's key/value cache, the projector's causal state and the decoder's memory carry from the first frame to the last, as the makers' streaming pipeline does, so there are no independent chunks to seam. Noise for each latent frame comes from its own key derived from the seed. During a run MLX's freed-buffer cache is held to 2 GiB.

Jobs: long operations use the universal endpoint with `Prefer: respond-async`; the host supplies durable jobs, polling and cancellation. The adapter checks cancellation between source frames and output frames; a chunk completes before cancellation is checked again, and the first chunk (21 output frames) is the longest, about 30 s at 1280×960 on an M5 Max. The runner is `native` (Swift, restoration worker, sequential only, macOS 26 or later).

Verification: `FlashVSRTests.fp16MatchesBF16Release` runs each FP16 component on the BF16 run's own inputs (25 real frames, first chunk) with bounds fixed before the run; `results/m5-max.md` holds its numbers and the end-to-end smoke. The port's author reports component parity with the makers' PyTorch code on the CPU at FP32; that reference has not been rerun here, so the gate (`gate/gate.toml`) is `pending-reference`.
