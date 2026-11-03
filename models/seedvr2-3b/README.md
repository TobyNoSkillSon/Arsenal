# SeedVR2 3B

SeedVR2 is ByteDance Seed's one-step diffusion model for video and image restoration (ICLR 2026). It is a 3-billion-parameter diffusion transformer with adaptive window attention, paired with a causal 3D-convolution video VAE and a fixed precomputed text embedding. It was made by adversarial post-training against real data so that a single diffusion step restores detail that multi-step diffusion restorers need many steps for. Arsenal runs the MLX-Swift conversion from mlx-community, FP16 by default.

## What it is for

The makers built it for real-world video restoration and super-resolution at high resolution in one step, aiming to match or beat multi-step video restoration methods at a fraction of the inference cost.

- Enlarging low-resolution or compressed photos while restoring texture and edges
- Upscaling and cleaning degraded video clips, keeping their soundtrack
- Restoring old or noisy footage to a sharper version at the same size (`scale` 1)

## Inputs and outputs

### `upscale_image` (default)

Restore and enlarge one image.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file `{"path":"/absolute/path"}` | yes | `image/png` or `image/jpeg`; path 1 to 4096 characters |
| `scale` | number | no | 1 to 8, default 2; output height is source height times `scale` |
| `target_height` | integer | no | 2 to 8192; when given it replaces `scale`, and width follows the source aspect ratio |
| `seed` | integer | no | 0 to 4294967295; omitted means a fresh random seed each run |
| `softness` | number | no | 0 to 1, default 0; blurs the enlarged input before restoration |
| `color_correction` | boolean | no | default true; transfers the source's LAB colour base onto the result |

Returns `files`: one PNG (`id` `image`, `mime` `image/png`, name `upscaled.png`) with `path`, `bytes`, `sha256`, `width` and `height`. The PNG is 8-bit RGB with an opaque alpha channel.

### `upscale_video`

Restore and enlarge a video clip, keeping its audio.

| Field | Type | Required | Notes |
|---|---|---|---|
| `video` | file `{"path":"/absolute/path"}` | yes | `video/*`, any file AVFoundation can decode; path 1 to 4096 characters |
| `scale` | number | no | 1 to 8, default 2 |
| `target_height` | integer | no | 2 to 8192; replaces `scale` when given |
| `seed` | integer | no | 0 to 4294967295; omitted means a fresh random seed; one seed serves the whole clip |
| `softness` | number | no | 0 to 1, default 0 |
| `color_correction` | boolean | no | default true |

Returns `files`: one MP4 (`id` `video`, `mime` `video/mp4`, name `upscaled.mp4`) with `path`, `bytes` and `sha256`. Video is H.264, one output frame per source frame, at the source presentation timestamps shifted so the first frame is at zero, with the source's preferred orientation. Every source audio track that overlaps the video is kept and aligned to the first video frame: copied unchanged when MP4 can carry its codec, otherwise the file is re-exported at the highest-quality preset.

The source image or each video frame may be at most 16 megapixels, and the output frame at most 16 megapixels and 8192 pixels per side; larger requests fail rather than shrink. Output height is `target_height`, or source height times `scale` rounded; width is source width times that height ratio; both are floored to even pixels. Arsenal decodes the input to 8-bit RGB (transparency is flattened onto black), enlarges it to the output size with Lanczos, applies `softness` as a Lanczos downscale by a factor of 1 + 7 × `softness` and back up, pads right and bottom with black to a multiple of 16, restores at that size in one diffusion step (timestep 1000), crops the padding and applies colour correction. The model therefore works at the output resolution, not a fixed internal size; latents are one eighth of the padded size. Video is read frame by frame and restored in chunks of 4n+1 frames: the largest of 21, 17, 13, 9 or 5 frames for which frames × output pixels stays within 21 × 1280 × 720, and never fewer than five (21 at 720p output, five at 4K); the last chunk repeats its final frame to reach 4n+1, and the repeats are dropped. The VAE encodes and decodes each chunk in overlapping spatial tiles (1,408 px for an image, 640 px for five frames, 256 px for 21) blended with linear ramps. There is no text prompt: the checkpoint's fixed positive embedding is always used.

## Run it

MCP: `arsenal_run(model:"seedvr2-3b", input:{"image":{"path":"/absolute/photo.jpg"},"scale":2})`, and `arsenal_run(model:"seedvr2-3b", operation:"upscale_video", input:{"video":{"path":"/absolute/clip.mov"},"target_height":1080})`.

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/seedvr2-3b/ops/upscale_image" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/photo.jpg"},"scale":2}'`, and for video, which can run long, `curl -s "$ARSENAL_URL/arsenal/v1/models/seedvr2-3b/ops/upscale_video" -H 'Content-Type: application/json' -H 'Prefer: respond-async' -d '{"video":{"path":"/absolute/clip.mov"},"target_height":1080}'`, then poll `GET $ARSENAL_URL/arsenal/v1/jobs/<id>`.

There is no step count or prompt to set; `seed` makes a run repeatable. Each operation has a one-hour timeout. No run time has been measured.

## Limits

- Generative restoration invents plausible detail; it is not a faithful enlargement and should not be used where exact detail matters.
- The makers report it is sometimes not robust to heavy degradation and very large motion, and tends to over-generate detail on lightly degraded inputs.
- 16 megapixels per input and output frame, at most 8192 pixels per side; output files at most 4 GiB.
- Video chunks are separate diffusion passes sharing one noise draw; upstream restores a whole clip in one pass, so seams between chunks are possible and have not been checked on trained weights.
- VAE tiling is an Arsenal memory bound that upstream does not use; tile seams have not been checked on trained weights.
- Transparency is not kept; the output is opaque.
- FP16 output quality on trained weights and the int8 and int4 tiers are not yet qualified; the Swift port's author reports that int4 degrades this model.

## Licence and credit

Apache-2.0 for the model and weights: commercial use, modification and redistribution are allowed with the licence and notices kept and changes stated. The mlx-community weights are a format conversion of ByteDance Seed's release (via `numz/SeedVR2_comfyUI` and `filipstrand/mflux`), converted and ported to MLX-Swift by MVS Collective (xocialize); the port's code is MIT. Credit ByteDance Seed and cite the paper:

```bibtex
@inproceedings{wang2025seedvr2,
  title={SeedVR2: One-Step Video Restoration via Diffusion Adversarial Post-Training},
  author={Wang, Jianyi and Lin, Shanchuan and Lin, Zhijie and Ren, Yuxi and Wei, Meng and Yue, Zongsheng and Zhou, Shangchen and Chen, Hao and Zhao, Yang and Yang, Ceyuan and Xiao, Xuefeng and Loy, Chen Change and Jiang, Lu},
  booktitle={ICLR},
  year={2026}
}
```

## Links

- Model card: https://huggingface.co/mlx-community/SeedVR2-3B-mlx/tree/46f851fe23b2a9fbefee6419b94e4fdf07b92965
- Code: https://github.com/ByteDance-Seed/SeedVR
- Swift port: https://github.com/xocialize/seedvr2-mlx-swift
- Paper: https://arxiv.org/abs/2506.05301

## How Arsenal runs it

Checkpoint: `mlx-community/SeedVR2-3B-mlx@46f851fe23b2a9fbefee6419b94e4fdf07b92965`. All four files, SHA256s and exact byte counts are in `model.toml`. Transformer, VAE and positive embedding are FP16. Port: `xocialize/seedvr2-mlx-swift@1dfa161ba9522bdf1295015fd0b00209a4e30093`; the source closure and adaptations are listed in `Workers/Restoration/PORTED-FROM.md`. The port declares MIT in its headers and README, but its pinned repository has no standalone LICENSE file; the available notice is preserved in `Workers/Restoration/LICENSE-upstream-SeedVR2Native`. The native worker uses MLX Swift throughout model execution, with AVFoundation video I/O and CoreImage Lanczos image preparation. No upstream download path is imported.

Local adaptations: upstream restores a whole clip in one diffusion pass; Arsenal bounds memory by restoring 4n+1-frame chunks (21 source frames at 720p output, shortened in steps of four for larger frames, down to five at 4K), repeating the final frame to complete the last chunk as upstream `cut_videos` does. One noise draw of shape [1, 16, latent frames, H/8, W/8] for the longest chunk serves the whole clip, so every chunk starts from the same noise; shorter chunks take a prefix. `Upsample3D` applies upstream `remove_head` for every temporal length (the import kept only the one-latent-frame case, so five frames decoded to eight); group norms are per frame like upstream `causal_norm_wrapper`; `SeedVR2VAE.tiled` encodes and decodes overlapping spatial tiles blended with linear ramps so activation memory follows the tile rather than the frame (about 2,097,152 pixel-frames per tile, near 8 GiB of activations, derived, not measured). VAE activation casts and the time embedding use FP16; this adaptation is unqualified on trained weights. Colour correction is the port's wavelet/histogram LAB transfer. Video output uses `MediaBridge.keepAudio`, shared with RIFE, after the video writer finishes; `AudioMuxTests` covers it.

Jobs: long operations use the universal endpoint with `Prefer: respond-async`; the host supplies durable jobs, polling and cancellation. File inputs take any absolute path on this Mac; `arsenal_upload` is only for files that are not on this Mac. HTTP, MCP and tool-window fields come from the local operation schemas. The adapter checks cancellation between preparation, encoding, the one diffusion step, decoding, temporal batches and frame writes; an individual MLX kernel evaluation completes before cancellation is checked again. The runner is `native` (Swift, restoration worker, sequential only, macOS 26 or later).

Precision: FP16 is the default. `native:seedvr2` derives group-64 affine int8 and int4 for all eligible Linear matrices in both transformer and VAE; spatial convolutions, normalization vectors, the 132-input patch projection and the fixed positive embedding remain FP16. The 267 eligible matrices contain 3,969,155,072 of the checkpoint's 4,219,200,387 values (94.07%, derived from headers). No component has a claimed quality exception; int4 is unqualified. The upstream port retained VAE attention at half precision *(unverified)*, but the task requires testing every eligible component; these recipes include its eight eligible VAE matrices. Trained-weight checks must decide whether an int8 component needs a recorded FP16 fallback.

Verification: header-only Range inspection matched all 996 parameter names, shapes and dtypes against the full graph; no tensor payload was read. `gate/header-map.json` records the transformer, VAE and embedding namespaces. `gate/no-weights.json` exercises both operations through the real family worker using the same graph with two transformer blocks, 128 hidden dimensions, one 128-dimensional attention head and four reduced VAE blocks. Random smokes establish execution, transport, dimensions and precision wiring; they do not establish trained output or the quality of the FP16 VAE activation adaptation. The gate (`gate/gate.toml`, `seedvr2-3b-native-v1`) is `pending-trained-weights`; the bench (`bench/bench.toml`, seconds per output and peak RAM) is deferred.

Remaining qualification, after weights are approved: full-resolution FP16 component values against the independent upstream reference, temporal chunk boundaries, orientation and timing on real clips, cancellation under realistic load, and trained int8/int4 component and output quality. No speed or peak-memory figure is claimed.

Logs: `log/001-native.md` (source closure, adapter, header inventory; inconclusive pending trained weights), `log/002-keep-audio.md` (source soundtrack kept on upscaled video).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: the upstream inference example is for video; it does not say whether images are supported. [ByteDance-Seed/SeedVR README, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable; no prompt.
- Parameters people recommend: Not found in sources checked. The MLX card's Swift example uses `seed: 42`. [mlx-community/SeedVR2-3B-mlx model card, undated, first-hand]
- Arsenal call: `arsenal_run(model:"seedvr2-3b", operation:"upscale_image", input:{"image":{"path":"/absolute/path/old-photo.jpg"},"scale":2,"seed":42})`

**Good and bad**
- Good at: Not found beyond the maker's claims.
- Bad at: Lightly degraded and small inputs; the maker says it can over-generate detail and oversharpen, especially at 480p. It is also sometimes not robust to heavy degradation or very large motion. [ByteDance-Seed/SeedVR README, undated, first-hand]
- Against the obvious alternatives: ComfyUI docs say the 7B gives higher quality at more VRAM and time; no 3B-versus-7B test found. [ComfyUI docs, SeedVR2 tutorial, undated, first-hand (integrator)]

**Complaints, bugs and fixes**
- Complaint or known bug: In ComfyUI's tiling node (not this port), a guide reports visible seams and conflicting mask_blur advice (0, or 1 to 2). [RunComfy SeedVR2 Tiling Upscaler guide, updated 2025-12-06, opinion]
- Fix or workaround: Suggested, not confirmed: tile_padding 64 px or more; mask_blur 1 or 2. [RunComfy guide, updated 2025-12-06, opinion]
- Complaint or known bug: numz ComfyUI node issue "Runs out of memory on 128Gb Mac after processing..." (2026-06-19). Fix or workaround: none shown. [numz/ComfyUI-SeedVR2_VideoUpscaler issue list, 2026-06-19, opinion (title only)]

**Evidence**
- Strength: thin; maker and integrator documents only.
- Sources:
  - ByteDance-Seed/SeedVR, https://github.com/ByteDance-Seed/SeedVR, undated, first-hand (maker)
  - mlx-community/SeedVR2-3B-mlx, https://huggingface.co/mlx-community/SeedVR2-3B-mlx, undated, first-hand (port maker)
  - ComfyUI docs, SeedVR2, https://docs.comfy.org/tutorials/utility/seedvr2, undated, first-hand (integrator)
  - numz/ComfyUI-SeedVR2_VideoUpscaler issues, https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler/issues, 2026-06-19 (title only), opinion
  - RunComfy SeedVR2 Tiling Upscaler guide, https://www.runcomfy.com/comfyui-nodes/comfyui-seedvr2-tilingupscaler, updated 2025-12-06, opinion
