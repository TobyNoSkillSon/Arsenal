# RIFE 4.25

RIFE 4.25 is the September 2024 model of Practical-RIFE by hzwer, a project that makes the RIFE frame interpolation network from the ECCV 2022 paper more practical with new features and models. It is a small optical-flow network (5.7M parameters, about 23 MB of FP32 weights): IFNet runs five coarse-to-fine blocks that estimate the motion from each of two frames to a chosen moment between them, warp both frames there and blend them. It makes the frame that would sit between two real frames, at any position between 0 and 1, fast enough for real-time use.

## What it is for

The paper targets real-time interpolation and reports speeds 4 to 27 times faster than SuperSlomo and DAIN with better results. Practical-RIFE recommends 4.25 as the default model for most scenes; its release note says anime scenes improved markedly, and the maintainer says 4.24 and later suit post-processing some videos made by diffusion models.

- Raising a video's frame rate (24 to 48 or 60 fps, slow motion through a higher factor).
- Converting a clip to a different frame rate by resampling at uniform target timestamps.
- Making one or more in-between images from two stills, such as two photos of a burst or two animation keyframes.
- Smoothing short or low-frame-rate clips from generators before editing.

## Inputs and outputs

### `interpolate_video` (default)

Makes in-between frames from a pair of images, or re-times a whole video at a higher or different frame rate.

| Field | Type | Required | Notes |
|---|---|---|---|
| `frames` | array of file `{"path":"/absolute/path"}` | one of `frames` or `video` | Exactly 2 items (`minItems` 2, `maxItems` 2). `image/png` or `image/jpeg`. Both images must have the same width and height. Cannot be combined with `video`, `factor` or `target_fps`. |
| `t` | array of number | no | Pair mode only. 1 to 16 positions, each from 0 to 1. Default `[0.5]` (the midpoint). 0 returns the first image and 1 the second, without running the network. |
| `video` | file `{"path":"/absolute/path"}` | one of `frames` or `video` | `video/*`, any container and codec AVFoundation can decode (MP4, MOV, M4V). At least two frames; frame size must stay fixed and timestamps must increase. Cannot be combined with `frames` or `t`. |
| `factor` | integer | no | Video mode only. 2 to 16, default 2. Output rate is the source's nominal frame rate times `factor`. Cannot be combined with `target_fps`. |
| `target_fps` | number | no | Video mode only. 1 to 240. Sets the output rate directly; it may also be lower than the source rate. Cannot be combined with `factor`. |

Returns `files`. Pair mode gives one PNG per entry of `t`, in order, ids `frame-0`, `frame-1` and so on, each 8-bit RGB at the input size with `width` and `height` set. Video mode gives one file, id `video`, `interpolated.mp4`: H.264 video in an MP4 container at the source's decoded frame size, carrying the source's rotation (preferred transform). Every file carries `path`, `mime`, `bytes` and `sha256`.

Pair mode: images are decoded with ImageIO, up to 16 megapixels each, drawn into 8-bit device RGB; alpha is composited onto black (transparent areas come out black). EXIF orientation is not applied. The images are not resized.

Video mode: frames are decoded as 8-bit BGRA, one at a time, up to 16 megapixels each; the encoder accepts at most 8192 pixels per side. Output timestamps are `index / fps` from the first source frame. Each output frame is interpolated between the two source frames around it at the matching fraction, so with a constant-rate source and a whole-number factor the original frames pass through unchanged and the network fills only the gaps. Variable-rate sources use their real timestamps; `factor` uses the track's nominal rate, so a source whose nominal rate is unknown needs `target_fps`. The last source frame is held for one source frame interval, so the output keeps the source duration. Audio: every source audio track that overlaps the rendered span is copied onto the output, aligned to the first video frame, by passthrough when MP4 can carry the codec and by re-encoding the whole file otherwise; a source with no audio gives a silent MP4. A failed or cancelled run leaves no partial MP4.

Internally the network works at the input's own resolution: each frame pair is padded on the bottom and right to the next multiple of 64 pixels, run through five flow blocks at scales 1/16, 1/8, 1/4, 1/2 and 1, and cropped back. Pixels are scaled to 0–1 and run at FP16; output is rounded to 8 bits.

## Run it

MCP: `arsenal_run(model:"rife-4.25", input:{"video":{"path":"/absolute/path/clip.mp4"},"factor":2})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/rife-4.25/ops/interpolate_video" -H 'Content-Type: application/json' -d '{"frames":[{"path":"/absolute/path/a.png"},{"path":"/absolute/path/b.png"}],"t":[0.25,0.5,0.75]}'`

The model has no settings. Measured on an M5 Max at FP16 during a quick check on a busy Mac: one warm request on a 2-second 640×360 clip took 0.90 s including the API round trip, with 0.61 GB peak worker memory. Video runs report progress in frames.

## Limits

- Large motion fails: in a live FP16 check on 19 October a 24-pixel pan was interpolated, and a 300-pixel jump of a blank square failed.
- No scene-cut handling. Upstream's video script duplicates frames when two neighbours are dissimilar (SSIM below 0.2); Arsenal always interpolates, so frames across a cut come out as blends.
- Arsenal runs the pyramid at scale 1.0. Upstream recommends `--scale=0.5` for 4K input, and Arsenal does not expose that option, so very high-resolution or fast-moving footage may interpolate worse than upstream's recommended setting *(unverified)*.
- Video output is 8-bit H.264 and the writer sets no colour properties; 10-bit or HDR sources lose their extra range *(unverified for how AVFoundation maps HDR to 8-bit)*.
- `factor` times the source rate must stay within 240 fps, otherwise the request fails.
- Pair mode takes PNG and JPEG only; image sequences, more than two frames and transparency are not supported.
- Output video files count against the 4 GiB per-job artifact limit.

## Licence and credit

MIT for the Practical-RIFE code and its model files, and for the MLX conversion and the Swift port: free to use, change and sell, including commercially, provided the copyright and licence notice are kept. Upstream: Practical-RIFE, © hzwer, built on RIFE and SAFA. The weights are the official 4.25 release converted to MLX safetensors. Practical-RIFE asks for these citations:

```bibtex
@inproceedings{huang2022rife,
  title={Real-Time Intermediate Flow Estimation for Video Frame Interpolation},
  author={Huang, Zhewei and Zhang, Tianyuan and Heng, Wen and Shi, Boxin and Zhou, Shuchang},
  booktitle={Proceedings of the European Conference on Computer Vision (ECCV)},
  year={2022}
}

@inproceedings{huang2024safa,
  title={Scale-Adaptive Feature Aggregation for Efficient Space-Time Video Super-Resolution},
  author={Huang, Zhewei and Huang, Ailin and Hu, Xiaotao and Hu, Chen and Xu, Jun and Zhou, Shuchang},
  booktitle={Winter Conference on Applications of Computer Vision (WACV)},
  year={2024}
}
```

## Links

- Model card: https://huggingface.co/mlx-community/RIFE-4.25/tree/6b650eaa5664aab280119511fe944861491aa15e
- Code: https://github.com/hzwer/Practical-RIFE
- Swift port: https://github.com/xocialize/rife-mlx-swift
- Paper: https://arxiv.org/abs/2011.06294

## How Arsenal runs it

Checkpoint: `mlx-community/RIFE-4.25@6b650eaa5664aab280119511fe944861491aa15e`. Port: `xocialize/rife-mlx-swift@f926ca03a910f53a34cd025f03c0ae20e24d63a9`, MIT. Exact source files, sizes and hashes are in `model.toml`; source provenance is in `Workers/Restoration/PORTED-FROM.md`. The validated `architecture = "rife"` selector handles the checkpoint's missing model_type without modifying the source config. The native Swift runner in the Restoration worker runs requests one at a time.

The source is FP32. Get streams every tensor through `cast:restoration-rife` into FP16; the model folder and header map explicitly declare that cast. This convolution-only architecture has no Linear/Embedding matrices to quantize. The interpolation weights and timestep stay in the activation dtype.

Header-only Range reads matched all 158 names/shapes with 158 explicit FP32→FP16 Get casts. The real worker passed pair interpolation at 0, 0.25, 0.5 and 1 and video target-FPS conversion at every declared tier, including production artifact validation and clean unload. Earlier factor-2 tests also produced twelve frames at 48 fps from a six-frame 24 fps clip, while 60 fps produced fifteen frames; both retained the 0.25-second duration *(unverified: no log or test in this folder records it)*. Malformed inputs and cancellation after actual progress were checked; partial MP4 output was removed *(unverified)*. Source audio is kept through `MediaBridge.keepAudio`, covered by `AudioMuxTests`.

These are graph, layout, transport and precision-policy checks without trained weights. The upstream card calls coarse-to-fine flow FP16-sensitive: compare trained FP16 frames with an independent reference before qualifying output quality. The quality gate (`gate/gate.toml`) is pending that trained-output qualification, and the benchmark suite is deferred. Trained weights were later checked live at FP16 on 19 October: a 24-pixel pan was interpolated; a 300-pixel jump of a blank square failed. The 22 October quick check (`results/quick-m5-max-fp16-native-exact.json`) loaded in 0.15 s, answered warm in 0.90 s and peaked at 0.61 GB.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster. Research logs: `log/001-blocked.md`, `log/002-admitted.md`, `log/003-keep-audio.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: the upstream CLI sets the output rate with `--multi` (a frame-rate multiplier) or `--fps` (output FPS). [Practical-RIFE README, 2024-09-19, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable; an interpolator.
- Parameters people recommend: The maker recommends 4.25 "by default for most scenes". [Practical-RIFE README, 2024-09-19, first-hand]
- Arsenal call: `arsenal_run(model:"rife-4.25", input:{"video":{"path":"/absolute/path/clip24.mp4"},"target_fps":48})`

**Good and bad**
- Good at: Anime. The README's 4.25 entry says anime scenes "have been significantly improved". [Practical-RIFE README, 2024-09-19, first-hand]
- Bad at: Artifacts that persist in predictions. Issue #34 (opened 2023-09-29) got a 2026-10-20 reply from the maintainer, labelled AI-assisted, saying "Changing scene thresholds or target FPS can hide an artifact without repairing the underlying prediction." [hzwer/Practical-RIFE issue #34, reply 2026-10-20, first-hand (maintainer); reporter opinion]
- Against the obvious alternatives: In SVP forum tests (Oct 2024, Lite builds, not this checkpoint), one tester found 4.26 had many artifacts; two others found 4.25 Lite no better than 4.26. [SVP forum, 2024-10-21, opinion]

**Complaints, bugs and fixes**
- Complaint or known bug: In SVP tests with Lite builds, some output resize settings showed artifacts at both 4.25 Lite and 4.26. [SVP forum, 2024-10-21, opinion]
- Fix or workaround: Reported by one user, not confirmed: at another resize setting the same tester saw artifacts only at 4.26. [SVP forum, 2024-10-21, opinion]

**Evidence**
- Strength: thin. Maker docs and one maintainer reply; forum tests cover SVP Lite builds, not this checkpoint.
- Sources:
  - Practical-RIFE README and release notes, https://github.com/hzwer/Practical-RIFE, 2024-09-19, first-hand (maker)
  - mlx-community/RIFE-4.25 model card, https://huggingface.co/mlx-community/RIFE-4.25, undated, first-hand (port maker)
  - hzwer/Practical-RIFE issue #34, https://github.com/hzwer/Practical-RIFE/issues/34, opened 2023-09-29, maintainer reply 2026-10-20; reporter opinion, maintainer first-hand
  - SVP forum thread, https://www.svp-team.com/forum/viewtopic.php?pid=85829, 2024-10-18 to 2024-10-31, opinion
