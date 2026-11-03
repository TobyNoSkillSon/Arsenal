# MoGe-3 ViT-L

MoGe-3 ViT-L is a monocular geometry model from Microsoft Research's MoGe project. A DINOv2 ViT-Large encoder feeds heads for a point map, a validity mask, surface normals and a metric scale, and MoGe-3 adds a self-guided sparse volumetric refiner that sharpens fine geometry in a few steps. It has 370M parameters. From one photo it gives metric depth, a metric 3D point map in camera coordinates, normals and the camera's intrinsics in one pass.

## What it is for

The makers optimised MoGe-3 for fine-detail, metric-scale 3D geometry from a single open-domain image, at any aspect ratio from 2:1 to 1:2, with an optional known field of view to improve accuracy.

- Measure approximate distances in a single photo, in metres.
- Turn one photo into a camera-space point cloud or mesh source.
- Get surface normals for relighting, compositing or plane detection.
- Estimate a photo's focal length or field of view when the camera is unknown.
- Separate pixels with valid geometry from those without, such as sky *(unverified)*, using the mask.

## Inputs and outputs

### `estimate_depth` (default)

Predicts metric depth for one image and, on request, a point map, normals and camera intrinsics.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | `{"path":"/absolute/path"}`; accepts `image/png`, `image/jpeg`; path 1–4096 characters; at most 16 megapixels (width × height ≤ 16,000,000). |
| `mode` | string | no | Only `metric`. |
| `points` | boolean | no | Adds the metric point map tensor to the safetensors file. |
| `normals` | boolean | no | Adds the normal map tensor. |
| `intrinsics` | boolean | no | Adds the 3×3 normalised camera matrix to `views`. |
| `focal_length` | number | no | 0.001–100000. Known horizontal focal length in pixels of the original image. Arsenal converts it to the horizontal field of view 2·atan(width / (2 · focal_length)) and passes that to the model, which then solves only for the depth shift. Leave it out to let the model estimate focal length. |
| `resolution_level` | integer | no | 0–9, default 9. Selects the token count inside the checkpoint's `num_tokens_range` (1200–3600): tokens = 1200 + level/9 × 2400. Higher captures finer detail and runs slower; output size does not change. |
| `refine_steps` | integer | no | 0–16, default 3. Sparse refinement steps; 0 returns the unrefined prediction. |
| `apply_mask` | boolean | no | Default true. When true, pixels outside the predicted mask get +infinity in `depth` and `points` and zero normals. |

Returns `files`, `mode` (`metric`), `depth_unit` (`meters`) and `views` (one entry, `index` 0). Two files:

- `view-0.safetensors` (id `view-0`): tensors at the image's own size, H × W (after the 4096 px cap below). `depth` (H × W, FP16, metres), `mask` (H × W, uint8, 1 = valid geometry, 0 = invalid or non-positive depth), with `points:true` `points` (H × W × 3, FP16, metres) and with `normals:true` `normals` (H × W × 3, FP16, unit vectors). Points and normals are in OpenCV camera coordinates: x right, y down, z forward.
- `view-0-depth.png` (id `view-0-depth`): a 16-bit greyscale preview of `depth`, stretched from the smallest to the largest positive finite depth in this image (near is dark, far is bright); masked pixels are 0 (black). For viewing only.

The `views` entry has `file_ids`, `image_size` [width, height] of the original image and `depth_size` [width, height] of the maps. With `points:true` it has `point_frame` `camera_x_right_y_down_z_forward`. With `intrinsics:true` it has `intrinsics` (3×3, row-major) and `intrinsics_unit` `normalized`: fx is divided by the image width, fy by the image height, and the principal point is fixed at (0.5, 0.5). Multiply the first row by the width in pixels and the second row by the height to get a pixel camera matrix. Horizontal field of view is 2·atan(1 / (2·fx)).

Arsenal reads the image as 8-bit RGB scaled to [0, 1]. As in the makers' demo (`moge/scripts/app.py`), an image whose larger side exceeds 4096 px is area-downscaled to 4096 px on that side before inference, and its maps come back at that size; smaller images keep full-resolution maps. The encoder then resizes internally to the token grid with antialiased bilinear filtering and ImageNet normalisation; this does not change the output size. EXIF orientation is not applied: the image is read in its stored pixel orientation, so rotate a camera JPEG first if its orientation tag matters.

## Run it

MCP: `arsenal_run(model:"moge-3-vitl", input:{"image":{"path":"/absolute/path/photo.jpg"}, "points":true, "normals":true, "intrinsics":true})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/moge-3-vitl/ops/estimate_depth" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/photo.jpg"},"points":true}'`

There are no settings; `resolution_level`, `refine_steps` and `focal_length` go in each request. One 1024 × 1280 photo took 2.87 s warm (3.69 s first request) at the defaults and FP16 on an M5 Max, with a peak of 2.79 GB (quick check, 22 Oct 2026, Mac in use, measured). Requests run one at a time and time out after 180 s.

## Limits

- One image per request; no multi-view consistency.
- Metric scale is the model's estimate from image content; it can be off for unusual scenes, miniatures or photos of screens and pictures *(unverified)*.
- Images larger than 4096 px on the long side return maps at 4096 px.
- The principal point is always assumed at the image centre; lens distortion is not modelled *(unverified)*.
- A safetensors file for a 12-megapixel photo with depth and mask is about 36 MB; points and normals add about 72 MB each (derived from FP16 storage).
- The makers' per-step outputs, `num_tokens`, edge-removal threshold and mesh, glb and ply exports are not exposed.

## Licence and credit

MIT (the converted checkpoint's card and `model.toml`; the makers' code is MIT, with its DINOv2 code under Apache-2.0). Commercial use, modification and redistribution are allowed provided the copyright and licence notice are kept.

MoGe-3 is by Lingyu Kong, Ruicheng Li, Ruicheng Wang, Sicheng Xu, Chengtang Yao, Jianfeng Xiang and Jiaolong Yang; the MLX conversion is by mlx-community with mlx-vlm. The makers ask to cite the MoGe papers; for MoGe-3:

```bibtex
@misc{kong2026finedetailmonoculargeometryestimation,
  title={Fine-Detail Monocular Geometry Estimation with Self-Guided Sparse Volumetric Refinement},
  author={Lingyu Kong and Ruicheng Li and Ruicheng Wang and Sicheng Xu and Chengtang Yao and Jianfeng Xiang and Jiaolong Yang},
  year={2026},
  eprint={2607.17967},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2607.17967}
}
```

## Links

- Model card: https://huggingface.co/mlx-community/moge-3-vitl-mlx-fp32/tree/f5ceca7512506731fb1c50307c313da76aac500f
- Original checkpoint: https://huggingface.co/Ruicheng/moge-3-vitl
- Code: https://github.com/microsoft/MoGe
- Paper: https://arxiv.org/abs/2607.17967
- Project page: https://qft-333.github.io/moge3page/

## How Arsenal runs it

Checkpoint: `mlx-community/moge-3-vitl-mlx-fp32@f5ceca7512506731fb1c50307c313da76aac500f`, converted by mlx-vlm from the official `model.pt`; the converter reports identical valid-pixel masks and under 0.2% median relative depth error against the PyTorch reference on CPU. The published checkpoint is FP32; Get casts tensor by tensor to FP16. Spatial kernels, normalization and non-matrix tensors stay FP16. Checked live at FP16 on 19 Oct: metric depth with sharp edges, near and far in the right order. Quantization quality needs the approved trained-weight trial.

Swift port: `mnmly/mlx-swift-MoGe@2925d7bbf821c8f7e9cb4d7c6cdc5b6061b6bce2`, checked against `microsoft/MoGe@74fbce054ebed49800de42d0ad0e83495065719a`, run in Arsenal's native geometry worker. The family retains licences and detailed adaptations beside its imported sources (antialiased encoder resize, half-to-even token grid, pixel-centre UV, focal/shift recovery solved to its optimum; on the regression maps it agrees with the reference's `ftol=1e-3` stop within 1e-4). Source pins, exact byte sizes and checkpoint SHA256 are in `model.toml`. The model folder selects its registered architecture explicitly through the dispatch seam; the original checkpoint config remains intact, and folder architecture selects the adapter even when the checkpoint carries an unrelated model_type.

Input files are given as absolute paths anywhere on this Mac; `arsenal_upload` is only for files that are not on this Mac. Depth, points, normals and companion maps with NaN, infinity outside MoGe's mask, or no positive finite depth fail with `invalid_model_output` before any file is written. Maps are stored as FP16 and the mask as uint8, and a value beyond FP16's range fails rather than being written as infinity.

All 607 active parameter names and shapes match the pinned header after the explicit model-local map. 0 tensors are excluded. All 607 active source tensors are FP32 and require the declared FP16 cast; a shape match is not cast-quality evidence. Exact Range responses read 66574 header bytes and no payload bytes. Explicit per-tensor header cast maps record the F32-to-F16 Get plan; they do not establish trained cast quality.

FP16/int8/int4 random graphs pass every declared operation through the native worker and the production artifact validator. Int8/int4 pack every eligible matrix in the shrunk graph; norms and spatial kernels remain half. Ten family tests pass, including fixed transposed-convolution orientation, bicubic half-precision behavior, and stored int8/int4 conversion/load with nontrivial biases. Four malformed requests, cancellation and unloading pass. Random graphs prove wiring and shapes only; trained output, FP16 casting and quantization quality are not established by these checks. Current checks use the tracked FP32 source metadata, the shared FP16/int8/int4 serving-tier enumeration, original checkpoint configs and immediate requests without pacing.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.

Measured figures: `results/quick-m5-max-fp16-native-exact.json` and `results/quick-checks/night-2026-10-22.json` (one Load in 1.17 s, then two requests on a 1024 × 1280 photo, 3.69 s and 2.87 s; peak 2661 MiB summed worker footprint; Mac in use, not the interleaved protocol). The benchmark suite in `bench/bench.toml` is deferred. Integration record: `log/001-integration.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: one image; the MLX card's example returns points, depth, mask, normals and intrinsics. [MLX model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: the MLX example sets `resolution_level=9`; the paper's default is 3 refinement iterations, and local metrics saturate around 5. [MLX model card, undated, first-hand; MoGe-3 paper, submitted 20 Jul 2026, first-hand]
- Arsenal call: `arsenal_run(model:"moge-3-vitl", input:{"image":{"path":"/absolute/path/photo.jpg"},"points":true,"intrinsics":true,"resolution_level":9,"refine_steps":3})`

**Good and bad**
- Good at: the paper reports global metrics over nine datasets and says MoGe-3 substantially outperforms all baselines at the strict δ0.01 threshold. [MoGe-3 paper, submitted 20 Jul 2026, first-hand]
- Bad at: metric scale on KITTI. One user's ViT-L run scored δ1 55.3 against the paper's 85.1, traced to predicted scale (ground truth 1.2 to 1.3 times the prediction). [MoGe issue #173, opened 2026-10-21, first-hand]
- Against the obvious alternatives: boundary F1 ranks third overall and is comparable to Depth Pro, which runs at 1536² pixels against MoGe-3's 840². [MoGe-3 paper, submitted 20 Jul 2026, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: the official PyTorch code does not reproduce the paper's ViT-L KITTI δ1 (55.3 against 85.1). Fix or workaround: none found; no comments at read time. [MoGe issue #173, opened 2026-10-21, first-hand]
- Complaint or known bug: fly-points, which the paper's limitations section says the regression approach introduces. Fix or workaround: the paper proposes generative formulations as future work, not a fix. [MoGe-3 paper, submitted 20 Jul 2026, first-hand]

**Evidence**
- Strength: thin. The maker paper and MLX card, plus one open issue with no comments at read time.
- Sources re-opened: the paper (arxiv.org/abs/2607.17967); the MLX card (huggingface.co/mlx-community/moge-3-vitl-mlx-fp32); issue #173 (github.com/microsoft/MoGe); `ops/estimate_depth.input.json`.
