# Depth Anything 3 Small

Depth Anything 3 Small (DA3-Small) is the smallest model of the Depth Anything 3 main series from the ByteDance Seed team. It is a plain vision transformer (a DINOv2-style encoder) with a dual DPT head that predicts a depth map and a per-pixel ray map for each view, the "depth-ray" representation the makers use in place of separate task heads. The pinned checkpoint holds 34M parameters (137 MB at FP32); the model card lists 0.08B. Its strength is consistent relative depth across one to eight views of the same scene, with camera intrinsics and poses recovered from the predicted rays.

## What it is for

The makers built the main series as one model for monocular depth, multi-view depth, pose-conditioned depth and camera pose estimation; Small is the fastest size, trained on public academic datasets only.

- Order a scene by distance (near versus far) from one photo, without physical scale.
- Get mutually consistent depth maps for up to eight overlapping photos of one scene.
- Recover approximate camera intrinsics and relative camera poses for a small photo set.
- Build a rough camera-space point cloud, in relative units, for layout or masking work.

## Inputs and outputs

### `estimate_depth` (default)

Predicts a relative depth map per view and, on request, camera intrinsics, camera poses and a point map.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | one of `image` or `images` | `{"path":"/absolute/path"}`; accepts `image/png`, `image/jpeg`; path 1–4096 characters. A single view. |
| `images` | list of files | one of `image` or `images` | 1–8 items, each `{"path":"/absolute/path"}`, `image/png` or `image/jpeg`. Views of one scene; all must give the same processed size (same aspect ratio, see below). |
| `mode` | string | no | Only `relative`. |
| `intrinsics` | boolean | no | Adds each view's 3×3 camera matrix to `views`. |
| `poses` | boolean | no | Adds each view's 4×4 world-to-camera matrix to `views`. |
| `points` | boolean | no | Adds a camera-space point map tensor to each view's safetensors file. |

Send exactly one of `image` or `images`. Each image is at most 16 megapixels (width × height ≤ 16,000,000).

Returns `files`, `mode` (`relative`), `depth_unit` (`relative`) and `views`. For each view `n` (0-based, in input order) there are two files:

- `view-n.safetensors` (id `view-n`): tensors at the processed depth size, H × W. `depth` (H × W, FP16), `confidence` (H × W, FP16, the model's depth confidence; values come from an exp(x)+1 activation, so they are ≥ 1 and larger means more confident *(unverified)*), `rays` (the ray head's output, six channels per pixel), and with `points:true` `points` (H × W × 3, FP16). DA3-Small predicts no validity mask and no sky map, so neither tensor appears.
- `view-n-depth.png` (id `view-n-depth`): a 16-bit greyscale preview of `depth`, H × W, stretched per view from the smallest to the largest positive finite depth (near is dark, far is bright). It is for viewing only; values are not comparable across views or runs.

Each `views` entry has `index`, `file_ids` (both file ids), `image_size` [width, height] of the original image, and `depth_size` [width, height] of the maps. With `intrinsics:true` it also has `intrinsics` (3×3, row-major) and `intrinsics_unit` `pixels`: focal lengths and principal point in pixels of the depth map, not of the original image. With `poses:true` it has `extrinsics` (4×4, last row 0 0 0 1) and `pose_convention` `world_to_camera`; the world frame is set by the model's reference view *(unverified)*. With `points:true` it has `point_frame` `camera_x_right_y_down_z_forward`: each point is (x, y, z) in that view's camera frame, z equal to the relative depth, so the cloud has the same arbitrary scale as the depth.

Depth is relative: larger means farther, but the unit is arbitrary and changes from request to request. Do not read it as metres. For metric depth from one photo use `moge-3-vitl`.

Preprocessing follows the makers' defaults (`process_res` 504, `upper_bound_resize`): Arsenal scales each image so its longer side is 504 px, rounds both sides to the nearest multiple of 14 (the patch size), and normalises with ImageNet mean and standard deviation. A 1280 × 1024 photo therefore gives 504 × 406 maps (derived from the resize rule). Outputs stay at that processed size; Arsenal does not resize them back to the original. In a multi-view request every image must reach the same processed size, so use photos with one aspect ratio; otherwise the request is refused. EXIF orientation is not applied: the image is read in its stored pixel orientation, so rotate a camera JPEG first if its orientation tag matters. Camera intrinsics, poses or depth cannot be supplied as inputs.

## Run it

MCP: `arsenal_run(model:"da3-small", input:{"images":[{"path":"/absolute/path/view-a.jpg"},{"path":"/absolute/path/view-b.jpg"}], "intrinsics":true, "poses":true})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/da3-small/ops/estimate_depth" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/photo.jpg"}}'`

There are no settings. One 1024 × 1280 photo took 0.45 s warm (0.57 s first request) at FP16 on an M5 Max, with a peak of 0.80 GB (quick check, 22 Oct 2026, Mac in use, measured). Requests run one at a time and time out after 180 s.

## Limits

- Relative depth only, with no physical scale; values from different requests cannot be compared.
- At most eight views per request, all with the same processed size.
- Maps come back at about 504 px on the longer side, whatever the input size.
- Intrinsics and poses come from geometric recovery on the predicted rays; the learned camera encoder and decoder are not ported, so camera poses cannot be given as conditioning, and accuracy of the recovered cameras has not been checked in Arsenal.
- The makers' 3D Gaussian and export outputs (glb, ply, npz, video) are not offered.
- The makers note it is trained on academic datasets and may do worse on domain-specific images, and that results vary with image quality, lighting and scene complexity.

## Licence and credit

Apache-2.0 (model card and `model.toml`). Commercial use, modification and redistribution are allowed; keep the licence and notices, and state changes. The vendored Swift port keeps its own licence beside its sources.

Made by Haotong Lin, Sili Chen, Jun Hao Liew, Donny Y. Chen, Zhenyu Li, Guang Shi, Jiashi Feng and Bingyi Kang (ByteDance Seed). The makers ask to cite:

```bibtex
@article{depthanything3,
  title={Depth Anything 3: Recovering the visual space from any views},
  author={Haotong Lin and Sili Chen and Jun Hao Liew and Donny Y. Chen and Zhenyu Li and Guang Shi and Jiashi Feng and Bingyi Kang},
  journal={arXiv preprint arXiv:2511.10647},
  year={2025}
}
```

## Links

- Model card: https://huggingface.co/depth-anything/DA3-SMALL/tree/e08cab65ca0ec38e7826075418411ab90cab4da3
- Code: https://github.com/ByteDance-Seed/depth-anything-3
- Paper: https://arxiv.org/abs/2511.10647
- Project page: https://depth-anything-3.github.io

## How Arsenal runs it

Checkpoint: `depth-anything/DA3-SMALL@e08cab65ca0ec38e7826075418411ab90cab4da3`. The published checkpoint is FP32; Get casts tensor by tensor to FP16. Spatial kernels, normalization and non-matrix tensors stay FP16. Checked live at FP16 on 19 Oct: in a living-room photo, near and far were ordered correctly.

Swift port: `mnmly/mlx-swift-da3@496dfb1d1f0969c5dc2518aad59ea21abb1ea88a`, run in Arsenal's native geometry worker. The family retains licences and detailed adaptations beside its imported sources. Source pins, exact byte sizes and checkpoint SHA256 are in `model.toml`. The model folder selects its registered architecture explicitly through the dispatch seam; the original checkpoint config remains intact, and folder architecture selects the adapter even when the checkpoint carries an unrelated model_type.

Input files are given as absolute paths anywhere on this Mac; `arsenal_upload` is only for files that are not on this Mac. Depth, points, normals and companion maps with NaN or infinity, or with no positive finite depth, fail with `invalid_model_output` before any file is written; every view is checked before the first file is written, so a failed view leaves no partial output. Maps are stored as FP16, and a value beyond FP16's range fails rather than being written as infinity.

All 363 active parameter names and shapes match the pinned header after the explicit model-local map. 74 tensors (the learned camera encoder and decoder) are excluded; Gaussian splatting heads are not imported either. All 363 active source tensors are FP32 and require the declared FP16 cast; a shape match is not cast-quality evidence. Exact Range responses read 51088 header bytes and no payload bytes. Explicit per-tensor header cast maps record the F32-to-F16 Get plan; they do not establish trained cast quality.

FP16/int8/int4 random graphs pass every declared operation through the native worker and the production artifact validator. Int8/int4 pack every eligible matrix in the shrunk graph; norms and spatial kernels remain half. Ten family tests pass, including fixed transposed-convolution orientation, bicubic half-precision behavior, and stored int8/int4 conversion/load with nontrivial biases. Four malformed requests, cancellation and unloading pass. Random graphs prove wiring and shapes only; these checks do not establish trained output or precision quality. Current checks use the tracked FP32 source metadata, the shared FP16/int8/int4 serving-tier enumeration, original checkpoint configs and immediate requests without pacing.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.

Measured figures: `results/quick-m5-max-fp16-native-exact.json` and `results/quick-checks/night-2026-10-22.json` (one Load in 1.21 s, then two requests on a 1024 × 1280 photo, 0.57 s and 0.45 s; peak 760 MiB summed worker footprint; Mac in use, not the interleaved protocol). The benchmark suite in `bench/bench.toml` is deferred. Integration record: `log/001-integration.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The DA3-SMALL card lists relative depth, pose estimation and pose conditioning [DA3-SMALL model card, undated, first-hand].
- Prompts, reference audio or voice settings: Not applicable.
- Parameters people recommend: The maker's FAQ says the ray-head pose option (use_ray_pose) is "generally slightly slower, but more accurate", and defaults to False. Arsenal does not expose it. No small-model advice found [DA3 GitHub README FAQ, undated, first-hand].
- Arsenal call: `arsenal_run(model:"da3-small", input:{"images":[{"path":"/absolute/path/view-a.jpg"},{"path":"/absolute/path/view-b.jpg"}],"intrinsics":true,"poses":true})`

**Good and bad**
- Good at: The maker says the family "significantly outperforms" Depth Anything 2 on monocular depth and VGGT on multi-view depth and pose. It gives no small-model number [DA3-SMALL model card, undated, first-hand, vendor claim].
- Bad at: The card warns of limits on domain-specific images and says results vary with image quality, lighting and scene complexity [model card, undated, first-hand]. A feature request open since 2026-06-15 asks for Small and Base metric checkpoints; it says the smaller ones are relative-only [DA3 issue #261, 2026-06-15, opinion].
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: The reporter says alignment runs Umeyama alignment even with the align flag set, "causing inevitable failures for 2-view inputs". The issue names no checkpoint. Fix or workaround: none offered. A second user reports the same problem on 2026-03-26, the newest comment read; no maintainer reply is visible [DA3 issue #241, 2026-03-19, first-hand].

**Evidence**
- Strength: thin. The sources are the maker's card, README and claims; no independent accuracy test was found.
- Sources:
  - depth-anything/DA3-SMALL model card, https://huggingface.co/depth-anything/da3-small, undated, first-hand (vendor).
  - ByteDance-Seed/Depth-Anything-3 GitHub README, https://github.com/ByteDance-Seed/Depth-Anything-3, undated, first-hand (vendor).
  - Issue #241, https://github.com/ByteDance-Seed/Depth-Anything-3/issues/241, 2026-03-19, first-hand.
  - Issue #261, https://github.com/ByteDance-Seed/Depth-Anything-3/issues/261, 2026-06-15, opinion.
