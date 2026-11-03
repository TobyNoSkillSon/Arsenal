# Depth Anything 3 Metric Large

Estimates how far everything in one photo is, in metres when the camera's focal length is known. The monocular metric model of ByteDance Seed's Depth Anything 3 family (334.2M parameters): a DINOv2 ViT-L backbone with a depth head and a sky head.

Input: one photo (`image`, PNG or JPEG), and optionally `focal_length`, the focal length in the photo's own pixels. Without it, the photo's EXIF 35 mm equivalent focal length is used when present. Output: `view-0-depth.png`, a 16-bit grayscale preview (brighter is nearer), and `view-0.safetensors` with FP16 tensors `depth` [height, width] and `sky` (sky probability). `depth_unit` is `meters` when a focal length was known, else `relative`; `"mode": "relative"` skips the scaling, and `"mode": "metric"` with no focal length gives `invalid_request`.

Example request:

```json
{"image": {"path": "/absolute/path/to/photo.jpg"}, "focal_length": 1450}
```

Limits: metres follow the maker's rule, depth times focal length in pixels divided by 300, so a wrong focal length scales every distance. EXIF orientation is not applied (as in the maker's code), so a rotated phone photo gives a rotated map. Maps come at the processed size, 504 pixels on the long side. Sky pixels take the 99th percentile of the other depths, computed over every pixel; the maker samples 100,000 pixels at random, so sky values can differ by a few percent. One photo per request; no poses, intrinsics or points.

Parity: against the maker's code (depth_anything_3 at 3d835ec, MPS, FP16 autocast), depth cosine 0.99968 and 0.99985 on two photos, sky cosine 0.9999. Details in `log/003-parity.md`.

Runner: the kit runner with Arsenal's vendored MLXDA3 sources (from mnmly/mlx-swift-da3, as in the geometry worker). Source: [depth-anything/DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE/tree/4010e39f3634a45bc60553321fb49fb760bd594e), Apache-2.0. Runs as FP16 through MLX.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: A list of image paths, PIL images or NumPy arrays goes to `inference`. [DA3METRIC-LARGE model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: No tuned settings. The README FAQ gives metric depth = focal length in pixels × output ÷ 300. [Depth Anything 3 GitHub README, 2025-12-11, first-hand]
- Arsenal call: `POST /arsenal/v1/models/da3metric-large/ops/estimate_depth`, or the same operation through MCP. Vendor example: `model = DepthAnything3.from_pretrained("depth-anything/da3metric-large")`, then `model.inference(images, export_dir="output", export_format="glb")`. [DA3METRIC-LARGE model card, undated, first-hand]

**Good and bad**
- Good at: Metric scale once a focal length is supplied. [Depth Anything 3 GitHub README, 2025-12-11, first-hand]
- Bad at: The card warns of limits on "certain domain-specific images" and varying lighting and scene complexity. [DA3METRIC-LARGE model card, undated, first-hand]
- Against the obvious alternatives: The card says DA3 "significantly outperforms" Depth Anything 2 on monocular depth, with no number. [DA3METRIC-LARGE model card, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: pyslam's DA3 wrapper returned raw output as metres, off by the focal factor. Fix or workaround: Reported by one user, not confirmed: multiply by the mean focal length scaled to 504 px, divide by 300 (author's TUM median ratios: 1.02 to 1.13). [pyslam PR #115, opened 2026-10-17, first-hand]
- Complaint or known bug (nested DA3 model, not this one): Gaussian point clouds are not scaled in depth alignment. Fix or workaround: Not found in sources checked; the reporter proposes a patch. [Depth Anything 3 issue #242, 2026-03-19, first-hand]

**Evidence**
- Strength: thin. The formula is documented; no independent accuracy test found.
- Sources: model card (huggingface.co/depth-anything/DA3METRIC-LARGE); README (github.com/ByteDance-Seed/Depth-Anything-3); pyslam PR #115 (github.com/sjulier/pyslam/pull/115); issue #242 (same repo).
