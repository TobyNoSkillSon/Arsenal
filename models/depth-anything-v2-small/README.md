# Depth Anything V2 Small

Estimates how near or far everything in a photo is. A small DINOv2 encoder with a DPT decoder (24.8M parameters), trained by the Depth Anything team on about 595K synthetic and 62M real photos.

Input: one photo (`image`, PNG or JPEG). Output: two files for the photo, at its own size: `view-0-depth.png`, a 16-bit grayscale depth map (brighter is nearer, stretched to the map's own range), and `view-0.safetensors`, the raw values as FP16 tensor `depth` [height, width]. `mode` and `depth_unit` are `relative`.

Example request:

```json
{"image": {"path": "/absolute/path/to/photo.jpg"}}
```

Limits: relative inverse depth only, so larger values mean nearer and there is no unit or metric scale. One photo per request; no camera intrinsics, poses or points. The photo is resized to about 518 pixels on its fitted side (a multiple of 14), so very fine detail is lost. A tiny photo still answers (a 33×17 image gives a 33×17 map); one whose short side would fall under 14 pixels after resizing is refused.

Parity: against the maker's transformers code (DepthAnythingForDepthEstimation, MPS, FP16) on a fixed 518×294 photo, cosine 0.99996, mean absolute difference 0.014 on a 0 to 6.6 range; the largest differences (up to 1.4) sit on single pixels at sharp object edges. Details in `log/003-parity.md`.

Source: [depth-anything/Depth-Anything-V2-Small-hf](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf/tree/5426e4f0f36572d16453bbda7a8389317b1bef99), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: A PIL image goes to the Transformers depth-estimation pipeline, which returns a dict with a `depth` key. [Depth-Anything-V2-Small-hf model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `POST /arsenal/v1/models/depth-anything-v2-small/ops/estimate_depth`, or the same operation through MCP. Vendor example: `pipe = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf")`, then `depth = pipe(image)["depth"]`. [Depth-Anything-V2-Small-hf model card, undated, first-hand]

**Good and bad**
- Good at: The card says the family is 10x faster and more lightweight than SD-based models, and more robust than Depth Anything V1. [Depth-Anything-V2-Small-hf model card, undated, first-hand]
- Bad at: Not found in sources checked. The card lists no limits and does not say whether this checkpoint's output is relative or metric.
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Not a bug report. A user asks how to get centimetres from relative output, and says some calibration is needed. Fix or workaround: Reported by one user, not confirmed: the author min-max normalises and inverts the map, then interpolates four reference pairs (0.04 to 0.92 mapped to 85 to 180 cm). The author says it "works very well"; least squares gave poor results. No reply on the issue page. [Depth-Anything-V2 issue #301, 2025-10-03, first-hand report]

**Evidence**
- Strength: thin. The card is the maker's own. I found no independent test of the Small model, and the one issue is an unanswered calibration question.
- Sources: model card (huggingface.co/depth-anything/Depth-Anything-V2-Small-hf); issue #301 (github.com/DepthAnything/Depth-Anything-V2/issues/301).
