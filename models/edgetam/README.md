# EdgeTAM On-Device Track Anything Model

EdgeTAM is Meta's on-device variant of SAM 2 (arXiv 2501.07256). This entry runs its image path: a point or box prompt on a photo returns up to three candidate masks with predicted IoU scores.

Inputs: `image` (a photo), `points` (x and y from 0 to 1 of the image size, label `foreground` or `background`), `boxes` (one box, `xmin`, `ymin`, `xmax`, `ymax` from 0 to 1), and `multimask_output` (default true). At least one point or box is required.

Outputs: `masks`, a list of up to three PNG files (white is the object) each with `path`, `width`, `height`, `bytes`, `sha256`, `score` (predicted IoU) and `index`, and `best`, the index of the highest score.

Example request: `{"image": {"path": "/abs/truck.jpg"}, "boxes": [{"xmin": 0.04, "ymin": 0.23, "xmax": 0.95, "ymax": 0.72}]}`. On the card's truck photo this returned a clean whole-truck mask with score 0.98.

With `multimask_output` false it returns one mask, chosen as the maker does: the single-mask output when it is stable, otherwise the best of the three alternatives.

Limits: a single click can be ambiguous and give partial masks with speckled edges; a box is the reliable prompt. Only the image path is loaded: the video memory and tracking modules (the planned perception `track_video`) are not, so it segments one photo at a time. Masks are predicted at 256 pixels and upsampled. There is no language input or output.

Parity: against the maker's transformers code (EdgeTamModel from yonigozlan/EdgeTAM-hf, MPS, FP16) on five photo and prompt pairs, the mask logits match with cosine 0.99999 or better when both get the same pixels, and 0.99996 or better through each side's own image loading. Predicted IoU scores and the best mask agree. Details in `log/003-parity.md`.

Source: [mlx-community/EdgeTAM-fp16](https://huggingface.co/mlx-community/EdgeTAM-fp16/tree/e7770c15b913841cb657e6d2fbb33a6541c3bb32), Apache-2.0 (converted from [facebookresearch/EdgeTAM](https://github.com/facebookresearch/EdgeTAM)). Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: An image, then prompts; the README shows only the placeholders `predictor.set_image(<your_image>)` and `predictor.predict(<input_prompts>)`. [EdgeTAM GitHub README, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: The README's usage snippet is `with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):`. [EdgeTAM GitHub README, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/edgetam/ops/segment`, or the same operation through MCP. Vendor example: `predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint))`, `predictor.set_image(<your_image>)`, `masks, _, _ = predictor.predict(<input_prompts>)`. [EdgeTAM GitHub README, undated, first-hand]

**Good and bad**
- Good at: Maker's claim: 22 times faster than SAM 2, and 16 FPS on an iPhone 15 Pro Max. [EdgeTAM GitHub README, undated, first-hand]
- Bad at: The reporter says image encoding is outside the timing loop; including it cuts FPS about 15 times. [EdgeTAM issue #24, 2026-01-15, opinion]
- Against the obvious alternatives: SA-23 image 1-click mIoU is 56.0 against SAM 2.1's 63.3, per the maker's table. [EdgeTAM GitHub README, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: The 16 FPS headline differs from the README's iPhone 15 Pro Max figures, 15.7 (VOS table) and 40.4 (SA table); the page does not say why. Fix or workaround: Not found in sources checked; issue #24 has no maintainer reply. [EdgeTAM GitHub README, undated, first-hand; EdgeTAM issue #24, 2026-01-15, opinion]
- Complaint or known bug: Different `frame_idx` values "does not behave as expected" (video path). No fix or reply. [EdgeTAM issue #9, 2025-06-18, opinion]

**Evidence**
- Strength: thin. Maker's README is first-hand; the two issues have no maintainer replies; no independent test was opened.
- Sources:
  - EdgeTAM GitHub README, https://github.com/facebookresearch/EdgeTAM, undated, first-hand.
  - EdgeTAM issue #24, https://github.com/facebookresearch/EdgeTAM/issues/24, 2026-01-15, opinion.
  - EdgeTAM issue #9, https://github.com/facebookresearch/EdgeTAM/issues/9, 2025-06-18, opinion.
