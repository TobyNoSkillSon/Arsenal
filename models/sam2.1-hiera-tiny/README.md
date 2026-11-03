# SAM 2.1 Hiera Tiny

Cuts out the object you click or box in a photo. This is Meta FAIR's SAM 2.1 at the tiny size (39.0M parameters): a Hiera image encoder, a prompt encoder and a two-way mask decoder. Only the image predictor runs here; the video tracking modules are not loaded. SAM 2.1 Hiera Large is the same network at 224M parameters and scores higher on the maker's benchmarks (79.5 against 76.5 J&F on SA-V test).

Inputs: `image` (PNG or JPEG), and `points` (x and y from 0 to 1 of the image, label `foreground` or `background`) and/or `boxes` (`xmin`, `ymin`, `xmax`, `ymax` from 0 to 1). One box can be combined with points. Several boxes without points give one mask per box. `multimask` (default true) returns three alternative masks for one prompt; false returns one, chosen as the maker does (the single-mask output when it is stable, otherwise the best alternative).

Outputs: `files`, one 8-bit PNG mask per answer at the photo's own size (white is the object), labelled "Mask 1..3", "Mask" or "Box N", and `scores`, the model's predicted IoU for each mask in the same order.

Example request:

```json
{"image": {"path": "/absolute/path/to/truck.jpg"}, "points": [{"x": 0.28, "y": 0.31}]}
```

On the SAM 2 notebook's truck photo this click on the rear side window returns three masks; the best-scored one is that window pane. Two boxes around the wheels return one wheel mask each.

Limits: it needs a click or box and does not find objects by name. A single click can be ambiguous, so a box or a few clicks is more reliable. Masks are predicted at 256x256 and upsampled, so fine edges are soft. No video tracking. No language input or output.

Parity: against the maker's transformers code (Sam2Model, MPS, FP16) on six photo and prompt pairs (clicks, boxes, a box with a background click, two boxes, a 33x17 image), the mask logits match with cosine 0.999989 or better when both get the same pixels and 0.99991 or better through each side's own image loading; predicted IoU scores and the chosen masks agree. Details in `log/003-parity.md`.

Source: [facebook/sam2.1-hiera-tiny](https://huggingface.co/facebook/sam2.1-hiera-tiny/tree/de431c4043854a71d8101e17995dfe596bf101a5), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Image, then prompts in pixel coordinates in the Transformers example (`input_points=[[[[500, 375]]]]`, `input_labels=[[[1]]]`, boxes `[x_min, y_min, x_max, y_max]`). [facebook/sam2.1-hiera-tiny model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: No advice given. The card's multi-object and batched examples pass `multimask_output=False` with no explanation; the single-point example omits it. [facebook/sam2.1-hiera-tiny model card, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/sam2.1-hiera-tiny/ops/segment`, or the same operation through MCP. Vendor example: `Sam2Model.from_pretrained("facebook/sam2.1-hiera-tiny")` with `Sam2Processor`, `model(**inputs)`, then `processor.post_process_masks(outputs.pred_masks.cpu(), inputs["original_sizes"])[0]`. [facebook/sam2.1-hiera-tiny model card, undated, first-hand]

**Good and bad**
- Good at: Maker's table: 91.2 FPS on an A100 (torch 2.5.1, CUDA 12.4). [SAM 2 GitHub README, 2024-09, first-hand]
- Bad at: Not found in sources checked; the model card lists no limitations. [facebook/sam2.1-hiera-tiny model card, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked. [SAM 2 GitHub README, 2024-09, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: On video, with a box on frame 0, the same prompt worked on a single image but gave poor masks with strange patterns. Fix or workaround: Suggested, not confirmed: tighter boxes or point prompts (a commenter, not shown as a maintainer); no maintainer reply, and the issue is open. [SAM 2 GitHub issue #533, 2025-01-14, opinion]

**Evidence**
- Strength: thin. Only the maker's card, README and one issue were opened; no independent tiny-model test was opened.
- Sources:
  - facebook/sam2.1-hiera-tiny model card, https://huggingface.co/facebook/sam2.1-hiera-tiny, undated, first-hand.
  - SAM 2 GitHub README, https://github.com/facebookresearch/sam2, 2024-09, first-hand.
  - SAM 2 GitHub issue #533, https://github.com/facebookresearch/sam2/issues/533, 2025-01-14, opinion.
