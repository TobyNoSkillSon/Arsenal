# SAM 2.1 Hiera Large

Cuts out the object you click or box in a photo. This is Meta FAIR's SAM 2.1 at the large size (224.4M parameters): a Hiera image encoder, a prompt encoder and a two-way mask decoder. Only the image predictor runs here; the video tracking modules are not loaded. SAM 2.1 Hiera Tiny is the same network at 39M parameters, with a smaller memory footprint and slightly lower benchmark scores (76.5 against 79.5 J&F on SA-V test).

Inputs: `image` (PNG or JPEG), and `points` (x and y from 0 to 1 of the image, label `foreground` or `background`) and/or `boxes` (`xmin`, `ymin`, `xmax`, `ymax` from 0 to 1). One box can be combined with points. Several boxes without points give one mask per box. `multimask` (default true) returns three alternative masks for one prompt; false returns one, chosen as the maker does (the single-mask output when it is stable, otherwise the best alternative).

Outputs: `files`, one 8-bit PNG mask per answer at the photo's own size (white is the object), labelled "Mask 1..3", "Mask" or "Box N", and `scores`, the model's predicted IoU for each mask in the same order.

Example request:

```json
{"image": {"path": "/absolute/path/to/truck.jpg"}, "points": [{"x": 0.28, "y": 0.31}]}
```

On the SAM 2 notebook's truck photo this click on the rear side window returns three masks; the best-scored one is that window pane. Two boxes around the wheels return one wheel mask each.

Limits: it needs a click or box and does not find objects by name. A single click can be ambiguous, so a box or a few clicks is more reliable. Masks are predicted at 256x256 and upsampled, so fine edges are soft. No video tracking. No language input or output.

Parity: against the maker's transformers code (Sam2Model, MPS, FP16) on six photo and prompt pairs (clicks, boxes, a box with a background click, two boxes, a 33x17 image), the mask logits match with cosine 0.999997 or better when both get the same pixels and 0.99990 or better through each side's own image loading; predicted IoU scores and the chosen masks agree. Details in `log/003-parity.md`.

Source: [facebook/sam2.1-hiera-large](https://huggingface.co/facebook/sam2.1-hiera-large/tree/665f8e2ad61cf5f53d65644ff27c8ee525124610), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: one image with point or box prompts; the card's code comment says it returns multiple masks ranked by quality score. [SAM 2.1 Hiera Large model card, undated, first-hand (maker)]
- Prompts, reference audio or voice settings, if relevant: the card's examples use one positive click at (500, 375). [SAM 2.1 Hiera Large model card, undated, first-hand (maker)]
- Parameters people recommend: Not found in sources checked. The card's code wraps predict in `torch.inference_mode()` and `torch.autocast("cuda", dtype=torch.bfloat16)`, with no advice on them. [SAM 2.1 Hiera Large model card, undated, first-hand (maker)]
- Arsenal call: `POST /arsenal/v1/models/sam2.1-hiera-large/ops/segment`, or the same operation through MCP. Vendor example: `predictor = SAM2ImagePredictor.from_pretrained("facebook/sam2.1-hiera-large")`, then `predictor.set_image(<your_image>)` and `masks, _, _ = predictor.predict(<input_prompts>)`. [SAM 2.1 Hiera Large model card, undated, first-hand (maker)]

**Good and bad**
- Good at: image segmentation, per the SAM 2 abstract: "more accurate and 6x faster than" SAM. The abstract gives no accuracy figure. [arXiv 2408.00714, submitted 2024-08-01, first-hand (maker)]
- Bad at: finding objects by name. The card shows only point and box prompts and automatic mask generation. [SAM 2.1 Hiera Large model card, undated, first-hand (maker)]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: loading fails with state-dict key errors (#704, created 2025-08-29) or a missing config (#721, created 2025-11-04). Fix or workaround: Not found in sources checked. [GitHub facebookresearch/sam2 #704 and #721, first-hand (users)]

**Evidence**
- Strength: thin. The paper's image claim is abstract-only, and the card has no benchmark numbers.
- Sources: SAM 2.1 Hiera Large model card (huggingface.co/facebook/sam2.1-hiera-large, undated); arXiv abstract 2408.00714; GitHub facebookresearch/sam2 #704 and #721.
