# ViTPose Base Simple

Finds 17 body keypoints (nose, eyes, ears, shoulders, elbows, wrists, hips, knees, ankles) for the person in a photo. A plain Vision Transformer backbone with a light heatmap decoder, from usyd-community's transformers port.

Input: one photo (`image`, PNG, JPEG or HEIC). Output: `keypoints`, a list of 17 objects with `name`, `x`, `y` and `score` in the photo's own pixels, plus `image_width` and `image_height`.

Example request:

```json
{"image": {"path": "/absolute/path/to/person.jpg"}}
```

Limits: the whole photo is treated as one person, since this runner has no person detector. Crop to one person for good results. Several people are not separated. Scores below about 0.4 are weak joints.

Parity: against the maker's transformers code (VitPoseForPoseEstimation, MPS, FP16) on a COCO photo of one skier, every joint within 2.2 pixels on the 640x425 photo and within 1 pixel on a crop of the person. Details in `log/003-parity.md`.

Source: [usyd-community/vitpose-base-simple](https://huggingface.co/usyd-community/vitpose-base-simple/tree/a93ac0c67e0b7e2c55287d21d4c460c8f3c54d45), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Top-down pose. Needs person boxes first; the processor takes boxes as COCO (x, y, width, height), and the output holds keypoints with scores. [Transformers ViTPose docs, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: The maker's example keeps person-detector boxes at threshold 0.3 (RT-DETR, label 0 is person). [Transformers ViTPose docs, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/vitpose-base-simple/ops/estimate_pose`, or the same operation through MCP. Vendor example: `image_processor(image, boxes=[person_boxes], return_tensors="pt")`, then `model(**inputs)`, then `image_processor.post_process_pose_estimation(outputs, boxes=[person_boxes])`, with `VitPoseForPoseEstimation.from_pretrained("usyd-community/vitpose-base-simple")`. [Transformers ViTPose docs, undated, first-hand]

**Good and bad**
- Good at: The card's headline is 81.1 AP on COCO Keypoint test-dev, and 80.9 AP for its largest model; no per-checkpoint figure appears in text. [usyd-community model card, undated, first-hand]
- Bad at: The card says its OCHuman and MPII evaluations use ground-truth person boxes, so detector errors are not in those numbers. [usyd-community model card, undated, first-hand]
- Against the obvious alternatives: The maker says ViTPose++ (`usyd-community/vitpose-plus-*`) adds a mixture-of-experts backbone, with `dataset_index` choosing an expert. No head-to-head result is on the page. [Transformers ViTPose docs, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Not found in sources checked; the model page lists one discussion, with no title shown. Fix or workaround: Not found in sources checked. [usyd-community model card, undated, first-hand]

**Evidence**
- Strength: thin. Only the maker's card and docs were opened; no independent test was opened.
- Sources:
  - ViTPose model card, https://huggingface.co/usyd-community/vitpose-base-simple, undated, first-hand.
  - Transformers ViTPose docs, https://huggingface.co/docs/transformers/model_doc/vitpose, undated (v5.22.1 page), first-hand.
