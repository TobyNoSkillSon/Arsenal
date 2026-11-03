# EoMT COCO Panoptic Large 640

Splits a photo into its objects and regions and labels each one. EoMT is a plain ViT-L (316.7M parameters, DINOv2 pre-training) whose last four layers also carry 200 learned queries; each query becomes a mask and a COCO class. This checkpoint does panoptic segmentation with the 133 COCO classes: 80 kinds of things (person, cat, car, remote) counted one by one, and 53 kinds of stuff (sky, road, grass, wall) as regions. The makers report 56.0 PQ on COCO val.

Input: `image` (PNG or JPEG). It takes no clicks, boxes or text.

Output: `files`, one 8-bit PNG mask per segment at the photo's size (white is the segment), labelled with its class ("cat 1", "cat 2", "couch"), and `scores`, the class confidence of each segment in the same order. When nothing passes the maker's 0.8 confidence, one empty mask labelled "Nothing found" comes back with score 0.

Example request:

```json
{"image": {"path": "/absolute/path/to/photo.jpg"}}
```

On the card's COCO photo of two cats on a couch this returned five segments: cat 1, cat 2, couch, remote 1 and remote 2, each traced on its object.

Limits: only the 133 COCO classes, so other things are left out or folded into the nearest class. The photo is resized to 640 pixels on its longest side, so small or thin things are missed and edges are soft. Above about 4 megapixels the map is computed at a scaled size and enlarged. Class names are English.

Parity: against the maker's transformers code (EomtForUniversalSegmentation and its image processor, MPS, FP16) on four images, every kept query, class label and panoptic segment is identical; mask logits match with cosine 0.99999 or better on the same pixels, and the panoptic maps agree on 99.8% of pixels or more. Details in `log/003-parity.md`.

Source: [tue-mps/coco_panoptic_eomt_large_640](https://huggingface.co/tue-mps/coco_panoptic_eomt_large_640/tree/dcd130bed9b1ebda7041fd660fddb16f905b9c3b), MIT. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: One image; the card's example passes only the image, with no prompts or boxes. [HF model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not relevant; the model takes no prompts.
- Parameters people recommend: The card sets no thresholds. [HF model card, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/eomt-coco-panoptic-large-640/ops/segment`, or the same operation through MCP. Vendor example from the card, abridged [HF model card, undated, first-hand]:
```python
from transformers import EomtForUniversalSegmentation, AutoImageProcessor
model_id = "tue-mps/coco_panoptic_eomt_large_640"
processor = AutoImageProcessor.from_pretrained(model_id)
model = EomtForUniversalSegmentation.from_pretrained(model_id)
inputs = processor(images=image, return_tensors="pt")
with torch.inference_mode():
    outputs = model(**inputs)
target_sizes = [(image.height, image.width)]
preds = processor.post_process_panoptic_segmentation(outputs, target_sizes=target_sizes)
```

**Good and bad**
- Good at: The paper reports 56.0 PQ for EoMT-L on COCO val2017, and 128 FPS against 32 FPS for ViT-Adapter with Mask2Former (ViT-B) in its introduction. [arXiv 2503.19108, 2025-03-24, first-hand]
- Bad at: Not found in sources checked; the card and repo README have no limitations section.
- Against the obvious alternatives: The README claims up to 4x faster with ViT-L; no independent timing was opened. [tue-mps/eomt README, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Not found in sources checked. Fix or workaround: Not found in sources checked. The repo lists 2 open pull requests; its issue count was not visible. [tue-mps/eomt GitHub, undated, first-hand]
- Caution: the README says DINOv3 configurations expect delta weights by default; its DINOv2 configs are separate. [tue-mps/eomt README, undated, first-hand]

**Evidence**
- Strength: thin. Maker's paper, card and repo only.
- Sources:
  - "Your ViT is Secretly an Image Segmentation Model", arXiv 2503.19108, 2025-03-24, first-hand.
  - "coco_panoptic_eomt_large_640", huggingface.co/tue-mps/coco_panoptic_eomt_large_640, undated, first-hand.
  - "tue-mps/eomt", github.com/tue-mps/eomt, undated, first-hand.
