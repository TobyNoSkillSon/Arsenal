# OWLv2 Base Patch16 Ensemble

Finds the things you name in a photo and draws a box around each one. Give it an image and a few short English phrases ("a photo of a cat", "a coffee mug"); it returns every box whose best-matching phrase scores above the threshold, with that phrase as the label. It is Google's open-vocabulary detector: a CLIP ViT-B/16 image encoder proposes one box per 16-pixel patch of a 960 by 960 view, and a CLIP text encoder scores each box against each phrase.

Input: `image` (PNG or JPEG), and the queries as `labels` (a list, one query each) or as `phrase` ("cat. remote control.", split on full stops, commas and semicolons); `threshold` (0 to 1, default 0.3). Output: `detections`, each with `label`, `score` and `box` (xmin, ymin, xmax, ymax as fractions of the photo's width and height), best first; `coordinate_system` is "normalized".

Example request:

```json
{"image": {"path": "/absolute/path/to/cats.jpg"}, "labels": ["a photo of a cat", "a photo of a dog"], "threshold": 0.1}
```

On the card's COCO photo of two cats on a sofa it returns two "a photo of a cat" boxes, one per cat, scoring 0.66 and 0.62 (the card reports 0.665 and 0.614), and no dog.

Limits: English queries only, not Polish. Each query is cut to 14 word pieces. As in the maker's post-processing there is no duplicate suppression, so one object can get two overlapping boxes. Scores run low; lower the threshold for hard queries. Small objects in a large photo get few patches.

Parity: on two photos (the card's cats, and a 1100 by 708 book photo that exercises the anti-alias blur), the boxes above 0.1 are the same set with the same labels as transformers' Owlv2ForObjectDetection on MPS fp16 (box corners within 0.0013, scores within 0.018); logits cosine 0.99994, boxes cosine 0.99995. Against an fp32 reference the MLX port is as close as PyTorch's own fp16. Details in log/003-parity.md.

Source: [google/owlv2-base-patch16-ensemble](https://huggingface.co/google/owlv2-base-patch16-ensemble/tree/cfd3195ba4ea9592eec887ded089f4c08eff231d), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: One image plus text queries; the card's example passes a list of queries per image. [HF model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: The card's example passes threshold 0.1 (not a stated recommendation), below this folder's 0.3 default. [HF model card, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/owlv2-base-patch16-ensemble/ops/detect`, or the same operation through MCP. Vendor example from the card, abridged [HF model card, undated, first-hand]:
```python
processor = Owlv2Processor.from_pretrained("google/owlv2-base-patch16-ensemble")
model = Owlv2ForObjectDetection.from_pretrained("google/owlv2-base-patch16-ensemble")
texts = [["a photo of a cat", "a photo of a dog"]]
inputs = processor(text=texts, images=image, return_tensors="pt")
outputs = model(**inputs)
target_sizes = torch.Tensor([image.size[::-1]])
results = processor.post_process_object_detection(outputs=outputs, target_sizes=target_sizes, threshold=0.1)
```

**Good and bad**
- Good at: Transformers docs say the query-free objectness score can rank or filter boxes. [HF Transformers OWLv2 docs, undated, first-hand]
- Bad at: The card has no limitations section; its data note says the training data is "more representative of people and societies most connected to the internet." [HF model card, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Not a confirmed bug. Discussion #7 (opened 30 Jan 2025) reports a local script finding one of two cats; changing the threshold switched which cat, never both. No reply is visible. [HF discussion #7, 2025-01-30, user's own run]
- Fix or workaround: Not found in sources checked.

**Evidence**
- Strength: thin. Maker's card, Transformers docs and one unanswered user discussion; no independent tests were opened.
- Sources:
  - "google/owlv2-base-patch16-ensemble", huggingface.co/google/owlv2-base-patch16-ensemble, undated, first-hand.
  - "OWLv2", huggingface.co/docs/transformers/model_doc/owlv2, undated, first-hand.
  - Discussion #7, huggingface.co/google/owlv2-base-patch16-ensemble/discussions/7, 2025-01-30, user's own run.
