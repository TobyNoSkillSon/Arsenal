# Grounding DINO Base

Finds the things you describe and draws a box around each one. Give it an image and what to look for ("a cat. a remote control.", or a list of labels); it returns every box whose best-matching words score above the threshold, labelled with the phrase it matched. IDEA Research's open-set detector: a Swin-B image backbone and a BERT text encoder whose features are fused through six encoder layers, then 900 candidate boxes refined by six decoder layers.

Input: `image` (PNG or JPEG), and what to find as `labels` (a list) or `phrase` (split on full stops, commas and semicolons); `threshold` (0 to 1, default 0.3). Output: `detections`, each with `label`, `score` and `box` (xmin, ymin, xmax, ymax as fractions of the photo's width and height), best first; `coordinate_system` is "normalized".

Example request:

```json
{"image": {"path": "/absolute/path/to/cats.jpg"}, "phrase": "a cat. a remote control."}
```

On the card's COCO photo of two cats on a sofa it returns the two cats (0.49 and 0.43) and the two remote controls (0.47 and 0.34), the same four boxes as the maker's code.

Limits: English only, not Polish. Labels are lowercased and joined as "a. b." for the model, at most 128 things and 256 tokens per request. As in the maker's post-processing there is no duplicate suppression. Each box gets the one phrase whose words score highest, where the maker's own post-processing can return a partial phrase ("cartoon" for "cartoon drawing"). Needs more memory than OWLv2.

Parity: on two photos (the card's cats and a book photo), the encoder's image and text states match transformers' GroundingDinoForObjectDetection on MPS fp16 with cosine 0.99987 or better, and the detections above the threshold are the same boxes with the same labels; against the maker's fp32 run the scores differ by at most 0.0024 and the boxes by at most 0.0011. Details in log/003-parity.md.

Source: [IDEA-Research/grounding-dino-base](https://huggingface.co/IDEA-Research/grounding-dino-base/tree/12bdfa3120f3e7ec7b434d90674b3396eccf88eb), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Lowercased text ending in a dot; the card's code comment says so. [IDEA-Research/grounding-dino-base model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: box_threshold 0.4, text_threshold 0.3 on the card; the repo demo uses 0.35 and 0.25. [model card, undated, first-hand; GroundingDINO GitHub README, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/grounding-dino-base/ops/detect`, or the same operation through MCP. Vendor example (docs' tiny ID): `processor(images=image, text=[["a cat", "a remote control"]], return_tensors="pt")`, `model(**inputs)`, then `processor.post_process_grounded_object_detection(outputs, inputs.input_ids, threshold=0.4, text_threshold=0.3, target_sizes=[image.size[::-1]])`. [Transformers Grounding DINO docs, undated, first-hand]

**Good and bad**
- Good at: Maker's headline: 52.5 AP on COCO zero-shot; no figure for this checkpoint. [model card, undated, first-hand]
- Bad at: Labels can be cut to part of a phrase; the maker says text_threshold picks the tokens ("a chest of drawers" came back as "a chest"). [discussion #9, 2024-10-23, user report, maintainer reply]
- Against the obvious alternatives: Not found in sources checked. [model card, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Partial labels, as above. Fix or workaround: None confirmed; the maker says they "could probably" return both prompts. [discussion #9, 2024-10-23, opinion]
- Complaint or known bug: Comma-separated labels gave no detections, or only the last one. Fix or workaround: None stated; open, no reply. [GroundingDINO GitHub issue #432, 2025-09-28, opinion]

**Evidence**
- Strength: thin. Maker card, docs and README are first-hand; both complaints lack a maintainer fix. No independent test was opened.
- Sources:
  - IDEA-Research/grounding-dino-base model card, https://huggingface.co/IDEA-Research/grounding-dino-base, undated, first-hand.
  - Transformers Grounding DINO docs, https://huggingface.co/docs/transformers/en/model_doc/grounding-dino, undated, first-hand.
  - GroundingDINO GitHub README, https://github.com/IDEA-Research/GroundingDINO, undated, first-hand.
  - Discussion #9, https://huggingface.co/IDEA-Research/grounding-dino-base/discussions/9, 2024-10-23, opinion.
  - GroundingDINO issue #432, https://github.com/IDEA-Research/GroundingDINO/issues/432, 2025-09-28, opinion.
