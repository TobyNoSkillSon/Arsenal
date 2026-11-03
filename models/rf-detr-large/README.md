# RF-DETR Large

RF-DETR Large is the most accurate Apache-licensed size of Roboflow's RF-DETR real-time detection transformers (ICLR 2026). It joins a DINOv2-style ViT backbone that alternates windowed and full attention, a multi-scale projector and a four-layer deformable DETR decoder. It has 33.9M parameters and works at 704 × 704. It finds and boxes objects from the 80 COCO classes, end to end and without non-maximum suppression. It is the same network as RF-DETR Nano at a larger input and with two more decoder layers.

## What it is for

The makers report 56.5 COCO AP50:95 and 75.1 AP50 at 6.8 ms on an NVIDIA T4 (Nano: 48.4 and 67.6 at 2.3 ms). These are their figures, not reproduced here.

- Find and count people, vehicles, animals and everyday objects in photos, including smaller or more distant ones than Nano finds.
- Pre-filter large photo sets by what they contain.
- Get boxes to crop, blur or feed into a segmentation model such as SAM.

## Inputs and outputs

### `detect` (default)

Finds objects of the 80 COCO classes in one image and returns their labels, scores and boxes.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | `{"path":"/absolute/path"}`; accepts `image/png`, `image/jpeg`; path 1–4096 characters; at most 16 megapixels (width × height ≤ 16,000,000). |
| `labels` | list of strings | no | 1–512 items, each one of the 80 COCO class names below. Keeps only these classes; leave out for all. |
| `threshold` | number | no | 0–1, default 0.5. Minimum score to keep a detection. |

The class names, exactly as accepted and returned: person, bicycle, car, motorcycle, airplane, bus, train, truck, boat, traffic light, fire hydrant, stop sign, parking meter, bench, bird, cat, dog, horse, sheep, cow, elephant, bear, zebra, giraffe, backpack, umbrella, handbag, tie, suitcase, frisbee, skis, snowboard, sports ball, kite, baseball bat, baseball glove, skateboard, surfboard, tennis racket, bottle, wine glass, cup, fork, knife, spoon, bowl, banana, apple, sandwich, orange, broccoli, carrot, hot dog, pizza, donut, cake, chair, couch, potted plant, bed, dining table, toilet, tv, laptop, mouse, remote, keyboard, cell phone, microwave, oven, toaster, sink, refrigerator, book, clock, vase, scissors, teddy bear, hair drier, toothbrush.

Returns JSON with `coordinate_system` `normalized` and `detections`, sorted by score, highest first, at most 300 entries. Each detection has:

- `label`: the COCO class name.
- `score`: 0–1, the sigmoid of the class logit.
- `box`: `xmin`, `ymin`, `xmax`, `ymax`, each 0–1 as a fraction of the original image's width (x) or height (y), origin at the top-left corner, y pointing down, clamped to the image. Multiply by the image's pixel width and height to get pixels.

There is no `point` field, no mask and no keypoints. The detector scores every one of its 300 object queries against every class, and Arsenal keeps each query-class pair above the threshold without non-maximum suppression, as DETR models are trained to work; occasionally one object can appear twice with different labels or overlapping boxes.

Arsenal reads the image as 8-bit RGB, scales it to 0–1 and then resizes it to 704 × 704 with bilinear interpolation (no antialiasing, aspect ratio not kept), exactly as the checkpoint's own processor does, and normalises with ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). Because boxes are normalised to the whole image, the squash does not distort them. EXIF orientation is not applied: the image is read in its stored pixel orientation, so rotate a camera JPEG first if its orientation tag matters.

## Run it

MCP: `arsenal_run(model:"rf-detr-large", input:{"image":{"path":"/absolute/path/photo.jpg"}, "labels":["person","dog"], "threshold":0.35})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/rf-detr-large/ops/detect" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/photo.jpg"}}'`

There are no settings. Requests run one at a time and time out after 180 s. Speed and peak memory are not measured yet.

## Limits

- Only the 80 COCO classes; it cannot find anything named in free text.
- Very wide or tall images are squashed to a square, which can lower accuracy *(unverified)*.
- No segmentation masks or keypoints from this checkpoint.
- The model card's example keeps detections above 0.35; Arsenal's default of 0.5 drops more weak detections.

## Usage notes (2 Nov 2026)

- What it does: one `detect` call per image; returns COCO labels, scores and normalised boxes, highest score first.
- Inputs: a PNG or JPEG path, optional `labels` to keep, optional `threshold` (0 to 1, default 0.5). The image is scaled to 0–1, resized to 704 × 704 with bilinear interpolation (no antialiasing) and normalised with ImageNet mean and standard deviation, as the checkpoint's own processor does.
- Precision: FP16, a plain cast of the pinned FP32 checkpoint (largest weight 36.9, far inside FP16's range). No 8-bit or 4-bit copies.
- Parity with the maker (measured 2 Nov 2026, Apple M5 Max): the reference is the model card's route, Transformers 5.22.1 `RfDetrForObjectDetection` with `model.half()` on the Apple GPU, on the card's own COCO two-cats photo. Each part was fed the maker's own input tensors. Relative L2 differences: backbone 0.0014 to 0.0023 across its four outputs, projector 0.0009, proposal scores 0.0002, decoder logits 0.0052, decoder boxes 0.0002 for confident queries (0.030 over all 300 queries, where background boxes move with half-precision rounding; the maker's own BF16 and FP16 runs differ by 0.26 there). End to end, both find the same five objects (two cats, two remotes, a couch): scores within 0.005 and boxes within 0.0003 of the image size. Bounds and their history are in `gate/golden/golden.json`.
- Real run (2 Nov 2026, after the preprocessing fix): Arsenal's own service on an isolated profile; Get cloned the pinned files from the local Hugging Face cache (0 bytes downloaded) and cast them to FP16; `detect` on the same photo returned the maker's five objects (cat 0.948, remote 0.941, cat 0.939, remote 0.834, couch 0.605; maker 0.949, 0.942, 0.941, 0.834, 0.606), scores within 0.002 and boxes within 0.0005. Preprocessing matches the maker's processor within 1e-5 on PNG fixtures; on a JPEG the two decoders can differ by a level or two.
- Fixed while porting: the runner hard-coded Nano's 384 px and two decoder layers, and its top-300 query selection ordered tied FP16 scores differently from the maker. Each rank carries its own learned query, so that order changes detections. Both fixes also apply to Nano.
- The model card's printed example (cat 0.986, remote 0.99) was made with `stevenbucaille/rf-detr-large` at unknown precision. This checkpoint's maker run gives 0.949 and 0.942 on the same photo; the gap is not explained *(unverified)*.

## Licence and credit

Apache-2.0. Roboflow states that RF-DETR Nano, Small, Medium and Large detection are commercially usable, including in closed-source products, without a licence fee. Keep the licence and notices and state changes. (XLarge and larger sizes use Roboflow's Platform Model License and are not shipped here.)

Made by Isaac Robinson, Peter Robicheaux, Matvei Popov, Deva Ramanan and Neehar Peri (Roboflow); contributed to Transformers by stevenbucaille. The makers ask to cite:

```bibtex
@misc{robinson2026rfdetrneuralarchitecturesearch,
  title={RF-DETR: Neural Architecture Search for Real-Time Detection Transformers},
  author={Isaac Robinson and Peter Robicheaux and Matvei Popov and Deva Ramanan and Neehar Peri},
  year={2026},
  eprint={2511.09554},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://huggingface.co/papers/2511.09554}
}
```
