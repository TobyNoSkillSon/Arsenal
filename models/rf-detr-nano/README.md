# RF-DETR Nano

RF-DETR Nano is the smallest Apache-licensed size of Roboflow's RF-DETR real-time detection transformers. It joins a DINOv2-style ViT backbone that alternates windowed and full attention, a multi-scale projector and a deformable DETR decoder; the family's sizes were found by neural architecture search over accuracy and latency. It has 30.5M parameters and works at 384 × 384. It finds and boxes objects from the 80 COCO classes, end to end and without non-maximum suppression, in a few hundredths of a second.

## What it is for

The makers optimised it for real-time speed at strong COCO accuracy: they report 48.4 COCO AP50:95 and 67.6 AP50 at 2.3 ms on an NVIDIA T4 (TensorRT, FP16, batch 1).

- Find and count people, vehicles, animals and everyday objects in photos.
- Pre-filter large photo sets by what they contain (for example, all photos with a dog).
- Get boxes to crop, blur or feed into a segmentation model such as SAM.
- Run quick detection on many frames where a larger open-vocabulary detector would be too slow.

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

Arsenal reads the image as 8-bit RGB, scales it to 0–1 and then resizes it to 384 × 384 with bilinear interpolation (no antialiasing, aspect ratio not kept), exactly as the checkpoint's own processor does, and normalises with ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). Because boxes are normalised to the whole image, the squash does not distort them. EXIF orientation is not applied: the image is read in its stored pixel orientation, so rotate a camera JPEG first if its orientation tag matters.

## Run it

MCP: `arsenal_run(model:"rf-detr-nano", input:{"image":{"path":"/absolute/path/photo.jpg"}, "labels":["person","dog"], "threshold":0.4})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/rf-detr-nano/ops/detect" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/photo.jpg"}}'`

There are no settings. One 1024 × 1280 photo took 0.044 s warm (0.61 s first request) at FP16 on an M5 Max, with a peak of 0.39 GB (quick check, 22 Oct 2026, Mac in use, measured). Requests run one at a time and time out after 180 s.

## Check, 2 Nov 2026

Preprocessing now follows the checkpoint's own processor (Transformers `RfDetrImageProcessor`: scale to 0–1, bilinear resize without antialiasing, normalise); it used to resize the 8-bit image with CoreGraphics' filter. The query selection also orders tied scores as the maker does. Headless check on the installed FP16 weights against the maker's Transformers code (FP16, Apple GPU) on the COCO two-cats photo: the same four objects (cat 0.962, remote 0.913, remote 0.910, cat 0.888; maker 0.962, 0.912, 0.910, 0.892), boxes within 0.002.

## Limits

- Only the 80 COCO classes; it cannot find anything named in free text. Use an open-vocabulary detector for other objects.
- The fixed 384 × 384 input makes small or distant objects in large photos easy to miss *(unverified)*.
- Very wide or tall images are squashed to a square, which can lower accuracy *(unverified)*.
- No segmentation masks or keypoints from this checkpoint; Roboflow ships those as separate models.
- The upstream example uses a 0.35 threshold; Arsenal's default of 0.5 drops more weak detections.

## Licence and credit

Apache-2.0. Roboflow states that RF-DETR Nano, Small, Medium and Large detection are commercially usable out of the box, including in closed-source and self-hosted products, without a licence fee. Keep the licence and notices and state changes. (Other RF-DETR sizes, not shipped here, use Roboflow's Platform Model License.)

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

## Links

- Model card: https://huggingface.co/Roboflow/rf-detr-nano/tree/9f201577d9415f0378084748d27e68586e6b3600
- Code: https://github.com/roboflow/rf-detr
- Paper: https://huggingface.co/papers/2511.09554
- Transformers integration: https://github.com/huggingface/transformers/pull/36895

## How Arsenal runs it

Checkpoint: `Roboflow/rf-detr-nano@9f201577d9415f0378084748d27e68586e6b3600`. Exact filenames, sizes and SHA256 hashes are recorded in `model.toml`. No weight payload was downloaded during integration. Labels come from the checkpoint's `id2label`, which keeps COCO's original sparse 91-index label space; the unused indices (`N/A`) are never returned.

All checkpoint tensors are loaded intact after the mandatory FP16 cast. Norms and spatial kernels remain half. Checked live at FP16 on 19 Oct: found both cats and both remotes in the COCO two-cats photo.

Swift port: `mnmly/mlx-swift-rf-detr@83941f0461313cdd1991d6873e205f613ff1be31`, run in Arsenal's native perception worker. Adapted files and source terms: `Workers/Perception/licenses/MLXRFDETR/PORTED-FROM.md`.

Input files are given as absolute paths anywhere on this Mac; `arsenal_upload` is only for files that are not on this Mac. Non-finite scores or boxes fail the request instead of returning detections.

Header inspection matches all 464 runtime parameter names and shapes. The production header checker passes with explicit per-tensor FP32→FP16 Get cast plans; these plans do not establish trained cast quality. The full graph has 159 eligible group-64 matrices (89.90% of learned parameters, derived from the inventory); all eligible matrices in its shrunk graph execute at int8 and int4. See `gate/header-check.json` and `gate/no-weights-results.json`.

The unused pretrained DINO mask token is explicitly ignored; it is absent from the detection forward path. The checkpoint uses two decoder layers, 300 queries, a 384-pixel input and the original sparse COCO indices. The pinned config independently selects feature layers `[2,5,8,11]` and global attention layers `[3,6,9]` (zero-based). Fixed tensor regressions check window/global attention against scalar reference math and positional embeddings against rf-detr `5b22dfc` in `y,x,w,h` order.

Random graph, header-only inventory and precision coverage establish wiring, not trained model quality. See `log/001-integration.md` for the actual checks and remaining gaps.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.

Measured figures: `results/quick-m5-max-fp16-native-exact.json` and `results/quick-checks/night-2026-10-22.json` (one Load in 1.07 s, then two requests on a 1024 × 1280 photo, 0.61 s and 0.044 s; peak 370 MiB summed worker footprint; Mac in use, not the interleaved protocol). The benchmark suite in `bench/bench.toml` is deferred.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: the model card's example keeps detections above 0.35 (a code comment). [Roboflow/rf-detr-nano model card, published 2025-11-12, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable; a detector.
- Parameters people recommend: Raising `num_select` alone returns duplicate boxes (same object, different label), not new objects; `num_queries` must be raised too. [RF-DETR FAQ, 2026-09-23, first-hand]
- Arsenal call: `arsenal_run(model:"rf-detr-nano", input:{"image":{"path":"/absolute/path/street.jpg"},"labels":["person","car"],"threshold":0.35})`

**Good and bad**
- Good at: Not found in sources checked.
- Bad at: Not found in sources checked.
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Upstream rf-detr v1.6.5.post0 failed to load `RFDETRSegNano` at resolution 1008 (issue opened 2026-05-02). A 10 May comment by yeldarby says the same crash hits `RFDETRNano(resolution=...)`. [roboflow/rf-detr issue #1023, 2026-05-02 to 2026-05-13, first-hand]
- Fix or workaround: Closed 2026-05-13 as completed in PR #1026, a regression-test PR. The 10 May comment names PR #840 (in 1.7.0rc0) as the fix. Whether either touches the 384 px path is not checked. [same issue, first-hand]
- Complaint or known bug: At 512 px Nano gives "significantly higher latency", per a commenter the mirror identifies as the RF-DETR author. Fix or workaround: none tested; the comment recommends larger variants. [r/computervision ExecuTorch post via a third-party mirror, 2025-08 inferred, opinion]

**Evidence**
- Strength: thin. Maker docs, one bug thread, one Reddit comment; no independent accuracy test.
- Sources:
  - Roboflow/rf-detr-nano model card, https://huggingface.co/Roboflow/rf-detr-nano, published 2025-11-12 per the paper listing, first-hand (maker)
  - RF-DETR FAQ, https://rfdetr.roboflow.com/latest/faq/, 2026-09-23, first-hand (maker)
  - roboflow/rf-detr issue #1023, https://github.com/roboflow/rf-detr/issues/1023, 2026-05-02 to 2026-05-13, first-hand
  - r/computervision ExecuTorch post, via a third-party mirror (reddit.sentinel-team.org), 2025-08 inferred; first-hand for the post, opinion for the 512 px comment
