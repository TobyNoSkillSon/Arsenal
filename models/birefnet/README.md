# BiRefNet

BiRefNet (Bilateral Reference Network) is a 220M-parameter segmentation model by Peng Zheng and colleagues, built for high-resolution dichotomous image segmentation: splitting an image into one foreground object and its background at pixel level. It pairs a Swin Transformer encoder with a decoder that uses a bilateral reference to the source image to keep fine detail. Arsenal uses it to cut the main subject out of a photo and return a transparent PNG plus its alpha mask.

## What it is for

The makers optimised it for accurate, detailed masks at high resolution; the pinned checkpoint is the standard DIS model, trained on DIS-TR and validated on DIS-TEs and DIS-VD, and runs at 1024 × 1024. The same architecture reports state-of-the-art results on high-resolution salient object detection and camouflaged object detection.

- Remove the background from a product photo, portrait or object shot.
- Produce an alpha mask for compositing, cut-outs or selective edits in another tool.
- Isolate the salient object in an image before further processing.

## Inputs and outputs

### `remove_background` (default)

Separates the main subject from the background and returns the cut-out and its mask.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | `{"path":"/absolute/path"}`; accepts `image/png`, `image/jpeg`; path 1–4096 characters; at most 16 megapixels (width × height ≤ 16,000,000) |

Returns two PNG files, both at the input's own width and height, in this order:

- `foreground` (order 0, label "Foreground"): 8-bit RGBA PNG. The input's colours with alpha set to the predicted mask; an alpha channel already in the input is multiplied by the mask.
- `mask` (order 1, label "Alpha mask"): 8-bit greyscale PNG, white = foreground, black = background, with soft grey values at edges.

Each file entry carries `id`, `path`, `mime` (`image/png`), `width`, `height`, `order`, `label`, `bytes` and `sha256`; `metadata` carries `model`, `revision`, `dtype` and `engine` (`mlx-swift`).

The model always works at 1024 × 1024. Arsenal reads the image's straight 8-bit RGB (alpha is ignored for the prediction; greyscale and other colour formats are converted to RGB), resizes it to 1024 × 1024 without keeping the aspect ratio using Pillow-compatible antialiased bilinear filtering, and normalises it with ImageNet mean (0.485, 0.456, 0.406) and standard deviation (0.229, 0.224, 0.225). The 1024 × 1024 sigmoid mask is quantised to 8 bits and resized back to the source size with bicubic filtering, matching the makers' ToPILImage-then-resize route. EXIF orientation is not applied: the image is processed in its stored pixel orientation, so the caller should rotate a camera JPEG first if its orientation tag matters. Files other than `.png`, `.jpg` or `.jpeg`, or above 16 megapixels, are refused.

## Run it

MCP: `arsenal_run(model:"birefnet", input:{"image":{"path":"/absolute/path/photo.jpg"}})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/birefnet/ops/remove_background" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/photo.jpg"}}'`

There are no settings. One image takes about 0.45 s warm at FP16 on an M5 Max with a peak of about 5.8 GB (quick check, 16 Oct 2026, busy Mac); a 1024 × 1280 photo took 0.48 s and 5.7 GB (22 Oct 2026). Requests run one at a time and time out after 180 s.

## Limits

- One foreground per image: it predicts a single subject-versus-background mask, not separate objects or classes.
- Fine detail is limited by the fixed 1024 × 1024 working size; very large or very wide images are squashed to that square, so thin structures can soften. The makers publish separate higher-resolution and dynamic-resolution checkpoints that Arsenal does not ship.
- It is a segmentation model, not a matting model: semi-transparent material such as glass, smoke or fine veils is not given true partial transparency *(unverified)*.
- EXIF orientation is ignored (see above).
- Memory is dominated by the 1024 × 1024 activations, about 5–6 GB regardless of input size.

## Licence and credit

MIT. Commercial use, modification and redistribution are allowed provided the copyright and licence notice are kept. The Swift MLX port is also MIT; Pillow-derived resampling code is under Pillow's licence (`runners/swift/LICENSE-Pillow`).

Made by Peng Zheng, Dehong Gao, Deng-Ping Fan, Li Liu, Jorma Laaksonen, Wanli Ouyang and Nicu Sebe. The makers ask to cite:

```bibtex
@article{zheng2024birefnet,
  title={Bilateral Reference for High-Resolution Dichotomous Image Segmentation},
  author={Zheng, Peng and Gao, Dehong and Fan, Deng-Ping and Liu, Li and Laaksonen, Jorma and Ouyang, Wanli and Sebe, Nicu},
  journal={CAAI Artificial Intelligence Research},
  volume = {3},
  pages = {9150038},
  year={2024}
}
```

## Links

- Model card: https://huggingface.co/ZhengPeng7/BiRefNet/tree/e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4
- Code: https://github.com/ZhengPeng7/BiRefNet
- Paper: https://arxiv.org/pdf/2401.03407
- Swift MLX port: https://github.com/mnmly/mlx-swift-BiRefNet

## How Arsenal runs it

Source: `ZhengPeng7/BiRefNet` at `e2bf8e4460fc8fa32bba5ea4d94b3233d367b0e4`. The pinned checkpoint (`model.safetensors`, 444,473,596 bytes) and `config.json` stay byte-identical and are hash-checked on Get. The Swift MLX engine is bundled as native family `birefnet`; see `runners/swift/PORTED-FROM.md` and its licences. Python remains a reference-only tool.

Input files are given as absolute paths anywhere on this Mac; Arsenal stages them for the worker. `arsenal_upload` is only for files that are not on this Mac. Results are returned as file paths, and can also be fetched through their HTTP URLs or MCP resource URIs.

FP16 is the source tier and the only one offered. Int8 and int4 are withdrawn: a quick check on the M5 Max (22 Oct 2026, Mac in use) found them no faster and at most 9% smaller, because activations dominate memory: FP16 0.48 s per image and 5.7 GB peak, int8 0.47 s and 5.3 GB, int4 0.47 s and 5.2 GB (`log/006-tiers-withdrawn.md`).

The 16 Oct intake that qualified FP16 also derived the withdrawn tiers with `intake-v1:birefnet` (affine group-64, every weight matrix with at least 65,536 parameters and input width divisible by 64; OIHW convolutions, norms, biases and small matrices stayed FP16). Run through `arsenal-stats intake` on a model-folder copy under `lab/models/birefnet`, it captured 20 outputs at both Swin scales and the squeeze/decoder boundaries on exact saved FP16 inputs, with relative L2 ≤0.05 / cosine ≥0.999 quantized bounds frozen first, no fallback search and no held-out claim, plus a diagnostic source-size mask check (mean absolute error ≤0.02) on three fixed synthetic images. Int8 passed all 20 outputs; int4 failed its component gate. The FP16 table result (0.45 s per image, 5.76 GB peak; warm served ABBA BAAB on a fixed 512 × 384 image, busy Mac, quick check) and the schema-3 quality binding come from that run's completion, on the combined native worker on M5 Max/40 GPU cores, with zero capture requests in serving workers.

FP16 preservation is separately recorded in `gate/evidence/native/fp16-preservation.json`: the unchanged b77f8aa5 component probe and loader versus the final implementation consume the same 16 saved input tensors. All 20 output tensors are byte-identical (relative L2 0, cosine 1, maximum absolute error 0), including a bridge to the intake's saved outputs. Both source closures, executables, input/output identities, keys and shapes are retained. The current/current A/A reproducibility check remains separately labelled; neither replaces the vendor gate.

Gate: `gate/evidence/native/components.json` (a lab record, not shipped in this folder) records 12 independent boundary calls, 20 exact-shape FP16 outputs against saved vendor PyTorch/MPS FP16 outputs. Every output is finite; worst relative L2 is below 0.007 (decoder 0.000435, largest stage 0.006828). This is a component correctness check, not broad segmentation-quality evaluation. The original lab API screenshot is recorded in `gate/evidence/native/api.json` and the review-fix rerun in `api-r1.json` (lab records, not shipped in this folder). Image boundaries use Pillow-compatible bilinear antialiasing for straight RGB input and byte-quantized bicubic output-mask resizing. Small fixed fixtures check these boundaries against Pillow; internal feature-map interpolation retains the vendor's aligned-corner geometry.

Lab host: use the repository's headless `arsenal-testserver --real` with isolated ARSENAL_SUPPORT_DIR/ARSENAL_CACHE_DIR. Copy the existing signed catalog plus public key into support/real-resources; drop this folder into support/local-models, approve its exact content digest, Add, Get FP16 from the verified local donor, Load and call the op. Do not use --prepare-real: it signs a new QA catalog and has a fixed model allowlist. The lab smoke script `gate/smoke-http.py` (not shipped in this folder) implements this route; it runs one inference then retrieves its two files through HTTP. MCP covers discovery, approval, Add/Get/Load and operation discovery; the recorded smoke does not cover MCP result-resource retrieval. Lab weights and caches are deleted afterward.

Research logs: `log/001-integration.md` through `log/006-tiers-withdrawn.md`. Historical Python-path work is superseded; see commits 76aee14a and 94bb5b1a. That attempt's quantized results do not qualify or describe this native engine.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The maker's example also uses 1024 × 1024 and resizes the mask back to the source size [BiRefNet model card, undated, first-hand].
- Prompts, reference audio or voice settings: Not applicable.
- Parameters people recommend: Run FP16. The maker says to "Validate the success of FP16 inference with ~0 decrease of performance and better efficiency" [BiRefNet GitHub README, news entry dated 2025-01-06, first-hand].
- Arsenal call: `arsenal_run(model:"birefnet", input:{"image":{"path":"/absolute/path/photo.jpg"}})`

**Good and bad**
- Good at: On 126 anime character images, BiRefNet scored 95.3% pixel accuracy and 88.5% mean boundary IoU. Pose (99.1%) and factory (98.7%) were strongest [ToonOut, 2025-09, first-hand].
- Bad at: Action scenes scored 76.8% pixel accuracy on that set [ToonOut, 2025-09, first-hand]. In a dealership team's pipeline, thin antennas were missed or broken up, and racks and tubes had soft edges [BiRefNet issue #315, 2026-07-22, first-hand].
- Against the obvious alternatives: ViTMatte with a trimap is described as more accurate on hair, fur and translucent edges, but the trimap is a hard requirement [Background Removal Library Landscape, 2026-04, opinion].

**Complaints, bugs and fixes**
- Complaint or known bug: Thin structures are missed or broken up, with soft edges. Fix or workaround: none confirmed. The reporter asks whether higher-resolution training, trimap refinement or tiled inference would help; no comments or maintainer reply [BiRefNet issue #315, 2026-07-22, first-hand].

**Evidence**
- Strength: strong. The maker documents the model, and one outside test covers 126 anime images. The rest are user reports.
- Sources:
  - ToonOut, https://arxiv.org/html/2509.06839v1, 2025-09-08, first-hand.
  - BiRefNet GitHub README, https://github.com/ZhengPeng7/BiRefNet, undated (quoted news entry 2025-01-06), first-hand.
  - BiRefNet model card, https://huggingface.co/ZhengPeng7/BiRefNet, undated, first-hand.
  - BiRefNet issue #315, https://github.com/ZhengPeng7/BiRefNet/issues/315, 2026-07-22, first-hand.
  - Background Removal Library Landscape, https://ice-ice-bear.github.io/posts/2026-04-13-matting-libraries, 2026-04-13, opinion.
