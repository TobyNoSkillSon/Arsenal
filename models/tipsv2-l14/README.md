# TIPSv2 L/14

TIPSv2 (Text-Image Pre-training with Spatial awareness) L/14 is a contrastive image-text model from Google: a 24-layer ViT-L vision encoder with 14-pixel patches (303M parameters) and a 12-layer text encoder (184M), 488M in all, that put images and short English texts in one 1024-number vector space. Besides one vector for the whole image it gives a vector for every patch that lines up with text, so it can say not only whether a picture shows "a bicycle" but roughly where.

## What it is for

The makers optimised it for dense patch-text alignment, through patch-level distillation, an iBOT++ masked-image objective in which unmasked tokens also count, and synthetic captions at several levels of detail. They report results on par with or better than recent vision encoders across 9 tasks and 20 datasets covering classification, retrieval, segmentation and depth.

- Search a photo collection by typing what you are looking for, or by giving a similar photo.
- Label a photo against a list of candidate tags without training (zero-shot classification).
- Mark which pixels of a photo belong to each of several named things (zero-shot segmentation).

## Inputs and outputs

### `detect` (default)

Score an image against text labels and return the labels that pass the threshold. Despite the name, it returns no boxes.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file `{"path":"/absolute/path"}` | yes | `image/png` or `image/jpeg`; path 1 to 4096 characters. |
| `labels` | array of strings | yes | 1 to 256 labels, each at least 1 character. |
| `threshold` | number | no | -1 to 1. Default 0. |

Returns `detections`: `{"label","score"}` for every label whose score is at or above `threshold`, highest first, and `coordinate_system` `normalized`. The score is the cosine similarity in [-1, 1] between the image vector and the label's text vector, not a probability; compare labels with each other rather than reading the value as a confidence. Each label is embedded exactly as written, as in the model card's classification example, so `a photo of a dog` and `dog` can score differently.

### `embed`

Turn texts and images into vectors in the shared space.

| Field | Type | Required | Notes |
|---|---|---|---|
| `input` | array | yes | 1 to 256 items. Each item is a string (1 to 32768 characters), `{"type":"text","text":"..."}` (1 to 32768 characters) or `{"type":"image","image":{"path":"/absolute/path"}}` (`image/png`, `image/jpeg`). |
| `normalize` | boolean | no | Arsenal's default is `true` (unit-length vectors). |

Returns `embeddings`: one array of 1024 numbers per input, in input order, and `input_tokens`: the text tokens read (images count 0). An image vector is the image's class token; a text vector is the text encoder's masked mean over its tokens.

### `segment`

Assign every pixel of an image to the best-matching of the given labels.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file `{"path":"/absolute/path"}` | yes | `image/png` or `image/jpeg`; path 1 to 4096 characters. |
| `labels` | array of strings | yes | 1 to 256 labels, each at least 1 character. |

Returns `files`: one PNG mask per label, in label order, at the input image's width and height, 8-bit greyscale with 255 where that label won and 0 elsewhere; `id` is `label-<n>` from 0 and `name` is `<label>.png`. Every pixel belongs to exactly one label, so include a `background` label or similar when the labels do not cover the whole picture. The decision is made on a 32 × 32 patch grid and upsampled bilinearly, so edges are soft and small objects can vanish.

Images, for all three operations: decoded at up to 16,000,000 pixels, stretched to 448 × 448 without keeping the aspect ratio (as the card's `Resize((448, 448))`), scaled to 0 to 1 per channel with no mean or standard-deviation normalisation. Images should already be upright. Text: lowercased, NFKC-normalised, whitespace folded, then SentencePiece-encoded and cut to the first 64 tokens without warning; write short captions, not paragraphs.

To search photos by meaning: embed every photo once as `{"type":"image",...}` and store the vectors. At search time embed the typed query (or a reference photo) and score each stored photo by cosine similarity, the dot product for the default unit vectors; keep the top results. Vectors from this model are comparable only with vectors from this model.

## Run it

MCP: `arsenal_run(model:"tipsv2-l14", input:{"image":{"path":"/absolute/path/photo.jpg"},"labels":["a dog","a cat","a bicycle"]})` (add `operation:"embed"` or `operation:"segment"` for the others).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key):

- `curl -s "$ARSENAL_URL/arsenal/v1/models/tipsv2-l14/ops/detect" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/photo.jpg"},"labels":["a dog","a cat"],"threshold":0.1}'`
- `curl -s "$ARSENAL_URL/arsenal/v1/models/tipsv2-l14/ops/embed" -H 'Content-Type: application/json' -d '{"input":["a red bicycle against a wall",{"type":"image","image":{"path":"/absolute/path/photo.jpg"}}]}'`
- `curl -s "$ARSENAL_URL/arsenal/v1/models/tipsv2-l14/ops/segment" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/photo.jpg"},"labels":["sky","building","road","person"]}'`

There are no settings. A warm `embed` of three short sentences took 0.04 s on an M5 Max in use (quick check, measured). Each operation times out after 180 s.

## Limits

- Short English text: 64 tokens, the card's maximum. The card names no languages; Arsenal lists English only *(unverified for other languages)*.
- Images are squashed to a square, so very wide or tall images lose detail on their long side.
- `detect` gives tags and cosine scores, not boxes or calibrated confidences.
- `segment` is coarse (14 × 14 px patches at the 448 px working size) and always picks one of the given labels.
- At most 256 inputs or labels per request.
- The trained DPT heads for depth, surface normals and segmentation are separate checkpoints and are not included.

## Licence and credit

Apache-2.0. Commercial use is allowed; keep the licence and notices, and state changes if you redistribute modified weights. Made at Google by Bingyi Cao, Koert Chen, Kevis-Kokitsi Maninis, André Araujo and coauthors. Cite:

```bibtex
@inproceedings{cao2026tipsv2,
  title     = {{TIPSv2: Advancing Vision-Language Pretraining with Enhanced Patch-Text Alignment}},
  author    = {Cao, Bingyi and Chen, Koert and Maninis, Kevis-Kokitsi and Chen, Kaifeng and Karpur, Arjun and Xia, Ye and Dua, Sahil and Dabral, Tanmaya and Han, Guangxing and Han, Bohyung and Ainslie, Joshua and Bewley, Alex and Jacob, Mithun and Wagner, Rene and Ramos, Washington and Choromanski, Krzysztof and Seyedhosseini, Mojtaba and Zhou, Howard and Araujo, Andre},
  booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2604.12012}
}
```

## Links

- Model card: https://huggingface.co/google/tipsv2-l14/blob/52847a71eb02a3082c370fbcd98cd799687bd4d0/README.md
- Code: https://github.com/google-deepmind/tips
- Paper: https://arxiv.org/abs/2604.12012
- Demo: https://huggingface.co/spaces/google/TIPSv2

## How Arsenal runs it

Checkpoint: `google/tipsv2-l14@52847a71eb02a3082c370fbcd98cd799687bd4d0` (`model.safetensors`, `config.json`, `tokenizer.model`). Exact filenames, sizes and SHA-256 hashes are in `model.toml`. The source tensors are FP32; Get casts them once to FP16 (`cast:tipsv2-affine-v1`), and all checkpoint tensors load intact after that cast, with norms and spatial kernels kept half. Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.

Runner: the native Swift MLX perception worker, sequential requests only, macOS 26 or later. Swift port: `mnmly/mlx-swift-tips@222f5fa5133b6ee9272f2878fd84b7ec3cff8a17`; adapted files and source terms are in `Workers/Perception/licenses/MLXTIPS/PORTED-FROM.md`. Image vectors use the class token and text vectors the masked mean. Zero-shot labels follow the card's classification example: the label text itself, cosine similarity against the first class token, every label at or above `threshold` returned in score order. Zero-shot segmentation follows the official recipe (`pytorch/TIPS_zeroshot_segmentation.ipynb`): last-block value-attention patch features, each label embedded as the mean of the nine TCL prompt templates, cosine scores bilinearly upsampled to the image and the best label taken per pixel, one PNG mask per label. The image is stretched to 448 × 448 as in the card; the notebook's 336-pixel sliding window over a short-side resize is not used.

Tokeniser: the native SentencePiece reader comes from Arsenal's existing MLXAudioCore implementation. NFKC normalisation, whitespace folding and lowercasing precede its 64-token, no-BOS/no-EOS encoding. Exact vendor tokenizer parity, especially for unusual Unicode and unknown tokens, remains a live-test risk. Embedding batches are bounded to 256 inputs so 1,024-dimensional JSON vector replies fit the 16 MiB worker frame limit. Preprocessing uses the decoded pixel extent.

Verification (`log/001-integration.md`, `gate/header-check.json`, `gate/no-weights-results.json`): header inspection matches all 491 runtime parameter names and shapes; the production header checker passes with explicit per-tensor FP32-to-FP16 cast plans. The full graph has 145 eligible group-64 matrices (99.55% of learned parameters, derived from the inventory). All three operations passed at FP16, int8 and int4 on the real worker with a shrunk graph, as did malformed-input, cancellation, session-closure and unload checks. This establishes wiring, not trained quality; trained-weight vendor parity is pending (`gate/gate.toml`).

Quick check, 22 Oct 2026, M5 Max, Mac in use: load 0.42 s; warm `embed` on three short sentences 0.04 s; peak memory 2.20 GB summed over Arsenal's worker processes (`results/`).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: The paper's segmentation protocol compares upscaled patch features with each class name's text embedding by cosine similarity. [arXiv 2604.12012, 2026-04, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: Not a recommendation. The paper says DINOv2 and SILC report sliding-window segmentation numbers, which generally boost their scores; its DINOv3 comparison reproduces that setup. [arXiv 2604.12012, 2026-04, first-hand]
- Arsenal call: `arsenal_run(model:"tipsv2-l14", operation:"segment", input:{"image":{"path":"/absolute/path/photo.jpg"},"labels":["sky","building","road","person","background"]})`

**Good and bad**
- Good at: Zero-shot segmentation mIoU of 37.1 on PC59, 44.4 on VOC21 and 24.7 on ADE150 (maker's table, not reproduced). [arXiv 2604.12012, 2026-04, first-hand]
- Bad at: Not found in sources checked.
- Against the obvious alternatives: In the paper's zero-shot segmentation table, TIPSv2 L/14 scores 37.1 on PC59 against 33.5 for the earlier TIPS L/14 and 30.9 for DINOv2 (dino.txt) L/14, which uses the sliding-window protocol. [arXiv 2604.12012, 2026-04, first-hand (maker)]

**Complaints, bugs and fixes**
- Complaint or known bug: None found. The google-deepmind/tips repo showed no open issues, and the Hugging Face google/tipsv2-l14 page shows one merged pull request, "Update files for transformers integration" (2026-06-22), and no open discussion. Fix or workaround: none needed in sources checked. [github.com/google-deepmind/tips/issues, 2026-10-31, first-hand]

**Evidence**
- Strength: thin. Results come from the maker's paper; no outside test of segmentation or detection found. Reddit was not reachable from search.
- Sources:
  - TIPSv2 paper, https://arxiv.org/abs/2604.12012, submitted 2026-04-13, first-hand (maker)
  - google/tipsv2-l14 discussions, https://huggingface.co/google/tipsv2-l14/discussions, 2026-06-22, first-hand (one merged pull request)
  - google-deepmind/tips issues, https://github.com/google-deepmind/tips/issues, 2026-10-31, first-hand (no open issues)
