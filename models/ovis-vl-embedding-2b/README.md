# Ovis VL Embedding 2B

Ovis-VL-Embedding-2B is a vision-language embedding model from Alibaba's ATH-MaaS team, built on Qwen3.5-2B: 2.21B parameters, a native vision encoder feeding a 24-layer language backbone that alternates three Gated DeltaNet layers with one full-attention layer. It maps text, images, PDF pages, video and mixed content into one shared 2048-number vector space, taken from the last token's hidden state and L2-normalised, so a sentence can find a photo, a slide or a clip that means the same thing.

## What it is for

The makers optimised it for cross-modal retrieval in one compact model, training in three stages (multimodal contrastive pretraining with focal contrastive learning and similarity distillation, single-dataset fine-tuning batches, then embedding distillation). On MMEB-v2 (78 datasets) they report 77.46 overall: image 80.62, video 67.12, visual document 80.47.

- Search a photo library with a sentence ("receipt from the hardware store"), or with another photo.
- Find the PDF page or slide that answers a question, from the page image rather than extracted text.
- Find a video clip by describing what happens in it.
- Group or deduplicate mixed collections of screenshots, scans, photos and notes by meaning.

## Inputs and outputs

### `embed` (default)

Turn each input into one vector, whatever its modality.

| Field | Type | Required | Notes |
|---|---|---|---|
| `input` | array | yes | 1 to 2048 items; each returns one vector. An item is one of: a string; `{"type":"text","text":"..."}`; `{"type":"image","image":{"path":"/absolute/path"}}` (`image/png`, `image/jpeg`); `{"type":"video","video":{"path":"/absolute/path"}}` (`video/*`); `{"type":"document","document":{"path":"/absolute/path"}}` (`application/pdf`); `{"type":"content","content":[...]}` with 1 to 32 parts of the first five kinds, embedded together as one interleaved input. Paths are 1 to 4096 characters. |
| `task` | string | no | `query` or `document`. Accepted, but Arsenal frames both the same way; use `instruction` to mark queries. |
| `instruction` | string | no | Up to 32768 characters. Replaces the system message `Represent the user's input.` for every item in the request. |
| `dimensions` | integer | no | Only `2048`. |
| `normalize` | boolean | no | Only `true`. |

Returns `embeddings`: one array of 2048 numbers per input item, unit length, in input order, and `input_tokens`: the total tokens read, image, page and frame tokens included.

What each kind of input becomes:

- Image: decoded at up to 16,000,000 pixels, resized so each side is a multiple of 32 px and the area lies between 65,536 and 16,777,216 pixels, normalised with mean and standard deviation 0.5. Every 32 × 32 px block is one token, so a 1024 × 1024 photo costs 1,024 tokens (derived). Images should already be upright.
- PDF (`document`): 1 to 128 pages. Each page is rendered on white at up to twice its point size, longest side at most 2048 px, and all pages go in as images of one input, giving one vector for the whole file. To search page by page, send each page as its own file.
- Video: frames are sampled uniformly across the whole clip at 2 per second, at least 4 and at most 768 (so clips longer than 384 s are sampled more sparsely), each pair of frames getting a timestamp. All sampled frames together are resized to fit 234,881,024 pixels; at 768 frames that is about 300,000 pixels, roughly 640 × 480, per frame (derived). The audio track is ignored.
- `content`: text and media parts in the order given, in one user turn, one vector.

Every input is wrapped in the maker's chat framing: a system message (default `Represent the user's input.`, or your `instruction`), the user turn with the content, an empty non-thinking assistant turn, then `<|endoftext|>`, whose hidden state is the vector.

To search by meaning: embed the collection once (images, pages, clips, notes) with no instruction and store the vectors. At search time embed the query, text or image, with a one-sentence `instruction` that describes the task, for example `Find the photo that matches this description.` Score each stored item by cosine similarity, which for these unit vectors is the dot product, and keep the top results. A text reranker such as `qwen3-reranker-8b` can reorder text candidates; it cannot read images. Vectors from this model are comparable only with vectors from this model.

## Run it

MCP: `arsenal_run(model:"ovis-vl-embedding-2b", input:{"input":["a dog catching a frisbee on a beach"],"instruction":"Find the photo that matches this description."})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/ovis-vl-embedding-2b/ops/embed" -H 'Content-Type: application/json' -d '{"input":[{"type":"image","image":{"path":"/absolute/path/photo.jpg"}},{"type":"document","document":{"path":"/absolute/path/slides.pdf"}}]}'`

Precision (FP16, int8, int4) is chosen in Arsenal's model table; embed a collection and its queries at the same precision. Warm requests on three short sentences took 0.11 to 0.13 s on an M5 Max in use (quick checks, measured); images, PDFs and video take longer in proportion to their tokens. Inputs in one request run one after another; the request times out after 180 s.

## Limits

- No audio; the makers point to their Omni embedding model for that.
- One input holds at most 131,072 tokens (Arsenal's bound; the makers publish none), about 68 letter-size PDF pages at the 2048 px render. Longer inputs are refused before rendering.
- One vector per PDF, not per page.
- No smaller vector sizes.
- The makers say results depend on task-appropriate instructions and the native preprocessing, that video question answering, general video retrieval and ViDoRe-V2 trail the strongest specialist models, and that benchmark scores may not transfer to new domains.
- Languages are not stated by the makers.
- Int4 quality has not been compared with FP16 on trained weights.

## Licence and credit

Apache-2.0. Commercial use is allowed; keep the licence and notices, and state changes if you redistribute modified weights. Made by the ATH-MaaS team (Ovis-Embedding Team). Cite their technical report: Ovis-Embedding Team, "Ovis-Embedding: Pushing the Frontiers of Universal Omni-Modal Embeddings", arXiv:2609.25165, 2026.

## Links

- Model card: https://huggingface.co/ATH-MaaS/Ovis-VL-Embedding-2B/tree/855b64a8246cf83b34730e4470ef66949e030659
- Code: https://github.com/ATH-MaaS/Ovis-VL-Embedding
- Paper: https://arxiv.org/abs/2609.25165

## How Arsenal runs it

Source: `ATH-MaaS/Ovis-VL-Embedding-2B@855b64a8246cf83b34730e4470ef66949e030659`. Immutable filenames, byte sizes and SHA-256 hashes are in `model.toml` and `gate/source.json`. The source revision predates the repository's folder move and has exactly the same weight SHA-256 as the current nested checkpoint. All source tensors are BF16. The serving default is FP16: Get streams each BF16 tensor into the FP16 base snapshot (`cast:embed-ovis-v2`). Int8 and int4 derive every eligible 2-D Linear/Embedding matrix with MLX affine group size 64; learned norms and spatial kernels stay half.

Runner: the native Swift MLX embed worker, sequential requests only, macOS 26 or later, ported from `ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57` and adapted inside `Workers/Embed` (see its `PORTED-FROM.md` and `LICENSE-upstream`). Text, images, PDF page images, sampled video and interleaved content share the Qwen3.5 hybrid DeltaNet/full-attention backbone; the last non-padding hidden state is pooled and L2-normalised, with no projection head and no audio tower. Query framing follows the maker's ms-swift `qwen3_5_emb` template (`args.json`): default system `Represent the user's input.`, empty non-thinking assistant turn, pooled `<|endoftext|>` suffix. Video uses the pinned `processor_config.json` video section: uniform sampling over the complete clip, the final frame repeated only when needed to fill a temporal pair, each temporal group with its reference timestamp and its own vision block, and resize limits applied to the complete sampled clip. Vision attention runs per page or frame segment, without a global square mask. The 131,072-token bound admits every input the maker's processor configuration produces (the whole-clip video budget of 234,881,024 px is 114,688 tokens); the model's position table is 262,144.

Review corrections (`log/002-review.md`): vision MLP uses tanh GELU; video framing, three-axis positions, seven resize cases and three sampling cases match small reference tensors captured once from Transformers v5.12.1 without weights; a four-frame clip catches zero-sample compressed reader markers; cancellation is checked after each evaluated language layer. All 13 family tests pass, and release worker smokes pass at FP16, int8 and int4 with text, image, PDF, video and interleaved fixtures; malformed-input, cancellation, packed-coverage and unload checks pass. Real header names and shapes match 617 served parameters.

Checked live on 20 Oct 2026 at FP16, int8 and int4: text-to-photo search found the right photo 3 of 3 times. Quick checks, 22 Oct 2026, M5 Max, Mac in use, three short sentences per request (`results/`): FP16 load 2.76 s, warm 0.11 s, peak 5.99 GB; int8 load 2.76 s, warm 0.12 s, peak 3.29 GB; int4 load 2.35 s, warm 0.13 s, peak 1.92 GB, peaks summed over Arsenal's worker processes.

Gate (`gate/gate.toml`): exact checkpoint inventory and every declared operation on synthetic weights pass; trained-weight reference and precision qualification are pending, so FP16 conversion and quantized retrieval quality are unqualified, as is long-context memory use. Logs: `log/001-port.md`, `log/002-review.md`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: text, images, visual documents, video and interleaved content in one 2048-number space; no audio. [model card, ATH-MaaS/Ovis-VL-Embedding-2B, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Retrieval quality depends on task-appropriate query instructions and the native preprocessing and chat template. [model card, undated, first-hand]
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `arsenal_run(model:"ovis-vl-embedding-2b", input:{"input":["receipt from the hardware store"],"instruction":"Find the photo that matches this description."})` [ops/embed.input.json, checked 2026-10-31, first-hand]

**Good and bad**
- Good at: Image and visual-document retrieval on MMEB-v2, 80.62 and 80.47, against the card's best compared baselines of 77.41 and 79.86. [model card, undated, first-hand]
- Bad at: The MMEB-v2 video group, 67.12 against 68.84 for the best baseline; the card warns that scores may not predict a new domain. [model card, undated, first-hand]
- Against the obvious alternatives: Overall 77.46 against 75.42 for the best compared baseline, in the card's table. No third-party comparison found. [model card, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: Not found in sources checked. The only GitHub issue (#1, opened 28 Oct 2026, open) asks for a recipe to reproduce the MMEB numbers and reports no fault. Fix or workaround: Not found in sources checked. [ATH-MaaS/Ovis-VL-Embedding issues, 2026-10-28, opinion]

**Evidence**
- Strength: thin. All figures are the maker's. The arXiv abstract (submitted 21 Sep 2026) claims state of the art on MMEB-v2; the PDF was not opened. The discussions tab is empty. Reddit not checked.
- Sources:
  - Paper abstract, https://arxiv.org/abs/2609.25165, submitted 2026-09-21, first-hand (maker).
  - Model card, https://huggingface.co/ATH-MaaS/Ovis-VL-Embedding-2B, undated, first-hand (maker).
  - Discussions, https://huggingface.co/ATH-MaaS/Ovis-VL-Embedding-2B/discussions, checked 2026-10-31, none.
  - Issues, https://github.com/ATH-MaaS/Ovis-VL-Embedding/issues, 2026-10-28, opinion.
