# d1 · 3B

d1-3B is Liquid AI's 3.12B-parameter decision model, post-trained from the LFM2.5-VL-3B vision-language model: a hybrid language model of 30 convolution and attention layers with a SigLIP2 NaFlex vision encoder (400M), a 32,768-token context and a 128,000-token vocabulary. Given a state (text, JSON, images, or a mix) and a set of typed questions, it reads the answer from the logits of a single forward pass and generates no tokens. It is the only Arsenal decision model that can look at pictures, and the one with the longest context.

## What it is for

Liquid AI post-trained it for single-pass, calibrated decisions and reports it as the best model under 10B on the Decision Index 0.2.1 (48.57) *(vendor figure)*. The card recommends it wherever a pipeline needs a yes/no, a pick from named options or a rating; it is not a chat model and does not write text. Its advertised languages are Arabic, Chinese, English, French, German, Hindi, Indonesian, Italian, Japanese, Korean, Polish, Portuguese, Russian, Spanish, Thai and Vietnamese.

- Routing, triage and intent or topic classification.
- Moderation, agent guardrails and extraction checks.
- Reranking and LLM-as-a-judge scoring.
- Visual inspection: questions about photos, screenshots or scans, with or without text beside them.
- Long states: documents, transcripts or JSON records up to the 32,768-token context.

## Inputs and outputs

### `decide` (default)

Answers up to 64 named, typed questions about one state, with optional pictures, in one pass.

| Field | Type | Required | Notes |
|---|---|---|---|
| `state` | any JSON value | yes | Text/JSON state; `null` for images alone. Objects with `image`, `images`, `audio`, `video` or `videos` keys are refused: pictures go in `images`. |
| `questions` | object | yes | Question name → question, 1 to 64 (`minProperties` 1, `maxProperties` 64). Names come back as the answers' keys; they are not shown to the model. Each question is described below. |
| `images` | array of `{"path":"/absolute/path"}` | no | At most 10 (`maxItems` 10). Existing absolute local image paths, any file on this Mac. |
| `model` | string | no | `minLength` 1. `d1-3b`; omit on this model's operation endpoint. |
| `bits` | integer or null | no | One of `0`, `4`, `8`, `16`, `null`. Required precision; 0 accepts the loaded dtype, 16 matches bf16, 8 matches int8 and 4 matches int4. A mismatch returns 409 without changing the loaded dtype. |
| `merge` | boolean | no | Must be `true` (`const`): the vendor shared-state decision tree is always used. |

A question is an object with these fields (unknown fields are ignored):

| Field | Type | Required | Notes |
|---|---|---|---|
| `type` | string | yes | `noul` = yes/no (answer P(yes)); `choice` = one of named options; `score` = an ordered level. |
| `instructions` | string, object or array | no | The question, as text or structured JSON. Defaults to empty. |
| `criteria` | depends on `type` | `choice` and `score`: yes | `noul`: optional object with only `true` and `false`, each text, JSON or null. `choice`: object, option label → description (text, JSON, or null for the bare label), 1 to 64 (`maxProperties` 64); labels are compared as exact strings. `score`: array of level descriptions, lowest first, 1 to 10 (`maxItems` 10); levels are answered as `"0"`, `"1"`, …. |

Returns JSON `{"answers": {...}, "model": "d1-3b", "usage": {"input_tokens": n, "output_tokens": 0}}`. Each answer has `type` and:

- `noul`: `noul`, the probability of yes, 0 to 1.
- `choice`: `choice` (the label with the highest probability), `confidence` (that highest probability) and `probabilities` (label → probability, summing to 1 over the options sent).
- `score`: `score`, the expected level (the sum of level index × probability), `confidence` (the probability of the most likely level), `probabilities` (`"0"`, `"1"`, … → probability) and `legend` (`"0"`, `"1"`, … → the level descriptions as sent).

How d1 reads a question, from the vendor's prompt code, which Arsenal reproduces exactly: the user turn holds the pictures, then the state (a string as is, any other JSON as `json.dumps` with indent 2, keys in the order sent), then `QUESTION:` and the question block, then the assistant header, where the answer token would go. A choice block is the instructions, then `Options:` with one line per option, `<code> <text>`, and "Reply with the option code only."; the codes are the labels themselves when every label is a single letter, otherwise A, B, C… (00, 01… past 26 options), and the text is the description, or the label with underscores read as spaces when the description is null. A noul block is the instructions, then `Yes: <true>` and `No: <false>` when criteria are given, and "Reply with yes or no only."; a score block is the instructions, one `<index> <description>` line per level and "Reply with a single digit 0-N only.". The probabilities are a softmax over each option's best answer-token logit (the option code, `yes`/`Yes`/`YES` against `no`/`No`/`NO`, or the level digit), computed in FP32. Several questions over one state form one tree: the state is read once and each question branches from it. So write the instructions as the question, give options short labels with plain descriptions, and keep a score to at most 10 levels.

Pictures: each file is decoded by macOS ImageIO to 8-bit RGB (alpha dropped), capped at 1,048,576 pixels with the vendor's bicubic resize, then prepared by the checkpoint's LFM2-VL processor: a large picture is split into a grid of 512-pixel tiles (2 to 10) plus a thumbnail, a small one is resized whole, at 64 to 256 image tokens per piece. The pictures of one request may retain at most 128 MiB of capped pixels; a picture past that budget fails as `invalid_image`. State, pictures and the longest question must fit in 32,768 tokens, otherwise the request is refused with `context_limit`.

### `judge`

Asks the same questions about up to 256 items, each its own state with its own pictures.

| Field | Type | Required | Notes |
|---|---|---|---|
| `items` | array | yes | 1 to 256 (`maxItems` 256). One text/JSON state per item; `null` with images. Input order is preserved. |
| `questions` | object | yes | As in `decide`, 1 to 64. Batch extension: a `choice` question's `criteria` may also be a plain list of labels (strings). |
| `images` | array of arrays of `{"path":"/absolute/path"}` | no | One image list per item, in the same order; its length must match `items`. At most 256 lists of at most 10 pictures. |
| `model` | string | no | `d1-3b`; omit on this model's operation endpoint. |
| `bits` | integer or null | no | As in `decide`. |

Returns JSON `{"results": [...]}`, one per item in item order. A result carries the same `answers` and `usage` as `decide` plus `model` and `ms` (the time of the item's pass divided by its items); an item that cannot be prepared (bad picture, too long) gets `{"error": "...", "model": null, "ms": 0}` while the others are answered. Items run in passes of up to 32.

## Run it

MCP: `arsenal_run(model:"d1-3b", input:{"state":null,"images":[{"path":"/absolute/path/photo.jpg"}],"questions":{"cats":{"type":"choice","instructions":"How many cats are there?","criteria":{"one":"One","two":"Two","more":"Three or more"}}}})` (add `operation:"judge"` with `items` for a batch).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/d1-3b/ops/decide" -H 'Content-Type: application/json' -d '{"state":"I was charged twice this month, please refund one of them.","questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}}'`

TypeSafe System One route (batches at `/v1/judge`), which needs `model`: `curl -s "$ARSENAL_URL/v1/systemone" -H 'Content-Type: application/json' -d '{"model":"d1-3b","state":"I was charged twice this month, please refund one of them.","questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}}'`

The model has no settings; precision (BF16, int8, int4) is chosen in Arsenal's model card. Measured on an M5 Max in a quick check on a busy Mac with one 33-token decision: BF16 21.3 ms and 5.90 GB peak, int8 17.5 ms and 3.49 GB, int4 20.4 ms and 2.15 GB (`results/intake-*.json`). A request with one picture took 0.13 to 0.89 s in an earlier quick check, depending on the picture's size (`log/007-swift-runner.md`).

## Limits

- No audio or video, and no remote image URLs: pictures are local files only.
- Decisions only: it does not write text or explain its answer.
- Scores take at most 10 levels: each level is read as a single digit. The vendor defines scores for 2 to 10 levels; Arsenal also accepts one.
- JPEG pictures decode slightly differently from the vendor's Pillow path (up to 43/255 at sharp colour edges on a synthetic drawing); PNG decoding is bit-exact.
- int4 fails Arsenal's component quality bound (largest error in the vision projector) and is offered with a quality warning; int8 passes.
- On a 32-ad Polish job-ad probe, d1 found 0 of 11 mandatory requirements with a six-way choice and 6 of 11 with a yes/no question; do not use it as a literal requirement check (`log/002-polish-probe.md`, corrected in `log/004-mlx-runner.md`).
- Vendor speed and benchmark figures (30 ms per decision on an M5 Pro, Decision Index 48.57) are Liquid AI's, not Arsenal measurements.

## Licence and credit

LFM Open License v1.0 (`LicenseRef-LFM-Open-1.0`), retained at `runners/swift/LICENSE-LFM-Open-1.0` and pinned in the checkpoint; the ported prompt code is marked modified. Commercial use is licensed only below a US$10 million annual-revenue threshold for the legal entity, including controlled affiliates; above it a separate Liquid AI licence is required. The text defines the threshold as $10M or more while section 5 says "exceeds"; at the boundary ask Liquid AI. Redistribution must include the licence, preserve attribution and mark modified files. This is not an unrestricted open-source licence. Made by Liquid AI, who ask for these citations:

```bibtex
@article{liquidAI2026opend1,
  author  = {Liquid AI},
  title   = {Open d1: Edge decision models for text, vision, and audio},
  journal = {Liquid AI Blog},
  year    = {2026},
  note    = {https://www.liquid.ai/blog/d1-open},
}

@article{liquidai2025lfm2,
  title   = {LFM2 Technical Report},
  author  = {Liquid AI},
  journal = {arXiv preprint arXiv:2511.23404},
  year    = {2025}
}
```

## Links

- Model card: https://huggingface.co/LiquidAI/d1-3B/blob/051bcc464b01b9f92942b364d9586b0ef5912432/README.md
- Blog post: https://www.liquid.ai/blog/d1-open (the card's own `/blog/open-d1` link returned 404 when checked)
- Base model: https://huggingface.co/LiquidAI/LFM2.5-VL-3B
- Paper (LFM2 Technical Report): https://arxiv.org/abs/2511.23404

## How Arsenal runs it

Source: `LiquidAI/d1-3B` at `051bcc464b01b9f92942b364d9586b0ef5912432`, vendor BF16 (`model.safetensors` 6.25 GB plus tokenizer, processor config and chat template; exact sizes and hashes in `model.toml`). Get installs the verified BF16 weights; int8 and int4 are derived at Get as affine group-64 packs (`mlx-affine-g64:d1-3b`) streamed from the BF16 source, covering 305 matrices and 2,988,652,544 parameters including the token embedding; small tensors, positions, convolutions, norms and 27 vision down-projections whose width is not divisible by 64 stay BF16 (`log/009-int8-intake.md`). There is no Python environment.

Runner: the Swift package in `runners/swift`, linked into the app's native worker as family `d1` (`log/007-swift-runner.md`). It ports the vendor's `prompt.py` and `api.py`, including Python's `json.dumps(indent=2)`, `str()`/`repr()` of non-text instructions, option aliases and readout ids; its byte-level BPE tokenizer matched `tokenizers` 0.22 and the vendor prompt code on 3,029 stress strings and all 139 gate and probe requests (25,013 tokens). The forward is MLX BF16: packed decision trees with per-tree attention, a pruned tail after the last attention layer, fused `w1|w3` and `q|k|v` GEMMs and a fused short-convolution Metal kernel (bit-identical to stock ops, with a load self-test). Option logits are read in FP32 because the vendor's BF16 head rounds near ties into exact ties. The SigLIP2 tower, projector and image processor run in MLX BF16, loaded on the first picture. Exact and Fast use the same arithmetic. The earlier torch-MPS and Python MLX runners are removed: torch 2.12's MPS causal attention is wrong in BF16 (0.41–0.46 relative error from about 100 tokens), so the vendor code on MPS is not a reference (`log/001-port.md`, `log/003-reference-fp32.md`, `log/004-mlx-runner.md`).

Gate (`gate/gate.toml`, `gate/parity.py`): the runner over ArsenalWire against the pinned vendor code in FP32 on the CPU, saved once in `gate/reference-fp32.json`. Max |Δp| 0.0039 and 24/24 answers on the gate set; pictures 0.0218, 20/20 (one three-way near tie); prompt edge cases 0.0072, 10/10; probe ads 0.0122, 94/96 (two genuine near ties); requirement sections 0.0187 and 0.0106, 72/72 (`gate/parity-swift.json`, unchanged after the review fixes in `gate/parity-swift-fixes.json`, `log/008-swift-review-fixes.md`). The BF16 floor, the same vendor code in BF16 on the CPU, is 0.0172 on the gate set. `gate/reference-check.json` compares MPS, CPU BF16, MLX BF16 and quantized variants with FP32.

Speed (`log/007-swift-runner.md`, 8 October 2026, M5 Max, Mac in use, interleaved over the wire): one decision of one job ad × 3 questions 61.9 ms median (44.4 ms best); 32 ads × 3 questions 67.4 ms per ad; 32 ads × 1 question 29.0 ms per ad; pictures 64×64, 640×480 and 1600×1000 took 264, 130 and 891 ms; load 0.52 s; footprint 5.84 GB after load, 6.97 GB after pictures. Precision intake (`log/010-default-load-readout-intake.md`, 15 October 2026, busy Mac, 93 saved-input components plus three probability vectors): BF16 21.257 ms and 5.896 GB; int8 passes (max relative L2 0.0195, max |Δp| 0.0009, 17.479 ms, 3.493 GB); int4 fails the component bound (0.2433, largest at the projector; max |Δp| 0.0035; 20.397 ms, 2.153 GB). Earlier trials of int8 and int4 over the language model alone were slower than BF16 on batches and int4 changed 6 of 96 ad answers (`log/005-quantized-weights.md`); a split-K down projection was rejected (`log/006-split-down-projection.md`). The suite-v1 quick recipe result is in `results/m5-max.json` (not a controlled benchmark).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: One forward pass with zero output tokens, from a state that can be text, JSON or images [LiquidAI/d1-3B model card, undated, first-hand].
- Prompts, reference audio or voice settings: The card gives no wording advice beyond the typed-question format [model card, undated, first-hand].
- Parameters people recommend: Not found in sources checked for sampling; the model generates no tokens. The card requires transformers 5.14 or later; its example uses bf16 on any non-CPU device [model card, undated, first-hand].
- Arsenal call: `arsenal_run(model:"d1-3b", input:{"state":"I was charged twice this month, please refund one of them.","questions":{"refund":{"type":"noul","instructions":"Is the customer asking for a refund?"}}})`

**Good and bad**
- Good at: The card's uses are routing, moderation, classification, reranking and judging. Its vision average is 74.1, against 73.9 for the base model [model card, undated, first-hand, vendor-run].
- Bad at: The card's table puts it below Decider 4B on HelpSteer2 (36.7 vs 42.0), BoolQ (86.7 vs 89.0) and XNLI (85.0 vs 88.6). Without images its vision questions score 45.1 [model card, undated, first-hand, vendor-run].
- Against the obvious alternatives: The vendor says it ranks first under 10B on Decision Index 0.2.1, at 48.57 [model card, undated, first-hand, vendor-run].

**Complaints, bugs and fixes**
- Complaint or known bug: A KuCoin news item dated 2026-09-30 gives d1 a Decision Index score of 58.9, attributed to a Liquid re-evaluation. The card gives 48.57. Fix or workaround: none found; cite the card [KuCoin flash news, 2026-09-30, opinion].

**Evidence**
- Strength: thin. Only vendor documentation and vendor-run results were found. Standard and extended searches found no independent evaluation.
- Sources:
  - LiquidAI/d1-3B model card, https://huggingface.co/LiquidAI/d1-3B, undated, first-hand (vendor).
  - Open d1, https://huggingface.co/blog/LiquidAI/open-d1, 2026-10-07, first-hand (vendor).
  - KuCoin flash news, https://www.kucoin.com/news/flash/liquid-ai-s-d1-decision-model-surpasses-jev-in-hugging-face-evaluation, 2026-09-30, opinion.
