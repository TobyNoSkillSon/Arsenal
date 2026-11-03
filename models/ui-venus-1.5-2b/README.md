# UI Venus 1.5 2B

UI-Venus-1.5-2B is the smallest dense model of inclusionAI's UI-Venus-1.5 GUI agent family. It is a Qwen3-VL vision-language model (2.4B parameters by Arsenal's count) further trained with large-scale GUI mid-training, offline and online reinforcement learning and model merging. Given a screenshot and a plain-language instruction, it answers with the location of the screen element the instruction refers to; Arsenal turns that into a click point in screenshot pixels.

## What it is for

The makers trained UI-Venus-1.5 as one end-to-end GUI agent for grounding, mobile and web navigation; Arsenal uses only its grounding skill. On ScreenSpot-Pro the makers report 57.7% for the 2B model (68.4% for 8B, 69.6% for 30B-A3B).

- Find where to click for an instruction such as "the Save button" or "open the settings menu" on a desktop, web or mobile screenshot.
- Drive UI automation or testing from screenshots without accessibility data.
- Check whether a described element is present on screen (an infeasible instruction returns no detection).

## Inputs and outputs

### `ground_screen` (default)

Locates the one screen element an instruction describes and returns its point, plus a box when the model gives one.

| Field | Type | Required | Notes |
|---|---|---|---|
| `image` | file | yes | `{"path":"/absolute/path"}`; accepts `image/png`, `image/jpeg`; path 1–4096 characters; at most 16 megapixels (width × height ≤ 16,000,000). The screenshot. |
| `instruction` | string | yes | 1–10000 characters. What to find, in plain words: an element description or a short task ("the search field", "turn on Wi-Fi"). |
| `max_tokens` | integer | no | 1–1024, default 256. Output budget; an answer is a few tokens. |

The instruction is placed in the makers' grounding prompt (from `ui_venus1_5_gd.py`), with one trailing full stop removed: "Output the center point of the position corresponding to the following instruction: \n{instruction}. \n\nThe output should just be the coordinates of a point, in the format [x,y]. Additionally, if the task is infeasible (e.g., the task is not related to the image), the output should be [-1,-1]." There is no system prompt, thinking is off and decoding is greedy.

Returns JSON with:

- `coordinate_system`: `pixels`.
- `image_size`: `width` and `height` of the screenshot in pixels, after its EXIF orientation is applied. All coordinates refer to this size, origin at the top-left corner, y pointing down.
- `detections`: one entry, or empty when the model answers `[-1,-1]` (infeasible). The entry has `label` (the instruction as sent), `point` (`x`, `y`: the element's centre) and, when the model answered with a box rather than a point, `box` (`xmin`, `ymin`, `xmax`, `ymax`) with `point` at its centre. There is no `score`.
- `finish_reason`: `stop`, `length` or `repetition`.

The model answers on a 0–1000 grid over the whole screenshot. Arsenal accepts the formats the makers' parsers read (`[x,y]`, `[x1,y1,x2,y2]`, `[[x1,y1],[x2,y2]]`, or bare numbers), scales x by width / 1000 and y by height / 1000, and clamps to the image; because the grid covers the whole image, the processor's internal resize does not affect the result. Pass the screenshot at its real pixel size (on a Retina display, physical pixels): to click in screen points, divide by the display's scale factor. An answer that cannot be parsed fails with `invalid_model_output`.

Arsenal reads the image with its EXIF orientation applied. The Qwen3-VL processor resizes it to between 65,536 and 12,845,056 pixels on a 32 px grid (16 px patches, merged 2 × 2) and normalises with mean and standard deviation 0.5; larger screenshots are scaled down, which can make small controls harder to hit *(unverified)*.

## Run it

MCP: `arsenal_run(model:"ui-venus-1.5-2b", input:{"image":{"path":"/absolute/path/screenshot.png"}, "instruction":"the Save button"})`

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/ui-venus-1.5-2b/ops/ground_screen" -H 'Content-Type: application/json' -d '{"image":{"path":"/absolute/path/screenshot.png"},"instruction":"the Save button"}'`

There are no settings. The precision is the one selected for the model in Arsenal (FP16, int8 or int4); requests cannot change it. One request on a 1024 × 1280 image took 0.80 s warm at FP16, 0.76 s at int8 and 0.74 s at int4 on an M5 Max, with peaks of 8.49, 5.90 and 4.37 GB (quick check, 22 Oct 2026, Mac in use, measured).

## Limits

- One element per request; it does not list every element or return several matches.
- Grounding only: the makers' multi-step mobile and web navigation (actions, trajectories) is not exposed.
- No confidence score; check an answer by looking at the point when it matters.
- The converter of the MLX checkpoint reports that the 2B model is noticeably weaker than the 8B at precise instruction-following: it can point at a section label instead of the exact control, or at a plausible rather than the best target.
- `model.toml` declares English. The makers report navigation across more than 40 Chinese mobile apps, so Chinese instructions may also work *(unverified)*.

## Licence and credit

Apache-2.0 (upstream model card, the MLX conversion's card and `model.toml`). Commercial use, modification and redistribution are allowed; keep the licence and notices and state changes.

Made by the Venus Team at inclusionAI (Changlong Gao, Zhangxuan Gu, Yulin Liu and colleagues); converted to MLX by mlx-community with mlx-vlm. The makers ask to cite:

```bibtex
@misc{venusteam2026uivenus15technicalreport,
  title={UI-Venus-1.5 Technical Report},
  author={Venus Team and Changlong Gao and Zhangxuan Gu and Yulin Liu and others},
  year={2026},
  eprint={2602.09082},
  archivePrefix={arXiv},
  primaryClass={cs.CV},
  url={https://arxiv.org/abs/2602.09082}
}
```

## Links

- Model card: https://huggingface.co/mlx-community/UI-Venus-1.5-2B-bf16/tree/d5286cd3d2580c331e2f57a976987744cd65e51d
- Original model: https://huggingface.co/inclusionAI/UI-Venus-1.5-2B
- Code: https://github.com/inclusionAI/UI-Venus
- Paper: https://arxiv.org/abs/2602.09082

## How Arsenal runs it

Maker: inclusionAI (`inclusionAI/UI-Venus-1.5-2B` model card). Converter: mlx-community (mlx-vlm 0.3.12; `tie_word_embeddings` set to false before conversion to match the published tensors). Source: `mlx-community/UI-Venus-1.5-2B-bf16@d5286cd3d2580c331e2f57a976987744cd65e51d`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`, in Arsenal's native Swift/MLX language-family worker. FP16 is stored at Get, with group-64 affine int8 and int4 derived locally. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified.

Capabilities and licence source: the inclusionAI model card. Checked live on 20 Oct at FP16, int8 and int4: both test clicks landed inside their targets.

Input files are given as absolute paths anywhere on this Mac; `arsenal_upload` is only for files that are not on this Mac.

Screen-element grounding uses the Qwen3-VL vision/text graph; normalized 0–1000 point, box-center and two-corner responses map into original screenshot pixels, independent of processor resizing. The prompt is the maker's grounding prompt with the `[-1,-1]` infeasible option (UI-Venus `192a9247ad1129279ba1d6c263d4c9e7ecef3644`, `models/grounding/ui_venus1_5_gd.py`); an infeasible answer returns no detection. Screenshots resize to at most 12,845,056 pixels, the grounding client's `max_pixels`. Interactive agent loops are not declared.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 2.4B, derived from the complete default text/image graph inventory. Source checkpoint: 2,438,696,960 stored values; imported graph: 2,438,696,960 values.

Measured figures: `results/quick-m5-max-{fp16,int8,int4}-native-exact.json` and `results/quick-checks/night-2026-10-22.json`: one Load (1.76 s FP16, 1.79 s int8, 1.49 s int4), then two Ground screen requests on a 1024 × 1280 image (first 1.39 s, 0.81 s, 0.76 s; warm 0.80 s, 0.76 s, 0.74 s); peak memory 8.49 GB FP16, 5.90 GB int8, 4.37 GB int4 (sum of worker footprints, Mac in use, quick check, not the interleaved protocol).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Not found in sources checked.
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: The README table lists the 2B model at 64.6 on ScreenSpot-Pro with "+ ZoomIn" and 57.7 without; ZoomIn is not described, and Arsenal's ground_screen has no zoom option. [inclusionAI/UI-Venus GitHub README, undated, first-hand] Issue #9 asks for offline settings; the maintainer's reply names only a vLLM 0.8.3 setup. [inclusionAI/UI-Venus #9, 2025-08-21, first-hand]
- Arsenal call: `arsenal_run(model:"ui-venus-1.5-2b", operation:"ground_screen", input:{"image":{"path":"/absolute/path/screen.png"},"instruction":"the search field"})`

**Good and bad**
- Good at: Not found beyond the maker's ScreenSpot-Pro figure, 57.7 without ZoomIn in the paper's Table 1. [arXiv 2602.09082, 2026-02-09, first-hand (maker)]
- Bad at: Navigation results were not reproduced by one user on Odyssey and Android_Control; a second user (4 Nov 2025) reported 0.78 action accuracy, low against the paper. Navigation tests, not grounding. [inclusionAI/UI-Venus #18, 2025-09-05, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: The navigation reproduction gap above. Fix or workaround: none shown; no maintainer reply. [inclusionAI/UI-Venus #18, 2025-09-05, first-hand]
- Complaint or known bug: None of the 11 open issue titles is a grounding complaint. Fix or workaround: not applicable. [inclusionAI/UI-Venus issues, checked 2026-10-31, first-hand]

**Evidence**
- Strength: thin. Grounding numbers come from the maker only. An open Hugging Face pull request titled "Add ScreenSpot-Pro evaluation result" (2026-03-17) shows no visible numbers. Reddit unreachable from search.
- Sources:
  - UI-Venus-1.5 Technical Report, https://arxiv.org/abs/2602.09082, 2026-02-09, first-hand (maker)
  - inclusionAI/UI-Venus README, https://github.com/inclusionAI/UI-Venus/blob/UI-Venus-1.5/README.md, undated, first-hand (maker)
  - inclusionAI/UI-Venus issues #9 and #18, https://github.com/inclusionAI/UI-Venus/issues, first-hand
  - UI-Venus-1.5-2B pull request, https://huggingface.co/inclusionAI/UI-Venus-1.5-2B/discussions, 2026-03-17, title only
