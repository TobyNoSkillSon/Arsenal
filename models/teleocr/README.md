# TeleOCR

TeleOCR (formerly NaviDC-OCR) is a small vision-language model for document parsing by Peng Cai and colleagues, built on the Qwen2.5-VL architecture: a vision encoder feeding a small language model that writes out what it reads. Arsenal counts 1.4B parameters in the pinned checkpoint (the makers describe it as about 1.2B). It reads both clean digital pages and photographed, curved or skewed pages in one model, turning a page into layout blocks, text, HTML tables, LaTeX formulas and code, in reading order.

## What it is for

The makers optimised it for full-page document parsing across digital and camera-captured documents in Chinese and English, including layout analysis of distorted pages without a separate dewarping step; they report the top overall score among the models they compare on OmniDocBench v1.6 (96.87) and Wild_OmniDocBench (88.53).

- Turn a PDF or a scanned or photographed page into Markdown in reading order.
- Extract tables as HTML, including tables implied by charts in scientific figures.
- Transcribe formulas to LaTeX and code screenshots to text.
- Find the layout of a page (titles, text, tables, figures, captions, headers and footers) with pixel boxes.
- Read the text of seals and stamps.

## Inputs and outputs

### `read_document` (default)

Reads a PDF or an image page by page in one of eleven modes and returns per-page Markdown, layout blocks and tables.

| Field | Type | Required | Notes |
|---|---|---|---|
| `document` | file | yes | `{"path":"/absolute/path"}`; accepts `application/pdf`, `image/png`, `image/jpeg`; path 1–4096 characters. A PDF has at most 1024 pages; an image is at most 16 megapixels (width × height ≤ 16,000,000) and is one page. |
| `pages` | list of integers | no | 1–10000 items, each 0–10000. Zero-based PDF page indices, processed in the order given; leave out for every page. Ignored for images. |
| `mode` | string | no | `ocr` (default), `document`, `distorted_document`, `text`, `table`, `formula`, `code`, `seal`, `scientific_figure`, `layout`, `distorted_layout`. See below. |
| `max_tokens` | integer | no | 1–32768, default 4096. Output budget for each single generation: each layout call and each block's recognition call gets its own budget, not the page as a whole. |

The modes fall into three groups.

- Full-page parsing: `ocr`, `document` (the same pipeline) and `distorted_document`. TeleOCR first analyses the layout of a 1036 × 1036 bicubic resize of the page (`distorted_document` uses the makers' multi-point segmentation prompt, which returns polygons for curved or skewed regions). Arsenal then cuts each block from the full-resolution page (polygon or rectangle, rotated upright by the predicted orientation), and runs a recognition call per block with the prompt for its type. This follows the makers' client (NaviDC-OCR `TeleOCR_client.py`).
- Whole-page single tasks: `text`, `table`, `formula`, `code`, `seal` and `scientific_figure` send the whole page with one prompt and return the model's answer. Use them on an image that holds only that content (a cropped table, one formula, one code block, one seal).
- Layout only: `layout` and `distorted_layout` return the layout blocks with boxes and no recognised text.

Returns JSON `pages`, one entry per page, each with:

- `index`: the zero-based page index (0 for an image).
- `markdown`: the page as Markdown. In full-page parsing, blocks are joined in reading order; titles become `#` headings, code is fenced, formulas are wrapped in `$$…$$`, tables are HTML `<table>` markup, and scientific-figure text appears only when longer than five characters. Page furniture (`header`, `footer`, `page_number`, `aside_text`, `page_footnote`), `seal`, `unknown`, `image` and `list` blocks are left out of the Markdown but kept in `blocks`. In single-task modes it is the model's answer: plain text for `text`, `code` and `seal`, raw LaTeX without `$$` for `formula`, and an HTML table (converted from the model's OTSL output) for `table` and `scientific_figure`. Empty in layout modes.
- `blocks` (full-page parsing and layout modes): each with `category`, `text` (recognised content; empty in layout modes and for `image` and `list` blocks, which are not sent to recognition), `coordinate_system` `pixels` and `box` (`xmin`, `ymin`, `xmax`, `ymax`). Boxes are in pixels of the page image described by `dimensions`, origin top-left, y down; for a polygon the box is its bounding rectangle. The model emits coordinates on a 0–1000 grid over the page and Arsenal scales them to pixels. Categories: text, title, table, image, code, algorithm, header, footer, page_number, page_footnote, aside_text, equation, ref_text, list, phonetic, table_caption, image_caption, code_caption, table_footnote, image_footnote, unknown, seal, char (a scientific figure's implied table). `equation_block` blocks are dropped.
- `tables` (full-page parsing): each table or scientific-figure block's HTML as `{id, content}`, `id` `table-<block number>`. Empty in layout modes; not present in single-task modes.
- `dimensions`: `width` and `height` in pixels of the page as the model saw it (for an image, its own size after EXIF orientation; for a PDF, the rendered page).
- `truncated` and `finish_reason` (`length` or `repetition`): present only when a generation on that page stopped at `max_tokens` or at a detected loop.

PDF pages are rendered at 144 dpi (2 px per PDF point), capped at 8192 px on the long side and 16,777,216 pixels in total. Image files are read with their EXIF orientation applied. Each image sent to the model (page, crop or layout square) is resized by the Qwen2.5-VL processor to between 3,136 and 12,845,056 pixels on a 28 px grid and normalised with the CLIP mean and standard deviation. Before recognition, crops with an aspect ratio beyond 50:1 are padded with white, and crops whose short side is under 28 px are upscaled.

## Run it

MCP: `arsenal_run(model:"teleocr", input:{"document":{"path":"/absolute/path/report.pdf"}, "pages":[0,1]})` (add `"mode":"table"` and the like for a single task)

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/teleocr/ops/read_document" -H 'Content-Type: application/json' -d '{"document":{"path":"/absolute/path/page.png"},"mode":"ocr"}'`

There are no settings; decoding is fixed to the makers' client: greedy, the system prompt "You are a helpful assistant.", presence penalty 1.0 with frequency penalty 0.05 on text, formula and code calls and 0.005 on table, scientific-figure and seal calls, no repetition penalty, and a 100-token no-repeat n-gram block on every call including layout. The precision is the one selected for the model in Arsenal (FP16, int8 or int4); requests cannot change it. A short fixed test page took 0.76 s warm at FP16, 0.79 s at int8 and 0.79 s at int4 on an M5 Max (quick check, 22 Oct 2026, Mac in use, measured); full pages with many blocks take one generation per block and run much longer.

## Limits

- Chinese and English only (model card).
- Full-page parsing does not recognise `list` and `image` blocks: their text is empty and they are left out of the Markdown, following the makers' pipeline. Figures are not returned as image files.
- A page whose layout output holds no valid lines fails with `invalid_model_output`; malformed individual lines are skipped.
- Single-task modes expect an image of just that content; on a full page they can miss or mix content *(unverified)*.
- PDFs are rasterised; embedded PDF text is not used.
- Pages over 12,845,056 pixels are downscaled before reading, so very small print on huge scans can be lost *(unverified)*.
- Seal recognition comes from the makers' client prompt; the model card does not describe or benchmark it.

## Licence and credit

Apache-2.0 (model card and `model.toml`). Commercial use, modification and redistribution are allowed; keep the licence and notices and state changes.

Made by Peng Cai, Zhaofan Zou, Shifa Liu, Yikun Wang, Jiawei Tang, Kaicheng Yang, Meng Tong, Zhongjiang He and Hao Sun; the makers credit Qwen2.5-VL, Qwen3 and MinerU. They ask to cite:

```bibtex
@article{teleocr,
  title={TeleOCR: Navigating Document Parsing Across Digital and Camera-Captured Documents},
  author={Cai, Peng and Zou, Zhaofan and Liu, Shifa and Wang, Yikun and Tang, Jiawei and Yang, Kaicheng and Tong, Meng and He, Zhongjiang and Sun, Hao},
  journal={arXiv preprint arXiv:2608.12898},
  year={2026}
}
```

## Links

- Model card: https://huggingface.co/XingChen-AGI/TeleOCR/tree/70a19ec6a3b036311af19ae5e762e650f41a5980
- Code: https://github.com/caipeng328/TeleOCR
- Paper: https://arxiv.org/abs/2608.12898

## How Arsenal runs it

Source: `XingChen-AGI/TeleOCR@70a19ec6a3b036311af19ae5e762e650f41a5980`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`, in Arsenal's native Swift/MLX language-family worker. FP16 is stored at Get, with group-64 affine int8 and int4 derived locally. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified.

Capabilities, language and licence source: the pinned model card. Checked live on 20 Oct at FP16, int8 and int4: three text lines and a three-column table read exactly.

Input files are given as absolute paths anywhere on this Mac; `arsenal_upload` is only for files that are not on this Mac.

Each document mode preserves the maker's prompt. Full-page modes run layout → polygon/rectangle crops and rotation → type-specific recognition → reading-order Markdown, tables and pixel blocks. Orchestration follows NaviDC-OCR `1e71f4fe792d12bbb86d2671c5ed6f3a1499b27d`, Markdown assembly its `vlm_middle_json_mkcontent.py`. Document orchestration details from the same reference: malformed layout lines are skipped (a page with none is `invalid_model_output`); crops narrower than 1:50 are padded with white and short edges below 28 px are upscaled before recognition; Markdown turns titles into headings, fences code and leaves page furniture (headers, footers, page numbers, side text, footnotes), seals and short figure text in `blocks` only.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 1.4B, derived from the complete default text/image graph inventory. Source checkpoint: 1,415,072,768 stored values; imported graph: 1,415,072,768 values.

Live fixes, 20 Oct 2026 (`log/002-decode-positions.md`): decode steps now keep the prefill's MROPE position deltas (before this, the first text line was lost and table cells shifted); text RoPE runs in the activation dtype, which halves KV memory to about 115 KB per token; the MLX buffer cache is capped at 256 MB so long generations stay flat; decoding penalties match the makers' client. After the fix, two pages (three text lines; a 4 × 3 table) read exactly at every tier, with peak worker footprints of 5.66 GB at FP16, 4.26 GB at int8 and 3.54 GB at int4 (max_tokens 512, quick check, M5 Max).

Measured figures: `results/quick-m5-max-{fp16,int8,int4}-native-exact.json` and `results/quick-checks/night-2026-10-22.json`: one Load (1.12 s FP16, 1.91 s int8, 1.79 s int4), then two Read document requests on a short fixed test input (warm 0.764 s, 0.785 s, 0.791 s); peak memory 5.89 GB FP16, 4.29 GB int8, 3.61 GB int4 (sum of worker footprints, Mac in use, quick check, not the interleaved protocol).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Not found in sources checked.
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `arsenal_run(model:"teleocr", operation:"read_document", input:{"document":{"path":"/absolute/path/page.png"},"mode":"ocr"})`

**Good and bad**
- Good at: Not found in sources checked. No source opened checks the maker's OmniDocBench scores.
- Bad at: Full-page parsing can drop text: one reporter's page kept about 57% of the JSON text. The maintainer's first-glance guess is a footer misread. [caipeng328/TeleOCR #9, 2026-09-21, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: Missing text on full pages, as above. Fix or workaround: none shown; the maintainer will try to reproduce it. [caipeng328/TeleOCR #9, 2026-09-21, first-hand]
- Complaint or known bug: guess_language_by_text raises UnboundLocalError on a code block with no language tag, failing the whole multi-page document. Fix or workaround: Reported by one user, not confirmed: setting lang to the default first; the reporter says it then completed. [caipeng328/TeleOCR #10, 2026-09-29, first-hand]
- Complaint or known bug: A cropped table gained a header absent from the image; the reporter also raised possible OmniDocBench overlap. Fix or workaround: the maintainer's malicious-crop test found no untrue transcription, and SSIM and text checks found no obvious overlap; open. [caipeng328/TeleOCR #15, 2026-10-23 to 10-27, first-hand]

**Evidence**
- Strength: thin. The maker's benchmark is not independently checked; outside material is three user-filed 2026 issues. A write-up cited earlier could not be opened (HTTP 403) and was dropped. Reddit unreachable from search.
- Sources:
  - TeleOCR issue #9, https://github.com/caipeng328/TeleOCR/issues/9, first-hand
  - TeleOCR issue #10, https://github.com/caipeng328/TeleOCR/issues/10, first-hand
  - TeleOCR issue #15, https://github.com/caipeng328/TeleOCR/issues/15, first-hand (maintainer replies to 2026-10-27)
