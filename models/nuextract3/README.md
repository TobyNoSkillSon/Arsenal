# NuExtract3

NuExtract3 is NuMind's 4B vision-language reasoning model for document understanding, fine-tuned from Qwen3.5-4B (4.5B values in the full text and image graph). Given a document as text, images or PDF pages and a JSON template, it writes the filled-in JSON; it also converts document images to Markdown. Its strength is template-driven structured extraction: on NuMind's internal benchmark of about 600 documents it scores 0.651, ahead of the larger general models NuMind compared *(vendor figure)*.

## What it is for

NuMind built it for structured information extraction and image-to-Markdown conversion in one model, for extraction pipelines, OCR and RAG preprocessing of scans, receipts, forms, invoices, contracts and tables, in many languages, with optional reasoning. NuMind found small models' reasoning often loops on long outputs and recommends starting without it.

- Filling a fixed JSON record from invoices, receipts, contracts or forms (numbers, dates, currencies, line items).
- Pulling named entities or lists from free text into a typed structure.
- Drafting an extraction template from a plain description or a sample document.
- Converting scanned or photographed pages to Markdown with HTML tables, for search or later processing.

## Inputs and outputs

### `extract` (default)

Fills a NuExtract template from text, an image or a document, or drafts a template.

| Field | Type | Required | Notes |
|---|---|---|---|
| `text` | string | one of `text`, `image`, `document` | The input text. May be combined with `image` or `document`; the pictures come first, then the text. |
| `image` | file `{"path":"/absolute/path"}` | one of `text`, `image`, `document` | `image/png` or `image/jpeg`. Cannot be combined with `document`. |
| `document` | file `{"path":"/absolute/path"}` | one of `text`, `image`, `document` | `application/pdf`, `image/png` or `image/jpeg`; path at most 4096 characters. Cannot be combined with `image`. |
| `pages` | array of integer | no | 1 to 99 page indexes (`minItems` 1, `maxItems` 99), each 0 to 10000, zero-based. Default: every page. |
| `template` | object | yes in `structured` mode | The NuExtract template (see below). Key order is kept and becomes the output's key order. |
| `instructions` | string | no | Extra rules on formats or values, for example "only output the date component". |
| `mode` | string | no | `structured` (default) fills `template`; `template-generation` drafts a template from the input instead. |
| `thinking` | boolean | no | Default `false`. Reasoning before the answer; must be `false` in `template-generation` mode. |
| `max_tokens` | integer | no | 1 to 32768, default 4096. Output budget, reasoning included. |
| `temperature` | number | no | 0 to 2. When unset, 0.2, or 0.6 with `thinking` (the maker's settings). |
| `examples` | array | no | At most 16 in-context examples, each `{"text": string, "image": {"path"}, "output": <expected JSON>}` with `output` and at least one of `text` or `image` (`image/png` or `image/jpeg`). |

Returns JSON `{"data": <the filled JSON>, "finish_reason": "stop" | "length" | "repetition"}`. In `template-generation` mode `data` is the drafted template. A field the model cannot find comes back as `null` or `[]`. If the model's output is not valid JSON the request fails with `invalid_model_output`, and the message says when it stopped at `max_tokens` (raise it) or because the output was repeating. With `thinking` on, the reasoning before `</think>` is dropped and only the answer is parsed.

How a template is written: it has exactly the shape of the JSON you want back, and each leaf names the type of value to extract, not a value. For example `{"invoice_number": "verbatim-string", "invoice_date": "date", "total": "number", "currency": "currency", "line_items": [{"description": "verbatim-string", "quantity": "number"}]}`. The types, from the maker's specification:

- `verbatim-string`: copied exactly from the input (whitespace normalised); `string`: free text, may paraphrase or be inferred.
- `integer`, `number`, `boolean`.
- `date`, `time`, `date-time`, `duration`: ISO 8601, reduced precision allowed (`2024-01`, `--12-25`).
- Codes: `country` (ISO 3166-1 alpha-2), `currency` (ISO 4217), `language` (ISO 639-3), `language-tag` (BCP 47), `script` (ISO 15924), `unit-code` (UCUM), `region:XX` (ISO 3166-2 subdivisions for US, FR, IE, GB, IT, ES, DE, PT, CA, MX, BR, AU, JP, KR).
- `url`, `email-address`, `phone-number` (E.164 when the region is known), `iban`, `bic`.
- Constructors: `["string"]` is a list of that type; `["yes", "no", "maybe"]` is an enum (one of the values); `[["A", "B", "C"]]` is a multi-enum (several of the values); nested objects and lists of objects follow the same rules.

A NuExtract template is not a JSON Schema: a schema's `type`, `properties` and `required` keywords would be read as field names. NuMind's Python SDK converts JSON Schema or Pydantic models to templates (`numind.nuextract_utils.convert_json_schema_to_nuextract_template`). Examples are rendered as the maker's `developer` messages before the input, each example's output written in the template's key order.

Pictures and pages: images are decoded with their EXIF orientation applied. PDF pages are rendered at 144 dpi, bounded to 8192 pixels on the long side and 16,777,216 pixels in all; image files are used at their own size. The checkpoint's Qwen3-VL processor then scales each picture to between 65,536 and 16,777,216 pixels and reads it as one token per 32 × 32-pixel block, so a page costs roughly its pixel count / 1024 tokens. At most 99 images go into one request, examples included; a longer PDF needs `pages`. Before running, Arsenal estimates the prompt (text, template, instructions, pages) plus `max_tokens` against the 262,144-token context and refuses a request that would not fit, naming the page where it overflows.

### `read_document`

Converts each page of a document to Markdown.

| Field | Type | Required | Notes |
|---|---|---|---|
| `document` | file `{"path":"/absolute/path"}` | yes | `application/pdf`, `image/png` or `image/jpeg`; path at most 4096 characters. PDFs of up to 1024 pages. |
| `pages` | array of integer | no | 1 to 10000 zero-based page indexes, each 0 to 10000. Default: every page. |
| `mode` | string | no | `ocr` only (default). |
| `max_tokens` | integer | no | 1 to 32768, default 4096, per page. |

Returns JSON `{"pages": [...]}`, one entry per page in order: `index` (zero-based), `markdown` and `dimensions` (`width`, `height` of the rendered page in pixels). A page cut off at `max_tokens` or by a repetition loop also carries `"truncated": true` and `finish_reason` (`length` or `repetition`). Per the maker, the Markdown uses Markdown for text and headings, HTML for tables, LaTeX for maths and `<figure data-type="image" data-id="img_n">` elements with a description for pictures. Each page is a separate generation, without reasoning.

## Run it

MCP: `arsenal_run(model:"nuextract3", input:{"document":{"path":"/absolute/path/invoice.pdf"},"template":{"invoice_number":"verbatim-string","invoice_date":"date","total":"number","currency":"currency"}})` (add `operation:"read_document"` and only `document` for Markdown).

HTTP (`ARSENAL_URL` is `http://127.0.0.1:<port>`, the port from `status.json` in Arsenal's support folder; local HTTP needs no key): `curl -s "$ARSENAL_URL/arsenal/v1/models/nuextract3/ops/extract" -H 'Content-Type: application/json' -d '{"text":"Yesterday I bought apples and coffee at the corner shop for a total of $12.40.","template":{"store":"verbatim-string","total":"number","items":[{"name":"verbatim-string","price":"number"}]}}'`

HTTP, Markdown: `curl -s "$ARSENAL_URL/arsenal/v1/models/nuextract3/ops/read_document" -H 'Content-Type: application/json' -d '{"document":{"path":"/absolute/path/scan.pdf"},"pages":[0]}'`

The model has no settings. Measured on an M5 Max at FP16 in a quick check on a busy Mac (22 October 2026): Load 2.7 s; one short `read_document` request 3.3 s warm; 12.9 GB peak worker memory in that check, 17.2 GB in the 1 Nov check (`results/quick-m5-max-fp16-native-exact.json`). Runs go one at a time.

## Limits

- In the only live check with the trained weights (22 October 2026), `extract` failed with "The model did not return valid JSON" on its short test input while `read_document` worked; extraction quality on real documents has not been checked in Arsenal.
- The template is a NuExtract template, not a JSON Schema.
- At most 99 images per `extract` request, examples included.
- `image` and `document` cannot be combined in one request; the maker's model accepts several separate images, which Arsenal takes only as pages of one document.
- PDF pages are rendered at 144 dpi; the maker's own example renders at 170 dpi, so very small print may read slightly worse *(unverified)*.
- `read_document` offers only the maker's Markdown mode; the maker's `content` mode is not exposed.
- Reasoning (`thinking`) can fall into repetition loops on long outputs, which the maker observed in small models; start without it.
- FP16 only: int8 and int4 changed the `read_document` text too much (similarity 0.26 and 0.20 to FP16).
- Multilingual, with task quality that varies by language; NuMind publishes no language list.

## Licence and credit

Apache-2.0: free to use, modify and redistribute, including commercially, provided the licence and notices are kept and changes are stated; it includes a patent grant. Made by NuMind; the base model is Qwen3.5-4B. NuMind asks users to cite NuMind and link to the model page:

```bibtex
@misc{nuextract3,
  title = {NuExtract3},
  author = {NuMind},
  year = {2026},
  url = {https://nuextract.ai/}
}
```

## Links

- Model card: https://huggingface.co/numind/NuExtract3/blob/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1/README.md
- Template types: https://huggingface.co/numind/NuExtract3/blob/c99dc8f5641b866aa0192b6ea78f84bf9f3535f1/TYPES.md
- Code: https://github.com/numindai/nuextract
- Swift runtime: https://github.com/ml-explore/mlx-swift-lm

## How Arsenal runs it

Source: `numind/NuExtract3` at `c99dc8f5641b866aa0192b6ea78f84bf9f3535f1` (`model.safetensors` 9.08 GB plus tokenizer, processor config, chat template and generation config; exact sizes and hashes in `model.toml`). Get casts the vendor BF16/F32 tensors one by one to FP16; that conversion's numerical quality has not been qualified. Upstream runtime: `ml-explore/mlx-swift-lm` at `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`, run in the native Swift language-family worker (`runners/native/language.json`: Qwen3-VL processor, sampling temperature 0.2, 0.6 with thinking). Requests run one at a time. The checkpoint's own chat template builds the prompt, with `mode`, `template`, `instructions` and `enable_thinking` passed to it as the maker documents; the template text is forwarded with its original key order.

Integration checks (`log/001-integration.md`, 18 October 2026, before any trained weights were downloaded): exact checkpoint headers fetched by HTTP Range reads match the graph, 723 tensors with none ignored, with the BF16/F32 → FP16 casts declared (`gate/header-check.json`, `gate/header-source.json`); a reduced random-weight graph passed all 18 mode/input fixtures, malformed input, cancellation and unload (`gate/no-weights.json`); parser fixtures cover schema-valid structured results and rejection of random output (`gate/parser-fixtures.json`); the offline tokenizer passed ASCII encode/decode and the real text and image chat templates, which is not vendor token-ID parity (`gate/tokenizer-check.json`). Random tensors prove neither task accuracy nor quantization quality. The 4.5B parameter label is the stored value count of the default text and image graph: 4,539,265,536 values in the source and in the imported graph.

Precision (`log/002-tiers-withdrawn.md`, 22 October 2026, M5 Max, Mac in use, one Load and two requests per precision, one sample, not the controlled ABBA protocol): one `read_document` took 3.3 s at FP16 with a 12.9 GB peak, 2.2 s and 8.5 GB at int8, 1.7 s and 5.8 GB at int4, but int8 and int4 changed the text (similarity 0.26 and 0.20 to FP16), so both were withdrawn and the model ships FP16 only. The `extract` request in that run returned HTTP 500 (invalid JSON from the model) at every precision, FP16 included; that fault is open. Run rows: `results/quick-checks/night-2026-10-22.json`.

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: text, images or documents with a NuExtract template; the card's overview lists multilingual documents. [NuExtract3 model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: the card passes in-context examples (input and expected output) as `developer` messages. [NuExtract3 model card, undated, first-hand]
- Parameters people recommend: start without thinking at temperature 0.2; thinking uses 0.6; the card's Transformers example uses `max_new_tokens=4096`. [NuExtract3 model card, undated, first-hand]
- Arsenal call: `arsenal_run(model:"nuextract3", input:{"document":{"path":"/absolute/path/receipt.jpg"},"template":{"merchant":"verbatim-string","date":"date","total":"number","currency":"currency"},"thinking":false,"temperature":0.2})`

**Good and bad**
- Good at: on the maker's internal test of about 600 documents (not public), 0.651 with 27 failed outputs, against 0.538 with 31 for gemma-4-E4B-it and 0.479 with 170 for Qwen3.5-9B. This is a vendor figure. [NuExtract3 model card, undated, first-hand]
- Bad at: not found in the card or the discussion titles. [NuExtract3 model card, undated, first-hand; discussions, undated]
- Against the obvious alternatives: the maker's comparison is above. No independent comparison was found in the card or discussion titles. [NuExtract3 model card, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: none found in the card or the two open discussion titles. Fix or workaround: not found in sources checked. [NuExtract3 model card, undated; discussions, undated, titles only]

**Evidence**
- Strength: thin. The one number is the maker's unpublished benchmark. The discussions were read by title only, with relative dates. The Reddit launch thread could not be opened in this audit, so its qualitative claims were removed.
- Sources re-opened in this audit:
  - NuExtract3 model card, https://huggingface.co/numind/NuExtract3, undated, first-hand.
  - Discussions, https://huggingface.co/numind/NuExtract3/discussions, undated (relative dates only), first-hand.
  - `models/nuextract3/ops/extract.input.json`.
