# Chandra OCR 2

Read handwriting, forms and documents. Native Swift/MLX language-family worker at FP16 only. Int8 and int4 are withdrawn: a quick check on the M5 Max (22 Oct 2026, Mac in use) found int8 no faster (8.1 s per page against 7.8 s at FP16) and only 16% smaller, and int4 changed the OCR text (similarity 0.25 to FP16) (`log/002-tiers-withdrawn.md`).

Operations: `read_document`. Controls are narrowed in `ops/`; each document mode preserves the maker's prompt.

Source: `datalab-to/chandra-ocr-2@af93b47dba1b47b6640c86ccf487ed2260ab9a09`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the pinned model card. Weights use a modified OpenRAIL-M licence (code is Apache 2.0): free for research, personal use and startups under US$2M in revenue or funding, and not for uses that compete with Datalab's API. Commercial use beyond that needs a separate licence, so the catalog marks it not commercial. Languages: multilingual; task performance varies by language. Checked live on 20 Oct at FP16: a text page and a table read exactly.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passed at FP16 (and at the since-withdrawn int8 and int4), including explicit parser rejection for random structured output. Malformed input, cancellation and unload pass.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 4.5B, derived from the complete default text/image graph inventory. Source checkpoint: 5,295,564,288 stored values; imported graph: 4,539,265,536 values.

Handwriting, forms, tables, math/chemistry and diagrams through the maker's HTML layout prompt. Page Markdown follows the maker's `parse_markdown` (page headers, footers and blank pages dropped, math as `$…$`/`$$…$$`, tables kept as HTML), and each layout div becomes a page block with its label, inner HTML and pixel box as in `parse_layout`.
