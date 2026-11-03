# OvisOCR2

Read documents in reading order. Native Swift/MLX language-family worker; FP16 only.

Operations: `read_document`. Controls are narrowed in `ops/`; each document mode preserves the maker's prompt.

Source: `ATH-MaaS/OvisOCR2@1fc9221b7823a371d6e97f92d527cc847e24e107`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the pinned model card. Apache License 2.0. Languages: multilingual; task performance varies by language. Trained weights, tokenizer correctness and FP16 task quality await the approved live trial.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 853M, derived from the complete default text/image graph inventory. Source checkpoint: 852,985,920 stored values; imported graph: 852,985,920 values.

Reading-order Markdown, LaTeX formulas, HTML tables and figure regions. Figure rectangles are mapped to original pixel coordinates and emitted as cropped PNG payloads in `files[]`, linked by page image ID. A Markdown file is always included; the operation is a file result with structured pages.

Figure tags are rounded and clamped to the page as the card's renderer does; empty regions are skipped. The card's 448²–2880² pixel budget and its long-repeat cleanup for truncated output apply.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
