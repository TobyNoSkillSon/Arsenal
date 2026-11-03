# Infinity Parser2 Flash

Read documents, charts and boxes. Native Swift/MLX language-family worker; FP16 by default, with group-64 affine int8 and int4 derived locally.

Operations: `read_document`. Controls are narrowed in `ops/`; each document mode preserves the maker's prompt, and grounding coordinates are converted from the maker's grid into original screenshot pixels.

Source: `infly/Infinity-Parser2-Flash@cfddc4106b0abc4706f05575a164f7dd35d09c46`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the pinned model card. Apache License 2.0. Languages: multilingual; task performance varies by language. Trained weights, tokenizer correctness and FP16/int8/int4 task quality await the approved live trial.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Modes: JSON layout, direct Markdown and custom prompts (including chart and formula questions). JSON layout blocks retain their categories and text and convert 0–1000 boxes to original pixel coordinates. The blocks extension is model-local; generic tools may show JSON rather than overlays. English and Chinese are the primary languages; dense/rotated layouts remain difficult.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 2.2B, derived from the complete default text/image graph inventory. Source checkpoint: 2,213,241,664 stored values; imported graph: 2,213,241,664 values.

Reading-order document layout, JSON boxes, HTML tables, Markdown, formulas and custom visual questions (including charts and chemical formulas).

Post-processing follows INF-MLLM `011866e93026290ba39f05b796c92d58dd0ce40c`: a fenced ```json block is extracted, a truncated layout is cut back to its last complete element, and Markdown mode drops a surrounding code fence.
