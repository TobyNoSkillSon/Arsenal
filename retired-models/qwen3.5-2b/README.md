# Qwen3.5 2B

Describe images and answer questions. Native Swift/MLX language-family worker; FP16 by default, with group-64 affine int8 and int4 derived locally.

Operations: `chat`, `count_tokens`, `describe`, `tag`. Controls are narrowed in `ops/`.

Source: `mlx-community/Qwen3.5-2B-bf16@fb270110eb5a9af244937040c9c3e57addab8ee9`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the pinned model card. Apache License 2.0. Languages: multilingual; task performance varies by language. Trained weights, tokenizer correctness and FP16/int8/int4 task quality await the approved live trial.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 2.2B, derived from the complete default text/image graph inventory. Source checkpoint: 2,213,241,664 stored values; imported graph: 2,213,241,664 values.

Text chat, visual question answering, image description and prompt-driven tagging. Thinking is request controlled; no audio/video or function-call execution is declared. The model card warns that the 2B model is more prone to thinking loops than other Qwen3.5 models with its recommended sampling settings, which can stop generation from terminating.
