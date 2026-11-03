# LFM2.5 VL 450M

Describe images with a small model. Native Swift/MLX language-family worker; FP16 only.

Operations: `chat`, `count_tokens`, `describe`, `tag`. Controls are narrowed in `ops/`.

Source: `LiquidAI/LFM2.5-VL-450M-MLX-bf16@ed71acdae0799c98cb450787d049d1343b65cd52`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the pinned model card. LFM Open License v1.0. Commercial use is licensed only while the licensee's annual revenue is below US$10 million; above that threshold it is not licensed, and Qualified Non-Profit Organizations are exempt for non-commercial or research purposes. Languages: English, Arabic, Chinese, French, German, Japanese, Korean, Portuguese and Spanish (model card); task performance varies by language. Trained weights, tokenizer correctness and FP16 task quality await the approved live trial. Not intended for fine-grained OCR. Commercial use above the upstream revenue threshold requires a separate licence.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

The source stores two obsolete projector LayerNorm tensors while projector_use_layernorm=false. The explicit header map excludes exactly these inactive tensors; the serving graph does not apply them.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 449M, derived from the complete default text/image graph inventory. Source checkpoint: 448,724,992 stored values; imported graph: 448,718,848 values.

Text/image chat, captions and prompt-driven tags; SigLIP2 image patches and LFM2 convolution/attention blocks. Fine-grained OCR is an upstream limitation. Bounding boxes come only from a chat prompt that asks for JSON output with `bbox`; there is no detection operation.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
