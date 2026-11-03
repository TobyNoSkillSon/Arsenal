# PaddleOCR VL 1.6

Read documents, charts and seals. Native Swift/MLX language-family worker; FP16 only.

Operations: `read_document`. Controls are narrowed in `ops/`; each document mode preserves the maker's prompt, and grounding coordinates are converted from the maker's grid into original screenshot pixels.

Source: `PaddlePaddle/PaddleOCR-VL-1.6@c5630abae1d940eafe0697512a0325494b02ab42`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/mlx-community/paddleocr-vl.swift@b31d5ab0bdb87a77ae80b74ec3380fed99f5b706`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the pinned model card. Apache License 2.0. Languages: multilingual; task performance varies by language. Trained weights, tokenizer correctness and FP16 task quality await the approved live trial.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 959M, derived from the complete graph and checkpoint: 958,588,736 stored values, 620 tensors.

All six maker tasks: OCR, tables, formulas, charts, seals and text spotting. The 1.6 graph is adapted from the pinned PaddlePaddle reference: explicit 128-dimensional ERNIE attention, multimodal RoPE, separate-QKV SigLIP/NaViT vision, bilinear learned positions and 2×2 merge projector. Spotting follows the maker pre-upscale and larger pixel budget. The complete pooling-head inventory is retained, although OCR uses unpooled vision features.

Output post-processing follows PaddleX `c852371c701d83cd21955b2963834825920ffebb` (paddleocr_vl pipeline): table OTSL becomes HTML, `\(…\)`/`\[…\]` delimiters become `$`/`$$`, and text spotting returns one block per line with its four-point polygon and box in original page pixels (location tokens on the 0–1000 grid). Spotting output without location points is reported as `invalid_model_output`.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
