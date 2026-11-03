# Hy-MT2 7B

Translate between 33 languages. Native Swift/MLX language-family worker; FP16 by default, with group-64 affine int8 and int4 derived locally.

Operations: `translate`. Controls are narrowed in `ops/`.

Maker: Tencent (`tencent/Hy-MT2-7B` model card; prompt wording reference `Tencent-Hunyuan/Hy-MT2@ff1903ec` on GitHub). Converter: mlx-community. Source: `mlx-community/Hy-MT2-7B-bf16@3168d7f87d357415a4a12448c8385c3a2c4b8847`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@48114eedf3c3d4a682a6b1d644cc5af5c2945d10`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the Tencent model card. Apache License 2.0. Languages: the maker's 33 languages. Trained weights, tokenizer correctness and FP16/int8/int4 task quality await the approved live trial.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 7.5B, derived from the complete default text/image graph inventory. Source checkpoint: 7,504,568,320 stored values; imported graph: 7,504,568,320 values.

Translation across the maker's 33 languages. The local dense Hunyuan graph retains qk normalization and dynamic NTK-alpha RoPE from the pinned backport.

Prompt types from the card's instruction examples, one per request: default, terminology (`glossary`), style (`tone`), background (`context`), structured data (`format`), personalization (`instructions`, one preference per line) and delimiter preservation (`preserve_delimiters`). English prompt wording with the card's English language names is used for every pair, and target languages outside the card's 38 supported languages are refused. Decoding uses the card's recommended settings for the 7B model: temperature 0.7, top-p 0.6, top-k 20 and repetition penalty 1.05.
