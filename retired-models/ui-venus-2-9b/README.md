# UI Venus 2 9B

Locate screen elements from instructions. Native Swift/MLX language-family worker; FP16 by default, with group-64 affine int8 and int4 derived locally.

Operations: `ground_screen`. Controls are narrowed in `ops/`; grounding coordinates are converted from the maker's grid into original screenshot pixels. Screenshots resize to at most 12,845,056 pixels, the grounding client's `max_pixels`.

Source: `inclusionAI/UI-Venus-2-9B@90d6dfe13ece735f82793635b69bcc10c710e2ab`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the pinned model card. Model-weight licence pending upstream confirmation. Languages: multilingual; task performance varies by language. Checked live on 20 Oct at FP16, int8 and int4: both test clicks landed inside their targets.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 9.4B, derived from the complete default text/image graph inventory. Source checkpoint: 9,409,813,744 stored values; imported graph: 9,409,813,744 values.

Direct screen-element grounding, with thinking disabled and normalized 0–1000 coordinates converted to original image pixels (box/two-corner responses use their center). Closed-loop GUI actions and CAPTCHA sequences are not declared.
