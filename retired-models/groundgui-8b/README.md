# GroundGUI 8B

Locate screen elements from instructions. Native Swift/MLX language-family worker; FP16 by default, with group-64 affine int8 and int4 derived locally.

Operations: `ground_screen`. Controls are narrowed in `ops/`; grounding coordinates are converted from the maker's grid into original screenshot pixels.

Source: `PrentisAI/GroundGUI-8B@9db378189fddaabb5f6e347b9053858a392022ac`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities and licence source: the pinned model card. Apache License 2.0. Languages: the card names none, so the language is undetermined (`und`). Trained weights, tokenizer correctness and FP16/int8/int4 task quality await the approved live trial.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 8.8B, derived from the complete default text/image graph inventory. Source checkpoint: 8,767,123,696 stored values; imported graph: 8,767,123,696 values.

Screen-element localization with the maker's trained system prompts: `platform` selects desktop (`computer_use`, default), web (`browser_use`) or mobile (`mobile_use`), as in GroundGUI `scalecua_toolcall` evaluation (a1a02a93cb626b86c0ba829c38b613b8a40ecd0b). The tool call's 0–1000 `coordinate` maps to original screenshot pixels for any click-type action. If the model emits `abstain`, the port returns no detection; the card does not show that plain prompts elicit abstention, so how often that happens is unverified. Grounding only; it does not execute the emitted action.
