# Gemma 4 E4B

Chat about text and images. Native Swift/MLX language-family worker; FP16 by default, with group-64 affine int8 and int4 derived locally.

Operations: `chat`, `count_tokens`, `describe`, `tag`. Controls are narrowed in `ops/`.

Source: `google/gemma-4-E4B-it@ee0ef6023621cff504d758262d4e04895a5af4a2`. Checkpoint filenames, bytes and SHA256 values are pinned in `model.toml`. Upstream code: `https://github.com/ml-explore/mlx-swift-lm@bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. Vendor BF16/F32 values are cast tensor by tensor at Get; that conversion's numerical quality remains unqualified. No trained tensors have been downloaded.

Capabilities, language and licence source: the pinned model card. Apache License 2.0. Languages: multilingual; task performance varies by language. Trained weights, tokenizer correctness and FP16/int8/int4 task quality await the approved live trial. Text and image input only; audio tower is excluded.

`gate/no-weights.json` exercises reduced architecture settings and a lab byte tokenizer, not trained tokenizer fidelity. `gate/header-source.json` records exact Range-read results. Full header versus graph mapping, random operation smoke, malformed input, cancellation and tier checks are reported in `log/001-integration.md`; a missing pass is not qualification. Random tensors do not prove task accuracy or quantization quality.

Header check: all mapped names and shapes match; source BF16/F32 → FP16 storage conversion is explicitly declared at Get. Trained numerical quality remains unqualified.

Random worker: every declared mode/input fixture passes at FP16, int8 and int4, including explicit parser rejection for random structured output. Malformed input, cancellation, unload and complete eligible-matrix packing pass at all tiers.

The header map excludes 752 audio-tower tensors and 54 shared-KV projection tensors; those layers reuse earlier K/V, so the graph does not build them. Image-only operation does not load audio. Get still transfers the single monolithic source archive.

Offline tokenizer: ASCII encode/decode and the real local text/image chat templates pass. This is not independent vendor token-ID parity or trained output qualification.

Parameter label: 7.6B, derived from the complete default text/image graph inventory. Source checkpoint: 7,996,157,418 stored values; imported graph: 7,632,344,554 values. The card reports 4.5B effective parameters (8B with per-layer embeddings); the label is the imported graph's stored-value count, not the card's figure.

Text and image chat, captions and prompt-driven tags. Audio tensors and operations are excluded; real-weight clipping bounds and FP16 range remain unqualified.
