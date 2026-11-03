# LocateAnything 3B

Find objects from descriptions.

## Capabilities

Declared native operations: `detect`, `ground_screen`. Each operation has a narrowed schema in `ops/`, shared by HTTP, MCP and the native tool window.

## Source

Checkpoint: `mlx-community/LocateAnything-3B-bf16@b409aade8f6aae828a0c8e0f4190bf28eb8f0244`. Exact filenames, sizes and SHA256 hashes are recorded in `model.toml`. No weight payload was downloaded during integration.

All checkpoint tensors are loaded intact after the mandatory FP16 cast; MLX affine group-64 int8/int4 packs are derived locally. Norms and spatial kernels remain half. Trained-weight quality and BF16-to-FP16 parity have not been qualified.

## Limits

Supports open labels, referring phrases, custom localization instructions and points. Screen output includes original screenshot pixel coordinates. The model does not publish calibrated box probabilities; outputs therefore omit confidence scores. Native use is subject to NVIDIA research terms.

## Qualification

Random graph, header-only inventory and precision coverage establish wiring, not trained model quality. See `log/001-integration.md` for the actual checks and remaining gaps.

Swift port: `mnmly/mlx-swift-LocateAnything@5245a70e736fcc46ae1d0a8baf0732854b5ce577`. Adapted files and source terms: `Workers/Perception/licenses/MLXLocateAnything/PORTED-FROM.md`.

Header inspection matches all 769 runtime parameter names and shapes. All 769 BF16→FP16 dtype comparisons remain unqualified; the production checker deliberately rejects this cast until reference qualification. The short random generations are now rejected as incomplete/malformed, rather than misreported as no matches. Successful trained detection output remains unverified. The full graph has 336 eligible group-64 matrices (96.02% of learned parameters, derived from the inventory); all eligible matrices in its shrunk graph execute at int8 and int4. See `gate/header-check.json` and `gate/no-weights-results.json`.

Both Parallel Box Decoding and autoregressive fallback use the maker’s token grammar and prompts. Parsed box and point coordinates are divided by 1000 exactly once. Tokenizer JSON is synthesized in a private temporary directory, never in the verified snapshot, and removed at unload. Video input and visual crop prompts are not declared by this integration.

Task selection applies the GUI template to screen instructions and the pointing template when `output_mode = "points"`. Generation requires explicit termination and complete reference/box grammar. Only a closed `<box>None</box>` block represents a no-match result; a token cap, a repeating output (the same block written over 256 tokens) or malformed output returns an error. The default budget is the maker `predict()` default of 2,048 tokens; up to 8,192, the card's suggestion against truncated lists.
