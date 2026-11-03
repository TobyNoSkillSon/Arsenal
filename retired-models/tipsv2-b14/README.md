# TIPSv2 B/14

Search photos with text or images.

## Capabilities

Declared native operations: `embed`, `detect`, `segment`. Each operation has a narrowed schema in `ops/`, shared by HTTP, MCP and the native tool window.

## Source

Checkpoint: `google/tipsv2-b14@ed1e4dc6b74bf3935ae099e9d5eb30fa96528454`. Exact filenames, sizes and SHA256 hashes are recorded in `model.toml`. No weight payload was downloaded during integration.

All checkpoint tensors are loaded intact after the mandatory FP16 cast. Norms and spatial kernels remain half. Trained-weight quality and BF16-to-FP16 parity have not been qualified.

## Limits

English text, at most 64 SentencePiece tokens. Photo/query vectors use the image class token and the masked-mean text representation. Zero-shot labels use cosine similarity against `a photo of a LABEL`. Zero-shot segmentation assigns each patch to its most similar label and returns one PNG mask per label. Trained DPT depth, normals and segmentation heads are different checkpoints and are excluded.

## Qualification

Random graph, header-only inventory and precision coverage establish wiring, not trained model quality. See `log/001-integration.md` for the actual checks and remaining gaps.

Swift port: `mnmly/mlx-swift-tips@222f5fa5133b6ee9272f2878fd84b7ec3cff8a17`. Adapted files and source terms: `Workers/Perception/licenses/MLXTIPS/PORTED-FROM.md`.

Header inspection matches all 323 runtime parameter names and shapes. The production header checker passes with explicit per-tensor FP32→FP16 Get cast plans; these plans do not establish trained cast quality. The full graph has 97 eligible group-64 matrices (99.23% of learned parameters, derived from the inventory); all eligible matrices in its shrunk graph execute at int8 and int4. See `gate/header-check.json` and `gate/no-weights-results.json`.

The native SentencePiece reader comes from Arsenal’s existing MLXAudioCore implementation. NFKC normalization, whitespace folding and lowercase input precede its 64-token, no-BOS/no-EOS encoding. Exact vendor tokenizer parity, especially unusual Unicode and unknown tokens, remains a live-test risk. Classification scores are cosine similarities, not probabilities.

Embedding batches are bounded to 256 inputs so 1,024-dimensional JSON vector replies fit the 16 MiB worker frame limit. Images must already be upright; preprocessing uses the decoded pixel extent.

Arsenal offers this model at FP16 only: below 1B parameters, 8-bit would save under 1 GB and run no faster.
