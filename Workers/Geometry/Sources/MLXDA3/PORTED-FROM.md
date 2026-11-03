# MLXDA3

Source: https://github.com/mnmly/mlx-swift-da3
Revision: 496dfb1d1f0969c5dc2518aad59ea21abb1ea88a

Imported only the inference closure; no CLI, downloader, checkpoint pickle reader, app, streaming orchestration, or SALAD dependency. Arsenal owns checked streaming weight loading, cancellation, precision derivation and file outputs.

## Local adaptations

- Keep only the pure checkpoint mapping; remove whole-dictionary checkpoint loaders and convenience paths.
- Match the original Small checkpoint's auxiliary heads: only level 0 has trained LayerNorm parameters. Later levels use identity normalization.
- Accept the original Mono checkpoint's `scratch.sky_output_conv2` key spelling.
- Keep half activations after bicubic positional interpolation. Expose an explicit config-based factory for small lab graphs.
- Import `MLXDA3Streaming/RayPose.swift` into this module for ray-based camera recovery; only small geometric QR/SVD solves use CPU streams, with the neural graph on MLX GPU.
- Learned camera encoder/decoder tensors in DA3 Small are explicitly excluded by the model-local header map. The adapter exposes geometric ray-pose, not learned camera conditioning or decoding.
- Bilinear interpolation results return the input dtype; MLX `Upsample` otherwise promotes half activations to float32 before the following convolutions.
- Mono sky post-processing from ByteDance-Seed/Depth-Anything-3@3d835ec1a5802d64a8b8b15f817a1ab54809bfe4 `model/da3.py`: sky probability ≥ 0.3 takes the 99th percentile of non-sky depth when both populations exceed 10 pixels. The percentile uses every non-sky pixel rather than upstream's random 100,000-pixel sample, so results are deterministic. The sky tensor is returned unchanged.
- `predict(_:checkpoint:)` evaluates the backbone block by block and calls the caller's cancellation checkpoint between blocks and around the head.
