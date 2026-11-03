# MLXMoGe

Source: https://github.com/mnmly/mlx-swift-MoGe
Revision: 2925d7bbf821c8f7e9cb4d7c6cdc5b6061b6bce2

Imported only the inference closure; no CLI, downloader, checkpoint pickle reader, app, streaming orchestration, or SALAD dependency. Arsenal owns checked streaming weight loading, cancellation, precision derivation and file outputs.

## Local adaptations

- Removed convenience checkpoint loaders; GeometryWorker installs tensors through the shared streamed loader and tracks every stored learned tensor.
- The selected mlx-vlm checkpoint keeps the 37×37 DINOv2 position table and five-axis sparse kernels. The graph now retains those shapes and reshapes sparse kernels only for the contraction.
- Community ConvTranspose weights are ordinary MLX OHWI weights, rather than the original port converter's pre-flipped weights. Use MLX's native transposed convolution.
- Match PyTorch's bicubic positional sampling with offset 0.1, using the small implementation imported from the DA3 closure at its recorded revision; retain half activations after interpolation.
- Keep dense/refiner learned arithmetic half; float geometric binning and output recovery do not create FP32 learned weights.
- Allow explicitly shrunk encoder widths/depths in lab-only random construction. Native serving always uses the pinned model config.
- Upstream Metal search kernel is retained unchanged. No kernel tuning was performed.
- Reference: microsoft/MoGe@74fbce054ebed49800de42d0ad0e83495065719a. The encoder resizes with PyTorch's antialiased bilinear filter (`dinov2_encoder.py`), and the token grid uses Python's half-to-even `round` in double precision (`v3.py`).
- View-plane UV sits at pixel centres (`±span·(n−1)/n`, `geometry_torch.py`), in the model's pyramid and in focal/shift recovery. Recovery samples point map, UV and mask with 64×64 nearest interpolation and solves the reference's variable-projection objective (shift by Levenberg-Marquardt from 0, focal in closed form); fewer than two samples give focal 1, shift 0. The solver runs to its optimum instead of scipy's `ftol=1e-3` stop; on the regression maps the two agree within 1e-4.
- Interpolation results return the input dtype, so a half activation stays half through later learned layers.
- Encoder blocks, heads, refinement steps and sparse U-Net levels are evaluated one at a time and call the caller's cancellation checkpoint between them.
