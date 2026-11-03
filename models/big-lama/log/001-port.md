+++
hypothesis = "Big-LaMa runs at FP16 once its eval BatchNorms are folded into the convolutions before them."
change = "Vendor xocialize's MLX-Swift LaMa port into Workers/Restoration (LaMaNative) as a folded conv+bias graph; Get folds the float32 source (cast:restoration-lama-v1, [precision] source = \"raw\") and stores FP16; add the inpaint operation with the provider's predict pipeline."
measurement = "Fixed 256x256 window with a hole, Swift FP16 vs the provider graph (spandrel, MPS, FP32 capture): predicted image 38.0 dB over [0,1], max |d| 0.036. One 1024x576 photo end to end: 43.2 dB inside the hole, max 10 levels; outside the hole bit-identical to the input."
verdict = "kept"
numbers = { source_tensors = 837, folded_tensors = 378, parameters = 51057027, window_psnr_db = 38.0, hole_psnr_db = 43.2, hole_max_levels = 10 }
links = ["../README.md", "../gate/gate.toml"]
+++

Source check (22 Oct): all 837 tensors of `mlx-community/LaMa-fp32@92a388f` are bit-identical (NHWC layout) to the generator in the widely mirrored `big-lama.pt` (sha256 344c77bb…, Sanster's lama-cleaner release, read through spandrel).

Why folding: the checkpoint's BatchNorm running variances reach 4.8e5 (`model.5.conv1.bn_l.running_var`), past FP16's 65,504, and the activations before those norms reach thousands. The provider graph on MPS collapses at FP16 and at BF16 alike (the fill comes out black; predicted window 0.6 dB). Folding (scale = gamma/sqrt(var+eps), shift = beta - mean*scale) is exact algebra: the folded provider graph at FP32 matches the unfolded one to 107 dB. Folded, FP16 holds (MLX port above; the folded provider graph on MPS read 38.6 dB). Get's ordinary half normalization would overflow those variances, so the folder declares `[precision] source = "raw"`: Get hands the float32 file to the model's own recipe, which folds and stores FP16; the float32 download is then removed. The FFTs of the spectral blocks run in float32 (MLX has no half FFT; the provider's own half path does the same) and the spectral 1x1 convolutions in FP16.

Timing and memory (quick check, M5 Max, Mac in use, 22 Oct): load plus one 1024x576 photo in 1.4 s through the adapter in the heavy test; MLX peak 1.57 GB. Not a controlled measurement.

Large photo (22 Oct, after review): one 3840x2160 inpaint (the test photo upscaled, same mask) against the provider graph at FP32 on MPS: inside the hole 31.6 dB, max 18 levels, mean 5.5, a visible shift toward a warmer tint; outside the hole identical. Running the spectral 1x1 convolutions in float32 changed nothing (31.65 dB), so the FFT block is not the cause; the drift was not localized further. Quick check: 11.9 s for the run in a debug build, MLX peak 6.9 GB (provider on MPS: 2.3 s, 14.2 GB driver memory).

