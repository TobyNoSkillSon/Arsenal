+++
hypothesis = "xocialize's MLX-Swift DDColor port, run on mlx-community's FP16 conversion of ddcolor_modelscope.pth, reproduces the provider at FP16 inside the restoration worker."
change = "Vendor the port into Workers/Restoration (DDColorNative) with MLX's fused layer norm, precise softmax and the provider's ColorizationPipeline (cv2 bilinear resize, nearest ab resize, rounding); add the colorize operation."
measurement = "Fixed 512x512 input from a real photo, Swift FP16 vs the provider graph (spandrel, MPS, FP32 capture): ab 53.4 dB PSNR, max |d| 4.2 of 47; every component 51.6 dB or more. One 1024x576 photo end to end: 51.9 dB, max 7 levels, against the provider pipeline's own output."
verdict = "kept"
numbers = { tensors = 575, parameters = 227881750, ab_psnr_db = 53.4, image_psnr_db = 51.9, image_max_levels = 7 }
links = ["../README.md", "../gate/gate.toml"]
+++

Source check (22 Oct): every tensor of `mlx-community/DDColor-modelscope-fp16@479a367` equals the FP16 cast of the original `piddnad/DDColor-models` `ddcolor_modelscope.pth` (sha256 17c460d7…): 566 of 573 bit-exact, the other seven (spectral-norm convolutions resolved with the stored u/v) within 7.6e-6; the extra `mean`/`std` are the model's ImageNet buffers. No weight exceeds FP16's range.

Precision: the provider graph itself fails at FP16 on MPS (ab 7.7 dB against its FP32 capture; its LayerNorm overflows in the ConvNeXt stages) and holds at BF16 (43 dB). The MLX port at FP16 reaches 53.4 dB because MLX's fused layer norm accumulates in float32, so FP16 is kept; the Swift BF16 run read 35.2 dB. The colour-space math around the network (Lab conversions, resizes) runs in float32, as image processing, like the provider's cv2 code.

Timing and memory (quick check, M5 Max, Mac in use, 22 Oct): load plus one 1024x576 photo in 2.9 s through the adapter in the heavy test; MLX peak 1.39 GB. Not a controlled measurement.
