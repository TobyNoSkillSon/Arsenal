+++
hypothesis = "The trained DINOv3-L encoder can run its residual stream in FP16 like the rest of the graph."
change = "None to the math tonight. Records the float32 residual stream and LayerScale in DINOv3 (commit 2d963cd6) as a precision exception to the FP16-only rule."
measurement = "Stage-by-stage parity against the vendor TRELLIS.2 capture on one product cutout (22 Oct, M5 Max, machine in use); heavy TrellisParityTests and the CPU overflow regression in TrellisTests."
verdict = "kept"
numbers = { fp16_residual_peak = 153000, fp16_range = 65504, condition_cosine_fp32_residual = 0.9999996, occupancy_iou_32 = 0.9993 }
links = ["../README.md", "../../../Workers/Geometry/Sources/MLXTrellis/DINOv3.swift"]
+++

# Precision exception: DINOv3 residual stream in float32

2026-10-22. Arsenal runs every model as FP16, int8 or int4, with float32 allowed only for accumulators and numerical intermediates. DINOv3's residual stream is model state carried through all 24 blocks, so keeping it in float32 is an exception, recorded here so it stays visible.

Evidence: from the first block the trained encoder's residual reaches about 153,000 (an MLP output near 5e4 times layer_scale2), past FP16's 65,504. In FP16 the stream became inf, the final LayerNorm turned the condition into NaN, and the sparse-structure decoder produced no occupied voxel, so every image failed. With the residual add and LayerScale in float32 (weights, attention and MLP matmuls stay FP16; the MLP down projection takes an exact 1/8-scaled input) the condition matches the vendor capture at cosine 0.9999996, and the 32³ occupancy from the port's own sampled latent has IoU 0.9993. One image only; this is a parity check, not a speed or memory measurement.

Open: a pure-FP16 alternative (for example rescaling the residual by a power of two and folding the scale into LayerScale and the norms) has not been tried. Changing it needs the owner's go and the same parity check.
