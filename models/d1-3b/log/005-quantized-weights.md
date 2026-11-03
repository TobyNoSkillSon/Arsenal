+++
hypothesis = "int8 or int4 weights make d1-3B decisions faster on M5 Max."
change = "MLX affine group-64 int8 and int4 for every language-model Linear (embedding and readout bf16)."
measurement = "M5 Max, 8 Oct 2026, same harness: 32 ads x 3 questions bf16 39.3, int8 50.4, int4 47.2 ms/item; single 39.1 / 41.8 / 39.3 ms. vs fp32: int8 gate 0.0196 24/24, requirements 0.0625; int4 ads 0.279, 90/96."
verdict = "rejected"
numbers = { bf16_batch_ms_per_item = 39.3, int8_batch_ms_per_item = 50.4, int4_batch_ms_per_item = 47.2, int8_single_ms = 41.8, int4_single_ms = 39.3, int4_ads_argmax = 90, ads_questions = 96 }
links = ["gate/reference-check.json"]
+++
# Quantized weights

A decision reads hundreds of tokens per pass (M 370-3,800), where MLX's quantized GEMMs dequantize in software and miss the bf16 tensor-unit rate. Both were slower than bf16 on batches and no faster on single items; int4 also changed 6 of 96 ad answers. Weight-only quantization only saves memory here. Fidelity numbers were taken with the earlier bf16-rounded readout.
