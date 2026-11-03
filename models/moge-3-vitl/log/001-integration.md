+++
hypothesis = "The pinned Swift MLX graph can load the checkpoint layout and expose depth operations in FP16, int8 and int4."
change = "Vendor the inference closure and implement a native geometry adapter."
measurement = "No trained weights; exact header layouts and isolated random-weight worker checks."
verdict = "inconclusive"
links = ["../README.md", "../../../docs/INTEGRATING.md"]
numbers = { trained_weights_downloaded = 0, precision_tiers = 3 }
+++

# Native integration

2026-10-18. No trained weights downloaded. Source metadata and bounded safetensors headers inspected at mlx-community/moge-3-vitl-mlx-fp32@f5ceca7512506731fb1c50307c313da76aac500f. FP32 source is staged for one streamed FP16 cast; only FP16/int8/int4 are serving tiers. Trained output and quantization quality remain unqualified. See README and gate evidence for actual coverage.
