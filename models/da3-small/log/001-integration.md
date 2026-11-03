+++
hypothesis = "The pinned Swift MLX graph can load the checkpoint layout and expose depth operations in FP16, int8 and int4."
change = "Vendor the inference closure and implement a native geometry adapter."
measurement = "No trained weights; exact header layouts and isolated random-weight worker checks."
verdict = "inconclusive"
links = ["../README.md", "../../../docs/INTEGRATING.md"]
numbers = { trained_weights_downloaded = 0, precision_tiers = 3 }
+++

# Native integration

2026-10-18. No trained weights downloaded. Source metadata and bounded safetensors headers inspected at depth-anything/DA3-SMALL@e08cab65ca0ec38e7826075418411ab90cab4da3. FP32 source is staged for one streamed FP16 cast; only FP16/int8/int4 are serving tiers. Trained output and quantization quality remain unqualified. See README and gate evidence for actual coverage.
