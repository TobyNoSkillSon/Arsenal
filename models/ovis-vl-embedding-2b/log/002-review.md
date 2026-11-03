+++
hypothesis = "Review corrections match fixed references and keep worker lifecycle checks passing."
change = "Correct visual preprocessing and bound lazy execution before cancellation checks."
measurement = "Captured small reference tensors, family tests and release worker smokes; no trained weights."
verdict = "inconclusive"
numbers = { tensor_payload_bytes = 0, family_tests_passed = 13 }
links = ["../README.md"]
+++

# Review corrections, 18 October 2026

The five review findings are addressed in the Embed family. Vision MLP uses tanh GELU. Video prompts contain timestamped independent temporal groups, with matching split position grids. Video settings come from the pinned processor_config.json video section; uniform sampling spans the full clip and resize budgets include sampled frame count. Attention runs per cuSeqlens segment, without an N×N mask over concatenated pages. Ovis and Gemma language loops evaluate each layer before checking cancellation.

Transformers v5.12.1 reference functions were captured once on fixed small tensors without weights. Tests compare vision MLP and segmented attention outputs, exact mixed image/video token sequences and three-axis positions, seven resize cases and three sampling cases. A four-frame clip catches zero-sample compressed reader markers. Both cancellation tests cancel after the first actual layer evaluation and verify that later layers are not executed.

After rebase onto main 162b7b9b (containing 61ea9cb1), all 13 family tests pass. Ovis release worker smokes pass FP16/int8/int4 with text, image, PDF, video and interleaved input fixtures. Malformed-input, cancellation, packed coverage and unload checks pass. Real header names/shapes still match 617 served parameters; default FP16 conversion and quantized output quality remain unqualified on trained weights. The source manifest now declares BF16 and a model-owned FP16 Get cast, followed by int8/int4 derivation from the FP16 base.
