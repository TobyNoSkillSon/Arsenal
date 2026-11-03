+++
hypothesis = "A reduced-width random checkpoint can exercise the real TTS architecture without running vendor weights."
change = "Reduced transformer widths, codec channels and codec layers, kept main layer counts and selective quantization; generated BF16 random checkpoint."
measurement = "Real fromModelDirectory loader and generation emitted finite PCM; int8 derive/reload generated PCM passed under mlock gpulock."
verdict = "kept"
numbers = { selected_layers = 250, base_samples = 32, int8_samples = 32, vendor_checkpoint_runs = 0 }
links = ["gate/synthetic-proof.json", "Workers/Voices/Tests/VoicesTests/SyntheticSpeechTests.swift", "Workers/Voices/Tests/VoicesTests/Fixtures"]
+++

The stand-in preserves the model classes, source loader, tokenizer, codec, selective affine conversion and code-predictor path. Qwen Base additionally encodes a synthetic reference and uses its speaker encoder. Generation is bounded to two tokens; random weights do not represent speech quality or validate vendor checkpoint shapes, precision quality or speed. The vendor model stays Unverified: Not tried on this Mac yet.

Early 64-wide codec attempts exposed the loader’s convolution-layout heuristic threshold; fixtures use channels above 64 and invert the source codec naming before saving. Equal talker/predictor widths also removed Qwen’s projection and failed the 250-layer guard; the stand-in restores the original unequal-width relationship. Production numerics and layer-count guards were not loosened. No real model weights are read by these tests.
