# FlashVSR v1.1 on M5 Max (quick check, 4 Nov 2026)

Mac in use, on AC power. Isolated debug Arsenal with its own support folder, driven headlessly through MCP. FP16, scale 2, seed 0.

| Input | Output | Frames | First request (Load included) | Second request | Worker peak memory | Swap added |
|---|---|---:|---:|---:|---:|---:|
| 640×480, 2.0 s, 24 fps | 1280×960 H.264 (1280×1024 padded inside) | 48 → 48 | 69 s | 72 s | 20 GB | none |

- Peak memory is the restoration worker's lifetime `phys_footprint_peak` (Load and both requests). The port's author measured 9.2 GB at 640×384, 19.1 GB at 1280×768 and 33.7 GB at 1920×1152 output (BF16, M5 Max).
- About 1.5 s per output frame. Load was not timed on its own; the first request, Load included, was not slower than the second.
- Two runs with the same seed agree to 41.3 dB PSNR (minimum 35.7 dB per frame); the MP4 files are not byte-identical. Whether the hardware H.264 encoder or GPU arithmetic causes the difference was not checked.
- Against the port's own BF16 run of the same clip (lab, 3 Nov), 29.0 dB PSNR; against a bicubic enlargement of the input, 27.6 dB.
- FP16 against the BF16 release on fixed inputs: `log/001-native.md`.

Not checked: speed on an idle Mac, outputs above 1280×960, the 4.2-megapixel bound in practice, cancellation under load, a clip with audio (the shared `MediaBridge.keepAudio` path is covered by `AudioMuxTests`), and the makers' PyTorch code as a reference.
