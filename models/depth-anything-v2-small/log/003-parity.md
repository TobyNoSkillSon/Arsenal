+++
hypothesis = "The MLX kit runner matches the maker's transformers code on the same photo."
change = "None; comparison only."
measurement = "transformers 5.22.0 DepthAnythingForDepthEstimation + DPTImageProcessorPil + post_process_depth_estimation, torch 2.15.0 on MPS at float16, against the kit runner (FP16) on the same files, 2026-10-30, M5 Max."
verdict = "kept"
numbers = { cosine = 0.9999562, max_abs = 1.440, mean_abs = 0.0138, ref_max = 6.62 }
links = ["results/quick-checks/kit-2026-10-30.json"]
+++
# 003 Parity with the maker's code

Decisive output: the predicted relative depth at the photo's size (post_process_depth_estimation), float32 from each side.
Inputs: `photo-518x294.png` (a macOS Portrait background photo, a rainbow arch on a lawn, resized to 518x294 so the
processor's resize is the identity and only the network is compared), the same photo at 640x360 (full path: PIL-style
bicubic resize, network, bicubic upsample), and an awkward 33x17 version.

| input | pixels | cosine | max abs | mean abs | reference range |
|---|---|---|---|---|---|
| 518x294 (network only) | 152,292 | 0.9999562 | 1.440 | 0.0138 | 0 to 6.62 |
| 640x360 (full path) | 230,400 | 0.9999608 | 1.310 | 0.0130 | -0.44 to 6.63 |
| 33x17 (awkward) | 561 | 0.9997525 | 0.541 | 0.0302 | 0 to 4.47 |

Verdict: pass (cosine >= 0.999 on every input). The large maxima are single pixels on sharp depth edges (the arch
against the sky): 0.21% of pixels differ by more than 0.1 and the 99th percentile of |diff| is 0.070. A shift test
(comparing against the reference moved by one pixel in each direction) gives the lowest error unshifted, so there is
no misalignment; the edge pixels flip between the two half-precision implementations. Mean signed difference
-0.0097 (0.15% of the range).

The FP16-cast weights (tools/cast16.py) give a byte-identical map to the original FP32 files loaded as FP16.
Scripts: lab/night1/depth-anything-v2-small/{ref.py,drive.py,compare.py}.
