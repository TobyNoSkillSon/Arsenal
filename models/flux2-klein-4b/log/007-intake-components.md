+++
hypothesis = "Identical component inputs and frozen BF16 bounds can distinguish format errors from different generated noise."
change = "Capture independent component replays, activation checks and balanced warm samples."
measurement = "Busy M5 Max quick check; historical component policy, superseded by log 008."
numbers = "Per-component bounds, warm samples and physical peaks are retained in the linked intake result."
verdict = "inconclusive"
links = ["../results/intake/intake-2026-10-15T11-14-00Z-acb8d191-2607-48f4-8a93-5c251b3ac926.json"]
+++

# Intake component gate, 15 October 2026

Round 2 supersedes the precision conclusions of the round-1 PSNR screen, while preserving that historical capture. Equal seeds did not establish equal initial noise across activation dtypes. The corrected intake saves identical token IDs/mask, reference conditioning, latents and timestep/position inputs at steps 0/1/3, and the final decoder input. Each component is independently replayed. A/A reference repeats were identical; bounds were frozen from that floor plus one BF16 ULP before any candidate was evaluated. Internal activation maxima, pre-clamp finiteness and clamp counts are also gated.

Run: `intake-2026-10-15T11-14-00Z-acb8d191-2607-48f4-8a93-5c251b3ac926`. Full metrics/bounds and balanced warm samples are in `../results/intake/<run>.json`; `../gate/evidence/intake-latest.json` links the exact run and bounds hash. Tensor archives stay local, with their hashes and shapes in the run's fixture manifest.

BF16 passes precision parity. FP16 fails: denoiser relative L2 0.14783/0.16256/0.05571 against 0.005759/0.005763/0.005765 at steps 0/1/3, and non-finite pre-clamp activations at steps 1/3 (121/134 clamp events). The corresponding BF16 maximum reaches 98,816. This establishes an activation-range problem on the saved inputs; finite checkpoint conversion alone did not test it. FP16 prompt and decoder errors also exceed their frozen bounds.

Int8/int4 pack every eligible large transformer and Qwen matrix/embedding. Their denoiser and prompt errors exceed the same component bounds. The small VAE remains BF16 and is exact on the teacher-forced decoder input. Every converted archive hash matches round 1, so its recorded component coverage remains applicable. These are conservative numerical failures, not evidence that every generated image is perceptually unsuitable: the single saved image per tier is a plausible red teapot, with SSIM diagnostics and a region-by-region sweep saved separately.

Warm production 1024x1024/four-step medians (min–max): BF16 4.819 s (4.019–5.639), FP16 4.575 s (4.476–4.624), int8 5.051 s (4.584–5.581), int4 4.868 s (4.675–5.386). Each candidate has four balanced ABBA BAAB samples against a resident BF16 baseline; the baseline pool has twelve samples. Baseline drift reaches 12.5%, so no speed winner is established. Per-worker lifetime physical peaks through Load and execution: 20.047/20.105/12.809/9.079 GB. Load and cold latency remain separate in the run.

Original BF16 remains selected. No new FP16 or whole-part quant recipe is enabled. Independent stock/vendor fidelity remains unqualified; this run does not turn the constrained mflux capture into stock parity. The bounded run completed without swap growth; all lab weight copies were removed. No install or manifest change was made.
