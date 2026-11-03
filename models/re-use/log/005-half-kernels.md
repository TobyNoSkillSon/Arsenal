+++
hypothesis = "The Metal causal conv and selective scan can serve half activations if they round exactly where the stock MLX path rounds."
change = "Typed conv/scan kernels on the activation dtype; stock causal conv through MLX's depthwise kernel; MLX precise exp in the scan; half cases in the unit self-test."
measurement = "M5 Max, 40 GPU cores, 16 October 2026 local; busy-machine quick check through scripts/mlock gpulock."
numbers = { selftest_48k_db = 999, selftest_16k_db = 999, bf16_rtf = 0.06400574307190254, int8_max_relative_l2 = 0.013158087812270094, int4_max_relative_l2 = 0.10956251491804991, bf16_peak_gb = 1.247331576, added_swap_bytes = 0 }
verdict = "kept"
links = ["../results/intake/intake-2026-10-15T23-39-57Z-efa57184-1394-4c90-8609-f621539fc6e8.json", "../gate/evidence/quality-binding-2026-10-15.json", "../gate/evidence/source-get-2026-10-16.json", "../gate/evidence/cleanup-2026-10-16-r2.json"]
+++

Log 004 served stock MLX because the whole-model kernel self-test failed (BF16/int8/int4 38.2/38.5/20.3 dB against the 40 dB bound). The kernels had been qualified under FP32 execution and kept FP32 semantics after the half change. The Metal conv returned FP32, so SiLU, xProj and the scan's `u` saw unrounded values, and the scan computed `u * D` and `SiLU(z)` in FP32. Stock rounded each of these to BF16. Two further differences showed up only once those were aligned. The padded `Conv1d` call ran the stock depthwise convolution through MLX's grouped GEMM, which rounds half products more coarsely (BF16 relative L2 0.00236 against the exact convolution, compared with 0.00166 for one rounding). The kernel's `metal::exp` is also the fast form in MLX's JIT build, while MLX's Exp op is `metal::precise::exp`; the two differ on 74% of FP32 inputs in [-20, 5]. Teacher-forced, each block still differed in at most 10 of 25,920 values, but 30 free-running blocks amplified those one-ULP flips to 36.3 dB.

The kernels now follow the stock graph operation by operation: FP32 taps, softplus, decay, state and sequential sum (permitted numerical accumulators, FMA contraction off), half `u * D`, MLX's SiLU formula in half and one output rounding. The stock conv pads explicitly and uses MLX's depthwise kernel. Learned parameters stay half. Fixed-tensor tests show the conv, conv+SiLU, scan and one ReUseMamba layer bit-identical to stock in BF16 and FP16, and in-kernel SiLU bit-identical over all 65,536 half patterns. On the real BF16 weights the whole model matched stock on every sample at 48 and 16 kHz.

Two failed runs are kept as history. In `ac600967`, the first typed unit self-test fed FP16 a scale-8 case that overflows on both paths, so every load fell back to stock. In `cc674a2f` the fast exp remained, giving 39.6 / 36.3 / 20.3 / 20.6 dB, and every load again served stock. Their table rows were replaced by `efa57184`.

Final run `efa57184` downloaded the pinned files again through the lab Get, with the same hashes as 15 October. It re-ran the fresh source-format qualification on the current implementation; FP16 fails again and BF16 stays selected. Every BF16, int8 and int4 load passed the self-test: kernels exact, model 999 dB at both rates. Ordinary Load serves the Metal path for all three tiers. FP16 is a qualification trial only; its self-test reported non-finite output and it does not serve. The `arsenal-stats` binary was built from `dl-pipeline` 535f5bcc, which contains the fresh-BF16 fix for review S2 and does not touch the worker. The worker is this branch's build (Sound identity c4d5928e…).

| Tier | Verdict against BF16 | Max rel L2 | Min cosine | Log-mel diagnostic dB | Warm RTF s/s | Peak GB |
|---|---|---:|---:|---:|---:|---:|
| bf16 | Pass | 0.000000000 | 1.000000000 | 0.000000000 | 0.064006 | 1.247332 |
| int8 | Pass | 0.013158088 | 0.999914262 | 0.338612906 | 0.064282 | 1.244104 |
| int4 | Fail; offered with warning | 0.109562515 | 0.993985103 | 1.883825161 | 0.065052 | 1.238091 |

The workload is the shared three-second 24-kHz tone, so RTF is the warm median divided by 3 s. The old stock RTF was 2.47; this one is 0.064, about 39 times faster on busy-machine quick checks with different worker builds. That makes it a quick comparison, not a controlled speed claim. Peaks are fresh serving workers' lifetime physical footprints; dropping the stock scan's sequential intermediates halves them. The run took 33.3 s, and system swap stayed at 1,023,279,104 bytes. Pinned source and the lab diagnostic BF16 copy were deleted after hash verification. The installed app and profile were not touched.
