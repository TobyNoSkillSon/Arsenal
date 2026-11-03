+++
hypothesis = "On single items MLX routes the K > N Linears (mlp.Wo 2624->1024, head linear2 4096->1024) to its NAX split-K GEMM: two dispatches, an fp32 scratch and a separate residual add. Doing exactly MLX's arithmetic in one dispatch is faster and bit-identical (an Exact kernel)."
change = "kernels/any/splitk_exact{.h,}.metal, exact@* on site linear.down: MLX's fragment layout and matmul2d(16,32,16, relaxed, multiply_accumulate) descriptor, MLX's partition size and order, 0.0f + p0 + p1 reduction, MLX's axpby bias epilogue and a T add for the residual; host routes only shapes MLX itself splits (FolderKernelRoute.splitKPlan). Gated at load by ArsenalKernels.ExactGate (bitwise, 9 check outputs incl. one 14-row chunk)."
measurement = "Same setup as 004: arsenal-kernels-smoke --bench (64-step dependent chains, bit-identity after the chain) and kernels/probe.py (Exact recipe with kernel vs config.kernels = [], 4 alternated pairs), M5 Max, gpulock quick checks."
verdict = "kept"
numbers = { attempt1_32x32_per_simdgroup_m96_speedup = 0.54, attempt2_16x32_per_simdgroup_m96_speedup = 1.75, attempt2_m192_speedup = 1.27, attempt2_m640_speedup = 1.04, single_median_ms_stock = [7.81, 7.41, 8.24, 8.16], single_median_ms_exact = [7.15, 7.21, 7.67, 7.98], mean_change_pct = -5.1, wire_answers_identical = "202/202", load_gate = "bit-identical, 9 outputs" }
links = ["../kernels/any/splitk_exact.metal", "../../../Workers/Decisions/Sources/VerdictEngine/FolderKernelRoute.swift", "../../../docs/evidence/mb-v2-kernels/gemm-bench-1-splitk-32x32-per-simdgroup.txt", "../../../docs/evidence/mb-v2-kernels/gemm-bench-2-splitk-16x32-per-simdgroup.txt", "../../../docs/evidence/mb-v2-kernels/laya-probe-2.json", "../../../docs/evidence/mb-v2-kernels/laya-probe-3.json"]
+++

# splitk-exact (Exact, every NAX chip)

What MLX 0.32 does for x[96, 2624] · W[1024, 2624]ᵀ on an M5 Max (`matmul.cpp`): NAX split-K (K > 2·max(M, N), max ≤ 1024),
partition 2048 → two partitions [0, 2048) and [2048, 2624), 64×64 tiles of 2×2 simdgroups each owning 32×32, K walked in
32-wide steps of two 16-wide `matmul2d(16, 32, 16)` accumulations into fp32, partials written to an fp32 scratch
[2, M, N]; `gemm_splitk_accum` sums `0 + p0 + p1` and rounds to fp16; Laya then adds the residual in fp16.

Each output element's value depends only on its own sequence of mma steps and the partition order, not on the tiling, so
any tiling that keeps those is exact. Attempt 1 (one simdgroup per partition per 32×32 tile) was bit-identical but 0.54×
(M 96): 192 simdgroups each walking 1,312 K serially. Attempt 2 gives each simdgroup 16×32 (MLX's same 16×32 mma per
element): 384 simdgroups, 1.75× at M 96, 1.27× at M 192, 1.04× at M 640, bit-identical after 64 chained steps at
every M.

End to end (Exact recipe, 202 single items alone): median 7.91 → 7.50 ms (−5.1 % over 4 alternated pairs, every pair
faster), p90 lower in all 4; wire answers byte-identical to the stock arm on all 202 items. Batched requests: no
measurable change (larger M, smaller per-GEMM gain).

Both English K > N Linears are split by MLX at every M ≤ 1024 (head linear2 4096→1024 too, with its bias), so the kernel
serves them in single items and batches. Multilingual's 1152→768 is not split by MLX (K < 2·768) and stays stock; a
replay of MLX's regular NAX GEMM with the residual folded in is the next exact candidate there.
