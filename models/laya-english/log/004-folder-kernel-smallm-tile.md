+++
hypothesis = "The family small-M tile kernel (Fast) can come from the model folder as Metal source, compiled at load, with no loss against the compiled-in copy."
change = "SmallMGEMM tile-1 header and body moved verbatim (plus comments) to kernels/m5/smallm_tile{.h,}.metal, declared fast@m5 on site linear.down; VerdictSmallMGEMM.tile = the folder kernel (ArsenalKernels.FolderMLXKernel: compile preflight per instantiation, persisted verdicts); kernel stage of SmallMGate runs in process (the preflight replaces the gate child); SmallMChild removed."
measurement = "arsenal-kernels-smoke + kernels/probe.py under scripts/mlock gpulock, M5 Max, macOS 26.6 build 25G5034-class, fp16, 202 Verdict fixture items judged alone, arms alternated A B A B over two probe runs (4 pairs); quick checks on a shared machine, not Measured results."
verdict = "kept"
numbers = { single_median_ms_stock = [7.36, 7.35, 7.54, 7.42], single_median_ms_fast = [6.78, 7.13, 6.85, 6.81], mean_change_pct = -7.1, changed_answers = 0, items = 202, max_dp = 0.0066, load_gate_logits_rel = 0.0016, gemm_m96_speedup = 2.36, gemm_m192_speedup = 1.23, first_probe_regression_ms = "8.36 -> 8.28 / 8.44 -> 7.33 before the per-call key fix" }
links = ["../kernels/manifest.toml", "../kernels/probe.py", "../../../docs/evidence/mb-v2-kernels/laya-probe-2.json", "../../../docs/evidence/mb-v2-kernels/laya-probe-3.json", "../../../docs/evidence/mb-v2-kernels/gemm-bench-2-splitk-16x32-per-simdgroup.txt", "../../../docs/KERNELS.md"]
+++

# smallm-tile from the folder (Fast, m5)

The tile kernel is the one Verdict qualified on 28 Sep (8f6f346, single items −11 %); only its delivery changed. The
load-time gate is unchanged (SmallMGate: kernel self-test + Verdict shapes, then logits rel RMS ≤ 1e-2 and unchanged
answers on the built-in items), now keyed by the folder source's digest (`tile-1:<digest>`), so an edited source
requalifies.

Fast vs the Fast recipe without the kernel (`config.kernels = []`), same binary, 202 single items: median 7.42 → 6.89 ms
(−7.1 % over 4 alternated pairs, every pair faster), p90 lower in 3 of 4; 0 changed answers, max |Δp| 0.0066 (Verdict's
ruling: ≤ 0.01, ≥ 99.5 % unchanged). Batched (64 items) unchanged: the kernel serves rows 9–256 only. Per GEMM
(mlp.Wo with residual, dependent chain): 2.36× at M 96, 1.23× at M 192.

Found on the way: the first probe showed no gain because the library's per-call memo built MLX's kernel name as a string
(~30 calls per forward). An integer key per instantiation fixed it (probe 1 vs 2/3 in docs/evidence/mb-v2-kernels).
A folder kernel's host cost per call matters as much as its GPU time on a 7 ms request.
