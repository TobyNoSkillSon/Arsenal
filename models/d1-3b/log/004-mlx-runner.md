+++
hypothesis = "An MLX port of the hybrid System One tree runs d1-3B much faster than torch MPS at the bf16 floor of fidelity."
change = "New default runner runners/mlx: the LFM2 hybrid stack in MLX with the vendor's prompt, readout and API; trees packed per pass; pruned tail layers; fused short-conv kernel; fp32 readout. Torch runner removed."
measurement = "M5 Max, 8 Oct 2026, interleaved with the torch MPS runner in one session (TMMTTM, daytime, Mac in use), in-process adapter calls: single decision 66.1 -> 38.5 ms, 32 ads x 3 questions 83.0 -> 39.5 ms/item, 32 ads x 1 question 56.1 -> 21.0 ms/item. Gate vs fp32 CPU: max |dp| 0.0039, 24/24."
verdict = "kept"
numbers = { torch_single_ms = 66.1, mlx_single_ms = 38.5, torch_batch32_3q_ms_per_item = 83.0, mlx_batch32_3q_ms_per_item = 39.5, torch_batch32_1q_ms_per_item = 56.1, mlx_batch32_1q_ms_per_item = 21.0, torch_load_s = 4.84, mlx_load_s = 0.52, torch_image_ms = 113.1, mlx_image_ms = 100.2, gate_max_dp = 0.0039, gate_argmax = 24, questions = 24 }
links = ["runners/mlx/engine.py", "runners/mlx/runtime.py", "gate/parity-bf16.json", "gate/reference-check.json", "results/m5-max.json"]
+++
# MLX runner

## What runs

`runners/mlx` keeps LiquidAI's prompt rendering, readout ids and `system_one` API (`vendor/prompt.py`, `vendor/api.py`, unmodified) and replaces the torch forward with `engine.py`:

- **Trees as the vendor defines them.** A request with several questions is one tree (state = trunk, questions = branches that see the trunk and their own earlier tokens, convolution continuing from the trunk's last two inputs, positions restarting at the trunk's end); a one-question request is one chain. Several trees share a pass up to 4,096 tokens, each attending only within itself.
- **Pruned tail.** After the last attention layer (27 of 30) only the answer slots and, through each 3-tap convolution, their two predecessors are computed. Same arithmetic per row, about 9 % fewer FLOPs.
- **Fused weights.** `w1|w3` as one GEMM and `q|k|v` as one: one wide product runs nearer the GEMM peak at small M.
- **Short-conv kernel.** One Metal kernel for B·x, the 3-tap conv over each token's predecessors and C·: bit-identical to the stock-op path (0 of 7.8M elements differ), 2.20 -> 0.36 ms per conv layer at 3,833 tokens. A load self-test falls back to stock ops if it fails (`engine_reason` says so).
- **fp32 readout.** Only the option tokens' logits are computed, in fp32. The vendor's bf16 head rounds logits to steps of 1/16-1/8, which turned the 0.46/0.54 near-tie of one ad into an exact 0.5/0.5; fp32 cut the gate's max |dp| vs fp32 from 0.0118 to 0.0039.
- **Pictures** keep the checkpoint's processor, SigLIP2 tower and projector in torch (MPS, bf16, non-causal attention, which is correct on MPS), loaded on the first image request; their features enter the MLX language model at the placeholders. Text-only use never imports torch.
- Batches: the adapter takes up to 32 items per runtime call, checks cancellation before every pass, and queues each pass while the previous one is read.

## Where the time went

Profile by ablation on 8 ads × 3 questions (3,833 tokens, one pass): the MLP is about two thirds of the pass and runs at ~55 TFLOPS; attention, convolution elementwise work and norms made up the rest. The whole pass is ~4.4 TFLOP, so ~40 ms per ad item is ~45 TFLOPS end to end, near the M5 Max bf16 GEMM ceiling (MLX GEMMs measure 50-62 TFLOPS at M ≥ 1,800). Per-change measurements (ABAB, medians, noisy daytime GPU):

| Change | Before | After |
|---|---:|---:|
| MLX port (unfused, global mask) vs torch MPS, 32 ads × 3 q | 101.5 ms/item | 49.9 ms/item |
| fused w1+w3 and q+k+v GEMMs, pruned tail, 32 ads × 3 q | 49.9 ms/item | 46.2 ms/item |
| per-tree attention instead of one block-diagonal mask, 8-ad pass | 428 ms | 405 ms |
| short-conv kernel, 8-ad pass | 417 ms | 396 ms |
| the two rows above plus 32-item groups with queued passes, 32 ads × 3 q | 46.2 ms/item | 39.3 ms/item |

## Fidelity

vs fp32 CPU: gate 0.0039 / 24 of 24; ads 0.012 / 94 of 96; ads one question 0.018 / 32 of 32; requirements 0.019 / 72 of 72. The two ad flips are genuine near ties (0.388 vs 0.380 choice; 0.499/0.501 Noul) of the kind the CPU bf16 floor also flips. Polish probe (in-process, same inputs and metrics): every question within one item of fp32; Polish Noul 24/32 (fp32 25, MPS runner 27, Laya 17).

## Rejected

int8 and int4 affine g64 weights (log/005), split-K down projection (log/006), buffer cache sizes (256 MB-2 GB all measured the same; 512 MB kept for footprint). Load time was not a target.
