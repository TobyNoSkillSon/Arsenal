+++
hypothesis = "Splitting the MLP down projection (K 10,752) into two K halves runs faster than MLX's single GEMM."
change = "w2 as two contiguous K halves, second product added through addmm."
measurement = "M5 Max, 8 Oct 2026, ABAB x6: 8-ad pass 356.7 -> 339.5 ms (-4.8 %), 16 one-question ads 356.5 -> 343.8 ms, single ad 41.4 -> 43.6 ms (+5 %)."
verdict = "rejected"
numbers = { pass_8ads_ms_before = 356.7, pass_8ads_ms_after = 339.5, single_ms_before = 41.4, single_ms_after = 43.6 }
links = ["runners/mlx/engine.py"]
+++
# Split-K down projection

Faster on batches, slower on single items, and inexact (one extra bf16 rounding of a half-sum). Inside daytime noise for the batch gain and a loss on the single decision, which matters as much; not kept.
