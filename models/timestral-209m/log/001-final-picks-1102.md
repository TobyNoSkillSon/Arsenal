+++
hypothesis = "A Swift MLX kit runner reproduces the maker's TimestralForecaster for mistralai/Timestral-209M at FP16."
change = "Added the folder: the network in Model.swift, the maker's host code (scaling, encoding, block decoding, refine frame, clampsym) in Foundation-only Request.swift, the embedded local.forecast kind of t0-gamma, schemas declaring every documented capability, the golden and a folder test."
measurement = "TimestralForecaster (timestral a534bdbd, torch 2.15.0) on MPS with load_model(dtype=float16), captured once on six fixed requests; the runner over the wire on the same requests; one isolated end-to-end run from the FP16 cast; m5-max, 2026-11-02, Mac in use."
verdict = "kept"
links = ["../gate/golden/golden.json"]
numbers = { airline_cosine = 0.99999996, worst_case_rel_l2 = 0.0004, worst_gap_fraction_of_ref_max = 0.00205, worst_component_rel_l2 = 0.00234, host_checks = 39, fp16_cast_mb = 418.6, peak_footprint_mb = 1007 }
+++
# 001 Port and parity (final picks, 2 Nov 2026)

Runtime choice (AGENTS.md order): no bundled runner or Swift port serves Timestral, and the maker publishes only PyTorch code, so this is our own MLX port as a kit runner, in the forecast contract t0-gamma established (the same embedded kind, byte for byte).

Reference: the maker's package from its GitHub repository at `a534bdbd`, in an isolated venv, `load_model(..., device="mps", dtype=torch.float16)` and `TimestralForecaster` with its defaults. MPS has no float64, so three weight-free host steps written in float64 (causal statistics, the refine frame, the categorical encoding) were wrapped to run on the CPU and return; nothing in the network moved.

Port notes:
- The maker's attention gate is stored as `_gate`; MLX Swift's module tree skips keys that start with an underscore, so it loads as `gate` (the first load named all 30 unused tensors).
- The fused QKV weight's layout is (qkv, head_dim, heads), not (qkv, heads, head_dim).
- RoPE rotates halves with only the first 16 of 32 frequencies nonzero (rope_partial_factor 0.5); the rest is the identity. MLXFast.RoPE's partial mode rotates a different pairing, so the tables are built explicitly.
- The two-stream attention (context queries over context keys, horizon queries over all keys) equals one attention with the full mask, because the context-to-horizon quadrant is empty and every row keeps its diagonal.

Results: every final quantile of the six requests within cosine 0.9999999 and 0.2% of the largest value; components at most 0.23% relative error; the forecasts from the FP16 cast copy equal those from the at-load cast exactly. Fixture: gate/golden/fixtures.json; raw captures and scripts in lab/evidence/timestral-209m (not tracked).

Not checked: benchmark scores, requests near 32 series or 32,768 steps with covariates (memory), speed.
