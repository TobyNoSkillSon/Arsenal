+++
hypothesis = "The kit runner serves t0-gamma at FP16 with covariates, joint variates and any quantile levels."
change = "Kit runner (kit/runner template unchanged) with Model.swift ported from the maker's MLX runtime (structure) and the shared host code of tfc-t0 0.6.0 (input layout, scaler, masks, levels, adjustments)."
measurement = "Isolated profile, 2026-11-02, m5-max: the runner's convert op cast the cloned FP32 snapshot to FP16, Load from the cast copy, two fixture forecasts and three awkward inputs through scripts/mlock gpulock."
verdict = "kept"
numbers = { fp16_file_bytes = 511264312, tensors_kept_float32 = 0, checks_passed = 6, checks_total = 6 }
links = ["../gate/golden/golden.json"]
+++
# 002 Port and the real run

Port: one Swift module per maker module (patch encoder, time and group attention with per-head RMSNorm, SwiGLU,
24-layer stack, decoder). Rows are targets first, then known-future covariates, as `TimeSeries.from_array` builds them;
time attention runs per row (causal for targets, bidirectional for covariates), group attention per patch across rows
(a covariate reads only covariates). The window follows `prepare_rollout_buffer`: left padding to the patch grid, the
forecast region withheld for targets and copied from the covariates. Scaling (causal Welford statistics with the
std clamp for targets, global statistics for covariates), the quantile head, rescaling, level interpolation, the
exponential tails, symmetric averaging, clipping and integer snapping run on the host in Double.

Not ported: the autoregressive rollout past 1024 steps, trim_leading_missing, explicit PAD masks and batches of
independent series (one target series per request).

Real run (isolated ARSENAL_SUPPORT_DIR and ARSENAL_CACHE_DIR, weights APFS-cloned from the Hugging Face cache): the
convert op wrote a 511,264,312-byte FP16 model.safetensors with no tensor kept in FP32 (none beyond FP16's range);
the airline forecast (24 steps, 21 levels) and the covariates forecast (2 known-future covariates, 1 past covariate,
one missing value) returned finite, non-decreasing quantiles with the right counts and matched the parity run exactly;
a single value answered; horizon 1100, an all-null series and a covariate of the wrong length were refused with
invalid_request. No speed figures were taken.
