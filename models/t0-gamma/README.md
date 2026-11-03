# T0 Gamma

Forecasts the next values of a numeric series as quantiles, at 21 trained levels or any levels you ask for, and can
use known future inputs (calendar features, holidays, promotions, weather forecasts) and related series. The weights
come from The Forecasting Company (theforecastingcompany/t0-gamma, Apache-2.0), the successor of t0-alpha, and the
Arsenal kit runner serves them as FP16 on MLX.

Example request:

```json
{"values": [10.0, 10.1, null, 10.3, 10.4, 10.5, 10.6, 10.7], "horizon": 4, "quantile_levels": [0.1, 0.5, 0.9]}
```

Output: `median`, `quantiles` (one row per step, one value per level), `quantile_levels`, and
`past_covariate_forecasts` when past covariates were sent.

On the 144 monthly airline passenger counts (1949 to 1960) the 12-month median runs 448, 433, 494, 493, 505, 581,
669, 667, 566, 489, 433, 475: the summer peak continues, with growth.

## Usage notes (2 Nov 2026)

What it does: a probabilistic forecast of one target series, up to 1024 steps ahead in one pass.

Inputs:
- `values`: the past observations, oldest first, at most 8192; `null` marks a missing observation.
- `horizon`: 1 to 1024 steps.
- `quantile_levels` (optional): ascending levels in (0, 1). Default: the 21 trained levels 0.01, 0.05 to 0.95 in
  steps of 0.05, and 0.99. Levels between them are interpolated; levels beyond them follow exponential tails.
- `covariates` (optional): known-future series, each `len(values) + horizon` long (past and horizon), `null` for
  missing. The model conditions on them and does not forecast them.
- `past_covariates` (optional): related series observed over the same past steps. They attend to the target and are
  forecast jointly, as the maker's multivariate input does; their forecasts come back in `past_covariate_forecasts`.
- `symmetric_averaging`, `clip_nonnegative`, `snap_integers` (optional, off by default): the maker's adjustments.
  Symmetric averaging runs the model twice; the maker recommends all of its adjustments on. As in the maker's code,
  the history tests read the series as float32, and snapping rounds half to even (2.5 becomes 2).
- `median` is always the model's own 0.5 level from the same pass, also when `quantile_levels` leaves 0.5 out.
- Bad input is refused, as the schema says: a fractional, string or true/false `horizon`, true/false levels or numbers
  for the on/off options, more than 99 levels, or more than 31 series in either covariate list.

Limits: horizons above 1024 (the maker's autoregressive rollout) are not ported; one target series per request, with up to 31
past and 31 known-future covariates; trim_leading_missing, explicit padding masks and the separate historical-covariate role are not
exposed. A reply holds at most 640,000 forecast numbers (series x horizon x levels with the median, 16 MiB as JSON), so 99
levels at horizon 1024 allow 6 series; larger asks are refused before the model runs. Values must lie inside the float32
range (about 3.4e38), and levels must not round to 0 or 1 in float32, nor with `symmetric_averaging` their mirrors 1 - q (1e-7 and 0.9999999 are fine). A request body is at
most 1 MiB of JSON over HTTP and MCP, so the full 8192-step context with 62 covariates only fits with short numbers.
Forecasts are estimates, not guarantees: backtest on your own history.

Parity: matches the maker's tfc-t0 0.6.0 code (T0Forecaster on Apple GPU under FP16 autocast) on five fixed requests,
covering covariates, a missing value, interpolated and tail levels and the adjustments: cosine at least 0.9999996 and
a largest gap of 1.43 on forecasts up to 912 (airline), 0.0095 on the covariates request. Every component (input,
scaler statistics, patch encoder, each layer, decoder) agrees within relative L2 0.004. The maker's own MLX and torch
runtimes differ from each other by up to 0.93 on the same airline request. Details in `log/003-parity.md` and
`gate/golden/golden.json`; the maker's benchmark figures (fev-bench 49.42, GIFT-Eval CRPS 0.4520) are not reproduced.

Source: [theforecastingcompany/t0-gamma](https://huggingface.co/theforecastingcompany/t0-gamma/tree/e08aee083778f1f32da4974c0a34276776a35c52),
Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).
