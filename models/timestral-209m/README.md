# Timestral 209M

Mistral AI's largest time-series forecaster reads a numeric history and returns a range of likely futures: 21 quantiles from 0.01 to 0.99 for every future step. A 24-layer patch transformer (209 million parameters) cuts the history into 32-step patches and alternates attention along time with attention across related series. It reads up to 32,768 past steps and forecasts 1,024 steps per pass. Arsenal runs a Swift and MLX port of the maker's own code in FP16.

## What it is for

- Probabilistic forecasts of load, sales, traffic, prices or counts, with no training on your data
- Forecasts that use known future inputs: calendar features, holidays, promotions, planned prices, weather forecasts, numeric or categorical
- Several related series forecast together, plus covariates known only in the past
- Long histories and long horizons: 32,768 past steps, up to 4,096 future steps
- Series with gaps: missing observations are null

The maker reports zero-shot GIFT-Eval MASE 0.658 and CRPS 0.446, TIME MASE 0.626 and CRPS 0.522, and a fev-bench skill score of 50.39, ahead of t0-gamma, TimesFM-3, Toto-2.0 and Chronos-2 on all three. These are the maker's figures, not reproduced here.

## Inputs and outputs

### `forecast`

| Field | Type | Notes |
|---|---|---|
| `values` | array of number or null | Required. The target's history, oldest first; null marks a missing step. 1 to 32,768 steps, at least one a number. |
| `horizon` | integer 1–4,096 | Required. Steps to forecast. Up to 1,024 come from one pass; longer horizons are forecast in blocks of 1,024, each block's median appended to the history, as the maker does. |
| `quantile_levels` | array | Which of the 21 trained levels to return (0.01, 0.05, 0.1 … 0.9, 0.95, 0.99), in any order; the reply lists them ascending. All 21 by default. No interpolation between them. |
| `co_targets` | array of series | Further target series over the same past steps, forecast jointly with `values` (the maker's multivariate mode). |
| `past_covariates` | array of series | Numeric covariates known only over the past (same length as `values`). Read, not forecast. |
| `covariates` | array of series | Numeric known-future covariates over `len(values) + horizon` steps. |
| `categorical_past_covariates`, `categorical_covariates` | arrays of code series | Category codes (any numbers; null is its own category), past-only or known-future. The maker's target encoding turns them into numbers against the target's history; with `co_targets` the codes are used as they are. |
| `symmetric_averaging` | boolean | Average with the mirrored forecast of the negated series (two passes). Default true, as the maker. |
| `clip_nonnegative` | boolean | Clip at zero a series whose observed history is non-negative. Default true, as the maker. |

Each list of series holds at most 31, and a request at most 32 series in all. A reply holds at most 640,000 forecast numbers (forecast series x horizon x levels with the median, 16 MiB as JSON): at horizon 4,096 with all 21 levels that is the target and 6 co-targets; larger asks are refused before the model runs. A request body is at most 1 MiB of JSON over HTTP and MCP (about 100,000 numbers in all), so a 32,768-step context fits for one or two series, not for 32. A covariate value beyond the float32 range (about 3.4e38) counts as missing, as a target value does.

Returns `horizon`, `quantile_levels`, `median` (the model's own 0.5 level, also when not requested) and `quantiles` (one row per step, one value per level). With `co_targets`, `co_target_forecasts` holds a `median` and `quantiles` per co-target.

## Usage notes (2 Nov 2026)

What it does: zero-shot probabilistic forecasting exactly as the maker's `TimestralForecaster.forecast` with its defaults: causal per-patch scaling with an asinh squash, the [context ; forecast-query] two-stream pass with its masks, the monotone quantile head, the advancing refine frame inside a block, block-by-block decoding past 1,024 steps, the quantile cap, symmetric averaging and non-negative clipping. Inputs: one target series per request, with optional co-targets and covariates. Limits: the maker's batching of independent series in one call is not exposed (send one request per series); only the 21 trained levels come back; at most 32 series per request; horizons up to 4,096 steps; at most 640,000 forecast numbers per reply and 1 MiB of request JSON.

Source: `mistralai/Timestral-209M` at `9874d252`, one FP32 `model.safetensors` (837 MB). Get casts it to FP16 (`cast:arsenal-kit-half-v1`, 419 MB); no tensor is beyond FP16's range. The network runs in FP16 with norms reduced in float32 and the head's output in float32; the scaling statistics run in float64 and the forecast arithmetic in float32 on the CPU, as the maker's code does.

Parity: checked against the maker's own code (github.com/mistralai/timestral at `a534bdbd`), run once on the Mac's GPU with the weights in FP16, on six fixed requests: the airline counts, a 600-step series with gaps at horizon 100, a series with two known-future covariates (one a 0/1 indicator), one past covariate and one categorical covariate, two series forecast jointly, a 7-step series, and a 2,000-step series at horizon 1,100 (two blocks). Final quantiles: cosine at least 0.9999999, relative error at most 0.04%, largest gap at most 0.2% of the largest forecast value. Per component on each request's first pass (patch embedding, each of the 24 layers, output norm, quantile head): relative error at most 0.23%. The maker's host functions (scaling, quantile head, target encoding, refine frame, covariate statistics) are matched on fixed inputs by a weight-free test. Bounds were set before the comparison (cosine 0.9999, relative error 0.5%, gap 0.5%, components 1%). Not checked: the maker's benchmark scores, very wide requests near 32 series, speed and memory figures (none claimed).

## Licence and credit

Modified MIT (Mistral AI). Free use, changes and redistribution with the attribution notice "2026 - Mistral AI" and the licence kept, except that no rights apply to a company whose global consolidated monthly revenue (or its employer's) exceeded US$20 million in the preceding month, for the model and any derivative such as Arsenal's FP16 copy. Arsenal has no revenue; the terms stay visible as a condition. See `licence/LICENSE` and `licence/ATTRIBUTION.md`.
