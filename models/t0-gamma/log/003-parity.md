+++
hypothesis = "The MLX kit runner reproduces the maker's tfc-t0 T0Forecaster forecast for t0-gamma."
change = "None to the port; measurement only. A lab-only capture (ARSENAL_T0_CAPTURE_DIR) saves each component's output."
measurement = "T0Forecaster.predict (tfc-t0 0.6.0, torch 2.15, MPS, dtype=float16: fp32 weights under torch.autocast fp16) vs the kit runner over the wire on five fixed requests, m5-max, 2026-11-02."
verdict = "kept"
links = ["../gate/golden/golden.json"]
numbers = { airline_cosine = 0.99999996, airline_max_abs_diff = 1.327, levels_max_abs_diff = 1.430, covariates_cosine = 0.99999999, covariates_max_abs_diff = 0.0095, adjusted_max_abs_diff = 1.0, short_max_abs_diff = 0.0272, worst_component_rel_l2 = 0.004, maker_mlx_vs_maker_torch_max_abs_diff = 0.927 }
+++
# 003 Parity with the maker's code

Reference: tfc-t0 0.6.0 from PyPI in an isolated venv, `T0Forecaster.from_pretrained` on the pinned FP32 snapshot
e08aee0 with `dtype=torch.float16` (the maker's half-precision path), on MPS. Captured once; raw tensors stay in
lab/evidence/t0-gamma. Fixture: gate/golden/fixtures.json (sha256 in golden.json).

Requests: the 144 airline counts at horizon 24 with the 21 trained levels; the same with levels 0.005, 0.1, 0.33,
0.5, 0.9, 0.995 (interpolation and both tails); a 100-step synthetic series with one missing value, two known-future
covariates over 120 steps and one past covariate with a missing value, horizon 20; the airline counts at horizon 12
with symmetric averaging, clipping and integer snapping (two passes); 7 values at horizon 5.

Final quantiles, every level of every step (and of the past covariate's forecast):

| request | values | largest gap | of reference max | cosine |
|---|---|---|---|---|
| airline, 21 levels | 504 | 1.327 | 885 | 0.99999996 |
| airline, 6 levels with tails | 144 | 1.430 | 912 | 0.99999994 |
| covariates, 2 rows | 120 | 0.0095 | 19.5 | 0.99999999 |
| adjusted (snapped to integers) | 36 | 1.0 | 698 | 0.99999961 |
| 7 values | 105 | 0.0272 | 39.9 | 0.99999977 |

Per component (each side builds its own tensors from the same fixture): scaled network input relative L2 at most
1.6e-4 (FP16 rounding of the input), location and scale 1e-7, patch encoder 2.5e-4, the 24 layers at most 4.0e-3,
output norm 2.2e-3, decoder 9.4e-4. Tolerance (golden.json): component relative L2 0.01;
outputs cosine 0.9999, relative L2 0.005, largest gap 0.5% of the reference maximum, plus one unit where snapping
rounds. Verdict: pass.

For scale, the maker's own MLX runtime (t0.mlx 0.6.0, weights cast to FP16) differs from its torch path by up to
0.927 on the airline request, and from this port by up to 2.25 (cosine at least 0.9999993).

Scripts (not tracked): lab/evidence/t0-gamma/scripts/{fixtures,ref,mlxref,drive,compare,realtest}.py.

## Review fixes, 2 Nov 2026

A coordinator review found four places where the runner differed from the maker or its own schema; all are fixed and
checked by `T0GammaFolderTests.requestLogicFollowsTheMakerAndTheSchema`, which compiles the new Foundation-only
`Request.swift` with `Tests/KitModels/t0-gamma/main.swift` (21 checks, no MLX, no weights):
- `snap_integers` rounded half away from zero; it now rounds half to even, as np.round and torch.round do.
- The clip and snap history tests read Double values; they now read the float32 values the maker's tensors hold.
- With `quantile_levels` lacking 0.5 the median was interpolated from the requested levels; 0.5 now always joins the
  pass, so the median is the model's own 0.5 level.
- Validation now matches ops/forecast.input.json: integer `horizon` only (no booleans, fractions or strings), boolean
  flags only, at most 99 levels, at most 31 series in each covariate list (the old total cap of 32 refused requests the
  schema allows).

The five golden requests were re-driven after the change on the same FP16 weights: every output identical to the
earlier port run to the printed precision (largest gaps 1.33, 1.43, 1.0 at a snapping flip, 0.0095, 0.027).
