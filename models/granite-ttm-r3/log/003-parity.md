+++
hypothesis = "The MLX kit runner reproduces the maker's granite-tsfm TTM R3 forecast."
change = "None: measurement only."
measurement = "Maker code vs kit runner on the same fixed inputs, Apple M5 Max GPU, 2026-10-30."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-30.json"]
numbers = { airline_cosine = 0.99999998, airline_max_abs_diff = 0.288, airline_ref_max = 663.1, short_cosine = 0.99999996, short_max_abs_diff = 0.037 }
+++
# 003 Parity with the maker's code

Reference: granite-tsfm at git 5991892 (`tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py`,
`TinyTimeMixerForDecomposedPrediction.from_pretrained` on the pinned FP32 snapshot ea17cfd), run in PyTorch 2.15 on MPS.
The trend and residual forecaster networks ran in float16. The decomposed model's glue (std scaler, `robust_lowess_like`,
residual detrend, `combine_quantiles_fast`, inverse scaling) ran in float32, because the maker's scaler overflows
float16 on data in real units: the sum of squared deviations of the airline series passes 65,504, the scale becomes
inf and the whole forecast NaN (seen in a first all-fp16 run). An all-fp32 run of the same script reproduced the
maker's own `forward()` output exactly (median 417.2, 393.6, 426.0 ...).

Candidate: the kit runner (`runners/swift`, MLX, FP16 weights; scaler and trend split on the host in Double),
driven over the wire on the same inputs. Inputs: `airline.json` (the 144 monthly airline passenger counts 1949-1960,
horizon 24) and `awkward.json` (7 values 3,1,4,1,5,9,2, horizon 5). Compared: all quantiles of every step (9 levels).

| input | values | max abs diff | cosine | centred cosine |
|---|---|---|---|---|
| airline, vs fp16-network ref | 216 | 0.288 (0.04% of 663) | 0.99999998 | 0.9999994 |
| airline, vs all-fp32 ref | 216 | 0.255 | 0.99999999 | 0.9999985 |
| 7 values | 45 | 0.037 (0.1% of 37) | 0.99999996 | 0.9999999 |

Verdict: pass (cosine >= 0.999 on the forecast quantiles). The FP16 cast (cast:arsenal-kit-half-v1) changed no output
digit in the kit try that followed: the runner already computes in float16.

Scripts: lab/night1/granite-ttm-r3/{ref.py,drive.py,compare.py} (not tracked).
