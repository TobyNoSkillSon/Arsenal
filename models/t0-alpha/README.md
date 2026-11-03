# T0 Alpha

Forecasts the next values of one numeric series as quantiles: a 0.1, 0.25, 0.5, 0.75 and 0.9 range for each future step. The weights come from The Forecasting Company (theforecastingcompany/t0-alpha, Apache-2.0), and the Arsenal kit runner serves them as FP16 on MLX.

Input: `values` (the past observations, oldest first, at most 8192 numbers) and `horizon` (1 to 1024 steps).

Example request:

```json
{"values": [10.0, 10.1, 10.2, 10.3, 10.4, 10.5, 10.6, 10.7], "horizon": 4}
```

Output: `median`, `quantiles` (one row per step, one value per level) and `quantile_levels`.

On the 144 monthly airline passenger counts (1949 to 1960) it forecasts a median of 468, 461, 507, 502, 503, 567, 644, 627 for the next eight months: the summer peak continues, with growth.

Limits: univariate only; horizons above 1024 are not supported; only the trained quantile levels are returned; trends are under-extrapolated.

Parity: matches the maker's tfc-t0 code (T0Forecaster, fp16 autocast on Apple GPU) with cosine 0.99999996 and a largest difference of 0.68 on forecasts up to 740 (airline series, 24 steps), and 1.6 over 100 steps; details in `log/003-parity.md`.

Source: [theforecastingcompany/t0-alpha](https://huggingface.co/theforecastingcompany/t0-alpha/tree/741927847fe02367624749bb9290b2f09f128e7d), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: a numeric series, with NaN for missing values. Horizons up to 1024 decode in one pass; known future covariates are conditioned on, not forecast. [HF model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not applicable to a numeric forecaster.
- Parameters people recommend: install `tfc-t0[torch]` or `tfc-t0[mlx]>=0.6.0`; a bare install has no runtime. Trained quantile levels are 0.1, 0.25, 0.5, 0.75 and 0.9. [HF model card, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/t0-alpha/ops/forecast`, or the same operation through MCP. Vendor example: `model = T0Forecaster.from_pretrained("theforecastingcompany/t0-alpha").eval()`, then `model.predict(context, horizon=64, quantile_levels=[0.1, 0.5, 0.9])`. [HF model card, undated, first-hand]

**Good and bad**
- Good at: self-reported CRPS 0.4941 on GIFT-Eval and skill 42.2 on fev-bench. The paper gives no rank for t0-alpha. [arXiv 2609.24559, first submitted 2026-09-21, first-hand (maker)]
- Bad at: Not found in sources checked. The card says forecasts are "probabilistic estimates, not guarantees" and advises backtesting. [HF model card, undated, first-hand]
- Against the obvious alternatives: Not found in sources checked.

**Complaints, bugs and fixes**
- Complaint or known bug: none found; the tfc-t0 tracker showed 0 open issues on 2026-10-31. Fix or workaround: Not applicable. [tfc-t0 GitHub issues, first-hand]
- Complaint or known bug: Not found in sources checked. The card's two community discussions were not read. Fix or workaround: Not found in sources checked. [HF model card, undated, first-hand]

**Evidence**
- Strength: thin. Only the maker's paper and card; no independent test found.
- Sources:
  - t0 paper, https://arxiv.org/abs/2609.24559, first submitted 2026-09-21, first-hand (maker).
  - t0-alpha model card, https://huggingface.co/theforecastingcompany/t0-alpha, undated, first-hand (maker).
  - tfc-t0 issues, https://github.com/theforecastingcompany/tfc-t0/issues, checked 2026-10-31, first-hand.
