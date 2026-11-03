# Granite TimeSeries TTM R3

Forecasts the next values of one numeric series as nine quantiles (0.1 to 0.9) for each future step, up to 30 steps ahead. IBM's TinyTimeMixer R3 (1.4M parameters, this is its 512-30-dec-90 checkpoint) splits the series into a smooth trend and a residual, forecasts each with a small gated MLP mixer, and combines the two ranges.

Input: `values` (the past observations, oldest first, 3 to 8192 numbers) and `horizon` (1 to 30 steps).

Example request:

```json
{"values": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118], "horizon": 12}
```

Output: `median`, `quantiles` (one row per step, one value per level) and `quantile_levels`.

On the 144 monthly airline passenger counts (1949 to 1960) it forecasts 417, 394, 426, 436, 455, 523, 611, 597, 489, 429, 364, 401 for the next twelve months: the summer peak and winter dip continue.

Limits: univariate only (the model's multivariate and exogenous modes are not ported); at most 30 steps, no rollout; the trend sees the last 512 values and the residual the last 90; only the trained quantile levels are returned.

Parity: matches the maker's granite-tsfm code on Apple GPU (networks in fp16, scaler and trend split in fp32 because the maker's scaler overflows fp16) with cosine 1.0000000 and a largest difference of 0.29 on forecasts up to 663 (airline series); details in `log/003-parity.md`.

Source: [ibm-granite/granite-timeseries-ttm-r3](https://huggingface.co/ibm-granite/granite-timeseries-ttm-r3/tree/ea17cfd2e3edcaea21eb8dcecd18bf88971482fa), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: Not found in sources checked. The card has no usage code; its recipes section says "To be released" [ibm-granite/granite-timeseries-ttm-r3 model card, 2026-10-31 fetch, first-hand].
- Prompts, reference audio or voice settings, if relevant: Not relevant, numeric input only.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `POST /arsenal/v1/models/granite-ttm-r3/ops/forecast`, or the same operation through MCP. The maker gives no usage example, only "To be released" [ibm-granite/granite-timeseries-ttm-r3 model card, 2026-10-31 fetch, first-hand].

**Good and bad**
- Good at: Maker-reported GIFT-Eval results for the pre-trained model: MASE 0.724, CRPS 0.520, setup not given [ibm-research/ttm-r3 model card, 2026-10-31 fetch, first-hand]. The ibm-granite card gives MASE 0.727 and CRPS 0.523 for it [ibm-granite card, 2026-10-31 fetch, first-hand].
- Bad at: Not found in sources checked.
- Against the obvious alternatives: Not found in sources checked for R3.

**Complaints, bugs and fixes**
- Complaint or known bug: Not found in sources checked for R3. Open granite-tsfm issue #542 (2026-09-14) asks maintainers to verify eval results on Papers with Code. Closed #538 (2026-08-20) asks which optimizer R3 pretraining used; a collaborator answers AdamW [ibm-granite/granite-tsfm issues, read 2026-10-31, opinion].
- Fix or workaround: Not found in sources checked.

**Evidence**
- Strength: thin. Two maker cards give different numbers, with no usage code and no independent test. Hub metadata lists 1,414,514 parameters for both. The licences differ: Apache-2.0 on ibm-granite, cc-by-nc-sa-4.0 on ibm-research. Reddit not re-searched.
- Sources: both model cards and Hub metadata (fetched 2026-10-31, first-hand); granite-tsfm issues #538 and #542 (read 2026-10-31, opinion).
