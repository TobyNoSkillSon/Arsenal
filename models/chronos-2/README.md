# Chronos-2

Forecasts the next values of one numeric series as 21 quantiles (0.01 to 0.99) for each future step. Amazon's Chronos-2 (120M parameters, a T5-style encoder with group attention) reads the last 8192 values in patches of 16 and predicts up to 1024 steps in one pass.

Input: `values` (the past observations, oldest first, at most 8192 numbers) and `horizon` (1 to 1024 steps).

Example request:

```json
{"values": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118], "horizon": 12}
```

Output: `median`, `quantiles` (one row per step, one value per level) and `quantile_levels`.

On the 144 monthly airline passenger counts (1949 to 1960) it forecasts a median of 449, 429, 499, 494, 503, 575, 643, 644, 554, 484, 406, 453 for the next twelve months: the summer peak and winter dip continue, with growth.

Limits: univariate only (the maker's covariate and multivariate modes are not ported); horizon at most 1024; every value must be a number: the maker's code reads NaN in the context as missing, but this port has no way to send a gap, so fill or drop missing points first; only the trained quantile levels are returned.

Parity: matches the maker's chronos-forecasting code on Apple GPU in fp16 with cosine 0.99999997 and a largest difference of 0.89 on forecasts up to 804 (airline series, 24 steps); over 1024 steps the cosine is 0.9999965, the median stays within 6.6 of the reference and only the far 0.99 quantile, near 76,000, moves by up to 890. Details in `log/003-parity.md`.

Source: [amazon/chronos-2](https://huggingface.co/amazon/chronos-2/tree/29ec3766d36d6f73f0696f85560a422f50e8498c), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: in Arsenal, `values` (one numeric series, oldest first, every value a number) and `horizon`. The maker's library also takes covariates (`context_df`, `future_df`) and reads NaN in the context as missing; neither is ported here, so fill gaps before sending. [HF model card, 2026-06-05, first-hand; Arsenal runner]
- Prompts, reference audio or voice settings, if relevant: Not applicable to a numeric forecaster.
- Parameters people recommend: Not found in sources checked.
- Arsenal call: `arsenal_run(model:"chronos-2", operation:"forecast", input:{"values":[112,118,132,129],"horizon":12})`. Vendor example, for reference: `pipeline.predict_df(context_df, future_df=future_df, prediction_length=24, quantile_levels=[0.1, 0.5, 0.9], id_column="id", timestamp_column="timestamp", target="target")`. [HF model card, 2026-06-05, first-hand]

**Good and bad**
- Good at: self-reported state-of-the-art zero-shot accuracy on fev-bench, GIFT-Eval and Chronos Benchmark II; the abstract gives no figures. [arXiv 2510.15821, 2025-10-17, first-hand (maker)]
- Bad at: fine-tuning. Issue #457 (opened 2026-01-30, open) says it always runs on GPU 0; the workaround is `CUDA_VISIBLE_DEVICES`. Issue #285 (2025-02-25) is an EOFError on chronos-bolt-tiny, not Chronos-2. [chronos-forecasting issues #457, #285, first-hand]
- Against the obvious alternatives: self-reported fev-bench 47.28, below Timestral 50.39, t0-gamma 49.42 and TimesFM-3 48.99. [pasqualepillitteri.it, 2026-10-23, opinion]

**Complaints, bugs and fixes**
- Complaint or known bug: `predict_quantiles` documents its second value as the mean; for Chronos-2 it is the median. [issue #542, opened 2026-10-22, first-hand]
- Fix or workaround: none merged; no maintainer reply. A user proposed a docs-only fix on 2026-10-26. [issue #542, first-hand]
- Complaint or known bug: one tester found absent rows make chronos-forecasting forecast differently from Darts and AutoGluon. [DevelopersIO test, 2026-10-02, first-hand]
- Fix or workaround: Reported by one user, not confirmed: explicit NaN rows made all three agree in that test. [DevelopersIO test, first-hand]

**Evidence**
- Strength: thin. Maker's paper and card, one test, a few issues.
- Sources: paper https://arxiv.org/abs/2510.15821 (first submitted 2025-10-17); card https://huggingface.co/amazon/chronos-2 (2026-06-05); test https://dev.classmethod.jp/articles/how-chronos-2-forecast-changes-when-past-target-is-missing/ (2026-10-02); issues https://github.com/amazon-science/chronos-forecasting/issues (#285, #457, #542); article https://pasqualepillitteri.it/en/news/26634/time-series-ai-compared-timesfm-chronos-timestral (2026-10-23).
