# Chronos-Bolt Small

Forecasts the next values of one numeric series as nine quantiles (0.1 to 0.9) for each future step. Amazon's Chronos-Bolt Small (48M parameters, a T5 encoder-decoder) reads the last 2048 values in patches of 16 and predicts 64 steps in one pass; longer horizons, up to 1024, use the maker's rollout, which feeds each quantile back as context.

Input: `values` (the past observations, oldest first, at most 8192 numbers; only the last 2048 are used) and `horizon` (1 to 1024 steps).

Example request:

```json
{"values": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118], "horizon": 12}
```

Output: `median`, `quantiles` (one row per step, one value per level) and `quantile_levels`.

On the 144 monthly airline passenger counts (1949 to 1960) it forecasts 448, 463, 488, 506, 523, 574, 635, 639, 573, 498, 458, 490 for the next twelve months: the summer peak and winter dip continue, with growth.

Limits: univariate only (no covariates); past 64 steps quality may degrade, as the maker warns; only the trained quantile levels are returned.

Parity: matches the maker's chronos-forecasting code on Apple GPU in fp16 with cosine 1.0000000 and a largest difference of 0.47 on forecasts up to 742 (airline series), and 0.81 over a 100-step rollout; details in `log/003-parity.md`.

Source: [amazon/chronos-bolt-small](https://huggingface.co/amazon/chronos-bolt-small/tree/772f3d25d38aec6d914c8949dab4462e2d46f5d8), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: A 1D tensor, a list of 1D tensors, or a left-padded 2D batch; output [series, quantiles, horizon]. [HF model card, undated, first-hand]
- Prompts, reference audio or voice settings, if relevant: Not found in sources checked.
- Parameters people recommend: Maker's example: bfloat16, prediction_length 12, device cuda, cpu or mps. [HF model card, undated, first-hand]
- Arsenal call: `POST /arsenal/v1/models/chronos-bolt-small/ops/forecast`, or the same operation through MCP. Vendor example (call only):

```python
pipeline = BaseChronosPipeline.from_pretrained("amazon/chronos-bolt-small", device_map="cuda", torch_dtype=torch.bfloat16)
forecast = pipeline.predict(context=torch.tensor(df["#Passengers"]), prediction_length=12)
```

**Good and bad**
- Good at: Maker says up to 250 times faster, 20 times more memory-efficient than original Chronos, and more accurate. [HF model card, undated, first-hand (maker)]
- Bad at: Maker says the companion package is for research; it recommends AutoGluon for production. [HF model card, undated, first-hand]

**Complaints, bugs and fixes**
- Complaint or known bug: predict_quantiles documents its second value as the mean; for Chronos-Bolt it is the median. CPU repro by the reporter; no maintainer reply. [issue #542, opened 22 Oct 2026, first-hand (user repro)]
- Fix or workaround: No fix merged; docs-only PR #543 is open and unconfirmed.
- Complaint or known bug: Chronos Base batch and one-by-one results differed (Jan 2025); the maintainer blamed the T5 sampler's random draws. For Bolt, the reporter found their own series-order mismatch. [issue #262, first-hand (user report, maintainer comments)]
- Fix or workaround: Reported by one user, not confirmed by the maintainer: keep series labels aligned with the context.

**Evidence**
- Strength: thin. Maker's card and two GitHub threads; no independent tests.
- Sources:
  - https://huggingface.co/amazon/chronos-bolt-small, undated (maker)
  - https://github.com/amazon-science/chronos-forecasting/issues/542, opened 22 Oct 2026, newest comment read 26 Oct 2026
  - https://github.com/amazon-science/chronos-forecasting/issues/262, opened 15 Jan 2025, closed 16 Jan 2025
