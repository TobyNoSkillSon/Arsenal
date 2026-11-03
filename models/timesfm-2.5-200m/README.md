# TimesFM 2.5 200M

Forecasts the next values of one numeric series as nine quantiles (0.1 to 0.9) for each future step. Google Research's TimesFM 2.5 (a decoder-only transformer) reads the last 8192 values in patches of 32, decodes 128 steps per pass and continues autoregressively up to 1024 steps, with a separate head for the quantile spread.

Input: `values` (the past observations, oldest first, at most 8192 numbers) and `horizon` (1 to 1024 steps).

Example request:

```json
{"values": [112, 118, 132, 129, 121, 135, 148, 148, 136, 119, 104, 118], "horizon": 12}
```

Output: `median`, `quantiles` (one row per step, one value per level) and `quantile_levels`.

On the 144 monthly airline passenger counts (1949 to 1960) it forecasts a median of 441, 406, 459, 480, 479, 537, 619, 613, 519, 447, 389, 449 for the next twelve months: the summer peak and winter dip continue, with growth.

It runs the model card's settings: input normalisation, flip invariance, non-negative output for a non-negative history, the continuous quantile head and quantile-crossing repair.

Limits: univariate only (the maker's covariate regression is not ported); horizon at most 1024; missing values are not accepted.

Parity: matches the maker's timesfm code on Apple GPU in fp16 with cosine 0.99999996 and a largest difference of 0.53 on forecasts up to 697 (airline series, 24 steps), 1.2 over 300 steps and 4.3 over 1024 steps (cosine 0.999999). Details in `log/003-parity.md`.

Source: [google/timesfm-2.5-200m-pytorch](https://huggingface.co/google/timesfm-2.5-200m-pytorch/tree/1d952420fba87f3c6dee4f240de0f1a0fbc790e3), Apache-2.0. Runs as FP16 through the Arsenal kit runner (MLX).

## Agent usage notes

As of 31 Oct 2026, from this search.

**Use it**
- Input and format: One numeric series, oldest first [model card, 2026-10-31 fetch, first-hand].
- Prompts, reference audio or voice settings, if relevant: Not relevant.
- Parameters people recommend: Reinstall the latest timesfm package after the 2025-10-02 QKV update [model card, 2025-10-02, first-hand]. Compile example: max_context 1024, max_horizon 256, an example rather than a limit [model card, 2026-10-31 fetch, first-hand].
- Arsenal call: `POST /arsenal/v1/models/timesfm-2.5-200m/ops/forecast`, or the same operation through MCP. Vendor example: `timesfm.TimesFM_2p5_200M_torch.from_pretrained("google/timesfm-2.5-200m-pytorch", torch_compile=True)`, then `model.forecast(horizon=12, ...)`. The card labels the first quantile column as the mean [model card, 2026-10-31 fetch, first-hand].

**Good and bad**
- Good at: The paper's abstract claims zero-shot performance close to supervised models on each dataset. It predates 2.5 [arXiv 2310.10688, first submitted 2023-10-14, first-hand].
- Bad at: Latency: 86 ms warm median on a DGX Spark, against 3.7 to 6.5 ms for Chronos-2 [third-party test, 2026-05-25, first-hand].
- Against the obvious alternatives: On ETTh1 (horizon 96, context 512) it had the best MASE, 1.106, against 1.149 and 1.154 for Chronos-2 120M and 28M. It placed last on AirQuality [same test].

**Complaints, bugs and fixes**
- Complaint or known bug: from_pretrained on the 2.5 checkpoint raises a TypeError on an unexpected 'proxies' or 'paper_url' keyword (issue #412, 2026-04-30, first-hand). No maintainer reply.
- Fix or workaround: Reported by one user, not confirmed: a monkeypatch setting proxies and resume_download defaults, posted as "Try this" (#412, 2026-05-01). Open PR #531 (2026-09-24, unmerged) raises the huggingface_hub minimum to 0.30.0.

**Evidence**
- Strength: thin. Maker claims plus one hands-on test. Reddit not re-searched.
- Sources: model card (fetched 2026-10-31, first-hand); issue #412 and PR #531 (read 2026-10-31, first-hand); DGX Spark test (2026-05-25); arXiv 2310.10688 (first submitted 2023-10-14).
