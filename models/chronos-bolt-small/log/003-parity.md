+++
hypothesis = "The MLX kit runner reproduces the maker's chronos-forecasting Chronos-Bolt forecast, rollout included."
change = "None: measurement only."
measurement = "ChronosBoltPipeline.predict on MPS fp16 vs the kit runner on the same fixed inputs, m5-max, 2026-10-30."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-30.json"]
numbers = { airline_cosine = 0.99999998, airline_max_abs_diff = 0.467, rollout100_cosine = 0.99999997, rollout100_max_abs_diff = 0.811, short_max_abs_diff = 0.0102, ref_fp16_vs_fp32_max_abs_diff = 0.288 }
+++
# 003 Parity with the maker's code

Reference: chronos-forecasting at git 1224faf (`src/chronos/chronos_bolt.py`), `ChronosBoltPipeline.from_pretrained`
on the pinned FP32 snapshot 772f3d2 with `dtype=torch.float16`, model on MPS, transformers 5.22 T5Stack, PyTorch 2.15.
`pipeline.predict(context, prediction_length=H)`; the maker's own instance scaling runs in float32 inside the model.

Candidate: the kit runner (`runners/swift`, MLX, FP16 weights; scaling and the rollout's quantile reduction on the host
in Double), driven over the wire. Inputs: `airline.json` (144 monthly airline passenger counts 1949-1960, horizon 24),
`awkward.json` (7 values 3,1,4,1,5,9,2, horizon 5) and the airline series at horizon 100 (one 64-step pass plus the
maker's rollout: 9 contexts, 81 forecasts reduced by torch.quantile). Compared: all quantiles of every step.

| input | values | max abs diff | cosine | centred cosine |
|---|---|---|---|---|
| airline, H 24 | 216 | 0.467 (0.06% of 742) | 0.99999998 | 0.9999986 |
| 7 values, H 5 | 45 | 0.0102 | 0.99999995 | 0.9999999 |
| airline, H 100 (rollout) | 900 | 0.811 (0.07% of 1091) | 0.99999997 | 0.9999992 |

For scale, the maker's code in fp16 and in fp32 differ by up to 0.288 on the airline input; the runner sits at that
floor (0.347 against the fp32 reference). Verdict: pass (cosine >= 0.999). The FP16 cast (cast:arsenal-kit-half-v1)
left the kit try output unchanged.

Scripts: lab/night1/chronos-bolt-small/{ref.py,drive.py,compare.py} (not tracked).
