+++
hypothesis = "The MLX kit runner reproduces the maker's tfc-t0 T0Forecaster forecast for t0-alpha."
change = "None to the port: measurement only. Kit.swift and Wire.swift were refreshed to the current kit's copies (the 26 Oct ones lacked the build's GPU self-test op); Model.swift is unchanged."
measurement = "T0Forecaster.predict on MPS with dtype=float16 (fp32 weights, torch.autocast fp16) vs the kit runner on the same fixed inputs, m5-max, 2026-10-31."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { airline_cosine = 0.99999996, airline_max_abs_diff = 0.683, h100_cosine = 0.99999993, h100_max_abs_diff = 1.635, short_cosine = 0.99999998, short_max_abs_diff = 0.0053, ref_fp16_vs_fp32_max_abs_diff = 0.249 }
+++
# 003 Parity with the maker's code

Reference: tfc-t0 0.6.0 from PyPI (with rotary-embedding-torch 0.9.1), `T0Forecaster.from_pretrained` on the pinned FP32
snapshot 7419278 with `dtype=torch.float16`, which is the maker's half-precision path: weights stay fp32 and the forward
runs under `torch.autocast` fp16 on MPS. PyTorch 2.15. `model.predict(values, horizon, quantile_levels=(0.1, 0.25, 0.5, 0.75, 0.9))`,
other options at their defaults (no symmetric averaging, clipping or rounding).

Candidate: the kit runner (`runners/swift`, MLX, FP16 weights; scaling and the quantile head on the host in Double),
driven over the wire. Inputs: the 144 monthly airline passenger counts 1949-1960 at horizon 24 and at horizon 100, and
7 values 3,1,4,1,5,9,2 at horizon 5. Compared: every quantile of every step.

| input | values | max abs diff | cosine | centred cosine |
|---|---|---|---|---|
| airline, H 24 | 120 | 0.683 (0.09% of 740) | 0.99999996 | 0.9999951 |
| 7 values, H 5 | 25 | 0.0053 | 0.99999998 | 0.9999997 |
| airline, H 100 | 500 | 1.635 (0.15% of 1064) | 0.99999993 | 0.9999981 |

For scale, the maker's code in fp16 and in fp32 differ by up to 0.249 on the airline input; the runner is 0.564 from the
fp32 reference. Verdict: pass (cosine >= 0.999). Horizon 1100 is refused with invalid_request by the runner (the maker
would continue with its autoregressive rollout, which this runner does not port).

Scripts: lab/night2/t0-alpha/{ref.py,drive.py,compare.py} (not tracked).
