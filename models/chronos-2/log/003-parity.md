+++
hypothesis = "The MLX kit runner reproduces the maker's chronos-forecasting Chronos-2 forecast for one univariate series."
change = "None: measurement only."
measurement = "Chronos2Pipeline.predict on MPS fp16 vs the kit runner on the same fixed inputs, m5-max, 2026-10-31."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { airline_cosine = 0.99999997, airline_max_abs_diff = 0.889, h100_cosine = 0.9999999, h100_max_abs_diff = 2.45, h1024_cosine = 0.9999965, h1024_median_max_abs_diff = 6.63, h1024_max_abs_diff = 889, short_cosine = 0.9999997, short_max_abs_diff = 0.136, ref_fp16_vs_fp32_max_abs_diff = 0.937 }
+++
# 003 Parity with the maker's code

Reference: chronos-forecasting 2.3.3.dev0 (git 1224faf, `src/chronos/chronos2/`), `Chronos2Pipeline.from_pretrained`
on the pinned FP32 snapshot 29ec376 with `dtype=torch.float16` (model weights in fp16, SDPA attention), model on MPS,
transformers 5.22, PyTorch 2.15. `pipeline.predict([context], prediction_length=H)`; the maker's instance scaling runs in
float32 inside the model.

Candidate: the kit runner (`runners/swift`, MLX, FP16 weights; scaling and rescaling on the host in Double), driven over
the wire. Inputs: the 144 monthly airline passenger counts 1949-1960 at horizons 24, 100 and 1024, and 7 values
3,1,4,1,5,9,2 at horizon 5. Compared: all 21 quantiles of every step.

| input | values | max abs diff | cosine | centred cosine |
|---|---|---|---|---|
| airline, H 24 | 504 | 0.889 (0.11% of 804) | 0.99999997 | 0.9999989 |
| 7 values, H 5 | 105 | 0.136 | 0.9999997 | 0.9999996 |
| airline, H 100 | 2100 | 2.45 (0.16%) | 0.9999999 | 0.9999987 |
| airline, H 1024 | 21504 | 889 (0.96% of 93,083) | 0.9999965 | 0.9999959 |

At H 1024 the largest gap sits on the 0.99 quantile at step 1003 (76,618 vs 75,729): the arcsinh output scale turns
fp16 rounding into large absolute moves in the far tails. The median over 1024 steps differs by at most 6.6 (of 2039).
For scale, the maker's code in fp16 and in fp32 differ by up to 0.937 on the airline H 24 input; the runner is 0.805
from the fp32 reference. Verdict: pass (cosine >= 0.999).

Scripts: lab/night2/chronos-2/{ref.py,drive.py,compare.py} (not tracked).
