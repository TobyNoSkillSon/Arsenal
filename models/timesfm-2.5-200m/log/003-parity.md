+++
hypothesis = "The MLX kit runner reproduces the maker's timesfm TimesFM 2.5 200M forecast with the model card's forecast settings."
change = "None: measurement only."
measurement = "TimesFM_2p5_200M_torch.forecast on MPS under torch.autocast fp16 vs the kit runner on the same fixed inputs, m5-max, 2026-10-31."
verdict = "kept"
links = ["results/quick-checks/kit-2026-10-31.json"]
numbers = { airline_cosine = 0.99999996, airline_max_abs_diff = 0.525, h300_cosine = 0.99999986, h300_max_abs_diff = 1.18, h1024_cosine = 0.9999989, h1024_max_abs_diff = 4.27, h1024_median_max_abs_diff = 3.12, short_cosine = 0.9999998, short_max_abs_diff = 0.0076, ref_fp16_vs_fp32_max_abs_diff = 0.449 }
+++
# 003 Parity with the maker's code

Reference: the maker's timesfm 3.0.2 (PyPI, `timesfm_2p5/timesfm_2p5_torch.py`), `TimesFM_2p5_200M_torch` with
`torch_compile=False` loaded from the pinned FP32 snapshot 1d95242 and moved to MPS (the module itself picks only cuda or
cpu), forecast under `torch.autocast` float16 (also run in fp32). PyTorch 2.15. ForecastConfig as on the model card:
normalize_inputs, use_continuous_quantile_head, force_flip_invariance, infer_is_positive and fix_quantile_crossing all
True; max_context = series length rounded up to 32 and max_horizon = horizon rounded up to 128, as compile rounds them.
Inputs passed as float32 (the maker's code cannot send float64 to MPS).

Candidate: the kit runner (`runners/swift`, MLX, FP16 weights; statistics and post-processing on the host in Double, a
key/value cache for the autoregressive steps), driven over the wire. Inputs: the 144 monthly airline passenger counts
1949-1960 at horizons 24, 300 (two autoregressive steps) and 1024 (seven), and 7 values 3,1,4,1,5,9,2 at horizon 5.
Compared: the nine quantiles (0.1 ... 0.9) of every step.

| input | values | max abs diff | cosine | centred cosine |
|---|---|---|---|---|
| airline, H 24 | 216 | 0.525 (0.08% of 697) | 0.99999996 | 0.9999952 |
| 7 values, H 5 | 45 | 0.0076 | 0.9999998 | 0.9999996 |
| airline, H 300 | 2700 | 1.18 (0.09%) | 0.99999986 | 0.9999984 |
| airline, H 1024 | 9216 | 4.27 (0.17%) | 0.9999989 | 0.9999943 |

Over 1024 steps the median differs by at most 3.1 (of 1145). For scale, the maker's code in fp16 and in fp32 differ by
up to 0.449 on the airline H 24 input; the runner is 0.465 from the fp32 reference. Verdict: pass (cosine >= 0.999).

Scripts: lab/night2/timesfm-2.5-200m/{ref.py,drive.py,compare.py} (not tracked).
