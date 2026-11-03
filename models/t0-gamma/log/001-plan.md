+++
hypothesis = "t0-gamma can join Arsenal as a kit-runner folder beside t0-alpha, from its pinned Hugging Face metadata and the maker's current code."
change = "Folder with weight pins, licence, attribution and a plan; source and differences from t0-alpha read from the card, config.json and tfc-t0 0.6.0."
measurement = "Metadata only: repository tree, sizes and LFS sha256 at the pinned revision, the kit's read-only plan, config.json and the tfc-t0 0.6.0 wheel source."
verdict = "kept"
numbers = "No measurement: metadata only."
links = ["../model.toml"]
+++
# 001 plan (2026-11-02)

Approved by the owner on 2 Nov 2026 as the successor of t0-alpha (theforecastingcompany/t0-gamma, Apache-2.0).

Kit plan (read-only): theforecastingcompany/t0-gamma at e08aee083778f1f32da4974c0a34276776a35c52, Apache-2.0,
255,614,293 parameters counted from the safetensors header, model.safetensors FP32 on disk (1022 MB), Arsenal serves FP16
(cast once at Get). No bundled runner; a kit runner (Path B).

What changed from t0-alpha, from the card, config.json and the tfc-t0 0.6.0 source (t0/config.py, scaler.py, rope.py):

- Size: embed_dim 512 to 1024 (about 102M to 256M parameters); 24 layers, 8 heads, patch 32, MLP 2048 and the
  2-time-then-1-group layer pattern are unchanged.
- Quantiles: 5 trained levels (0.1 to 0.9) to 21 (0.01, 0.05 to 0.95 in steps of 0.05, 0.99). The maker interpolates
  any requested level between them and uses exponential tails beyond.
- Scaler: eps 0.1 added to the variance ("variance_offset") to eps 0.01 as a lower bound on the std ("std_clamp").
- Cells without an observation (padding, missing, forecast region) enter as the scaled value of zero
  (scaler_zero_no_value = false) instead of zero.
- xPos scale layout "halves": the per-pair rotary scales repeat over the two halves of the head dimension, while the
  rotation still pairs adjacent coordinates.
- Context and horizon: unchanged. 1024 steps decode in one pass; longer horizons use the autoregressive rollout. The
  card states no context limit; this runner keeps t0-alpha's 8192.
- Covariates and multivariate input: the card documents both for the family (known-future covariates, extra target
  variates forecast jointly, NaN as missing). The t0-alpha runner was univariate only; this one ports them.

Kind: the same embedded `local.forecast` tree as t0-alpha, byte for byte, so both folders can be installed together.
The model declares its own `[[op]] forecast` with model-local schemas (`ops/`) that add covariates, past covariates,
quantile levels, null for missing values and the maker's three output adjustments.
