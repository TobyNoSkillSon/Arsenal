# Real check, 2 Nov 2026

One isolated end-to-end run of the kit runner on the host, headless (ARSENAL_SUPPORT_DIR / ARSENAL_CACHE_DIR in a temporary profile), under the GPU lock. M5 Max, Mac in use. No speed figures are claimed.

- Get's cast through the runner's `convert` op on an APFS clone of the pinned files: `model.safetensors` 418.6 MB FP16, no tensor kept in float32.
- Load from the cast copy, then `forecast` on the airline counts (horizon 24) and on the covariate request (two known-future covariates, one past and one categorical, horizon 40): counts, keys, finite and non-crossing quantiles; both identical to the parity run.
- A one-value history (42 at horizon 3) answers 42, 42, 42.
- Refused as invalid_request: horizon 4097, an all-null history, a covariate shorter than len(values) + horizon.
- Peak footprint about 1.0 GB (runner status frames).
- Not checked: the app's own Get and Load (headless rule: the runner was driven directly), requests near the 32-series and 32,768-step limits.
