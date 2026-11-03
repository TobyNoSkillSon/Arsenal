+++
hypothesis = "Default served Load preserves the precision verdicts when qualification uses its exact kernels and settings."
change = "Share the served Load builder, bind folder kernel contents/settings per model, and generate binding receipts inside intake completion."
measurement = "M5 Max, 40 GPU cores, 15 October 2026; busy Mac; 64 saved-input components, three served probability vectors and ABBA BAAB warm requests."
verdict = "kept"
numbers = { fp16_max_relative_l2 = 0, fp16_min_cosine = 1, fp16_max_probability_delta = 0, fp16_milliseconds_per_decision = 8.40649998281151, fp16_peak_gb = 1.184269848, int8_max_relative_l2 = 0.01105382974925903, int8_min_cosine = 0.9999683689169698, int8_max_probability_delta = 0.0012012720108032227, int8_milliseconds_per_decision = 9.560562495607883, int8_peak_gb = 0.744342584, int4_max_relative_l2 = 0.18063032578271446, int4_min_cosine = 0.9875735679947254, int4_max_probability_delta = 0.035153716802597046, int4_milliseconds_per_decision = 9.525124973151833, int4_peak_gb = 0.5369375519999999 }
links = ["results/intake/intake-2026-10-15T18-20-40Z-b687a9b0-20f5-4416-a3ed-b77dee4288c4.json", "gate/evidence/intake/intake-2026-10-15T18-20-40Z-b687a9b0-20f5-4416-a3ed-b77dee4288c4/quality-binding.json"]
+++
# Default Load precision intake — 15 October 2026

Run `intake-2026-10-15T18-20-40Z-b687a9b0-20f5-4416-a3ed-b77dee4288c4` replaces the table figures from the preceding intake. The original captures and logs remain historical evidence.

The earlier intakes omitted the folder kernels and remain stock-matmul history, not default Fast Load evidence. This run sends the same absolute M5 `smallm-tile` kernel/header paths and `stock_mlx=false` load setting as ordinary Load. The worker reports the tile active for FP16; int8/int4 explicitly fall back to stock matmul because the tile accepts half weights only. Requested configurations and actual selections/fallback reasons are saved in every row. Language-graded states and the quality limits are unchanged.

| Tier | Max relative L2 vs half | Min cosine vs half | Max absolute probability delta vs half | Warm ms/decision | Peak GB per worker | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| fp16 | 0.00000000 | 1.00000000 | 0.00000000 | 8.406 | 1.184270 | pass |
| int8 | 0.01105383 | 0.99996837 | 0.00120127 | 9.561 | 0.744343 | pass |
| int4 | 0.18063033 | 0.98757357 | 0.03515372 | 9.525 | 0.536938 | fail |

Int8 passes; int4 fails its component bounds and remains offered with the failed-quality warning. Bounds remain relative L2 ≤0.05, cosine ≥0.999, finite activations/outputs, no new clamps, unchanged answers and max |dp| ≤0.017. This is quantization against the pinned served half implementation, not new independent vendor/task qualification.

The guarded intake took 19.07 seconds under `scripts/mlock gpulock`; aggregate worker cap 30.829 GB, observed aggregate peak 1.927 GB. Existing swap 1.023279 GB did not grow. Default-workload peaks cover fresh Load and decisions, after probe workers are discarded. Busy-machine timings are quick checks; no speed winner is claimed.

The installed/HF-cache originals were hashed before copying, APFS-cloned into the lab, checked unchanged afterwards, and all weight/cache/donor copies deleted. No download, install, signing, push or merge. `source-copy-receipt.json` records every original pin. `build-receipt.json` records the actual worker, source closure, effective load identity, MLX pin and privately rebuilt no-fast-math metallib. The shared metallib cache had a stale digest; rebuilding the pinned ten Metal sources reproduced the existing metallib bytes without changing the shared cache.

Intake completion wrote both the immutable per-run binding and the model binding directly from this exact result, worker identity, bounds and emitted table stats. No temporary binding generator or filename selection was used.
