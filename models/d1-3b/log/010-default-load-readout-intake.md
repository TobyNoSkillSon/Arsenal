+++
hypothesis = "Default served Load preserves the precision verdicts when qualification uses its exact kernels and settings."
change = "Share the served Load builder, bind folder kernel contents/settings per model, and generate binding receipts inside intake completion."
measurement = "M5 Max, 40 GPU cores, 15 October 2026; busy Mac; 93 saved-input components, three served probability vectors and ABBA BAAB warm requests."
verdict = "kept"
numbers = { bf16_max_relative_l2 = 0, bf16_min_cosine = 1, bf16_max_probability_delta = 0, bf16_milliseconds_per_decision = 21.25683333724737, bf16_peak_gb = 5.895720488, int8_max_relative_l2 = 0.01946498516749375, int8_min_cosine = 0.9998105641456941, int8_max_probability_delta = 0.0008521080017089844, int8_milliseconds_per_decision = 17.479083326179534, int8_peak_gb = 3.493382928, int4_max_relative_l2 = 0.24333150765468853, int4_min_cosine = 0.9709852340409445, int4_max_probability_delta = 0.003509223461151123, int4_milliseconds_per_decision = 20.397270855028182, int4_peak_gb = 2.153433224 }
links = ["results/intake/intake-2026-10-15T18-18-52Z-9d99a609-3ecc-4837-8ec8-449d524ab8ac.json", "gate/evidence/intake/intake-2026-10-15T18-18-52Z-9d99a609-3ecc-4837-8ec8-449d524ab8ac/quality-binding.json"]
+++
# Default Load precision intake — 15 October 2026

Run `intake-2026-10-15T18-18-52Z-9d99a609-3ecc-4837-8ec8-449d524ab8ac` replaces the table figures from the preceding intake. The original captures and logs remain historical evidence.

Both readouts now capture/replay the final normalized hidden input immediately before option-logit calculation: `case-0/readout/input` [1,2048] and `case-1/readout/input` [2,2048], BF16. The immutable fixture has 94 inputs (the prior 92 plus these two). The 93 component outputs and separate three whole-decision probability vectors are retained. The fixed-tensor regression changes the predecessor output and proves the readout still receives the saved input; missing/wrong-shape/wrong-dtype inputs fail before readout.

| Tier | Max relative L2 vs half | Min cosine vs half | Max absolute probability delta vs half | Warm ms/decision | Peak GB per worker | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| bf16 | 0.00000000 | 1.00000000 | 0.00000000 | 21.257 | 5.895720 | pass |
| int8 | 0.01946499 | 0.99981056 | 0.00085211 | 17.479 | 3.493383 | pass |
| int4 | 0.24333151 | 0.97098523 | 0.00350922 | 20.397 | 2.153433 | fail |

Int8 passes; int4 fails its component bounds and remains offered with the failed-quality warning. Bounds remain relative L2 ≤0.05, cosine ≥0.999, finite activations/outputs, no new clamps, unchanged answers and max |dp| ≤0.017. This is quantization against the pinned served half implementation, not new independent vendor/task qualification.

The guarded intake took 61.80 seconds under `scripts/mlock gpulock`; aggregate worker cap 28.081 GB, observed aggregate peak 9.936 GB. Existing swap 1.023279 GB did not grow. Default-workload peaks cover fresh Load and decisions, after probe workers are discarded. Busy-machine timings are quick checks; no speed winner is claimed.

The installed/HF-cache originals were hashed before copying, APFS-cloned into the lab, checked unchanged afterwards, and all weight/cache/donor copies deleted. No download, install, signing, push or merge. `source-copy-receipt.json` records every original pin. `build-receipt.json` records the actual worker, source closure, effective load identity, MLX pin and privately rebuilt no-fast-math metallib. The shared metallib cache had a stale digest; rebuilding the pinned ten Metal sources reproduced the existing metallib bytes without changing the shared cache.

Intake completion wrote both the immutable per-run binding and the model binding directly from this exact result, worker identity, bounds and emitted table stats. No temporary binding generator or filename selection was used.
