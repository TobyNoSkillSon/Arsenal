+++
hypothesis = "Laya int8 (quantized at load from FP16) keeps FP16 component quality and decision probabilities; int4 is measured alongside."
change = "Add a measurement_components adapter to the decisions worker (teacher-forced Laya components) and run the shared intake pipeline."
measurement = "M5 Max, 40 GPU cores, 15 Oct 2026; busy Mac. 52 component outputs and three served-path probability vectors. Default served workload: one refund decision, 41 input tokens, warm ABBA BAAB."
verdict = "kept"
numbers = { int8_max_relative_l2 = 0.014478731551605149, int8_min_cosine = 0.9998965433784406, int8_max_dp = 0.0005562473088502884, int8_peak_gb = 1.26653396, int8_passed = true, int4_max_relative_l2 = 0.10152351179195286, int4_min_cosine = 0.9948350225306589, int4_max_dp = 0.010619103908538818, int4_peak_gb = 1.199441408, int4_passed = false }
links = ["gate/evidence/quality-binding-2026-10-15.json", "results/intake/intake-2026-10-15T17-23-20Z-204651f0-99d0-4d70-96ac-6ce1d0750f9d.json"]
+++
# Int8 and int4 intake — 15 October 2026

Run `intake-2026-10-15T17-23-20Z-204651f0-99d0-4d70-96ac-6ce1d0750f9d` on the pinned FP16 revision through the production Get and the bundled decisions worker (Fast recipe), with int8/int4 quantized at load by the catalog recipe `mlx-affine-g64:verdict-at-load` (Linear layers with input width divisible by 64; embeddings, norms and biases stay FP16). This qualifies quantization against the FP16 served implementation; it is not a new vendor qualification.

The fixture has two one-row cases, unpadded: an English yes/no refund question and a German two-option choice. Each component (embedding, every encoder layer, every head layer, scorer) replays its saved FP16 input, so a candidate's error never feeds the next component: 52 outputs. The probability check runs the served path for three question vectors (the English yes/no, and the German yes/no and choice) and compares end-to-end probabilities with FP16.

| Tier | Max component relative L2 | Min cosine | Max absolute probability delta | Warm ms/decision | Default-workload peak GB | Check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 0.000000 | 1.000000 | 0.00000 | 3.965 | 1.3318 | pass |
| int8 | 0.014479 | 0.999897 | 0.00056 | 3.463 | 1.2665 | pass |
| int4 | 0.101524 | 0.994835 | 0.01062 | 3.421 | 1.1994 | fail |

Bounds: relative L2 ≤0.05, cosine ≥0.999, finite outputs/activations, no new clamps, all chosen answers unchanged and max |dp| ≤0.017 (the bound d1 uses; Laya's FP16 A/A repeat was exactly 0). int8 passes. int4 fails 16 components (case-0/layer-0, case-0/layer-10, case-0/layer-11, case-0/layer-5, case-0/layer-6, case-0/layer-7…). Under the precision rule int8 is offered; int4 stays offered with its failed-quality warning. The FP16 row's timing is the median of its paired baseline windows. All figures are quick checks on a busy Mac; spreads are in the run JSON, and a speed difference this small is not established.

Peaks are each worker's lifetime physical footprint over Load (including quantizing at load) and the timed decisions, from fresh workers started after the component probes. The intake ran 11.3 s under `scripts/mlock gpulock` with a 48.6 GB worker cap; aggregate worker peak 2.60 GB. Existing swap was 1.032 GB and did not grow. Source weights were APFS-cloned into a private lab, unchanged, and the lab removed afterwards; no downloads, signing or installation.
