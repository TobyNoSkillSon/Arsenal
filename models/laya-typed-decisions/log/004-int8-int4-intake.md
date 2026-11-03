+++
hypothesis = "Laya int8 (quantized at load from FP16) keeps FP16 component quality and decision probabilities; int4 is measured alongside."
change = "Add a measurement_components adapter to the decisions worker (teacher-forced Laya components) and run the shared intake pipeline."
measurement = "M5 Max, 40 GPU cores, 15 Oct 2026; busy Mac. 64 component outputs and three served-path probability vectors. Default served workload: one refund decision, 41 input tokens, warm ABBA BAAB."
verdict = "kept"
numbers = { int8_max_relative_l2 = 0.010280334447887904, int8_min_cosine = 0.9999713005500643, int8_max_dp = 0.0023348331451416016, int8_peak_gb = 0.8721051360000001, int8_passed = true, int4_max_relative_l2 = 0.18035602674795911, int4_min_cosine = 0.9875650767537998, int4_max_dp = 0.110878586769104, int4_peak_gb = 0.6891777520000001, int4_passed = false }
links = ["gate/evidence/quality-binding-2026-10-15.json", "results/intake/intake-2026-10-15T17-24-32Z-da17c900-d1b0-415c-83e8-7cb25295211b.json"]
+++
# Int8 and int4 intake — 15 October 2026

Run `intake-2026-10-15T17-24-32Z-da17c900-d1b0-415c-83e8-7cb25295211b` on the pinned FP16 revision through the production Get and the bundled decisions worker (Fast recipe), with int8/int4 quantized at load by the catalog recipe `mlx-affine-g64:verdict-at-load` (Linear layers with input width divisible by 64; embeddings, norms and biases stay FP16). This qualifies quantization against the FP16 served implementation; it is not a new vendor qualification.

The fixture has two one-row cases, unpadded: an English yes/no refund question and a German two-option choice. Each component (embedding, every encoder layer, every head layer, scorer) replays its saved FP16 input, so a candidate's error never feeds the next component: 64 outputs. The probability check runs the served path for three question vectors (the English yes/no, and the German yes/no and choice) and compares end-to-end probabilities with FP16.

| Tier | Max component relative L2 | Min cosine | Max absolute probability delta | Warm ms/decision | Default-workload peak GB | Check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 0.000000 | 1.000000 | 0.00000 | 6.294 | 1.1968 | pass |
| int8 | 0.010280 | 0.999971 | 0.00233 | 6.154 | 0.8721 | pass |
| int4 | 0.180356 | 0.987565 | 0.11088 | 6.039 | 0.6892 | fail |

Bounds: relative L2 ≤0.05, cosine ≥0.999, finite outputs/activations, no new clamps, all chosen answers unchanged and max |dp| ≤0.017 (the bound d1 uses; Laya's FP16 A/A repeat was exactly 0). int8 passes. int4 fails 7 components (case-0/layer-0, case-0/layer-5, case-0/scorer, case-1/layer-0, case-1/layer-4, case-1/layer-5…). Under the precision rule int8 is offered; int4 stays offered with its failed-quality warning. The FP16 row's timing is the median of its paired baseline windows. All figures are quick checks on a busy Mac; spreads are in the run JSON, and a speed difference this small is not established.

Peaks are each worker's lifetime physical footprint over Load (including quantizing at load) and the timed decisions, from fresh workers started after the component probes. The intake ran 11.5 s under `scripts/mlock gpulock` with a 48.7 GB worker cap; aggregate worker peak 2.07 GB. Existing swap was 1.032 GB and did not grow. Source weights were APFS-cloned into a private lab, unchanged, and the lab removed afterwards; no downloads, signing or installation.
