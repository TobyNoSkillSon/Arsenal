+++
hypothesis = "Laya int8 (quantized at load from FP16) keeps FP16 component quality and decision probabilities; int4 is measured alongside."
change = "Add a measurement_components adapter to the decisions worker (teacher-forced Laya components) and run the shared intake pipeline."
measurement = "M5 Max, 40 GPU cores, 15 Oct 2026; busy Mac. 64 component outputs and three served-path probability vectors. Default served workload: one refund decision, 41 input tokens, warm ABBA BAAB."
verdict = "rejected"
numbers = { int8_max_relative_l2 = 0.0031517335851340704, int8_min_cosine = 0.9999952502573909, int8_max_dp = 0.01737026870250702, int8_peak_gb = 0.8720395999999999, int8_passed = false, int4_max_relative_l2 = 0.05333709849225914, int4_min_cosine = 0.9987049006183433, int4_max_dp = 0.2518833875656128, int4_peak_gb = 0.6892268800000001, int4_passed = false }
links = ["gate/evidence/quality-binding-2026-10-15.json", "results/intake/intake-2026-10-15T17-24-18Z-3472a522-35dd-49f2-aa79-93a2ecf66d46.json"]
+++
# Int8 and int4 intake — 15 October 2026

Run `intake-2026-10-15T17-24-18Z-3472a522-35dd-49f2-aa79-93a2ecf66d46` on the pinned FP16 revision through the production Get and the bundled decisions worker (Fast recipe), with int8/int4 quantized at load by the catalog recipe `mlx-affine-g64:verdict-at-load` (Linear layers with input width divisible by 64; embeddings, norms and biases stay FP16). This qualifies quantization against the FP16 served implementation; it is not a new vendor qualification.

The fixture has two one-row cases, unpadded: an English yes/no refund question and a German two-option choice. Each component (embedding, every encoder layer, every head layer, scorer) replays its saved FP16 input, so a candidate's error never feeds the next component: 64 outputs. The probability check runs the served path for three question vectors (the English yes/no, and the German yes/no and choice) and compares end-to-end probabilities with FP16.

| Tier | Max component relative L2 | Min cosine | Max absolute probability delta | Warm ms/decision | Default-workload peak GB | Check |
| --- | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 0.000000 | 1.000000 | 0.00000 | 6.269 | 1.1968 | pass |
| int8 | 0.003152 | 0.999995 | 0.01737 | 6.214 | 0.8720 | fail |
| int4 | 0.053337 | 0.998705 | 0.25188 | 5.995 | 0.6892 | fail |

Bounds: relative L2 ≤0.05, cosine ≥0.999, finite outputs/activations, no new clamps, all chosen answers unchanged and max |dp| ≤0.017 (the bound d1 uses; Laya's FP16 A/A repeat was exactly 0). int8 fails only the probability bound (all components pass): the German yes/no case on this English-only model moves from 0.7629 to 0.7803 (0.01737 against 0.017); the English cases move 0.0018 and 0.0019, and no answer changes. A follow-up probe ran FP16 on the same fixture with bucket padding off and with padding to 128 (VERDICT_LAYA_PAD_MULTIPLE=1 and 128); both reproduced the reference FP16 probabilities to six decimals, so the delta comes from quantization, not from the padded served path. int4 fails 5 components (case-0/layer-0, case-0/layer-5, case-1/layer-0, case-1/layer-4, case-1/layer-5). Under the precision rule int8 is greyed with its reason; int4 stays offered with its failed-quality warning. The FP16 row's timing is the median of its paired baseline windows. All figures are quick checks on a busy Mac; spreads are in the run JSON, and a speed difference this small is not established.

Peaks are each worker's lifetime physical footprint over Load (including quantizing at load) and the timed decisions, from fresh workers started after the component probes. The intake ran 11.6 s under `scripts/mlock gpulock` with a 48.6 GB worker cap; aggregate worker peak 2.07 GB. Existing swap was 1.032 GB and did not grow. Source weights were APFS-cloned into a private lab, unchanged, and the lab removed afterwards; no downloads, signing or installation.
