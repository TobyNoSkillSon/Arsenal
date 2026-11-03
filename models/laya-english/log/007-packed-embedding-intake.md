+++
hypothesis = "Laya int8, now packing the token embedding as well as every Linear, keeps FP16 component quality and decision probabilities; int4 is measured alongside."
change = "Pack the token embedding at int8/int4 (row blocks read from the file, quantized lookup); grade each model only on inputs in its supported languages; rerun intake."
measurement = "M5 Max, 40 GPU cores, 15 Oct 2026; busy Mac. 64 component outputs and three served-path probability vectors. Default served workload: one refund decision, warm ABBA BAAB."
verdict = "kept"
numbers = { int8_max_relative_l2 = 0.007386113478691461, int8_min_cosine = 0.9999727222928155, int8_max_dp = 0.0037994980812072754, int8_peak_gb = 0.74004976, int8_peak_gb_linears_only = 0.8720395999999999, int4_max_relative_l2 = 0.11506579217454342, int4_min_cosine = 0.9933795842648891, int4_max_dp = 0.040938153862953186, int4_peak_gb = 0.529957872, int4_peak_gb_linears_only = 0.6892268800000001 }
links = ["gate/evidence/quality-binding-2026-10-15.json", "results/intake/intake-2026-10-15T17-48-31Z-6fd26dcb-39e5-47b5-90ef-a0fd5a5d0836.json", "log/006-int8-int4-intake.md"]
+++
# Int8 and int4 intake with a packed token embedding — 15 October 2026

Run `intake-2026-10-15T17-48-31Z-6fd26dcb-39e5-47b5-90ef-a0fd5a5d0836` supersedes `intake-2026-10-15T17-24-18Z-3472a522-35dd-49f2-aa79-93a2ecf66d46` (`log/006`) for the table. int8 and int4 now pack the token embedding too, as every large part must be: the loader reads it from the file in blocks of 8,192 rows, quantizes each block (affine, group 64, along the row, so blocks pack exactly as the whole table) and looks rows up from the packed table; the dense FP16 table never exists in memory. Norms, biases and other small tensors stay FP16. The recipe string stays `mlx-affine-g64:verdict-at-load`; the decisions worker's source identity changed, so earlier bindings no longer apply.

Component states: "The customer asks for a refund." (yes/no) and "The package has not arrived. Please check the delivery." (two-option choice). Both cases are English: the model is English-only, and a model is graded only on inputs in its supported languages (coordinator decision, 15 Oct).

The probe set changed after the round-2 result. Round 2 (`log/006`) graded this English-only model on a German case too, and its int8 failed only that case (max |dp| 0.01737 against 0.017; every component passed). The coordinator then decided that a model is graded only on inputs in its supported languages, so the German case was replaced by the English delivery case for Laya English (and, by the same rule, Laya Typed). This was decided after seeing 0.0174: it removes an out-of-domain input, it does not change the 0.017 limit, and the round-2 run stays on record as it was.

| Tier | Max component relative L2 | Min cosine | Max absolute probability delta | Warm ms/decision | Peak GB, Linears only (round 2) | Peak GB, with packed embedding | Check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 0.000000 | 1.000000 | 0.00000 | 6.348 | 1.197 | 1.197 | pass |
| int8 | 0.007386 | 0.999973 | 0.00380 | 6.377 | 0.872 | 0.740 | pass |
| int4 | 0.115066 | 0.993380 | 0.04094 | 6.538 | 0.689 | 0.530 | fail |

Bounds as before: relative L2 ≤0.05, cosine ≥0.999, finite, no new clamps, unchanged answers, max |dp| ≤0.017. int8 passes, so it is offered; int4 fails 6 components (case-0/embedding, case-0/layer-0, case-0/layer-5, case-1/embedding, case-1/layer-0, case-1/layer-5) and stays offered with its failed-quality warning. Peaks are each worker's lifetime physical footprint over Load and the timed decisions. A first pass that quantized the embedding as one whole tensor gave the same quality numbers but raised the int8/int4 Load footprint above the Linears-only build (Multilingual int8 1.46 GB: MLX held the dense and packed tables together during packing, and the footprint after Load stayed about 0.4 GB higher); it was discarded uncommitted in favour of the block reader. Timings are busy-Mac quick checks; the speed differences are within noise.

The intake ran 10.7 s under `scripts/mlock gpulock` with a 47.4 GB worker cap; aggregate worker peak 2.05 GB; existing swap 1.032 GB did not grow. Source weights were APFS-cloned into a private lab, hash-checked unchanged afterwards, and the copies deleted; no downloads, signing or installation.
