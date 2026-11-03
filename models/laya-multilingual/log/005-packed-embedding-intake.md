+++
hypothesis = "Laya int8, now packing the token embedding as well as every Linear, keeps FP16 component quality and decision probabilities; int4 is measured alongside."
change = "Pack the token embedding at int8/int4 (row blocks read from the file, quantized lookup); grade each model only on inputs in its supported languages; rerun intake."
measurement = "M5 Max, 40 GPU cores, 15 Oct 2026; busy Mac. 52 component outputs and three served-path probability vectors. Default served workload: one refund decision, warm ABBA BAAB."
verdict = "kept"
numbers = { int8_max_relative_l2 = 0.014478731551605149, int8_min_cosine = 0.9998965433784406, int8_max_dp = 0.000624852254986763, int8_peak_gb = 0.835585008, int8_peak_gb_linears_only = 1.26653396, int4_max_relative_l2 = 0.10152351179195286, int4_min_cosine = 0.9948350225306589, int4_max_dp = 0.009529471397399902, int4_peak_gb = 0.601899992, int4_peak_gb_linears_only = 1.199441408 }
links = ["gate/evidence/quality-binding-2026-10-15.json", "results/intake/intake-2026-10-15T17-48-19Z-ee47efc1-f89d-4129-9d5d-98a063da379f.json", "log/004-int8-int4-intake.md"]
+++
# Int8 and int4 intake with a packed token embedding — 15 October 2026

Run `intake-2026-10-15T17-48-19Z-ee47efc1-f89d-4129-9d5d-98a063da379f` supersedes `intake-2026-10-15T17-23-20Z-204651f0-99d0-4d70-96ac-6ce1d0750f9d` (`log/004`) for the table. int8 and int4 now pack the token embedding too, as every large part must be: the loader reads it from the file in blocks of 8,192 rows, quantizes each block (affine, group 64, along the row, so blocks pack exactly as the whole table) and looks rows up from the packed table; the dense FP16 table never exists in memory. Norms, biases and other small tensors stay FP16. The recipe string stays `mlx-affine-g64:verdict-at-load`; the decisions worker's source identity changed, so earlier bindings no longer apply.

Component states: "The customer asks for a refund." (yes/no) and "Das Paket ist nicht angekommen. Bitte prüfen Sie die Lieferung." (two-option choice). The second case is German, inside this model's supported languages.

| Tier | Max component relative L2 | Min cosine | Max absolute probability delta | Warm ms/decision | Peak GB, Linears only (round 2) | Peak GB, with packed embedding | Check |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| FP16 | 0.000000 | 1.000000 | 0.00000 | 4.088 | 1.332 | 1.332 | pass |
| int8 | 0.014479 | 0.999897 | 0.00062 | 3.616 | 1.267 | 0.836 | pass |
| int4 | 0.101524 | 0.994835 | 0.00953 | 3.640 | 1.199 | 0.602 | fail |

Bounds as before: relative L2 ≤0.05, cosine ≥0.999, finite, no new clamps, unchanged answers, max |dp| ≤0.017. int8 passes, so it is offered; int4 fails 18 components (case-0/embedding, case-0/layer-0, case-0/layer-10, case-0/layer-11, case-0/layer-5, case-0/layer-6…) and stays offered with its failed-quality warning. Peaks are each worker's lifetime physical footprint over Load and the timed decisions. A first pass that quantized the embedding as one whole tensor gave the same quality numbers but raised the int8/int4 Load footprint above the Linears-only build (Multilingual int8 1.46 GB: MLX held the dense and packed tables together during packing, and the footprint after Load stayed about 0.4 GB higher); it was discarded uncommitted in favour of the block reader. Timings are busy-Mac quick checks; the speed differences are within noise.

The intake ran 8.6 s under `scripts/mlock gpulock` with a 47.3 GB worker cap; aggregate worker peak 2.03 GB; existing swap 1.032 GB did not grow. Source weights were APFS-cloned into a private lab, hash-checked unchanged afterwards, and the copies deleted; no downloads, signing or installation.
