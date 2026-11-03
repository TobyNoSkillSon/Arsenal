+++
hypothesis = "Separate format and quantized-tier gates can qualify int8 without accepting FLUX FP16 activation overflow."
change = "Rescore saved component tensors and activation traces with the declared precision policies."
measurement = "Host-only rescore; retained busy M5 Max quick-check timings, no new model run."
numbers = "BF16 and int8 pass; FP16 and int4 fail. Full metrics and bounds remain in the linked result."
verdict = "kept"
links = ["../results/intake/intake-2026-10-15T11-14-00Z-acb8d191-2607-48f4-8a93-5c251b3ac926-rescore-08547261-ea74-4fb6-bbcc-31d9cceb01bb.json", "../gate/evidence/intake-latest.json"]
+++

# Qualified intake recipes — 15 October 2026

Current evidence: `../results/intake/intake-2026-10-15T11-14-00Z-acb8d191-2607-48f4-8a93-5c251b3ac926-rescore-08547261-ea74-4fb6-bbcc-31d9cceb01bb.json`; bounds and capture identity are linked by `../gate/evidence/intake-latest.json`. Earlier intake logs and candidate stats are historical. The format and quantized-tier policies are defined once in `docs/ADDING-A-MODEL.md` and implemented by `IntakeComponentGate`. Independent vendor qualification remains separate.

FLUX stays BF16. Its FP16 format check still fails: reference pre-clamp residual maxima reach 98,816, and FP16 produced non-finite residuals with new clamps. The saved round-2 component tensors and activation traces were re-scored on the host; no new FLUX model run was made. Transformer and Qwen3-4B matrices/embedding are packed by int8; the small VAE stays BF16. Int8 passes; int4 fails and is unavailable with “Did not pass the quality check”.

| Precision | Offered | Maximum component relative L2 | Peak GB | Warm median s |
|---|---|---:|---:|---:|
| bf16 | yes | 0.000000 | 20.047 | 4.8192 |
| fp16 | no | 0.162563 | 20.105 | 4.5747 |
| int8 | yes | 0.039887 | 12.809 | 5.0505 |
| int4 | no | 0.264390 | 9.079 | 4.8679 |

Busy M5 Max quick checks on 15 October 2026; FLUX is 1024×1024/four steps, audio five seconds/eight steps. Warm figures are same-session balanced ABBA BAAB medians; cold/load is separate and peak is each worker’s lifetime physical footprint including Load. Compare each candidate with its saved paired baseline; no speed winner is claimed. Per-precision schema-2 results live directly under `results/`, where the Models table reads them.

Upgrade behavior: catalog/selection/Load do not cast weights. Changed recipes make old snapshots stale; normal Get does the file-by-file in-place cast and derives the new int8 snapshot. Old DiT-only int8 cannot inherit the new recipe or measurements. No installed files or app bundle were touched; integration/install belong to the coordinator.
