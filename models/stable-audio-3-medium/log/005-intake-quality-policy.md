+++
hypothesis = "The accepted FP16 cast and explicit quantized-tier policy can qualify a whole-part int8 recipe."
change = "Keep decoder arithmetic in FP16 and qualify each precision against retained component evidence."
measurement = "Busy M5 Max quick check; balanced warm windows, cold/load separate and lifetime physical peaks."
numbers = "FP16 and int8 pass; int4 fails. Full metrics and bounds remain in the linked result."
verdict = "kept"
links = ["../results/intake/intake-2026-10-15T11-57-03Z-619ba9d4-cb51-46ca-a784-caed82dc2fd0.json", "../gate/evidence/intake-latest.json"]
+++

# Qualified intake recipes — 15 October 2026

Current evidence: `../results/intake/intake-2026-10-15T11-57-03Z-619ba9d4-cb51-46ca-a784-caed82dc2fd0.json`; bounds and capture identity are linked by `../gate/evidence/intake-latest.json`. Earlier intake logs and candidate stats are historical. The format and quantized-tier policies are defined once in `docs/ADDING-A-MODEL.md` and implemented by `IntakeComponentGate`. Independent vendor qualification remains separate.

The coordinator accepts the FP16 SAME-L cast on existing finite/no-clamp evidence and retained original PCM (spectral relative L2 0.0157; waveform 0.0212). No FP32 reference is permitted or was executed in this round. DiT, T5Gemma matrices/embedding and SAME-L eligible matrices are covered by `intake-v1:stable-audio-3-medium`; vectors, norms, convolutions, small or unaligned matrices stay FP16. Int8 passes; int4 fails and is unavailable with “Did not pass the quality check”.

Inspection also found FP32 decoder residual/attention accumulation despite half GEMMs. The production decoder now anchors its input to stored half weights and its sine constant to the gate dtype. A tiny one-block sine-path test asserts FP16 block/output dtypes, shape and finiteness. The intermediate run `intake-2026-10-15T11-51-42Z-eafc5d40-b860-474a-a6af-7b3a1116add8` caught the remaining sine promotion and is retained as superseded evidence. The final run confirms every decoder trace and output is FP16, finite and unclamped. On the exact same saved decoder input, final FP16 versus the previously accepted half-GEMM/FP32-stream capture is relative L2 0.005363, cosine 0.999989 (`decoder-half-bridge.json`).

The affected audio production paths were re-run after that arithmetic fix, under the 590-second ceiling and abort-on-any-swap-growth guard. Both short runs completed without guard stops; the final run is the table source. Numeric initial swap was not persisted by that executable, so its resource receipt records enforced guard completion rather than inventing a numeric reading. Lab donor/weight/cache copies were removed.

| Precision | Offered | Maximum component relative L2 | Peak GB | Warm median s |
|---|---|---:|---:|---:|
| fp16 | yes | 0.000000 | 6.159 | 0.2491 |
| int8 | yes | 0.015139 | 3.949 | 0.2832 |
| int4 | no | 0.228570 | 2.704 | 0.2664 |

Busy M5 Max quick checks on 15 October 2026; FLUX is 1024×1024/four steps, audio five seconds/eight steps. Warm figures are same-session balanced ABBA BAAB medians; cold/load is separate and peak is each worker’s lifetime physical footprint including Load. Compare each candidate with its saved paired baseline; no speed winner is claimed. Per-precision schema-2 results live directly under `results/`, where the Models table reads them.

Upgrade behavior: catalog/selection/Load do not cast weights. Changed recipes make old snapshots stale; normal Get does the file-by-file in-place cast and derives the new int8 snapshot. Old DiT-only int8 cannot inherit the new recipe or measurements. No installed files or app bundle were touched; integration/install belong to the coordinator.
