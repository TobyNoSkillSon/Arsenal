# Ported from Verdict

Source: `Projects/Verdict` at commit `f8177ed4c43ec164e6fc9600c15cf1293d5e8d61` (Apache-2.0; `LICENSE-verdict`, `NOTICE`).
Dependency pins are Verdict's parity-proven set (`Package.resolved` identical: mlx-swift `9019419`, swift-transformers 1.3.4,
swift-collections 1.7.0).

| Here | From | Changes |
|---|---|---|
| `Sources/VerdictEngine/**` | `Sources/VerdictEngine/**` | verbatim, except `SmallMRoute.swift` `SmallMGate.supportDirectory`: gate verdicts in `$ARSENAL_SUPPORT_DIR/state/smallm-gate.json`; kernel stage in process (folder kernels compile through a preflight, docs/KERNELS.md) and off without a folder tile kernel; `Laya/LayaNetwork.swift`, `Laya/LayaModel.swift`: folder kernel site `linear.down` (Exact gate, `FolderKernelRoute.swift`, Arsenal) |
| `Sources/VerdictSmallMGEMM/SmallMGEMM.swift` | `Sources/SmallMGEMM/SmallMGEMM.swift` (Vella's family kernel, vendored by Verdict) | tile-1 Metal source moved to `models/laya-english/kernels/m5/` (loaded as `SmallMGEMM.tile`); `revision` includes the source digest |
| `Sources/VerdictCore/Safetensors.swift` | `Sources/VerdictCore/Safetensors.swift` | verbatim |
| `Sources/VerdictCore/VerdictError.swift` | `Sources/VerdictCore/Core.swift` (`VerdictError` only) | extracted |
| `Sources/DecisionsProtocol/OrderedJSON.swift` | `Sources/VerdictHelper/OrderedJSON.swift` | types made public |
| `Sources/DecisionsProtocol/SystemOne.swift` | `Sources/VerdictHelper/SystemOne.swift` | public; `DecisionsCatalog` instead of `Catalog`; "Arsenal" in two messages; `SystemOneBatcher` not ported |
| `Sources/DecisionsProtocol/Runner.swift` | `Sources/VerdictHelper/Service.swift` (`judge`, `run`, `predict`, `finite`, `answer`, `item`, `questions`, `precisionConflict`, `wholeNumber`) | one loaded model per process; residency, eviction, micro-batching and status.json are the Arsenal supervisor's |
| `Sources/DecisionsProtocol/Catalog.swift` | `Sources/VerdictHelper/Catalog.swift` `ModelSpec` + `Resources/models.json` | the five local models, tiers and bits rules only |
| `Sources/arsenal-worker-decisions/main.swift`, `Worker.swift` | `Sources/VerdictHelper/main.swift`, `Registry.swift`, `Service.swift` (engine label, cache bound, TF32 off) | ArsenalWire ops instead of HTTP |
| `Sources/arsenal-worker-decisions/Convert.swift` | `Sources/VerdictHelper/Catalog.swift` `snapshot` (Von fp16 conversion) + models.json `convert_to` | `convert` op |
| `build.sh` | `Projects/Vella/Worker/build-split.sh` | worker name, bundle id |

Parity fixtures are read in place from `Projects/Verdict/lab/fixtures` (never copied, never regenerated here).
