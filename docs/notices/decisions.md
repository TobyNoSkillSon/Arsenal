# decisions (arsenal-worker-decisions)

Full texts: `Workers/Decisions/THIRD_PARTY_NOTICES.txt` (verbatim from Verdict f8177ed; identical package set), `Workers/Decisions/NOTICE`, `Workers/Decisions/LICENSE-verdict`. Ship them beside the worker in the app bundle.

| Component | Licence | Use |
|---|---|---|
| Verdict engine and helper rules (upstream revision f8177ed), incl. Laya port of laya-mlx 0.2.0 and Von port of von-sdk 1.1.1/1.2.2 | Apache-2.0 (NOTICE carries laya-mlx / Laya attribution) | the models' Swift implementation |
| SmallMGEMM (Vella family kernel) | Apache-2.0 (Verdict/Vella) | small-M GEMM, Fast path |
| mlx-swift 9019419 (+ vendored MLX, mlx-c, fmt, nlohmann/json, metal-cpp, pocketfft, ThreadPool, NVIDIA CCCL complex math) | MIT (+ Apache-2.0, BSD-3-Clause, zlib for the vendored parts) | runtime |
| swift-transformers 1.3.4, swift-huggingface, swift-jinja | Apache-2.0 | library tokenizer fallback |
| swift-collections 1.7.0, swift-numerics | Apache-2.0 with Runtime Library Exception | dependencies |
| swift-crypto, swift-asn1 | Apache-2.0 | transitive (swift-transformers) |
| EventSource, yyjson | MIT | transitive (swift-transformers) |

Model weights are not bundled; Get downloads them. Laya ×3: Convai Innovations, Apache-2.0 (LICENSE in each aac6fef/*-mlx repo, pinned in the model folders). Von 1.1/1.2: Victor Hugo Panisa (wfzyx/von), Apache-2.0 per the model card.

## Licence of the files in Workers/Decisions

Arsenal is MIT. Files written for Arsenal carry `SPDX-License-Identifier: MIT` (`Package.swift`, `build.sh` — adapted from Vella's `Worker/build-split.sh` (MIT) —, `DecisionsProtocol/Catalog.swift`, `Wire.swift`, `arsenal-worker-decisions/main.swift`, `Worker.swift`, `Convert.swift`, the tests and `scripts/*.py`; the last three of the Swift files incorporate Apache-2.0 code from Verdict's helper, noted in their headers). Ported files keep Verdict's Apache-2.0 terms unchanged (no SPDX line added; the provenance comment says Apache-2.0 where one was added): `Sources/VerdictEngine/**`, `Sources/SmallMGEMM/**`, `Sources/VerdictCore/**`, `DecisionsProtocol/OrderedJSON.swift`, `SystemOne.swift`, `Runner.swift`, `arsenal-worker-decisions/SmallMChild.swift`. Apache-2.0 code is combined into the MIT worker; its LICENSE/NOTICE texts ship with it.
