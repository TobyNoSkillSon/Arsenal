# Native dependency policy

All bundled families resolve one `mlx-swift` revision: `901941965d82e4a216d4d117231d847d194c563d`. Keep the no-fast-math metallib in `scripts/pkg-lib.sh`. Integrators do not change package pins, build scripts or dispatcher files. IntegrationKit depends on the wire/precision packages rather than the app Core library; Core is a dependency of `IntegrationKitTests` only, for exercising the ordinary Get/Load and production artifact validation paths. Worker-side file metadata is checked against that service policy in tests.

## One build graph

`Workers/Package.swift` is the only native package anything builds. Its dependencies are every family package plus the three pinned upstreams, so SwiftPM resolves one copy of each and compiles it once per configuration in `Workers/.build`; release links that copy once into `arsenal-worker-native`. The root also declares every family's test target (paths into `Workers/<Family>/Tests/`), the IntegrationKit service fixture and the lab entry points (`language-lab`, `geometry-lab`). Family manifests declare libraries only, keep their own pins identical to the root's and must not grow test or executable targets; the legacy gate builders (`Workers/Voices/scripts/build-split.sh`, `Workers/Decisions/build.sh`, `Workers/Kernels/build.sh`, and the D1/FLUX/BiRefNet runner folders' `build.sh`) remain separate, opt-in parity tools. The MLX metallib is compiled once per pin, Metal toolchain and flags into `~/Library/Caches/dev.arsenal.build/mlx-metallib/` (override with `ARSENAL_BUILD_CACHE`), shared by every worktree and by both the dev helpers and the bundle.

Adding a family: put its library package in `Workers/<Family>/`, add it to the root's dependencies, the `arsenal-worker-native` target, the dispatcher in `Workers/Native/Sources/arsenal-worker-native/Main.swift`, `NATIVE_FAMILIES` in `scripts/pkg-lib.sh` and `scripts/quality-identities.py`; declare its test target in its own section of the root manifest.

`mlx-swift-lm` remains `bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57`. The Language/Embed package pins make `MLXVLM` and `MLXEmbedders` available to integrators, but empty registries do not link their unused global architecture registrations. Chat keeps `MLXLLM`/`MLXLMCommon`; unused VLM/embed products were removed from its target closure. An integrator adds the needed product to its owned target when importing its first real adapter, without changing shared pins. The upstream main inspected on 18 October 2026 was `48114eedf3c3d4a682a6b1d644cc5af5c2945d10`: its package now requires Swift 6.2, MLX 0.32.3, a new Hugging Face/tokenizer boundary and new library targets. Bumping this package alone is not a compatible addition to the existing Swift Transformers tokenizer bridge. Keep the working pin; backport Hunyuan, EmbeddingGemma2 and Jina ranking into the owning family against it, with upstream source/revision recorded. These model adapters are integrator work, not implemented architectures in this foundation.

The installed CLT and Xcode both report Swift 6.4 (`swiftlang-6.4.0.34.1`, arm64 macOS 27). TRELLIS.2's stated Xcode 27/Swift 6.4 floor is therefore available here. This does not raise Arsenal's macOS 26 deployment floor or prove the port runs there; compile and weak-runtime checks still apply.

Vendor the smallest MLX source closure into the owning family, adapted to the common pin. Do not add whole upstream applications or packages with a competing MLX resolution. Each imported closure includes `LICENSE-upstream` and `PORTED-FROM.md` with repository, immutable revision, imported files, local adaptations and the matching checkpoint layout. Keep Core ML execution out of the new model paths. Missing upstream terms are recorded for review; a model's presence in the research list is not a claim about redistributable source.

| Source | Owner and integration boundary |
|---|---|
| `Blaizzy/mlx-audio-swift` STT/VAD/STS | Audio; reuse existing Core/Codecs targets when identical, import STT/VAD/STS closures only. |
| `soniqo/speech-swift` | Audio; MLX segmentation/embedding/diarization pieces only. Community-1 Core ML pieces cannot silently substitute. |
| `kylehowells/demucs-mlx-swift` | Audio; validate exact HTDemucs-ft bag and key map. |
| `arraypress/swift-music-generation` | Music; YuE2 and ACE-Step closures, including separate codec/VAE assets. |
| `xocialize/seedvr2-mlx-swift`, `xocialize/rife-mlx-swift` | Restoration; shared video I/O comes from IntegrationKit. |
| `thesepehrm/sam31-swift` | Perception; click/text/image-session and video-state ownership stay in the adapter. |
| `mnmly` RF-DETR, LocateAnything, TIPS, MoGe, DA3 ports | Perception/Geometry by operation; retain provenance and each source's terms; validate variant-specific shapes. |
| `xocialize/mlx-trellis2-swift` and mesh/xatlas pieces | Geometry; isolate optional mesh dependencies within that family. |
| `paddleocr-vl.swift` | Language; processor/model implementation for PaddleOCR, not a generic Qwen alias. |
| `kylehowells/Granite-MLX` | Audio; Granite Speech 5 CTC is distinct from Granite Speech 4 QFormer. |
| Vella `Worker/Sources` (MLXAudioSTT, MLXAudioCore, SmallMGEMM) | Audio; the `VellaSTT` target: Parakeet TDT and Qwen3-ASR with Vella's optimized paths, on the same pins. |
| `mnmly/mlx-swift-Qwen3-VL-Embedding` | Embed; VLM backbone/pooling adapter on the common pin. |

The foundation registries contain no imported model implementation. Each integrator freezes the exact port revision before adding its source. No additional runtime package manager or download service is introduced.

Sources inspected: https://github.com/ml-explore/mlx-swift-lm/blob/48114eedf3c3d4a682a6b1d644cc5af5c2945d10/Package.swift and the pinned local checkouts. Per-model source references are in the integration research supplied with the task.
