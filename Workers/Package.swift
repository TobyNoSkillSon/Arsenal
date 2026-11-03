// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
import PackageDescription

// The one native build graph. Every family keeps its own folder, package manifest (its libraries and pinned upstream
// closures), registry and tests; this root links them into the shipped worker and owns every family's test target, so
// MLX and every other identical pinned dependency compiles once per configuration here and links once into
// arsenal-worker-native (Vella's VellaWorker recipe). Family packages declare libraries only: a family package built on
// its own would compile a second MLX, so scripts never build one. One process per resident runner, as before: the
// binary runs exactly one family per process, chosen by its family link name.
//
// Family owners edit only their own section below (test target dependencies, lab tools); pins and the dispatcher belong
// to the contracts owner (docs/DEPENDENCIES.md).
let mlxSwift = "901941965d82e4a216d4d117231d847d194c563d"
let mlxSwiftLM = "bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57"
let swiftTransformers = "150169bfba0889c229a2ce7494cf8949f18e6906"

let mlx: [Target.Dependency] = [.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift")]
let kit: [Target.Dependency] = [
    .product(name: "IntegrationKit", package: "IntegrationKit"), .product(name: "ArsenalWire", package: "ArsenalWire"),
    .product(name: "ModelPrecision", package: "ModelPrecision")
]
// The kit models' libraries, as Kit/Package.swift declares them (the dispatcher imports each one).
let kitModels = [
    "KitAstrobert",
    "KitAudioboxAesthetics",
    "KitChronos2",
    "KitChronosBoltSmall",
    "KitClapHtsatFused",
    "KitDa3metricLarge",
    "KitDepthAnythingV2Small",
    "KitEdgetam",
    "KitEomtCocoPanopticLarge640",
    "KitEsm2T33650m",
    "KitEsm2T68m",
    "KitEsm3SmOpenV1",
    "KitFlorence2Base",
    "KitFlorence2Large",
    "KitGeneformerV2104m",
    "KitGlmOcr",
    "KitGraniteEmbeddingSmallEnglishR2",
    "KitGraniteTtmR3",
    "KitGroundingDinoBase",
    "KitMangaOcrBase",
    "KitMitraClassifier2",
    "KitMolformerXlBoth10pct",
    "KitOwlv2BasePatch16Ensemble",
    "KitRealEsrganX4plus",
    "KitSam21HieraLarge",
    "KitSam21HieraTiny",
    "KitT0Alpha",
    "KitT0Gamma",
    "KitTigerDnr",
    "KitTigerSpeech",
    "KitTimesfm25200m",
    "KitTimestral209m",
    "KitTriposr",
    "KitUltra50g",
    "KitUvdoc",
    "KitVitposeBaseSimple",
    "KitVjepa2VitlFpc64256",
    "KitVoiceGenderEcapa"
]
let v5: [SwiftSetting] = [.swiftLanguageMode(.v5)]
let v6: [SwiftSetting] = [.swiftLanguageMode(.v6)]

let package = Package(
    name: "ArsenalWorkers", platforms: [.macOS("26.0")],
    products: [
        .executable(name: "arsenal-worker-native", targets: ["arsenal-worker-native"])
    ],
    dependencies: [
        // Families.
        .package(path: "Voices"),
        .package(path: "Chat"),
        .package(path: "Sound"),
        .package(path: "Decisions"),
        // d1-3B's, FLUX.2 Klein 4B's and BiRefNet's runners live in their model folders; these are links to them.
        .package(path: "D1"),
        .package(path: "Elephant"),
        .package(path: "Flux"),
        .package(name: "BiRefNet", path: "BiRefNetEngine"),
        .package(path: "Language"),
        .package(path: "Embed"),
        .package(path: "Audio"),
        .package(path: "Music"),
        .package(path: "Restoration"),
        .package(path: "Perception"),
        .package(path: "Geometry"),
        .package(path: "QwenImage"),
        // Kit models: one library per model, each started by its own link (Kit/Package.swift).
        .package(path: "Kit"),
        // Shared native pieces.
        .package(path: "IntegrationKit"),
        .package(path: "Kernels"),
        .package(path: "Transport"),
        .package(name: "Arsenal", path: ".."),
        .package(path: "../Packages/ArsenalWire"),
        .package(path: "../Packages/ModelPrecision"),
        // Pinned third-party code, resolved once for the whole graph.
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: mlxSwift),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: mlxSwiftLM),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: swiftTransformers)
    ],
    targets: [
        // --- The shipped worker (contracts owner) ---------------------------------------------------------------------
        .executableTarget(
            name: "arsenal-worker-native",
            dependencies: [
                .product(name: "VoicesWorker", package: "Voices"),
                .product(name: "ChatWorker", package: "Chat"),
                .product(name: "SoundWorker", package: "Sound"),
                .product(name: "DecisionsWorker", package: "Decisions"),
                .product(name: "D1Worker", package: "D1"),
                .product(name: "ElephantWorker", package: "Elephant"),
                .product(name: "FluxWorker", package: "Flux"),
                .product(name: "BiRefNetWorker", package: "BiRefNet"),
                .product(name: "LanguageWorker", package: "Language"),
                .product(name: "EmbedWorker", package: "Embed"),
                .product(name: "AudioWorker", package: "Audio"),
                .product(name: "MusicWorker", package: "Music"),
                .product(name: "RestorationWorker", package: "Restoration"),
                .product(name: "PerceptionWorker", package: "Perception"),
                .product(name: "GeometryWorker", package: "Geometry"),
                .product(name: "QwenImageWorker", package: "QwenImage")
            ] + kitModels.map { .product(name: $0, package: "Kit") },
            path: "Native/Sources/arsenal-worker-native", swiftSettings: v6),
        .testTarget(
            name: "NativeModelContractTests",
            dependencies: [
                .product(name: "ArsenalAPI", package: "Arsenal"), .product(name: "ArsenalCore", package: "Arsenal"),
                .product(name: "VoicesWorker", package: "Voices"), .product(name: "SoundWorker", package: "Sound"),
                .product(name: "ChatWorker", package: "Chat"), .product(name: "DecisionsWorker", package: "Decisions"),
                .product(name: "D1Worker", package: "D1"), .product(name: "FluxWorker", package: "Flux")
            ],
            path: "Native/Tests/NativeModelContractTests", swiftSettings: v5),
        .testTarget(
            name: "NativeMeasurementTests",
            dependencies: [.product(name: "ArsenalCore", package: "Arsenal"), .product(name: "ChatWorker", package: "Chat")] + mlx,
            path: "Native/Tests/NativeMeasurementTests", swiftSettings: v6),

        // --- IntegrationKit (contracts owner) -------------------------------------------------------------------------
        // Synthetic graph used only by the kit's service tests; not bundled.
        .executableTarget(
            name: "native-integration-fixture", dependencies: kit + mlx,
            path: "IntegrationKit/Sources/native-integration-fixture", swiftSettings: v5),
        .testTarget(
            name: "IntegrationKitTests",
            dependencies: kit + mlx + ["native-integration-fixture", .product(name: "ArsenalCore", package: "Arsenal")],
            path: "IntegrationKit/Tests/IntegrationKitTests", swiftSettings: v5),

        // --- language (G1) ----------------------------------------------------------------------------------------------
        // Lab entry point: tokenizer/header inspection, otherwise the ordinary language worker. Never bundled.
        .executableTarget(
            name: "language-lab", dependencies: [.product(name: "LanguageWorker", package: "Language")],
            path: "Language/Sources/language-lab", swiftSettings: v5),
        .testTarget(
            name: "LanguageWorkerTests", dependencies: kit + mlx + [.product(name: "LanguageWorker", package: "Language")],
            path: "Language/Tests/LanguageWorkerTests", swiftSettings: v5),

        // --- embed (G2) -------------------------------------------------------------------------------------------------
        .testTarget(
            name: "EmbedWorkerTests",
            dependencies: kit + mlx + [
                .product(name: "EmbedWorker", package: "Embed"), .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"), .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm")
            ],
            path: "Embed/Tests/EmbedWorkerTests", resources: [.copy("Fixtures")], swiftSettings: v6),

        // --- audio (G3) -------------------------------------------------------------------------------------------------
        .testTarget(
            name: "AudioWorkerTests", dependencies: kit + mlx + [.product(name: "AudioWorker", package: "Audio"), .product(name: "VellaSTT", package: "Audio")],
            path: "Audio/Tests/AudioWorkerTests", resources: [.copy("Fixtures")], swiftSettings: v6),

        // --- music (G4) -------------------------------------------------------------------------------------------------
        .testTarget(
            name: "MusicTests", dependencies: kit + mlx + [.product(name: "MusicWorker", package: "Music")],
            path: "Music/Tests/MusicTests", swiftSettings: v5),

        // --- restoration (G4) -------------------------------------------------------------------------------------------
        .testTarget(
            name: "RestorationTests",
            dependencies: kit + mlx + [
                .product(name: "RestorationWorker", package: "Restoration"), .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Restoration/Tests/RestorationTests", swiftSettings: v5),

        // --- perception (G5) --------------------------------------------------------------------------------------------
        .testTarget(
            name: "PerceptionTests", dependencies: kit + mlx + [.product(name: "PerceptionWorker", package: "Perception")],
            path: "Perception/Tests/PerceptionTests", resources: [.copy("Fixtures")], swiftSettings: v5),

        // --- geometry (G6) ----------------------------------------------------------------------------------------------
        // Lab entry point: header-map and TRELLIS shape export, otherwise the ordinary geometry worker. Never bundled.
        .executableTarget(
            name: "geometry-lab", dependencies: [.product(name: "GeometryWorker", package: "Geometry")],
            path: "Geometry/Sources/geometry-lab", swiftSettings: v5),
        .testTarget(
            name: "GeometryWorkerTests", dependencies: kit + mlx + [.product(name: "GeometryWorker", package: "Geometry")],
            path: "Geometry/Tests/GeometryWorkerTests", swiftSettings: v6),

        // --- qwenimage (Qwen-Image-2.1 and Turbo) -------------------------------------------------------------------------
        // Lab entry point: standalone real-weight runs and component checks with an MLX memory limit. Never bundled.
        .executableTarget(
            name: "qwen-image-lab",
            dependencies: kit + mlx + [
                .product(name: "QwenImageWorker", package: "QwenImage"), .product(name: "QwenImageNative", package: "QwenImage"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "QwenImage/Sources/qwen-image-lab", swiftSettings: v5),
        .testTarget(
            name: "QwenImageTests",
            dependencies: kit + mlx + [
                .product(name: "QwenImageWorker", package: "QwenImage"), .product(name: "QwenImageNative", package: "QwenImage"),
                .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "ArsenalCore", package: "Arsenal")
            ],
            path: "QwenImage/Tests/QwenImageTests", swiftSettings: v5),

        // --- runners in model folders (their Package.swift files are links from Workers/<Family>; tests stay there) ---
        .testTarget(
            name: "BiRefNetTests",
            dependencies: kit + mlx + [.product(name: "MLXBiRefNet", package: "BiRefNet"), .product(name: "BiRefNetWorker", package: "BiRefNet")],
            path: "BiRefNetEngine/Tests/BiRefNetTests", resources: [.copy("Fixtures")], swiftSettings: v5),
        .testTarget(
            name: "D1Tests", dependencies: [.product(name: "D1Engine", package: "D1")],
            path: "D1/Tests/D1Tests", swiftSettings: v6),
        .testTarget(
            name: "ElephantTests", dependencies: mlx + [.product(name: "ElephantEngine", package: "Elephant"), .product(name: "ElephantWorker", package: "Elephant")],
            path: "Elephant/Tests/ElephantTests", resources: [.copy("Fixtures")], swiftSettings: v5),
        .testTarget(
            name: "Flux2KleinTests",
            dependencies: mlx + [.product(name: "Flux2Klein", package: "Flux"), .product(name: "FluxWorker", package: "Flux")],
            path: "Flux/Tests/Flux2KleinTests", swiftSettings: v6),
        // FLUX 3 Image Klein (an add-on folder served by the flux worker). The fixtures are a tiny random-initialised
        // checkpoint and mflux's outputs on it (lab/flux3/scripts/capture_tiny.py); the lab tool runs real-weight checks.
        .testTarget(
            name: "Flux3KleinTests",
            dependencies: mlx + [
                .product(name: "Flux3Klein", package: "Flux"), .product(name: "Flux2Klein", package: "Flux"), .product(name: "FluxWorker", package: "Flux"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ],
            path: "Flux/Tests/Flux3KleinTests", resources: [.copy("Fixtures")], swiftSettings: v6),
        .executableTarget(
            name: "flux3-lab",
            dependencies: mlx + [.product(name: "Flux3Klein", package: "Flux"), .product(name: "Flux2Klein", package: "Flux")],
            path: "Flux/Sources/flux3-lab", swiftSettings: v6),

        // --- voices -----------------------------------------------------------------------------------------------------
        .testTarget(
            name: "VoicesTests",
            dependencies: mlx + [
                .product(name: "VoicesWorker", package: "Voices"), .product(name: "VireoCore", package: "Voices"),
                .product(name: "ArsenalWire", package: "ArsenalWire"), .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "FishDrama3", package: "Voices"), .product(name: "VoxCPM2", package: "Voices")
            ],
            path: "Voices/Tests/VoicesTests", resources: [.process("Fixtures")], swiftSettings: v5),

        // --- chat -------------------------------------------------------------------------------------------------------
        .testTarget(
            name: "ChatContractTests", dependencies: [.product(name: "ChatWorker", package: "Chat")],
            path: "Chat/Tests/ChatContractTests", swiftSettings: v6),

        // --- sound ------------------------------------------------------------------------------------------------------
        .testTarget(
            name: "SoundTests",
            dependencies: mlx + [
                .product(name: "SoundWorker", package: "Sound"), .product(name: "VireoCore", package: "Voices"),
                .product(name: "ArsenalCore", package: "Arsenal"), .product(name: "ArsenalWire", package: "ArsenalWire"),
                .product(name: "WorkerTransport", package: "Transport")
            ],
            path: "Sound/Tests/SoundTests", swiftSettings: v5),

        // --- decisions --------------------------------------------------------------------------------------------------
        .testTarget(
            name: "DecisionsTests",
            dependencies: [
                .product(name: "DecisionsWorker", package: "Decisions"), .product(name: "ArsenalKernels", package: "Kernels"),
                .product(name: "Tokenizers", package: "swift-transformers"), .product(name: "Hub", package: "swift-transformers")
            ],
            path: "Decisions/Tests/DecisionsTests", resources: [.copy("Fixtures")], swiftSettings: v5),

        // --- kernels and transport (shared) -----------------------------------------------------------------------------
        .testTarget(
            name: "ArsenalKernelsTests",
            dependencies: [.product(name: "ArsenalKernels", package: "Kernels"), .product(name: "ArsenalKernelsCore", package: "Kernels")],
            path: "Kernels/Tests/ArsenalKernelsTests", swiftSettings: v6),
        .testTarget(
            name: "WorkerTransportTests",
            dependencies: [.product(name: "WorkerTransport", package: "Transport"), .product(name: "ArsenalWire", package: "ArsenalWire")],
            path: "Transport/Tests/WorkerTransportTests", swiftSettings: v6)
    ])
