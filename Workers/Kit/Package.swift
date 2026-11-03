// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
import PackageDescription

// Kit models: runners written with the new-model kit (kit/runner), bundled into the native worker. One library per model,
// so each keeps the kit's internal names (Wire, Model, Params) to itself; Entry.swift replaces the runner's main.swift and
// every other file is the runner's own. The dispatcher starts one by its link, arsenal-worker-kit-<id>.
let models = [
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
let dependencies: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXFFT", package: "mlx-swift"),
    .product(name: "Tokenizers", package: "swift-transformers")
]
let package = Package(
    name: "Kit", platforms: [.macOS("26.0")],
    products: models.map { .library(name: $0, targets: [$0]) },
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: models.map { .target(name: $0, dependencies: dependencies, path: "Sources/\($0)", swiftSettings: [.swiftLanguageMode(.v5)]) })
