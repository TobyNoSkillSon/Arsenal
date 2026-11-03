// swift-tools-version: 6.1
import PackageDescription
let mlx: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
    .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift"),
    .product(name: "IntegrationKit", package: "IntegrationKit")
]
let perceptionTargets: [Target] = [
    .target(name: "PerceptionSupport", dependencies: mlx),
    .target(name: "MLXTIPS", dependencies: mlx + ["PerceptionSupport"]),
    .target(name: "MLXRFDETR", dependencies: mlx + ["PerceptionSupport"]),
    .target(name: "SAM31", dependencies: mlx + ["PerceptionSupport"]),
    .target(
        name: "MLXLocateAnything",
        dependencies: mlx + ["PerceptionSupport", .product(name: "MLXLMCommon", package: "mlx-swift-lm"), .product(name: "Tokenizers", package: "swift-transformers")]),
    .target(
        name: "PerceptionWorker",
        dependencies: [.product(name: "IntegrationKit", package: "IntegrationKit"), "PerceptionSupport", "MLXTIPS", "MLXRFDETR", "SAM31", "MLXLocateAnything"])
]
for target in perceptionTargets { target.swiftSettings = [.swiftLanguageMode(.v5)] }
let package = Package(
    name: "Perception", platforms: [.macOS("26.0")], products: [.library(name: "PerceptionWorker", targets: ["PerceptionWorker"])],
    dependencies: [
        .package(path: "../IntegrationKit"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: perceptionTargets)
