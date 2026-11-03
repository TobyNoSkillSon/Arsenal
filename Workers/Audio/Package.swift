// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "Audio", platforms: [.macOS("26.0")],
    products: [
        .library(name: "AudioWorker", targets: ["AudioWorker"]),
        // Vella's dictation runtime (Parakeet TDT, Qwen3-ASR) with its optimized paths; see Sources/VellaSTT/PROVENANCE.md.
        .library(name: "VellaSTT", targets: ["VellaSTT"])
    ],
    dependencies: [
        .package(path: "../IntegrationKit"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(
            name: "VellaSTT",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"), .product(name: "Tokenizers", package: "swift-transformers")
            ], exclude: ["PROVENANCE.md"], swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(
            name: "AudioWorker",
            dependencies: [
                "VellaSTT",
                .product(name: "IntegrationKit", package: "IntegrationKit"),
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"), .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"), .product(name: "Hub", package: "swift-transformers")
            ],
            // The fast-path self-test clip (VellaFastPathGate), with its CC BY 4.0 attribution.
            resources: [.copy("Resources/selftest")], swiftSettings: [.swiftLanguageMode(.v5)])
    ])
