// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "Music", platforms: [.macOS("26.0")],
    products: [.library(name: "MusicWorker", targets: ["MusicWorker"])],
    dependencies: [
        .package(path: "../IntegrationKit"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(
            name: "MusicWorker",
            dependencies: [
                .product(name: "IntegrationKit", package: "IntegrationKit"),
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"), .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"), .product(name: "Hub", package: "swift-transformers")
            ], swiftSettings: [.swiftLanguageMode(.v5)])
    ])
