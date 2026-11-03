// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "Embed", platforms: [.macOS("26.0")], products: [.library(name: "EmbedWorker", targets: ["EmbedWorker"])],
    dependencies: [
        .package(path: "../IntegrationKit"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(
            name: "EmbedWorker",
            dependencies: [
                .product(name: "IntegrationKit", package: "IntegrationKit"),
                .product(name: "MLXEmbedders", package: "mlx-swift-lm"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers"),
                .product(name: "Hub", package: "swift-transformers")
            ], swiftSettings: [.swiftLanguageMode(.v5)])
    ])
