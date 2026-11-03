// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
import PackageDescription
let package = Package(
    name: "ArsenalChat", platforms: [.macOS("26.0")],
    products: [
        .executable(name: "arsenal-worker-chat", targets: ["arsenal-worker-chat"]),
        // The worker as a library, so Workers/Native links voices and chat into one binary with one MLX copy.
        .library(name: "ChatWorker", targets: ["ChatWorker"])
    ],
    dependencies: [
        .package(path: "../../Packages/ArsenalWire"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(name: "ChatContract"),
        .target(
            name: "ChatEngine",
            dependencies: [
                "ChatContract", .product(name: "ArsenalWire", package: "ArsenalWire"), .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]),
        .target(name: "ChatWorker", dependencies: ["ChatContract", "ChatEngine", .product(name: "ArsenalWire", package: "ArsenalWire")]),
        .executableTarget(name: "arsenal-worker-chat", dependencies: ["ChatWorker"])
    ], swiftLanguageModes: [.v6])
