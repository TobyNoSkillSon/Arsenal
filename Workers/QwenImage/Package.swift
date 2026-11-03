// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
import PackageDescription

// Qwen-Image-2.1 (base and 8-step Turbo) in Swift/MLX: the Qwen3-VL prompt encoder, the block-causal single-stream DiT,
// the RGBA VAE and the flow-matching scheduler. Libraries only: Workers/Package.swift owns the tests and the lab tool.
let mlx: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift"),
    .product(name: "MLXRandom", package: "mlx-swift")
]
let package = Package(
    name: "QwenImage", platforms: [.macOS("26.0")],
    products: [
        .library(name: "QwenImageWorker", targets: ["QwenImageWorker"]),
        .library(name: "QwenImageNative", targets: ["QwenImageNative"])
    ],
    dependencies: [
        .package(path: "../IntegrationKit"),
        .package(path: "../../Packages/ModelPrecision"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(
            name: "QwenImageNative", dependencies: mlx + [.product(name: "Tokenizers", package: "swift-transformers")],
            swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(
            name: "QwenImageWorker",
            dependencies: [.product(name: "IntegrationKit", package: "IntegrationKit"), .product(name: "ModelPrecision", package: "ModelPrecision"), "QwenImageNative"] + mlx,
            swiftSettings: [.swiftLanguageMode(.v5)])
    ])
