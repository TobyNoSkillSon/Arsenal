// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "Language", platforms: [.macOS("26.0")], products: [.library(name: "LanguageWorker", targets: ["LanguageWorker"])],
    dependencies: [
        .package(path: "../IntegrationKit"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(
            name: "LanguageWorker",
            dependencies: [
                .product(name: "IntegrationKit", package: "IntegrationKit"),
                .product(name: "MLXVLM", package: "mlx-swift-lm"),
                .product(name: "MLXLLM", package: "mlx-swift-lm"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ],
            exclude: [
                "Upstream/Hunyuan/LICENSE-upstream", "Upstream/Hunyuan/PORTED-FROM.md", "Upstream/TeleOCR/LICENSE-upstream", "Upstream/TeleOCR/PORTED-FROM.md",
                "Upstream/Paddle/LICENSE-upstream", "Upstream/Paddle/PORTED-FROM.md", "Upstream/LFM2VL/LICENSE-upstream", "Upstream/LFM2VL/PORTED-FROM.md",
                "Upstream/Gemma4/LICENSE-upstream", "Upstream/Gemma4/PORTED-FROM.md"
            ],
            swiftSettings: [.swiftLanguageMode(.v5)])
    ])
