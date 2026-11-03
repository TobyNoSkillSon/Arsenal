// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
// Arsenal new-runner kit. Same MLX pin as the app (scripts/pkg-lib.sh MLX_SWIFT_PIN), so the app's shaders fit.
// Do not edit: kit/new-model.py builds every model's runner from this exact manifest.
import PackageDescription
let package = Package(
    name: "ArsenalRunner", platforms: [.macOS("26.0")],
    products: [.executable(name: "runner", targets: ["Runner"])],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906"),
    ],
    targets: [
        .executableTarget(
            name: "Runner",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXFFT", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"),
            ])
    ],
    swiftLanguageModes: [.v5])
