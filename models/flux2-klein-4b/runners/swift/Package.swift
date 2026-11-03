// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
import PackageDescription
import Foundation
let precisionPackage = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath().deletingLastPathComponent()
    .appendingPathComponent("../../../../Packages/ModelPrecision").standardizedFileURL.path

// FLUX.2 Klein 4B and FLUX 3 Image Klein (models/flux3-image-klein) for Arsenal, in Swift/MLX. The worker library links into Workers/Native (one MLX copy for every family);
// arsenal-worker-flux and flux-lab are lab builds (scripts in this folder), never shipped on their own.
// Pins match Workers/Native: the same mlx-swift and swift-transformers revisions. No path dependencies: Workers/Flux is
// a link to this folder (SwiftPM names a path package after its directory, and every runner folder is `swift`).
let package = Package(
    name: "ArsenalFluxKlein", platforms: [.macOS("26.0")],
    products: [
        .library(name: "FluxWorker", targets: ["FluxWorker"]),
        .library(name: "Flux2Klein", targets: ["Flux2Klein"]),
        .library(name: "Flux3Klein", targets: ["Flux3Klein"]),
        .executable(name: "arsenal-worker-flux", targets: ["arsenal-worker-flux"]),
        .executable(name: "flux-lab", targets: ["flux-lab"])
    ],
    dependencies: [
        .package(path: precisionPackage),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(
            name: "Flux2Klein",
            dependencies: [
                .product(name: "ModelPrecision", package: "ModelPrecision"),
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"), .product(name: "Hub", package: "swift-transformers")
            ]),
        .target(
            name: "Flux3Klein",
            dependencies: [
                "Flux2Klein", .product(name: "ModelPrecision", package: "ModelPrecision"),
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"), .product(name: "Hub", package: "swift-transformers")
            ]),
        .target(
            name: "FluxWorker",
            dependencies: [
                "Flux2Klein", "Flux3Klein", .product(name: "MLX", package: "mlx-swift")
            ]),
        .executableTarget(name: "arsenal-worker-flux", dependencies: ["FluxWorker"]),
        .executableTarget(
            name: "flux-lab",
            dependencies: [
                "Flux2Klein", .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift")
            ]),
        .testTarget(name: "Flux2KleinTests", dependencies: ["Flux2Klein", "FluxWorker"])
    ], swiftLanguageModes: [.v6])
