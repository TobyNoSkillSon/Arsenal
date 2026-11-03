// swift-tools-version: 6.0
// SPDX-License-Identifier: MIT
import PackageDescription
import Foundation
let precisionPackage = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath().deletingLastPathComponent()
    .appendingPathComponent("../../../../Packages/ModelPrecision").standardizedFileURL.path

// d1-3B's Swift runner: LiquidAI's System One prompt and readout (a port of the checkpoint's prompt.py and api.py)
// over an MLX forward of the hybrid tree, and the SigLIP2 picture path. Workers/Native links D1Worker as family `d1`;
// build.sh builds the standalone arsenal-worker-d1 for the gate and the benchmark. mlx-swift is the native worker's pin,
// so the shipped binary carries one MLX. Workers/D1 is a link to this folder: SwiftPM names a path package after its
// directory, and every Swift runner folder is called `swift`.
let package = Package(
    name: "ArsenalD1", platforms: [.macOS("26.0")],
    products: [
        .library(name: "D1Worker", targets: ["D1Worker"]),
        .library(name: "D1Engine", targets: ["D1Engine"]),
        .executable(name: "arsenal-worker-d1", targets: ["arsenal-worker-d1"]),
        .executable(name: "d1-check", targets: ["d1-check"])
    ],
    dependencies: [
        .package(path: precisionPackage),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d")
    ],
    targets: [
        .target(
            name: "D1Engine",
            dependencies: [
                .product(name: "ModelPrecision", package: "ModelPrecision"), .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift")
            ]),
        .target(
            name: "D1Worker",
            dependencies: ["D1Engine", .product(name: "MLX", package: "mlx-swift")]),
        .executableTarget(name: "arsenal-worker-d1", dependencies: ["D1Worker"]),
        // Lab tool: tokens and rendered prompts for the tokenizer and prompt checks; not shipped.
        .executableTarget(name: "d1-check", dependencies: ["D1Engine", .product(name: "MLX", package: "mlx-swift")]),
        .testTarget(name: "D1Tests", dependencies: ["D1Engine"])
    ], swiftLanguageModes: [.v6])
