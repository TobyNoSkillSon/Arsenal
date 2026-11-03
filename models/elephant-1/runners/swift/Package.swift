// swift-tools-version: 6.0
// SPDX-License-Identifier: MIT
import Foundation
import PackageDescription

// elephant-1's Swift runner: the maker's System One decision path (OneJev qev + the checkpoint's serve.patch) over an MLX
// port of the Qwen3.5 text decoder. Workers/Native links ElephantWorker as family `elephant`; Workers/Elephant is a link
// to this folder (SwiftPM names a path package after its directory). Pins are the native graph's, so the shipped binary
// carries one MLX; build and test only through Workers/Package.swift.
let root = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath().deletingLastPathComponent()
func local(_ relative: String) -> String { root.appendingPathComponent(relative).standardizedFileURL.path }

let package = Package(
    name: "ArsenalElephant", platforms: [.macOS("26.0")],
    products: [
        .library(name: "ElephantWorker", targets: ["ElephantWorker"]),
        .library(name: "ElephantEngine", targets: ["ElephantEngine"]),
        .executable(name: "arsenal-worker-elephant", targets: ["arsenal-worker-elephant"])
    ],
    dependencies: [
        .package(path: local("../../../../Workers/IntegrationKit")),
        .package(path: local("../../../../Packages/ArsenalWire")),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(
            name: "ElephantEngine",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers"), .product(name: "Hub", package: "swift-transformers")
            ]),
        .target(
            name: "ElephantWorker",
            dependencies: [
                "ElephantEngine", .product(name: "MLX", package: "mlx-swift"),
                .product(name: "IntegrationKit", package: "IntegrationKit"), .product(name: "ArsenalWire", package: "ArsenalWire")
            ]),
        .executableTarget(name: "arsenal-worker-elephant", dependencies: ["ElephantWorker"])
    ], swiftLanguageModes: [.v5])
