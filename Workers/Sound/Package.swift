// swift-tools-version: 5.9
// SPDX-License-Identifier: MIT
import PackageDescription

let mlx: [Target.Dependency] = [.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift")]
let core: Target.Dependency = .product(name: "VireoCore", package: "Voices")
let package = Package(
    name: "ArsenalSound", platforms: [.macOS("26.0")],
    products: [
        .library(name: "SoundWorker", targets: ["SoundWorker"]),
        .executable(name: "arsenal-worker-sound", targets: ["arsenal-worker-sound"])
    ],
    dependencies: [
        .package(path: "../Voices"), .package(path: "../../Packages/ArsenalWire"),
        .package(path: "../Transport"), .package(path: "../../Packages/ModelPrecision"), .package(path: "../IntegrationKit"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d")
    ],
    targets: [
        .target(
            name: "StableAudio3",
            dependencies: mlx + [
                .product(name: "ModelPrecision", package: "ModelPrecision"),
                .product(name: "SmallMGEMM", package: "Voices"), .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift")
            ]),
        .target(
            name: "MossFormer2SE",
            dependencies: mlx + [.product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXFFT", package: "mlx-swift")]),
        .target(
            name: "UniPASE",
            dependencies: mlx + [
                .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXFFT", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift")
            ]),
        .target(
            name: "DeepFilterNet3",
            dependencies: mlx + [.product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXFFT", package: "mlx-swift")]),
        .target(
            name: "SoundEngine",
            dependencies: mlx + [
                .product(name: "ModelPrecision", package: "ModelPrecision"), .product(name: "IntegrationKit", package: "IntegrationKit"), core, "StableAudio3",
                .product(name: "ArsenalWire", package: "ArsenalWire"),
                "MossFormer2SE", "DeepFilterNet3", "UniPASE",
                .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift")
            ],
            exclude: ["Enhance/ReUse/UPSTREAM-README.md", "Enhance/ReUse/CHANGES.md"]),
        .target(
            name: "SoundWorker",
            dependencies: mlx + [
                core, "SoundEngine", .product(name: "ArsenalWire", package: "ArsenalWire"), .product(name: "WorkerTransport", package: "Transport"),
                .product(name: "IntegrationKit", package: "IntegrationKit")
            ]),
        .executableTarget(name: "arsenal-worker-sound", dependencies: ["SoundWorker"])
    ])
