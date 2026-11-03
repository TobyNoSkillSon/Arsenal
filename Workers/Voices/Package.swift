// swift-tools-version: 5.9
import PackageDescription
import Foundation

// Headless TTS family worker. Family pins and provenance: PORTED-FROM.md.
let mlx: [Target.Dependency] = [.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift")]
let vendor = "Sources/"

let package = Package(
    name: "ArsenalVoices", platforms: [.macOS("26.0")],
    products: [
        .executable(name: "arsenal-worker-voices", targets: ["arsenal-worker-voices"]),
        // The worker as a library, so Workers/Native links voices and chat into one binary with one MLX copy.
        .library(name: "VoicesWorker", targets: ["VoicesWorker"]),
        .library(name: "VireoCore", targets: ["VireoCore"]),
        .library(name: "SmallMGEMM", targets: ["SmallMGEMM"]),
        .library(name: "FishDrama3", targets: ["FishDrama3", "FishS1Codec"]),
        .library(name: "VoxCPM2", targets: ["VoxCPM2"])
    ],
    dependencies: [
        .package(path: "../../Packages/ModelPrecision"),
        .package(path: "../../Packages/ArsenalWire"),
        .package(path: "../Transport"),
        .package(path: "../Kernels"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/ml-explore/mlx-swift-lm.git", revision: "bd4b7434e6bdb588c7ef55706ff8904cb7fd4c57"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906")
    ],
    targets: [
        .target(name: "VireoCore"),
        .target(name: "MLXAudioCore", dependencies: mlx + [.product(name: "MLXFFT", package: "mlx-swift")], path: vendor + "MLXAudioCore"),
        .target(name: "MLXAudioCodecs", dependencies: ["MLXAudioCore", .product(name: "MLXLMCommon", package: "mlx-swift-lm")] + mlx, path: vendor + "MLXAudioCodecs"),
        // Small-M GEMM kernels, the V family's shared copy (Vella Worker/Sources/SmallMGEMM, verbatim; commit in Vendor/README.md).
        .target(
            name: "SmallMGEMM", dependencies: [.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift")],
            path: vendor + "SmallMGEMM"),
        .target(
            name: "MLXAudioTTS",
            dependencies: [
                "MLXAudioCore", "MLXAudioCodecs", "SmallMGEMM", .product(name: "ArsenalKernels", package: "Kernels"), .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXFFT", package: "mlx-swift"), .product(name: "MLXLMCommon", package: "mlx-swift-lm"),
                .product(name: "Transformers", package: "swift-transformers")
            ] + mlx,
            path: vendor + "MLXAudioTTS",
            exclude: ["Models/BreezeTTS/README.md", "Models/PocketTTS/README.md", "Models/Qwen3TTS/README.md", "Models/StyleTTS2/Kokoro/README.md"]),
        // Fish S1 DAC (Drama 3's codec): mlx-audio-swift 01dec7c9 plus the streaming decode of mlx-audio dad2087f.
        .target(name: "FishS1Codec", dependencies: [.product(name: "MLXFast", package: "mlx-swift")] + mlx, path: vendor + "FishS1Codec"),
        // Fish Audio Drama 3, ported from mlx-audio fish_drama3 dad2087f.
        .target(
            name: "FishDrama3",
            dependencies: [
                "FishS1Codec", "VireoCore", .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "MLXLMCommon", package: "mlx-swift-lm"), .product(name: "Transformers", package: "swift-transformers")
            ] + mlx),
        // OpenBMB VoxCPM2, from mlx-audio-swift e8450402 Models/VoxCPM2, checked against OpenBMB/VoxCPM.
        .target(
            name: "VoxCPM2",
            dependencies: [
                "VireoCore", .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift"),
                .product(name: "Transformers", package: "swift-transformers")
            ] + mlx),
        .target(
            name: "VireoEngine",
            dependencies: [
                .product(name: "ModelPrecision", package: "ModelPrecision"), "VireoCore", "MLXAudioTTS", "SmallMGEMM", .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift")
            ] + mlx),
        .target(
            name: "VoicesWorker",
            dependencies: [
                .product(name: "WorkerTransport", package: "Transport"), "VireoCore", "VireoEngine", "MLXAudioTTS", "FishDrama3", "VoxCPM2",
                .product(name: "ArsenalWire", package: "ArsenalWire")
            ]
                + mlx, resources: [.process("Resources")]),
        .executableTarget(name: "arsenal-worker-voices", dependencies: ["VoicesWorker"])
    ])
