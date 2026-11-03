// swift-tools-version: 6.0
// SPDX-License-Identifier: MIT
import PackageDescription

// Arsenal decisions worker: Verdict's Laya and Von engines behind the ArsenalWire runner protocol.
// Build with Workers/Decisions/build.sh (CLT Swift 6.3.3 + Xcode's Metal toolchain, as Verdict and Vella).
// mlx-swift is Verdict's parity-proven pin (Projects/Verdict/Package.swift @ f8177ed). swift-transformers follows the
// family pin Voices and Chat use (1.1.9) so Workers/Native resolves one copy; it only serves the stock-path library
// tokenizer (Exact and Fast use Verdict's fast tokenizers), requalified in models/laya-english/log/004-native-link.md.
// Changing either requalifies the Laya/Von fixture gate.
let package = Package(
    name: "ArsenalDecisions",
    platforms: [.macOS("26.0")],
    products: [
        .executable(name: "arsenal-worker-decisions", targets: ["arsenal-worker-decisions"]),
        // The worker as a library, so Workers/Native links decisions with voices and chat into one binary, one MLX copy.
        .library(name: "DecisionsWorker", targets: ["DecisionsWorker"])
    ],
    dependencies: [
        .package(path: "../../Packages/ModelPrecision"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(url: "https://github.com/huggingface/swift-transformers.git", revision: "150169bfba0889c229a2ce7494cf8949f18e6906"),
        .package(path: "../../Packages/ArsenalWire"),
        // Folder-delivered Metal kernels (models/<id>/kernels, docs/KERNELS.md).
        .package(path: "../Kernels")
    ],
    targets: [
        // Verbatim from Verdict (which vendors it from Vella). Module renamed from SmallMGEMM: Voices vendors a newer
        // revision under that name, and module names must be unique in Native's graph; this one is the revision
        // Verdict's gate qualified (tile kernel only, no GEMV path).
        .target(
            name: "VerdictSmallMGEMM",
            dependencies: [
                .product(name: "ArsenalKernels", package: "Kernels"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]),
        // Only Verdict's Safetensors writer (Von's fp32 -> fp16 conversion); the module keeps Verdict's name so the
        // engine sources stay verbatim.
        .target(name: "VerdictCore"),
        // Verdict's engine: Laya, Von, fast tokenizers, windowed attention, small-M route.
        .target(
            name: "VerdictEngine",
            dependencies: [
                .product(name: "ModelPrecision", package: "ModelPrecision"), "VerdictSmallMGEMM",
                "VerdictCore",
                .product(name: "ArsenalKernels", package: "Kernels"),
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "Tokenizers", package: "swift-transformers")
            ]),
        // System One / judge translation (Verdict's helper rules) and the wire framing.
        .target(name: "DecisionsProtocol", dependencies: ["VerdictEngine", "ArsenalWire"]),
        .target(
            name: "DecisionsWorker",
            dependencies: [
                "DecisionsProtocol", "VerdictEngine", "VerdictCore", "ArsenalWire",
                .product(name: "ArsenalKernels", package: "Kernels"),
                .product(name: "MLX", package: "mlx-swift")
            ]),
        .executableTarget(name: "arsenal-worker-decisions", dependencies: ["DecisionsWorker"])
    ],
    // Verdict's sources are Swift 5 mode (its Package.swift is tools 5.9); kept so the port stays verbatim.
    swiftLanguageModes: [.v5]
)
