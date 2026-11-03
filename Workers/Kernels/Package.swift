// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
import PackageDescription

// ArsenalKernels: folder-delivered Metal kernels for native runners (CONTRACTS §15, docs/KERNELS.md).
// ArsenalKernelsCore needs only Foundation + Metal (any engine: MLX, own Metal, C via a shim); ArsenalKernels adds the
// MLX custom-kernel wrapper. mlx-swift is pinned like every Arsenal family (changing it requalifies every gate).
let package = Package(
    name: "ArsenalKernels",
    platforms: [.macOS("26.0")],
    products: [
        .library(name: "ArsenalKernelsCore", targets: ["ArsenalKernelsCore"]),
        .library(name: "ArsenalKernels", targets: ["ArsenalKernels"]),
        .executable(name: "arsenal-kernels-smoke", targets: ["arsenal-kernels-smoke"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d")
    ],
    targets: [
        .target(name: "ArsenalKernelsCore"),
        // One C++ line: MLX's own Metal preamble (mlx::core::metal::utils()), so the compile preflight builds exactly
        // the source MLX will build, from the MLX that is linked.
        .target(name: "MLXPreamble", dependencies: [.product(name: "MLX", package: "mlx-swift")]),
        .target(
            name: "ArsenalKernels",
            dependencies: [
                "ArsenalKernelsCore", "MLXPreamble",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]),
        .executableTarget(
            name: "arsenal-kernels-smoke",
            dependencies: [
                "ArsenalKernels", "ArsenalKernelsCore",
                .product(name: "MLX", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ])
    ],
    swiftLanguageModes: [.v6],
    cxxLanguageStandard: .gnucxx20
)
