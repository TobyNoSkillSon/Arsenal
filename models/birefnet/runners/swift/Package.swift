// swift-tools-version: 6.1
import PackageDescription
import Foundation
let wirePackage = URL(fileURLWithPath: #filePath).resolvingSymlinksInPath().deletingLastPathComponent()
    .appendingPathComponent("../../../../Packages/ArsenalWire").standardizedFileURL.path
let package = Package(
    name: "BiRefNet", platforms: [.macOS("26.0")],
    products: [
        .library(name: "BiRefNetWorker", targets: ["BiRefNetWorker"]),
        .library(name: "MLXBiRefNet", targets: ["MLXBiRefNet"]),
        .executable(name: "arsenal-worker-birefnet", targets: ["arsenal-worker-birefnet"]),
        .executable(name: "birefnet-check", targets: ["birefnet-check"])
    ],
    dependencies: [
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d"),
        .package(path: wirePackage),
        .package(path: URL(fileURLWithPath: wirePackage).deletingLastPathComponent().appendingPathComponent("ModelPrecision").path)
    ],
    targets: [
        .testTarget(name: "BiRefNetTests", dependencies: ["MLXBiRefNet", "BiRefNetWorker"], resources: [.copy("Fixtures")]),
        .target(
            name: "MLXBiRefNet",
            dependencies: [
                .product(name: "ModelPrecision", package: "ModelPrecision"), .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"),
                .product(name: "MLXFast", package: "mlx-swift")
            ]),
        .target(name: "BiRefNetWorker", dependencies: ["MLXBiRefNet", "ArsenalWire"]),
        .executableTarget(name: "arsenal-worker-birefnet", dependencies: ["BiRefNetWorker"]),
        .executableTarget(name: "birefnet-check", dependencies: ["MLXBiRefNet"])
    ], swiftLanguageModes: [.v5])
