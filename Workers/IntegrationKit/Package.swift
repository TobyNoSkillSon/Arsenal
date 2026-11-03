// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "IntegrationKit", platforms: [.macOS("26.0")], products: [.library(name: "IntegrationKit", targets: ["IntegrationKit"])],
    dependencies: [
        .package(path: "../../Packages/ArsenalWire"), .package(path: "../../Packages/ModelPrecision"),
        .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d")
    ],
    targets: [
        .target(
            name: "IntegrationKit",
            dependencies: [
                .product(name: "ArsenalWire", package: "ArsenalWire"),
                .product(name: "ModelPrecision", package: "ModelPrecision"),
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift")
            ], swiftSettings: [.swiftLanguageMode(.v5)])
    ])
