// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "ModelPrecision", platforms: [.macOS("26.0")],
    products: [.library(name: "ModelPrecision", targets: ["ModelPrecision"])],
    dependencies: [.package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d")],
    targets: [
        .target(name: "ModelPrecision", dependencies: [.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift")]),
        .testTarget(name: "ModelPrecisionTests", dependencies: ["ModelPrecision"])
    ], swiftLanguageModes: [.v6])
