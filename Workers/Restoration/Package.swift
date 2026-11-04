// swift-tools-version: 6.1
import PackageDescription
let mlx: [Target.Dependency] = [
    .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift"),
    .product(name: "MLXRandom", package: "mlx-swift")
]
let package = Package(
    name: "Restoration", platforms: [.macOS("26.0")],
    products: [.library(name: "RestorationWorker", targets: ["RestorationWorker"])],
    dependencies: [.package(path: "../IntegrationKit"), .package(url: "https://github.com/ml-explore/mlx-swift.git", revision: "901941965d82e4a216d4d117231d847d194c563d")],
    targets: [
        .target(name: "RIFENative", dependencies: mlx, swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "SeedVR2Native", dependencies: mlx, swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "DDColorNative", dependencies: mlx, swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "GIMMNative", dependencies: mlx, swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "FlashVSRNative", dependencies: mlx, swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "LaMaNative", dependencies: mlx + [.product(name: "MLXFFT", package: "mlx-swift")], swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(
            name: "RestorationWorker",
            dependencies: [
                .product(name: "IntegrationKit", package: "IntegrationKit"), "RIFENative", "SeedVR2Native", "DDColorNative", "LaMaNative", "GIMMNative", "FlashVSRNative"
            ] + mlx,
            swiftSettings: [.swiftLanguageMode(.v5)])
    ])
