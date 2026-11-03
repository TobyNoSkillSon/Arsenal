// swift-tools-version: 6.1
import PackageDescription
let package = Package(
    name: "Geometry", platforms: [.macOS("26.0")],
    products: [.library(name: "GeometryWorker", targets: ["GeometryWorker"])],
    dependencies: [
        .package(path: "../IntegrationKit"),
        .package(path: "../Flux"),
        .package(url: "https://github.com/ml-explore/mlx-swift", revision: "901941965d82e4a216d4d117231d847d194c563d")
    ],
    targets: [
        .target(
            name: "MLXMoGe",
            dependencies: [.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift")],
            exclude: ["LICENSE-upstream", "PORTED-FROM.md"], swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(
            name: "MLXDA3", dependencies: [.product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift")],
            exclude: ["LICENSE-upstream", "PORTED-FROM.md", "THIRD-PARTY-NOTICES.md"], swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(name: "GeometryXatlas", path: "Sources/GeometryXatlas", exclude: ["LICENSE-upstream"], publicHeadersPath: "include", cxxSettings: [.define("NDEBUG")]),
        .target(
            name: "MLXMesh",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift"), .product(name: "MLXLinalg", package: "mlx-swift"), "GeometryXatlas"
            ], exclude: ["LICENSE-upstream", "PORTED-FROM.md"], swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(
            name: "MLXTrellis",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXLinalg", package: "mlx-swift"), .product(name: "MLXRandom", package: "mlx-swift"), "IntegrationKit", "MLXMesh"
            ], exclude: ["LICENSE-upstream", "LICENSE-Pillow", "PORTED-FROM.md"], swiftSettings: [.swiftLanguageMode(.v5)]),
        // IDOL: its denoiser and decoders, DINOv3 ViT-H+ and the PLY output; the FLUX.2 VAE encoder is Flux3Klein's.
        .target(
            name: "MLXIdol",
            dependencies: [
                .product(name: "MLX", package: "mlx-swift"), .product(name: "MLXNN", package: "mlx-swift"), .product(name: "MLXFast", package: "mlx-swift"),
                .product(name: "MLXRandom", package: "mlx-swift"), "IntegrationKit", "MLXTrellis",
                .product(name: "Flux3Klein", package: "Flux"), .product(name: "Flux2Klein", package: "Flux")
            ], exclude: ["PORTED-FROM.md"], swiftSettings: [.swiftLanguageMode(.v5)]),
        .target(
            name: "GeometryWorker", dependencies: [.product(name: "IntegrationKit", package: "IntegrationKit"), "MLXDA3", "MLXMoGe", "MLXTrellis", "MLXIdol"],
            swiftSettings: [.swiftLanguageMode(.v5)])
    ], cxxLanguageStandard: .cxx17)
