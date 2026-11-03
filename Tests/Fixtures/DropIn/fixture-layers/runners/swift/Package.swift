// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
import PackageDescription
// This deterministic protocol fixture needs no MLX dependencies. Model runners pin the app's MLX revisions.
let package = Package(
    name: "FixtureLayers", platforms: [.macOS("26.0")],
    products: [.executable(name: "fixture-layers", targets: ["FixtureLayers"])],
    targets: [.executableTarget(name: "FixtureLayers")], swiftLanguageModes: [.v6])
