// swift-tools-version: 6.1
// SPDX-License-Identifier: MIT
import PackageDescription

let package = Package(
    name: "ArsenalWorkerTransport", platforms: [.macOS("26.0")],
    products: [
        .library(name: "WorkerTransport", targets: ["WorkerTransport"])
    ],
    dependencies: [
        .package(path: "../../Packages/ArsenalWire")
    ],
    targets: [
        .target(name: "WorkerTransport", dependencies: [.product(name: "ArsenalWire", package: "ArsenalWire")])
    ], swiftLanguageModes: [.v6])
