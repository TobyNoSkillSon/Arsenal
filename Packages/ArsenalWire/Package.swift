// swift-tools-version: 6.0
// SPDX-License-Identifier: MIT
import PackageDescription
let package = Package(
    name: "ArsenalWire", platforms: [.macOS("26.0")], products: [.library(name: "ArsenalWire", targets: ["ArsenalWire"])],
    targets: [.target(name: "ArsenalWire"), .testTarget(name: "ArsenalWireTests", dependencies: ["ArsenalWire"])], swiftLanguageModes: [.v6])
