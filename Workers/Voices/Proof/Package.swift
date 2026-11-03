// swift-tools-version: 6.0
// SPDX-License-Identifier: MIT
import PackageDescription
let package = Package(
    name: "VoicesGetProof", platforms: [.macOS("26.0")],
    products: [.executable(name: "voices-get-proof", targets: ["voices-get-proof"]), .executable(name: "saved-design-proof", targets: ["saved-design-proof"])],
    dependencies: [.package(name: "Arsenal", path: "../../..")],
    targets: [
        .executableTarget(name: "voices-get-proof", dependencies: [.product(name: "ArsenalCore", package: "Arsenal")]),
        .executableTarget(name: "saved-design-proof", dependencies: [.product(name: "ArsenalCore", package: "Arsenal"), .product(name: "ArsenalAPI", package: "Arsenal")])
    ], swiftLanguageModes: [.v6])
