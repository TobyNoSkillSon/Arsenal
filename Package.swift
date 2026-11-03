// swift-tools-version: 6.0
// SPDX-License-Identifier: MIT
import PackageDescription
let package = Package(
    name: "Arsenal", platforms: [.macOS("26.0")],
    products: [
        .library(name: "ArsenalCore", targets: ["ArsenalCore"]), .library(name: "ArsenalAPI", targets: ["ArsenalAPI"]), .library(name: "ArsenalBench", targets: ["ArsenalBench"]),
        .executable(name: "ArsenalUpdateHelper", targets: ["ArsenalUpdateHelper"]),
        .executable(name: "Arsenal", targets: ["Arsenal"]), .executable(name: "arsenal-mcp", targets: ["arsenal-mcp"]),
        .executable(name: "arsenal-bench", targets: ["arsenal-bench"]),
        .executable(name: "arsenal-stats", targets: ["arsenal-stats"]),
        .executable(name: "arsenal-stub-worker", targets: ["arsenal-stub-worker"]),
        .executable(name: "arsenal-testserver", targets: ["arsenal-testserver"])
    ], dependencies: [.package(path: "Packages/ArsenalWire"), .package(path: "Packages/swift-sdk")],
    targets: [
        .target(name: "ArsenalCore", dependencies: [.product(name: "ArsenalWire", package: "ArsenalWire")]),
        .target(name: "ArsenalAPI", dependencies: ["ArsenalCore", "ArsenalBench", .product(name: "MCP", package: "swift-sdk")]),
        .target(name: "ArsenalBench", dependencies: ["ArsenalCore"]),
        .executableTarget(name: "ArsenalUpdateHelper", dependencies: ["ArsenalCore"]),
        .executableTarget(name: "Arsenal", dependencies: ["ArsenalCore", "ArsenalAPI", "ArsenalCommands"]), .executableTarget(name: "arsenal-mcp", dependencies: []),
        .executableTarget(name: "arsenal-bench", dependencies: ["ArsenalBench"]), .executableTarget(name: "arsenal-stats", dependencies: ["ArsenalCore"]),
        .target(name: "ArsenalCommands", dependencies: ["ArsenalCore", "ArsenalBench", .product(name: "ArsenalWire", package: "ArsenalWire")]),
        .executableTarget(name: "arsenal-stub-worker", dependencies: [.product(name: "ArsenalWire", package: "ArsenalWire")]),
        .executableTarget(name: "arsenal-testserver", dependencies: ["ArsenalAPI"]),
        .testTarget(name: "ArsenalRunnerCheckTests", dependencies: ["ArsenalCommands"]),
        .testTarget(name: "ArsenalCoreTests", dependencies: ["ArsenalCore"], exclude: ["Fixtures"]), .testTarget(name: "ArsenalAPITests", dependencies: ["ArsenalAPI"]),
        .testTarget(name: "ArsenalBenchTests", dependencies: ["ArsenalBench"]),
        .testTarget(name: "ArsenalAppTests", dependencies: ["Arsenal", "ArsenalCore"])
    ], swiftLanguageModes: [.v6])
