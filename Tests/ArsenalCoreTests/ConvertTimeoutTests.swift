// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct ConvertTimeoutTests {
    @Test func deadlineKeepsTheFloorAndGrowsWithBytes() {
        let gib = Int64(1) << 30
        #expect(WorkerPrecisionConverter.convertTimeout(bytes: 0) == 600)
        #expect(WorkerPrecisionConverter.convertTimeout(bytes: 2 * gib) == 600)
        #expect(WorkerPrecisionConverter.convertTimeout(bytes: 16 * gib) == 4_800)
    }

    @Test func treeBytesCountsEveryFileUnderEachPath() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("convert-bytes-\(UUID().uuidString)", isDirectory: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root.appendingPathComponent("sub"), withIntermediateDirectories: true)
        try Data(count: 3_000).write(to: root.appendingPathComponent("a.safetensors"))
        try Data(count: 5_000).write(to: root.appendingPathComponent("sub/b.safetensors"))
        // A lone file beside the tree, so it counts once when listed on its own.
        let single = FileManager.default.temporaryDirectory.appendingPathComponent("convert-bytes-\(UUID().uuidString).bin")
        defer { try? FileManager.default.removeItem(at: single) }
        try Data(count: 700).write(to: single)
        #expect(WorkerPrecisionConverter.treeBytes([root]) == 8_000)
        #expect(WorkerPrecisionConverter.treeBytes([root, single, root.appendingPathComponent("missing")]) == 8_700)
    }
}
