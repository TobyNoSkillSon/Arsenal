// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func logRetentionKeepsNewestWithinBudget() throws {
    let dir = URL(fileURLWithPath: "/tmp/arsenal-core/logs-\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    defer { try? FileManager.default.removeItem(at: dir) }
    for i in 0..<40 {
        let url = dir.appendingPathComponent("w-\(i).log"); try Data(count: 1024).write(to: url)
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: Double(1_000_000 + i))], ofItemAtPath: url.path)
    }
    try Data("keep".utf8).write(to: dir.appendingPathComponent("config.json"))
    LogRetention.prune(dir, keep: 10, maxBytes: 1 << 20)
    let left = Set(try FileManager.default.contentsOfDirectory(atPath: dir.path))
    #expect(left == Set((31..<40).map { "w-\($0).log" } + ["config.json"]))
    LogRetention.prune(dir, keep: 10, maxBytes: 3 * 1024)
    #expect(Set(try FileManager.default.contentsOfDirectory(atPath: dir.path)) == ["w-39.log", "w-38.log", "w-37.log", "config.json"])
}
