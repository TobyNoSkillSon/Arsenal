// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func verifiedInventoryRechecksEditsAdditionsLinksAndEmptySubtrees() throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-pol-core/inventory-\(UUID().uuidString)")
    let file = root.appendingPathComponent("voices/input.json")
    try FileManager.default.createDirectory(at: file.deletingLastPathComponent(), withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: root.appendingPathComponent("empty"), withIntermediateDirectories: true)
    let original = Data("{\"type\":\"object\"}".utf8)
    try original.write(to: file)
    let expected = [CatalogFile(path: "voices/input.json", sha256: FileDigest.sha256(original), bytes: Int64(original.count))]
    var cache = VerifiedInventoryCache()
    try cache.verify(directory: root, files: expected)
    try cache.verify(directory: root, files: expected)
    let attributes = try FileManager.default.attributesOfItem(atPath: file.path)
    try Data("{\"type\":\"string\"}".utf8).write(to: file)
    try FileManager.default.setAttributes([.modificationDate: attributes[.modificationDate]!], ofItemAtPath: file.path)
    #expect(throws: (any Error).self) { try cache.verify(directory: root, files: expected) }
    try original.write(to: file); try cache.verify(directory: root, files: expected)
    let unlisted = root.appendingPathComponent("empty/unlisted.json")
    try Data().write(to: unlisted)
    #expect(throws: (any Error).self) { try cache.verify(directory: root, files: expected) }
    // Move the rejected file out; tests preserve their artifacts under the private scratch root.
    try FileManager.default.moveItem(at: unlisted, to: root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString))
    try cache.verify(directory: root, files: expected)
    let outside = root.deletingLastPathComponent().appendingPathComponent(UUID().uuidString)
    try FileManager.default.moveItem(at: file, to: outside)
    try FileManager.default.createSymbolicLink(at: file, withDestinationURL: outside)
    #expect(throws: (any Error).self) { try cache.verify(directory: root, files: expected) }
}
