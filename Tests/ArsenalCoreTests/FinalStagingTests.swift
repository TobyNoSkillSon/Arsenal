// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func finalStagingSweepOwnsOnlyStaleNativeImportsAndQuarantines() throws {
    let paths = ArsenalPaths(support: URL(fileURLWithPath: "/tmp/arsenal-final-b/staging-\(UUID().uuidString)")); try paths.create()
    defer { try? FileManager.default.removeItem(at: paths.support) }
    let imports = paths.directory("weights/imports"), tools = paths.directory("tools"), downloads = paths.directory("weights/downloads")
    for dir in [imports, tools, downloads] { try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true) }
    let staleImport = imports.appendingPathComponent(UUID().uuidString + ".part")
    let recentImport = imports.appendingPathComponent(UUID().uuidString + ".part")
    let unknownImport = imports.appendingPathComponent("donor.part")
    let resume = downloads.appendingPathComponent(String(repeating: "a", count: 64) + ".part")
    let abandoned = downloads.appendingPathComponent(String(repeating: "a", count: 64) + ".part.old-" + UUID().uuidString)
    let badPart = downloads.appendingPathComponent(String(repeating: "b", count: 64) + ".part.bad-" + UUID().uuidString)
    let badJournal = downloads.appendingPathComponent(String(repeating: "b", count: 64) + ".json.bad-" + UUID().uuidString)
    let recentUv = tools.appendingPathComponent(".uv-0.12.0-" + UUID().uuidString)
    let linkedUv = tools.appendingPathComponent(".uv-0.12.0-" + UUID().uuidString)
    let uv = tools.appendingPathComponent(".uv-0.12.0-" + UUID().uuidString)
    let unknownUv = tools.appendingPathComponent(".uv-user-folder")
    for file in [staleImport, recentImport, unknownImport, resume, abandoned, badPart, badJournal] { try Data([1, 2, 3]).write(to: file) }
    for dir in [uv, unknownUv, recentUv, linkedUv] {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true); try Data([1, 2, 3]).write(to: dir.appendingPathComponent("uv"))
    }
    let external = paths.support.appendingPathComponent("external"); try Data([4]).write(to: external)
    try FileManager.default.createSymbolicLink(at: linkedUv.appendingPathComponent("external"), withDestinationURL: external)
    let link = imports.appendingPathComponent(UUID().uuidString + ".part"); try FileManager.default.createSymbolicLink(at: link, withDestinationURL: external)
    for item in [staleImport, unknownImport, resume, abandoned, badPart, badJournal, uv, unknownUv, linkedUv] {
        try FileManager.default.setAttributes([.modificationDate: Date().addingTimeInterval(-172_800)], ofItemAtPath: item.path)
    }
    #expect(try StagingCleanup.run(paths: paths) == 4)
    for item in [staleImport, abandoned, badPart, badJournal] { #expect(!FileManager.default.fileExists(atPath: item.path)) }
    for item in [recentImport, unknownImport, resume, uv, unknownUv, recentUv, linkedUv, link, external] { #expect(FileManager.default.fileExists(atPath: item.path)) }
    let receipts = try FileManager.default.contentsOfDirectory(at: paths.directory("logs/cleanup-receipts"), includingPropertiesForKeys: nil)
    #expect(receipts.count == 4)
    for file in receipts {
        let receipt = try WireJSON.decode(Data(contentsOf: file)); #expect(receipt["verified"] == true && receipt["files"]?.array?.count == 1 && receipt["unlinkedBytes"] == 3)
    }
}
