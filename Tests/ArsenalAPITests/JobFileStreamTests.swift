// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import ArsenalAPI

/// A job file larger than the 64 MiB inline cap is streamed from disk; one past the per-job artifact limit is refused.
@Test func jobFilesStreamPastTheInlineCapAndStopAtTheArtifactLimit() async throws {
    let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-stream-\(UUID().uuidString)", isDirectory: true)
    defer { try? FileManager.default.removeItem(at: support) }
    let job = support.appendingPathComponent("jobs/\(UUID().uuidString.lowercased())", isDirectory: true)
    try FileManager.default.createDirectory(at: job, withIntermediateDirectories: true)
    let long = job.appendingPathComponent("long.wav"), over = job.appendingPathComponent("over.wav")
    // Sparse files: the bytes read are zeros, and the oversized one occupies no disk space.
    let longBytes: UInt64 = 65 * 1024 * 1024 + 7, overBytes = UInt64(OperationArtifacts.maxArtifactBytes) + 1
    for (file, size) in [(long, longBytes), (over, overBytes)] {
        #expect(FileManager.default.createFile(atPath: file.path, contents: nil))
        let handle = try FileHandle(forWritingTo: file)
        try handle.truncate(atOffset: size)
        try handle.close()
    }
    let server = ArsenalAPIServer(service: StubModelsService(rows: [], operations: [:]), supportDirectory: support, port: 0)
    let port = try await server.start(); defer { server.stop() }
    func get(_ file: URL) async throws -> (Data, HTTPURLResponse) {
        let url = URL(string: "http://127.0.0.1:\(port)/arsenal/files/\(job.lastPathComponent)/\(file.lastPathComponent)")!
        let (data, response) = try await URLSession.shared.data(from: url)
        return (data, try #require(response as? HTTPURLResponse))
    }
    let (data, response) = try await get(long)
    #expect(response.statusCode == 200)
    #expect(data.count == Int(longBytes))
    let (_, refused) = try await get(over)
    #expect(refused.statusCode == 404)
}
