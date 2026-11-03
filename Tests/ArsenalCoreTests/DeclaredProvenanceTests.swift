// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct DeclaredProvenanceTests {
    private func git(_ arguments: [String], in directory: URL, committerDate: String? = nil) throws {
        let process = Process()
        process.executableURL = URL(fileURLWithPath: "/usr/bin/git"); process.arguments = ["-C", directory.path] + arguments
        var environment = ProcessInfo.processInfo.environment
        if let committerDate { environment["GIT_COMMITTER_DATE"] = committerDate; environment["GIT_AUTHOR_DATE"] = committerDate }
        process.environment = environment
        process.standardOutput = FileHandle.nullDevice; process.standardError = FileHandle.nullDevice
        try process.run(); process.waitUntilExit()
        #expect(process.terminationStatus == 0, "git \(arguments.first ?? "") failed")
    }

    @Test func declaredContextKeepsItsSourceDateAcrossCalls() async throws {
        let sourceDate = "2026-10-01T12:00:00+00:00"
        let fixture = try await makeLocalCatalogFixture(
            kinds: ["chat"], filePayloads: ["config.json": Data("{\"max_position_embeddings\":262144}".utf8)],
            artifacts: ProtocolFixtureArtifact(scenario: "normal"), context: 32768, nativeContext: 262144, sourceDate: sourceDate)
        defer { fixture.stop() }
        let model = try await fixture.service.measurementManifest(id: "fixture-chat")
        let first = await fixture.service.statsForRunner(model, runner: "native")
        // The second call happens a whole second later: a wall-clock stamp would differ here.
        try await Task.sleep(nanoseconds: 1_100_000_000)
        let second = await fixture.service.statsForRunner(model, runner: "native")
        #expect(first == second)
        let dates = first.values.compactMap { $0.value(.maximumContextTokens).provenance?.date }
        #expect(!dates.isEmpty && dates.count == first.count && dates.allSatisfy { $0 == sourceDate })
        await fixture.service.shutdown()
    }

    @Test func localFolderDateIsItsLastCommitOrModelFileTime() throws {
        let root = URL(fileURLWithPath: "/private/tmp/arsenal-declared-date-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: root) }
        // No git history: model.toml's modification time.
        let plain = root.appendingPathComponent("plain")
        try FileManager.default.createDirectory(at: plain, withIntermediateDirectories: true)
        let modelFile = plain.appendingPathComponent("model.toml")
        try Data("id = \"plain\"\n".utf8).write(to: modelFile)
        let modified = Date(timeIntervalSince1970: 1_790_000_000)
        try FileManager.default.setAttributes([.modificationDate: modified], ofItemAtPath: modelFile.path)
        #expect(DeclaredDate.localFolder(plain) == ISO8601DateFormatter().string(from: modified))
        // With git history: the committer time of the last commit touching the folder.
        let tracked = root.appendingPathComponent("tracked")
        try FileManager.default.createDirectory(at: tracked, withIntermediateDirectories: true)
        try git(["init", "-q"], in: tracked)
        try Data("id = \"tracked\"\n".utf8).write(to: tracked.appendingPathComponent("model.toml"))
        try git(["add", "."], in: tracked)
        try git(["-c", "user.name=test", "-c", "user.email=test@example.invalid", "commit", "-q", "-m", "model"], in: tracked, committerDate: "2026-10-02T09:30:00+02:00")
        #expect(DeclaredDate.localFolder(tracked) == "2026-10-02T09:30:00+02:00")
    }
}
