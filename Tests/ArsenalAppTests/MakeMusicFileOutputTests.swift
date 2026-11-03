// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// Arsenal 1.0, 3 Nov 2026: Make Music refused ACE-Step 1.5 Turbo with "declares no operation that makes
/// audio", because the window kept only operations whose output is `audio`, and ACE-Step, YuE2 and MiniMax Music
/// declare `generate_music` with `output = "file"` (a finished song from a job). The window is checked against the
/// folder as it is today, through the app's own API over the stub service, to a saved take.
@MainActor @Suite(.serialized) final class MakeMusicFileOutputTests {
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-music-file-\(UUID().uuidString)", isDirectory: true)
    deinit { try? FileManager.default.removeItem(at: scratch) }

    private static let kinds = SoundFixture.repo.appendingPathComponent("Resources/kinds")

    /// The folder's own declaration, as the service lists it.
    private static func operations(_ id: String) throws -> [ModelOperation] {
        let folder = shippedOrRetiredFolder(SoundFixture.repo, id)
        let manifest = try ModelManifest.load(folder, kindsDirectory: kinds)
        return manifest.operations.map { d in
            ModelOperation(
                name: d.name, title: d.title, inputSchema: SoundFixture.schema(id, d.name), output: d.output, stream: d.stream,
                policy: d.timeoutS.map { OperationPolicy(timeoutS: $0, firstResponseTimeoutS: d.firstResponseTimeoutS ?? $0) })
        }
    }

    @Test func everyMusicFolderThatAnswersWithFilesIsOfferedByMakeMusic() throws {
        let session = SoundSession(models: [], selected: "", client: { nil }, inbox: scratch, outputDirectory: scratch, task: .music)
        for id in ["ace-step-1.5", "yue2-3b", "minimax-music3", "stable-audio-3-medium"] {
            let ops = try Self.operations(id)
            let music = try #require(ops.first { $0.name == "generate_music" }, "\(id) declares no generate_music")
            let d = OperationDescriptor(name: music.name, title: music.title, inputSchema: music.inputSchema, output: music.output, stream: music.stream)
            #expect(session.offers(d), "\(id) generate_music (output \(music.output)) is refused")
        }
        // A file operation that is not a sound task's stays out (a window of sounds never offers a document).
        let other = OperationDescriptor(name: "read_document", title: "Read", inputSchema: [:], output: "file", stream: false)
        #expect(!SoundSession(models: [], selected: "", client: { nil }, inbox: scratch, outputDirectory: scratch).offers(other))
    }

    @Test func aceStepMakesASongInTheMakeMusicWindow() async throws {
        let id = "ace-step-1.5"
        var row = SoundFixture.row(id, "Music Fixture", kind: "sound", tools: ["sound"])
        row.name = "Music Fixture"
        let ops = try Self.operations(id)
        #expect(ops.map(\.name) == ["generate_music"] && ops[0].output == "file")
        let service = FixtureService(rows: [row], metadata: [id: ["tools": ["sound"]]], operations: [id: ops])
        await service.setRunner("generate_music") { params in
            let dir = URL(fileURLWithPath: params["out_dir"]?.string ?? NSTemporaryDirectory())
            let file = try CapabilityFixture.file(dir, name: "song", ext: "wav", mime: "audio/wav", id: "song") {
                try CapabilityFixture.tone($0, frequency: 330, seconds: 0.5)
            }
            return ["files": [file]]
        }
        let support = scratch.appendingPathComponent("support")
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        let port = try #require(api.port)
        let music = scratch.appendingPathComponent("Music")
        let session = SoundSession(
            models: [SoundFixture.model(row)], selected: id, client: { ArsenalClient(port: port) }, inbox: support.appendingPathComponent("inbox"),
            outputDirectory: music, task: .music)
        defer { session.close() }
        session.prepare()
        try await waitUntil { !session.loading }
        #expect(session.problem == nil, "\(session.problem?.message ?? "")")
        #expect(session.operations.map(\.name) == ["generate_music"] && session.operation == "generate_music")
        #expect(session.runVerb == SoundTask.music.runTitle)
        session.set("prompt", .text("Acoustic folk with a slow guitar"))
        #expect(session.canRun)
        session.start()
        try await waitUntil(seconds: 30) { !session.running }
        #expect(session.problem == nil, "\(session.problem?.message ?? "")")
        let take = try #require(session.takes.first)
        #expect(take.url.deletingLastPathComponent().resolvingSymlinksInPath() == music.resolvingSymlinksInPath() && take.url.pathExtension == "wav")
        #expect(abs(take.shape.seconds - 0.5) < 0.01)
        #expect(await service.runLog() == ["\(id) generate_music"])
    }

    /// ACE-Step, YuE2 and MiniMax Music take `batch_size`: a batch gives one take per song (a non-audio file beside them,
    /// such as a metadata JSON, is no take), the last one shown.
    @Test func aBatchOfSongsGivesOneTakeEach() async throws {
        let id = "ace-step-1.5"
        let row = SoundFixture.row(id, "Music Fixture", kind: "sound", tools: ["sound"])
        let service = FixtureService(rows: [row], metadata: [id: ["tools": ["sound"]]], operations: [id: try Self.operations(id)])
        await service.setRunner("generate_music") { params in
            let dir = URL(fileURLWithPath: params["out_dir"]?.string ?? NSTemporaryDirectory())
            let meta = try CapabilityFixture.file(dir, name: "meta", ext: "json", mime: "application/json", id: "meta") { try Data("{}".utf8).write(to: $0) }
            let songs = try (0..<3).map { i in
                try CapabilityFixture.file(dir, name: "song-\(i)", ext: "wav", mime: "audio/wav", id: "song-\(i)") {
                    try CapabilityFixture.tone($0, frequency: 220 * Double(i + 1), seconds: 0.25 * Double(i + 1))
                }
            }
            return ["files": .array([meta] + songs)]
        }
        let support = scratch.appendingPathComponent("support-batch")
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        defer { api.stop() }
        let port = try #require(api.port)
        let session = SoundSession(
            models: [SoundFixture.model(row)], selected: id, client: { ArsenalClient(port: port) }, inbox: support.appendingPathComponent("inbox"),
            outputDirectory: scratch.appendingPathComponent("Music-batch"), task: .music)
        defer { session.close() }
        session.prepare()
        try await waitUntil { !session.loading }
        session.set("prompt", .text("Three short tunes"))
        session.set("batch_size", .text("3"))
        session.start()
        try await waitUntil(seconds: 30) { !session.running }
        #expect(session.problem == nil, "\(session.problem?.message ?? "")")
        #expect(session.takes.count == 3)
        #expect(Set(session.takes.map(\.url)).count == 3 && session.takes.allSatisfy { $0.url.pathExtension == "wav" })
        #expect(session.takes.map { ($0.shape.seconds * 100).rounded() / 100 } == [0.25, 0.5, 0.75])
        #expect(session.shown == session.takes.last?.id)
        #expect(await service.runInputs.last?["batch_size"] == 3)
    }

    private func waitUntil(seconds: Double = 5, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            if Date() > deadline { Issue.record("timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }
}
