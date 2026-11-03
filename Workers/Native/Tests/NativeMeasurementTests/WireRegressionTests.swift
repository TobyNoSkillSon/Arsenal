// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
import ArsenalWire

@Suite(.serialized) struct WireRegressionTests {
    private func binary() throws -> URL {
        // Workers/: the one native graph. scripts/test.sh lays out its dev helpers (worker copy, metallib) there.
        let package = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .deletingLastPathComponent()
        let file =
            ProcessInfo.processInfo.environment["ARSENAL_TEST_NATIVE_WORKER"].map { URL(fileURLWithPath: $0) }
            ?? package.appendingPathComponent(".build/helpers/arsenal-worker-native")
        guard FileManager.default.isExecutableFile(atPath: file.path) else { throw ManifestError.invalid("Native test executable missing: " + file.path) }
        return file
    }
    @Test(.heavy) func normalFamilyHandshakesUseOrdinaryDispatchAndChatAddsOnlyLogits() async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-wire-stats-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        // Decisions initializes a tiny MLX array before hello; symlinks need the same metallib as the real helper.
        let testBinary = root.appendingPathComponent("arsenal-worker-native")
        try FileManager.default.copyItem(at: binary(), to: testBinary)
        let helper = try binary().deletingLastPathComponent()
        let metal = [helper.appendingPathComponent("mlx-swift_Cmlx.bundle/Contents/Resources/default.metallib"), helper.appendingPathComponent("mlx.metallib")]
        let library = try #require(metal.first { FileManager.default.fileExists(atPath: $0.path) })
        try FileManager.default.copyItem(at: library, to: root.appendingPathComponent("mlx.metallib"))
        let expected: [String: Set<String>] = [
            "chat": ["load", "chat", "count_tokens", "status", "cancel", "unload", "convert", "ping", "measurement_logits"],
            "decisions": ["load", "status", "unload", "ping", "cancel", "convert", "decide", "judge", "measurement_components", "measurement_source"],
            "d1": ["cancel", "measurement_components", "convert", "decide", "judge", "load", "ping", "status", "unload"],
            "flux": ["cancel", "convert", "generate", "measurement_components"],
            "voices": ["speak", "voices", "design_voice", "clone_voice", "convert", "measurement_components"],
            "sound": ["generate_sound", "generate_music", "enhance", "convert", "measurement_components", "measurement_source"]
        ]
        for (family, operations) in expected.sorted(by: { $0.key < $1.key }) {
            let link = root.appendingPathComponent("arsenal-worker-" + family)
            try FileManager.default.createSymbolicLink(at: link, withDestinationURL: testBinary)
            let worker = WorkerProcess()
            do {
                try await worker.start(
                    WorkerLaunch(executable: link, kind: family == "d1" ? "decisions" : family == "flux" ? "images" : family, log: root.appendingPathComponent(family + ".log")))
                let hello = try #require(await worker.hello)
                #expect(Set(hello.ops) == operations && hello.proto == 1 && hello.runner == "native")
                #expect(!hello.ops.contains("quality") && !hello.ops.contains("prepare"))
            } catch {
                await worker.unload()
                let log = (try? String(contentsOf: root.appendingPathComponent(family + ".log"), encoding: .utf8)) ?? ""
                throw ManifestError.invalid(family + " handshake: " + error.localizedDescription + " / " + log)
            }
            await worker.unload()
        }
        let worker = WorkerProcess()
        let chat = root.appendingPathComponent("arsenal-worker-chat")
        do {
            try await worker.start(WorkerLaunch(executable: chat, kind: "chat", log: root.appendingPathComponent("logits.log")))
            try await worker.verifyOperations(["load", "chat", "measurement_logits", "convert"])
            // Added capability refuses an unloaded model; it cannot load another model behind the supervisor.
            await #expect(throws: (any Error).self) { try await worker.request(op: "measurement_logits", timeout: 5) }
        } catch { await worker.unload(); throw error }
        await worker.unload()
    }
}
