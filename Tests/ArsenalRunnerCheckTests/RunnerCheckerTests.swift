// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalWire
import CryptoKit
@testable import ArsenalCommands

private var repository: URL { URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent() }
private func scratch() throws -> URL {
    let url = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-runners/tests/\(UUID().uuidString)")
    try FileManager.default.createDirectory(at: url.appendingPathComponent("jobs"), withIntermediateDirectories: true)
    return url
}
private func compile(_ args: [String], output: URL) throws {
    let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/env"); process.arguments = args
    process.currentDirectoryURL = repository
    var environment = ProcessInfo.processInfo.environment; environment["DEVELOPER_DIR"] = "/Library/Developer/CommandLineTools"; process.environment = environment
    let done = DispatchSemaphore(value: 0); process.terminationHandler = { _ in done.signal() }
    try process.run()
    guard done.wait(timeout: .now() + 45) == .success else { process.terminate(); throw CheckFailure("template compiler timed out") }
    process.waitUntilExit()
    #expect(process.terminationStatus == 0)
    #expect(FileManager.default.fileExists(atPath: output.path))
}
@Suite(.serialized) struct RunnerCheckerTests {
    @Test(arguments: ["swift", "c"])
    func templatesPassBothModes(language: String) throws {
        let support = try scratch()
        let binary = support.appendingPathComponent("toy")
        let command: [String]
        switch language {
        case "swift":
            try compile(["swiftc", "-swift-version", "6", "Tests/Fixtures/Templates/runners/swift/main.swift", "-o", binary.path], output: binary); command = [binary.path]
        default:
            try compile(["clang", "-std=c11", "-O2", "-Wall", "-Wextra", "-Werror", "-pthread", "Tests/Fixtures/Templates/runners/c/main.c", "-o", binary.path], output: binary)
            command = [binary.path]
        }
        let model = try CheckModel(folder: repository.appendingPathComponent("Tests/Fixtures/Templates/runners/toy"), kindDir: nil, support: support)
        for fast in [false, true] {
            let checks = RunnerChecks(command: command, model: model, support: support, weights: model.folder, timeout: 5, fast: fast)
            try checks.run()
            #expect(checks.failed == 0)
            #expect(checks.passed == 21)
        }
        let session = try ProcessSession(command: command, support: support, timeout: 5, label: "arithmetic")
        defer { session.stop() }
        try session.hello(kind: "toy", ops: ["evaluate"])
        try session.send(["id": 1, "op": "load", "config": ["settings": ["scale": 2]]])
        _ = try session.response(id: 1)
        try session.send(["id": 2, "op": "evaluate", "values": [1, 2, 3]])
        #expect(try session.response(id: 2).0["result"]?.number == 12)
        try session.send(["id": 3, "op": "unload"]); _ = try session.response(id: 3)
        #expect(try session.exitStatus() == 0)
    }
    @Test func schemaValidationFailsClosedAndReferencesWork() throws {
        let schema: JSONObject = ["type": "object", "required": ["n"], "properties": ["n": ["$ref": "#/$defs/positive"]], "$defs": ["positive": ["type": "integer", "minimum": 1]]]
        try CheckSchema.validate(["n": 2], schema: schema)
        #expect(throws: (any Error).self) { try CheckSchema.validate(["n": false], schema: schema) }
        #expect(throws: (any Error).self) { try CheckSchema.validate(["n": 0], schema: schema) }
        #expect(throws: (any Error).self) { try CheckSchema.validate([:], schema: ["anyOf": [["type": "object"], ["unevaluatedProperties": false]]]) }
        #expect(throws: (any Error).self) { _ = try CheckSchema.synthesize(["type": "string", "pattern": "^model-.*$"]) }
        let values = try CheckSchema.synthesize(schema)
        try CheckSchema.validate(values, schema: schema)
        #expect(values["n"]?.number == 1)
    }
    @Test func verifiedFilesAndStreamOrdering() throws {
        let support = try scratch(); let jobs = support.appendingPathComponent("jobs")
        let file = jobs.appendingPathComponent("audio.pcm"); let bytes = Data([0, 0, 1, 0]); try bytes.write(to: file)
        let hash = SHA256.hash(data: bytes).map { String(format: "%02x", $0) }.joined()
        let value: JSONValue = ["path": .string(file.path), "bytes": 4, "sha256": .string(hash)]
        #expect(try validateFiles(value, inside: jobs) == 1)
        #expect(throws: (any Error).self) { _ = try validateFiles(["path": .string(file.path), "bytes": 5, "sha256": .string(hash)], inside: jobs) }
        let op = OpCheck(name: "speak", schema: [:], output: "audio", stream: true, params: [:], outputSchema: nil)
        #expect(throws: (any Error).self) { try validateOutput(op, reply: ["id": 1, "ok": true], events: [.binary(stream: 1, bytes: bytes)], root: jobs) }
        try validateOutput(
            op, reply: ["id": 1, "ok": true],
            events: [.json(["event": "audio_begin", "stream": 1, "sample_rate": 16000, "channels": 1, "encoding": "pcm_s16le"]), .binary(stream: 1, bytes: bytes)], root: jobs)
        let escape = support.appendingPathComponent("escape.pcm"); try bytes.write(to: escape)
        #expect(throws: (any Error).self) { _ = try validateFiles(["path": .string(escape.path), "bytes": 4, "sha256": .string(hash)], inside: jobs) }
    }
    @Test func silentAndPartialFrameDeadlines() throws {
        let support = try scratch()
        for code in ["import time; time.sleep(20)", "import sys,time; sys.stdout.buffer.write(bytes([0,0,0,3,1])); sys.stdout.buffer.flush(); time.sleep(20)"] {
            let start = Date()
            let session = try ProcessSession(command: ["python3", "-c", code], support: support, timeout: 0.1, label: "timeout")
            #expect(throws: (any Error).self) { _ = try session.read() }
            session.stop()
            #expect(!session.process.isRunning)
            #expect(Date().timeIntervalSince(start) < 3)
        }
    }
    @Test func negativeMemoryAndMultilineErrorsFail() throws {
        let support = try scratch()
        let session = try ProcessSession(command: ["python3", "-c", "import time; time.sleep(20)"], support: support, timeout: 0.1, label: "memory")
        defer { session.stop() }
        #expect(throws: (any Error).self) { try session.observe(["memory": ["footprint_mb": -1, "mlx_active_mb": 0, "mlx_cache_mb": 0]]) }
        #expect(throws: (any Error).self) { try ProcessSession.error(["error": ["code": "bad", "message": "line\nbreak"]]) }
    }
    @Test func writeDeadlineAndWrongReplyID() throws {
        let support = try scratch()
        let silent = try ProcessSession(command: ["python3", "-c", "import time; time.sleep(20)"], support: support, timeout: 0.1, label: "write-timeout")
        defer { silent.stop() }
        #expect(throws: (any Error).self) { try silent.raw(Data(repeating: 1, count: 1024 * 1024)) }
        let code = "import sys,struct,time; b=b'\\x01{\"id\":999,\"ok\":true}'; sys.stdout.buffer.write(struct.pack('>I',len(b))+b); sys.stdout.buffer.flush(); time.sleep(20)"
        let bad = try ProcessSession(command: ["python3", "-c", code], support: support, timeout: 0.1, label: "wrong-id")
        defer { bad.stop() }
        #expect(throws: (any Error).self) { _ = try bad.response(id: 1) }
    }

    @Test func flatSamplesAndAuthoritativeCapabilities() throws {
        let support = try scratch()
        let folder = support.appendingPathComponent("toy")
        let kinds = support.appendingPathComponent("kinds")
        for path in [folder.appendingPathComponent("ops"), folder.appendingPathComponent("gate"), kinds.appendingPathComponent("toy/ops")] {
            try FileManager.default.createDirectory(at: path, withIntermediateDirectories: true)
        }
        let schema = try Data(contentsOf: repository.appendingPathComponent("Tests/Fixtures/Templates/runners/toy/ops/evaluate.input.json"))
        try schema.write(to: folder.appendingPathComponent("ops/evaluate.input.json"))
        try schema.write(to: kinds.appendingPathComponent("toy/ops/evaluate.input.json"))
        let kind = """
            [[op]]
            name = "optional"
            input = "ops/evaluate.input.json"
            output = "json"
            stream = false
            [[op]]
            name = "evaluate"
            default = true
            input = "ops/evaluate.input.json"
            output = "json"
            stream = false
            """
        try kind.write(to: kinds.appendingPathComponent("toy.toml"), atomically: true, encoding: .utf8)
        let descriptor = """
            kind = "toy"
            [[op]]
            name = "evaluate"
            input = "kinds/toy/ops/evaluate.input.json"
            output = "json"
            stream = false
            """
        try descriptor.write(to: folder.appendingPathComponent("model.toml"), atomically: true, encoding: .utf8)
        try WireJSON.encode(["evaluate": ["values": [7]]]).write(to: folder.appendingPathComponent("gate/runner-check.json"))
        let model = try CheckModel(folder: folder, kindDir: kinds, support: support)
        #expect(model.ops.map(\.name) == ["evaluate"])
        #expect(model.ops.first?.params["values"] == .array([7]))
        try "kind = \"toy\"".write(to: folder.appendingPathComponent("model.toml"), atomically: true, encoding: .utf8)
        #expect(try CheckModel(folder: folder, kindDir: kinds, support: support).ops.map(\.name) == ["evaluate"])
    }

}

@Test func checkerExplicitEmptyOperationsDoNotInheritKindDefaults() throws {
    let support = try scratch(), model = support.appendingPathComponent("model"), kinds = support.appendingPathComponent("kinds")
    try FileManager.default.createDirectory(at: model, withIntermediateDirectories: true)
    try FileManager.default.createDirectory(at: kinds.appendingPathComponent("toy/ops"), withIntermediateDirectories: true)
    try "[[op]]\nname = \"evaluate\"\ntitle = \"Evaluate\"\ninput = \"ops/evaluate.json\"\noutput = \"json\"\nstream = false\ndefault = true\n".write(
        to: kinds.appendingPathComponent("toy.toml"), atomically: true, encoding: .utf8)
    try Data(#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object"}"#.utf8).write(to: kinds.appendingPathComponent("toy/ops/evaluate.json"))
    let manifest = model.appendingPathComponent("model.toml")
    try "kind = \"toy\"\n".write(to: manifest, atomically: true, encoding: .utf8)
    #expect(try CheckModel(folder: model, kindDir: kinds, support: support).ops.map(\.name) == ["evaluate"])
    try "op = []\nkind = \"toy\"\n".write(to: manifest, atomically: true, encoding: .utf8)
    #expect(throws: CheckFailure.self) { _ = try CheckModel(folder: model, kindDir: kinds, support: support) }
}
