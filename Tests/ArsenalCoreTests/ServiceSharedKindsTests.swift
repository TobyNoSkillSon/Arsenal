// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func v2InstalledSharedKindSchemasLoadOfflineAndRejectTampering() async throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-v2-service/shared-kinds-\(UUID().uuidString)")
    let kindRoot = root.appendingPathComponent("kinds")
    let content: [String: Data] = [
        "voices.toml": Data("id=\"voices\"\ntitle=\"Voices\"\norder=0\ncolumns=[]\napi={}\nsuite=\"fixture\"\n".utf8),
        "voices/ops/inspect.input.json": Data(#"{"$schema":"https://json-schema.org/draft/2020-12/schema","type":"object","additionalProperties":false}"#.utf8)
    ]
    for (path, data) in content {
        let target = kindRoot.appendingPathComponent(path); try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
        try data.write(to: target)
    }
    let inventory = content.map { CatalogFile(path: $0.key, sha256: FileDigest.sha256($0.value), bytes: Int64($0.value.count)) }
    let op: TOMLValue = .table([
        "name": .string("test_config"), "title": .string("Inspect"), "input": .string("kinds/voices/ops/inspect.input.json"), "output": .string("json"), "stream": .bool(false)
    ])
    let extra: [String: TOMLValue] = ["schema": .integer(2), "selection": .table(["default": .string("native")]), "op": .array([op])]
    let f = try await makeLocalCatalogFixture(
        kinds: ["voices"], extraTable: extra, kindSpecs: KindSpec.load(directory: kindRoot), kindsDirectory: kindRoot, kindFiles: inventory, kindContents: content)
    defer { f.stop() }
    try await f.service.get(id: "fixture-voices", precision: "bf16")
    try await f.service.load(id: "fixture-voices")
    #expect(try await f.service.run(id: "fixture-voices", op: "test_config")["config"] != nil)
    await f.service.shutdown(); f.stop() // No network server remains; the next load must be entirely local.
    let reopened = try await f.reopen(threshold: 8_000_000_000)
    let cachedKinds = f.paths.directory("catalog").appendingPathComponent("kinds/" + f.revision)
    #expect(FileManager.default.fileExists(atPath: cachedKinds.appendingPathComponent("voices/ops/inspect.input.json").path))
    try await reopened.load(id: "fixture-voices")
    #expect(await reopened.operations(id: "fixture-voices").map(\.name) == ["test_config"])
    try await reopened.unload(id: "fixture-voices")
    try Data("{}".utf8).write(to: cachedKinds.appendingPathComponent("voices/ops/inspect.input.json"))
    await #expect(throws: (any Error).self) { try await reopened.load(id: "fixture-voices") }
    #expect(await reopened.operations(id: "fixture-voices").isEmpty)
    #expect(await reopened.memory().models.isEmpty)
    // A batch of admissions verifies the kinds directory once for its own task (A7): a caller outside the batch, even
    // one that runs while the batch is in flight, verifies the directory itself and sees the tampering.
    await #expect(throws: (any Error).self) { try await reopened.resolvedOperations(id: "fixture-voices") }
    await KindsBatch.$verified.withValue(cachedKinds) {
        #expect(KindsBatch.verified == cachedKinds)
        let outside = Task.detached { try await reopened.resolvedOperations(id: "fixture-voices") }
        await #expect(throws: (any Error).self) { try await outside.value }
    }
    #expect(KindsBatch.verified == nil)
    #expect(await reopened.admissions([AdmissionQuery(id: "fixture-voices", op: "test_config")]).first?.refusal != nil)
    await reopened.shutdown()
}
