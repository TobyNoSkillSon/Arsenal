// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct Round2FixETests {
    let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    @Test func largeRequestsRemainWithinBodyLimit() throws {
        let chat = try WireJSON.decode(Data(contentsOf: repo.appendingPathComponent("models/qwen3.8-27b/ops/chat.input.json")))
        let messages: [JSONValue] = (0..<2000).map { i in ["role": .string(i % 2 == 0 ? "user" : "assistant"), "content": .string(String(repeating: "word ", count: 20))] }
        let request: JSONObject = ["messages": .array(messages)]
        #expect(try WireJSON.encode(request).count < 1_048_576)
        try OperationSchemaValidator.validate(.object(request), schema: .object(chat))
        let judge = try WireJSON.decode(Data(contentsOf: repo.appendingPathComponent("Resources/kinds/decisions/ops/judge.input.json")))
        let items: [JSONValue] = (0..<5000).map { i in ["title": .string("Job \(i)"), "company": "Acme", "location": "Remote", "salary": 100000] }
        let batch: JSONObject = ["items": .array(items), "questions": ["relevant": ["instructions": "Is this about Swift?", "type": "noul"]]]
        #expect(try WireJSON.encode(batch).count < 1_048_576)
        try OperationSchemaValidator.validate(.object(batch), schema: .object(judge))
    }
    @Test func expiryIsBestEffortPerEntry() throws {
        let root = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-e/expiry-\(UUID().uuidString)")
        let bad = root.appendingPathComponent("jobs/" + UUID().uuidString), good = root.appendingPathComponent("jobs/" + UUID().uuidString)
        for dir in [bad, good] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 0)], ofItemAtPath: dir.path)
        }
        try FileManager.default.createSymbolicLink(at: bad.appendingPathComponent("out.wav"), withDestinationURL: URL(fileURLWithPath: "/etc/hosts"))
        try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: 0)], ofItemAtPath: bad.path)
        let fresh = root.appendingPathComponent("jobs/" + UUID().uuidString)
        try TransientModelJobs.prepare(fresh, support: root)
        #expect(FileManager.default.fileExists(atPath: fresh.path))
        #expect(!FileManager.default.fileExists(atPath: good.path))
        #expect(FileManager.default.fileExists(atPath: bad.path))
    }
    @Test func brokenLocalTableRowKeepsResidentAndRecovers() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let folder = f.paths.support.appendingPathComponent("local-models/local-voices")
        for (path, data) in f.source.files {
            let relative = String(path.dropFirst("models/fixture-voices/".count)), target = folder.appendingPathComponent(relative)
            try FileManager.default.createDirectory(at: target.deletingLastPathComponent(), withIntermediateDirectories: true)
            try (relative == "model.toml" ? Data(String(decoding: data, as: UTF8.self).replacingOccurrences(of: "fixture-voices", with: "local-voices").utf8) : data).write(
                to: target)
        }
        _ = await f.service.catalog(); try await f.service.add(id: "local-voices")
        try await f.service.get(id: "local-voices", precision: "bf16"); try await f.service.load(id: "local-voices")
        let original = try Data(contentsOf: folder.appendingPathComponent("model.toml"))
        var events = await f.service.events().makeAsyncIterator()
        _ = await events.next() // Initial catalog status.
        let notifications = Task {
            var count = 0
            while await events.next() != nil { count += 1 }
            return count
        }
        try Data("id = \"local-voices\"\nkind = ".utf8).write(to: folder.appendingPathComponent("model.toml"))
        _ = await f.service.catalog()
        notifications.cancel()
        #expect(await notifications.value == 0, "Discovery reads report the problem on the row without scheduling a refresh")
        let row = try #require(await f.service.rows().first { $0.id == "local-voices" })
        #expect(row.states[row.selectedPrecision]?.state == .failed)
        #expect(row.states[row.selectedPrecision]?.phase == "localFolderErrorLoaded")
        #expect(row.residentDType == "bf16")
        #expect(await f.service.catalog().first { $0.id == row.id }?.loaded == true)
        #expect(await f.service.supervisor.residentIDs().contains("local-voices"))
        #expect(try await f.service.run(id: "local-voices", op: "test_config")["config"] != nil)
        await #expect(throws: (any Error).self) { try await f.service.applySettings(id: "local-voices") }
        #expect(await f.service.supervisor.residentIDs().contains("local-voices"))
        try original.write(to: folder.appendingPathComponent("model.toml")); _ = await f.service.catalog()
        #expect(await f.service.rows().first { $0.id == "local-voices" }?.states["bf16"]?.state == .loaded)
        await f.service.shutdown()
    }
    @Test func musicHasItsOwnDefaultButSavedDurationWins() throws {
        let m = try ModelManifest.load(shippedOrRetiredFolder(repo, "stable-audio-3-medium"))
        var settings = ModelSettings()
        settings.declarations[m.id] = try WireJSON.decode(JSONEncoder().encode(m))["setting"]!.array!.compactMap(\.object)
        let schema = try WireJSON.decode(Data(contentsOf: shippedOrRetiredFolder(repo, "stable-audio-3-medium").appendingPathComponent("ops/generate_music.input.json")))
        #expect(settings.request(m.id, params: ["prompt": "piano"], schema: schema)["duration"] == 30)
        try settings.set(m.id, key: "duration", value: 12)
        #expect(settings.request(m.id, params: ["prompt": "piano"], schema: schema)["duration"] == 12)
        #expect(settings.request(m.id, params: ["prompt": "piano", "duration": 8], schema: schema)["duration"] == 8)
        let sfx = try WireJSON.decode(Data(contentsOf: shippedOrRetiredFolder(repo, "stable-audio-3-medium").appendingPathComponent("ops/generate_sound.input.json")))
        var defaults = settings; defaults.load([:])
        #expect(defaults.request(m.id, params: ["prompt": "rain"], schema: sfx)["duration"] == 5)
    }
    @Test func deletingDefaultVoiceResetsSettingAndWarns() async throws {
        let voice = "voice-" + UUID().uuidString.lowercased()
        var config = ArsenalConfiguration(); config.settings = ["fixture-voices": ["voice": .string(voice)]]
        let setting: TOMLValue = .table([
            "id": .string("voice"), "title": .string("Voice"), "type": .string("voice"), "default": .string("preset"), "applies": .string("request"),
            "maps_to": .string("request.voice")
        ])
        let f = try await makeLocalCatalogFixture(kinds: ["voices"], configuration: config, extraTable: ["setting": .array([setting])]); defer { f.stop() }
        let dir = f.paths.support.appendingPathComponent("voices/fixture-voices")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try WireJSON.encode(["id": .string(voice)]).write(to: dir.appendingPathComponent(voice + ".json"))
        try await f.service.deleteSavedVoice(id: "fixture-voices", voiceID: voice)
        #expect(try await f.service.settings(id: "fixture-voices").first?.value == "preset")
        #expect(await catalogRow(f.service, "fixture-voices")?.settingsWarning?.contains("was deleted") == true)
        #expect(try ArsenalConfiguration.read(f.paths.support.appendingPathComponent("config.json")).settings["fixture-voices"]?["voice"] == nil)
        await f.service.shutdown()
    }
    @Test func receiptsAndKindsHaveBoundedRetentionWithEvidence() throws {
        let support = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-e/retention-\(UUID().uuidString)")
        let receipts = support.appendingPathComponent("logs/cleanup-receipts")
        try FileManager.default.createDirectory(at: receipts, withIntermediateDirectories: true)
        for _ in 0..<6 { try Data("{}".utf8).write(to: receipts.appendingPathComponent(UUID().uuidString + ".json")) }
        try OwnedCacheCleanup.pruneReceipts(support: support, limit: 3)
        let after = try FileManager.default.contentsOfDirectory(at: receipts, includingPropertiesForKeys: nil)
        #expect(after.count == 3)
        let rotation = try #require(after.compactMap { try? WireJSON.decode(Data(contentsOf: $0)) }.first { $0["operation"] == "cleanup_receipt_retention" })
        #expect(rotation["verified"] == true && rotation["files"]?.array?.count == 4)
        let kinds = support.appendingPathComponent("catalog/kinds")
        for i in 0..<4 {
            let dir = kinds.appendingPathComponent(String(format: "%040x", i))
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data([1]).write(to: dir.appendingPathComponent("kind.toml"))
            try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: Double(i))], ofItemAtPath: dir.path)
        }
        try ModelFolderInstaller.pruneKinds(support: support, current: String(format: "%040x", 1))
        #expect(try Set(FileManager.default.contentsOfDirectory(atPath: kinds.path)) == Set([String(format: "%040x", 1), String(format: "%040x", 3)]))
        let records = try FileManager.default.contentsOfDirectory(at: receipts, includingPropertiesForKeys: nil).compactMap { try? WireJSON.decode(Data(contentsOf: $0)) }
        #expect(records.contains { $0["operation"] == "shared_kinds_retention" && $0["verified"] == true && $0["files"]?.array?.count == 2 })
        for _ in 0..<260 { try Data("{}".utf8).write(to: receipts.appendingPathComponent(UUID().uuidString + ".json")) }
        let owned = support.appendingPathComponent("old.txt"); try Data([1]).write(to: owned)
        try OwnedCacheCleanup.remove([owned], support: support, operation: "retention_test")
        #expect(try FileManager.default.contentsOfDirectory(atPath: receipts.path).count == 256)
    }
    @Test func brokenFolderSurvivesRestartInTable() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"], addToTable: false); defer { f.stop() }
        await f.service.shutdown()
        let dir = f.paths.support.appendingPathComponent("local-models/local-voices")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("kind = ".utf8).write(to: dir.appendingPathComponent("model.toml"))
        let file = f.paths.support.appendingPathComponent("config.json")
        var config = try ArsenalConfiguration.read(file); config.table = ["local-voices"]; try config.write(file)
        // A restart sees the saved membership and the already-existing broken folder together.
        let restarted = try await f.reopen(threshold: 8_000_000_000)
        #expect(await restarted.rows().map(\.id) == ["local-voices"])
        #expect(await restarted.catalog().first { $0.id == "local-voices" }?.added == true)
        try await restarted.remove(id: "local-voices")
        #expect(await restarted.rows().isEmpty)
        await restarted.shutdown()
    }
    @Test func aRejectedShadowFolderCannotBreakASignedTableRow() async throws {
        let f = try await makeLocalCatalogFixture(kinds: ["voices"]); defer { f.stop() }
        let dir = f.paths.support.appendingPathComponent("local-models/fixture-voices")
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        try Data("kind = ".utf8).write(to: dir.appendingPathComponent("model.toml"))
        let row = try #require(await catalogRow(f.service, "fixture-voices"))
        #expect(!row.local && row.states["bf16"]?.state != .failed)
        // Since 2 Nov a folder named like a signed model is superseded: ignored and kept, with no error row.
        #expect(await !f.service.catalog().contains { $0.id.hasPrefix("local-error-") })
        #expect(FileManager.default.fileExists(atPath: dir.appendingPathComponent("model.toml").path))
        await f.service.shutdown()
    }
    @Test func prepareKindsPrunesEvenWhenCurrentAlreadyVerified() async throws {
        let data = Data("id=\"voices\"\ntitle=\"Voices\"\norder=0\ncolumns=[]\napi={}\nsuite=\"fixture\"\n".utf8)
        let inventory = [CatalogFile(path: "voices.toml", sha256: FileDigest.sha256(data), bytes: Int64(data.count))]
        let f = try await makeLocalCatalogFixture(
            kinds: ["voices"], kindSpecs: [KindSpec(id: "voices", title: "Voices", order: 0, columns: [], suite: "fixture")], kindFiles: inventory,
            kindContents: ["voices.toml": data])
        defer { f.stop() }
        let root = f.paths.support.appendingPathComponent("catalog/kinds")
        for i in 0..<3 {
            let dir = root.appendingPathComponent(String(format: "%040x", i))
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data([1]).write(to: dir.appendingPathComponent("old.toml"))
            try FileManager.default.setAttributes([.modificationDate: Date(timeIntervalSince1970: Double(i))], ofItemAtPath: dir.path)
        }
        try await f.service.get(id: "fixture-voices", precision: "bf16")
        #expect(try Set(FileManager.default.contentsOfDirectory(atPath: root.path)) == Set([f.revision, String(format: "%040x", 2)]))
        await f.service.shutdown()
    }
}
