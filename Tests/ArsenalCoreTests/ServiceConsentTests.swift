// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

@Test func v2StubConsentRefusesGetUntilCorrectDigestApproved() async throws {
    let fixture = StubModelsService.localFixture, service = StubModelsService(rows: [fixture])
    #expect(await catalogRow(service, fixture.id)?.codeConsentRequired == true)
    do { try await service.get(id: fixture.id, precision: "bf16"); Issue.record("Unapproved stub loaded") } catch let e as WorkerFailure { #expect(e.code == "code_consent") }
    await #expect(throws: WorkerFailure.self) { try await service.approveLocalCode(id: fixture.id, contentDigest: "wrong") }
    try await service.approveLocalCode(id: fixture.id, contentDigest: fixture.contentDigest!)
    try await service.get(id: fixture.id, precision: "bf16")
    #expect(await service.rows().first?.codeConsentRequired == false)
}

@Test func v2RowsDecodeV1AndEncodeConsentAliases() throws {
    let legacy = Data(#"{"id":"old","name":"Old","kind":"voices","states":{"bf16":{"state":"ready"}}}"#.utf8)
    let row = try JSONDecoder().decode(ModelRow.self, from: legacy)
    #expect(!row.needsReload && !row.codeConsentRequired && row.contentDigest == nil)
    let encoded = try WireJSON.decode(JSONEncoder().encode(StubModelsService.localFixture))
    #expect(encoded["codeConsentRequired"] == true && encoded["needsCodeConsent"] == true)
    #expect(encoded["contentDigest"]?.string?.count == 64 && encoded["fastAvailable"] == true)
}

@Test func reviewFixApprovalProvenanceAndAtomicSettings() async throws {
    let row = StubModelsService.localFixture
    let declarations: JSONObject = [
        "setting": [["id": "speed", "title": "Speed", "type": "number", "default": 1, "min": 0.5, "max": 2, "applies": "request", "maps_to": "request.speed"]]
    ]
    let service = StubModelsService(rows: [row], metadata: [row.id: declarations])
    try await service.approveLocalCode(id: row.id, contentDigest: row.contentDigest!, approvedBy: .api)
    let approved = try #require(await catalogRow(service, row.id)?.approval)
    #expect(approved.by == .api && approved.digest == row.contentDigest)
    #expect(ISO8601DateFormatter().date(from: approved.at) != nil)
    await #expect(throws: WorkerFailure.self) { try await service.setSettings(id: row.id, values: ["speed": 1.5, "unknown": true]) }
    #expect(try await service.settings(id: row.id).first?.value == 1)
    try await service.setSettings(id: row.id, values: ["speed": 1.5])
    #expect(try await service.settings(id: row.id).first?.value == 1.5)
}
@Test func reviewFixDeleteSavedVoiceKeepsPresetAndForeignReferences() throws {
    let root = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-a/voices-\(UUID().uuidString)")
    let dir = root.appendingPathComponent("voices/test"), id = "voice-" + UUID().uuidString.lowercased()
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    let wav = dir.appendingPathComponent(id + ".wav"), json = dir.appendingPathComponent(id + ".json")
    try Data([1, 2]).write(to: wav)
    try WireJSON.encode(["id": .string(id), "reference": .string(wav.path)]).write(to: json)
    #expect(throws: WorkerFailure.self) { try SavedVoiceStore.delete(model: "test", voiceID: "af_heart", support: root) }
    try SavedVoiceStore.delete(model: "test", voiceID: id, support: root)
    #expect(!FileManager.default.fileExists(atPath: json.path) && !FileManager.default.fileExists(atPath: wav.path))
    let receipts = try FileManager.default.contentsOfDirectory(at: root.appendingPathComponent("logs/cleanup-receipts"), includingPropertiesForKeys: nil)
    let record = try WireJSON.decode(Data(contentsOf: receipts[0]))
    #expect(record["files"]?.array?.count == 2 && record["verified"] == true)
}
