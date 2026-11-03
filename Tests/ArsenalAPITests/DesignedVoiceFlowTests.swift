// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalAPI
@testable import ArsenalCore

@Test(arguments: [UInt32(12345), UInt32.max]) func elevenlabsPreviewSaveListAndExpiredPreview404(seed: UInt32) async throws {
    let support = URL(fileURLWithPath: "/tmp/arsenal-mb-v2-fix-f/adapter-\(UUID().uuidString)")
    let row = ModelRow(id: "designer", name: "Designer", kind: "voices", states: ["bf16": .init(state: .ready)])
    let ops = ["voices", "design_voice"].map {
        ModelOperation(
            name: $0, title: $0, inputSchema: ["type": "object", "properties": ["seed": ["type": "integer", "minimum": 0, "maximum": 4294967295]]], output: "json", stream: false)
    }
    let service = StubModelsService(rows: [row], operations: [row.id: ops], supportDirectory: support)
    let server = ArsenalAPIServer(service: service, supportDirectory: support, port: 0)
    let port = try await server.start(); defer { server.stop() }
    func call(_ path: String, body: JSONObject? = nil) async throws -> (JSONObject, Int) {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/providers/elevenlabs" + path)!)
        if let body { request.httpMethod = "POST"; request.httpBody = try WireJSON.encode(body); request.setValue("application/json", forHTTPHeaderField: "Content-Type") }
        let (data, response) = try await URLSession.shared.data(for: request)
        return (try WireJSON.decode(data), (response as? HTTPURLResponse)?.statusCode ?? 0)
    }
    let (previews, previewStatus) = try await call(
        "/v1/text-to-voice/create-previews", body: ["voice_description": "Warm narrator", "model_id": "designer", "seed": .number(Double(seed))])
    #expect(previewStatus == 200)
    let generated = try #require(previews["previews"]?.array?[1]["generated_voice_id"]?.string)
    let chosenAudio = try #require(previews["previews"]?.array?[1]["audio_base_64"]?.string)
    let records = try FileManager.default.contentsOfDirectory(at: support.appendingPathComponent("jobs"), includingPropertiesForKeys: nil).map {
        try WireJSON.decode(Data(contentsOf: $0.appendingPathComponent(".design-preview.json")))
    }
    #expect(records.count == 3)
    #expect(Set(records.compactMap { $0["design"]?["seed"]?.number }) == Set((0..<3).map { Double(seed &+ UInt32($0)) }))
    for invalid: JSONValue in [-1, 4294967296, 1.5, .null, "not a seed"] {
        let (_, status) = try await call("/v1/text-to-voice/create-previews", body: ["voice_description": "Warm narrator", "model_id": "designer", "seed": invalid])
        #expect(status == 422)
    }
    #expect(try FileManager.default.contentsOfDirectory(at: support.appendingPathComponent("jobs"), includingPropertiesForKeys: nil).count == 3)
    #expect(try SavedVoiceStore.list(model: row.id, support: support).isEmpty)
    let (saved, savedStatus) = try await call("/v1/text-to-voice", body: ["generated_voice_id": .string(generated), "voice_name": "Narrator"])
    #expect(savedStatus == 200, "\(saved)")
    let voice = try #require(saved["voice_id"]?.string)
    #expect(voice.hasPrefix("voice-"))
    let manifest = try #require(try SavedVoiceStore.list(model: row.id, support: support).first?.object)
    let savedAudio = try #require(manifest["preview"]?.string)
    #expect(try Data(contentsOf: URL(fileURLWithPath: savedAudio)).base64EncodedString() == chosenAudio)
    #expect(records.contains { $0["design"]?["seed"] == manifest["seed"] })
    #expect(manifest["modelRevision"] == "stub" && manifest["previewText"]?.string?.isEmpty == false)
    #expect(manifest["seed"] == .number(Double(seed &+ 1)))
    let (listed, listStatus) = try await call("/v2/voices")
    #expect(listStatus == 200)
    #expect(listed["voices"]?.array?.contains { $0["voice_id"]?.string == voice && $0["name"] == "Narrator" } == true)
    let expired = try #require(previews["previews"]?.array?.last?["generated_voice_id"]?.string)
    for directory in try FileManager.default.contentsOfDirectory(at: support.appendingPathComponent("jobs"), includingPropertiesForKeys: nil) {
        let file = directory.appendingPathComponent(".design-preview.json")
        if let data = try? Data(contentsOf: file) { var record = try WireJSON.decode(data); record["created"] = 0; try WireJSON.encode(record).write(to: file) }
    }
    let (error, expiredStatus) = try await call("/v1/text-to-voice", body: ["generated_voice_id": .string(expired), "voice_name": "Expired"])
    #expect(expiredStatus == 404)
    #expect(String(decoding: try WireJSON.encode(error), as: UTF8.self).contains("expired"))
    #expect(try SavedVoiceStore.list(model: row.id, support: support).count == 1)
}
