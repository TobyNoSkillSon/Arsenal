// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

struct SavedVoice: Codable {
    let id: String
    let name: String
    let description: String?
    let reference: String?
    let transcript: String?
    let language: String?
    var seed: UInt32?
    var modelRevision: String?
    var preview: String?
    var previewText: String?
    var dtype: String?
    var fast: Bool?
    var generation: JSONObject?

    var json: JSONValue {
        [
            "id": .string(id), "name": .string(name), "language": .string(language ?? "auto"),
            "description": .string(description ?? "Saved reference voice"), "saved": true
        ]
    }
}

/// Reference clips are copied into Arsenal; deleting the caller's original cannot change a saved voice.
struct SavedVoices {
    let directory: URL
    init(model: String, support: URL? = nil) {
        let root =
            support ?? ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"].map { URL(fileURLWithPath: $0) }
            ?? FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Arsenal")
        directory = root.appendingPathComponent("voices/" + model)
    }
    func list() throws -> [SavedVoice] {
        guard FileManager.default.fileExists(atPath: directory.path) else { return [] }
        return try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "json" }.sorted { $0.path < $1.path }
            .map { try JSONDecoder().decode(SavedVoice.self, from: Data(contentsOf: $0)) }
    }
    func save(name: String, description: String? = nil, audio: URL? = nil, transcript: String? = nil, language: String? = nil, design: JSONObject? = nil, preview: URL? = nil)
        throws -> SavedVoice
    {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let id = "voice-" + UUID().uuidString.lowercased()
        let owned = directory.appendingPathComponent(id + ".wav")
        if let source = audio ?? preview { try FileManager.default.copyItem(at: source.resolvingSymlinksInPath(), to: owned) }
        var voice = SavedVoice(id: id, name: name, description: description, reference: audio == nil ? nil : owned.path, transcript: transcript, language: language)
        if let design {
            voice.seed = design["seed"]?.number.map { UInt32($0) }; voice.modelRevision = design["model_revision"]?.string
            voice.previewText = design["preview_text"]?.string; voice.dtype = design["dtype"]?.string; voice.fast = design["fast"]?.bool
            voice.preview = preview == nil ? nil : owned.path
            voice.generation = design["generation"]?.object?.filter { $0.key != "max_tokens" }
        }
        try JSONEncoder().encode(voice).write(to: directory.appendingPathComponent(id + ".json"), options: .atomic)
        return voice
    }
    /// A design is a disposable preview unless the caller explicitly supplies a durable name.
    func designResult(request: JSONObject, preview: JSONObject, design: JSONObject) throws -> JSONObject {
        guard let seed = design["seed"]?.number, seed.isFinite, seed.rounded() == seed, (0...Double(UInt32.max)).contains(seed),
            design["model_revision"]?.string?.isEmpty == false, design["preview_text"]?.string?.isEmpty == false,
            let path = preview["path"]?.string
        else { throw WorkerFailure(code: "integrity", message: "Voice design returned no reproducible preview") }
        var result: JSONObject = ["voice_id": .string("preview-" + UUID().uuidString.lowercased()), "preview": .object(preview), "design": .object(design), "saved": false]
        if let name = request["name"]?.string?.trimmingCharacters(in: .whitespacesAndNewlines), !name.isEmpty {
            let saved = try save(name: name, description: request["description"]?.string, language: design["language"]?.string, design: design, preview: URL(fileURLWithPath: path))
            result["voice_id"] = .string(saved.id); result["saved"] = true
        }
        return result
    }
    func resolve(_ request: JSONObject, modelRevision: String? = nil) throws -> JSONObject {
        guard let id = request["voice"]?.string, id.hasPrefix("voice-") else { return request }
        guard let saved = try list().first(where: { $0.id == id }) else { throw WorkerFailure(code: "unknown_voice", message: "Saved voice does not exist for this model") }
        var request = request
        request.removeValue(forKey: "voice")
        if let revision = saved.modelRevision, let modelRevision, revision != modelRevision {
            throw WorkerFailure(code: "invalid_request", message: "Saved voice belongs to a different model revision; design it again on this version")
        }
        // The seed is part of this voice's identity, not a new random design for each utterance.
        if let seed = saved.seed { request["seed"] = .number(Double(seed)) }
        for (key, value) in saved.generation ?? [:] where key != "max_tokens" { request[key] = value }
        if let description = saved.description { request["instructions"] = .string(description) }
        if let reference = saved.reference {
            var ref: JSONObject = ["path": .string(reference)]
            if let transcript = saved.transcript { ref["text"] = .string(transcript) }
            request["reference"] = .object(ref)
        }
        if request["language"] == nil, let language = saved.language { request["language"] = .string(language) }
        return request
    }
}
