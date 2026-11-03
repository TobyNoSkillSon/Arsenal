// SPDX-License-Identifier: MIT
import Foundation

/// Only explicit saved-voice UUIDs are deletable. Presets and caller-owned reference clips are never touched.
enum SavedVoiceStore {
    static func saveDesigned(model: String, name: String, description: String, design: JSONObject, preview: URL, support: URL) throws -> JSONObject {
        guard let seed = design["seed"]?.number, seed.isFinite, seed.rounded() == seed, (0...Double(UInt32.max)).contains(seed),
            let revision = design["model_revision"]?.string, !revision.isEmpty,
            let text = design["preview_text"]?.string, !text.isEmpty
        else {
            throw WorkerFailure(code: "integrity", message: "This preview has no reproducible voice identity; design a voice again.")
        }
        let directory = try SafePath.resolve("voices/" + model, inside: support)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let id = "voice-" + UUID().uuidString.lowercased(), audio = directory.appendingPathComponent(id + ".wav")
        try FileManager.default.copyItem(at: preview, to: audio)
        // Seed + description reproduce the chosen design; the retained clip also survives transient-job cleanup.
        var voice: JSONObject = [
            "id": .string(id), "name": .string(name), "description": .string(description), "preview": .string(audio.path),
            "seed": .number(seed), "modelRevision": .string(revision), "previewText": .string(text)
        ]
        voice["language"] = design["language"]
        voice["generation"] = design["generation"]
        voice["dtype"] = design["dtype"]; voice["fast"] = design["fast"]
        try WireJSON.encode(voice).write(to: directory.appendingPathComponent(id + ".json"), options: .atomic)
        return ["voice_id": .string(id)]
    }
    static func list(model: String, support: URL) throws -> [JSONValue] {
        let directory = try SafePath.resolve("voices/" + model, inside: support)
        return try ((try? FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)) ?? []).filter { $0.pathExtension == "json" }.sorted {
            $0.path < $1.path
        }.map { file in
            var object = try WireJSON.decode(Data(contentsOf: SafePath.resolve(file.lastPathComponent, inside: directory))); object["saved"] = true; return .object(object)
        }
    }
    static func delete(model: String, voiceID: String, support: URL) throws {
        guard voiceID.hasPrefix("voice-"), UUID(uuidString: String(voiceID.dropFirst(6))) != nil else {
            throw WorkerFailure(code: "invalid_request", message: "Preset voices cannot be deleted.")
        }
        let directory = try SafePath.resolve("voices/" + model, inside: support)
        let manifest = try SafePath.resolve(voiceID + ".json", inside: directory)
        guard FileManager.default.fileExists(atPath: manifest.path) else { throw WorkerFailure(code: "unknown_voice", message: "Saved voice does not exist for this model.") }
        let object = try WireJSON.decode(Data(contentsOf: manifest))
        guard object["id"]?.string == voiceID else { throw WorkerFailure(code: "integrity", message: "Saved voice identity is invalid.") }
        var files = [manifest]
        if let reference = object["reference"]?.string {
            let expected = try SafePath.resolve(voiceID + ".wav", inside: directory)
            guard reference == expected.path else { throw WorkerFailure(code: "integrity", message: "Saved voice reference is outside its owned files.") }
            if FileManager.default.fileExists(atPath: expected.path) { files.append(expected) }
        }
        let audio = try SafePath.resolve(voiceID + ".wav", inside: directory)
        if FileManager.default.fileExists(atPath: audio.path), !files.contains(audio) { files.append(audio) }
        try OwnedCacheCleanup.remove(files, support: support, operation: "delete_saved_voice")
    }
}
