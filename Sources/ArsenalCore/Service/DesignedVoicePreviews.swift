// SPDX-License-Identifier: MIT
import Foundation

/// A preview token grants access only to service-recorded, still-existing output. It is not a saved voice.
enum DesignedVoicePreviews {
    static func record(model: String, request: JSONObject, result: JSONObject, support: URL) throws {
        guard result["saved"]?.bool != true, let id = result["voice_id"]?.string, id.hasPrefix("preview-"), UUID(uuidString: String(id.dropFirst(8))) != nil,
            let path = result["preview"]?["path"]?.string, let description = request["description"]?.string
        else { return }
        let file = URL(fileURLWithPath: path), directory = file.deletingLastPathComponent(), root = support.appendingPathComponent("jobs")
        guard directory.deletingLastPathComponent().standardizedFileURL == root.standardizedFileURL, UUID(uuidString: directory.lastPathComponent) != nil,
            try SafePath.resolve(file.lastPathComponent, inside: directory).path == path
        else { throw WorkerFailure(code: "integrity", message: "Voice preview is outside its Arsenal job.") }
        var record: JSONObject = [
            "id": .string(id), "model": .string(model), "path": .string(path), "description": .string(description), "created": .number(Date().timeIntervalSince1970)
        ]
        record["language"] = request["language"]
        record["design"] = result["design"]
        try WireJSON.encode(record).write(to: directory.appendingPathComponent(".design-preview.json"), options: .atomic)
    }
    static func save(model: String, previewID: String, name: String, support: URL, now: Date = Date()) throws -> JSONObject {
        guard !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw WorkerFailure(code: "invalid_request", message: "Name the saved voice.") }
        guard previewID.hasPrefix("preview-"), UUID(uuidString: String(previewID.dropFirst(8))) != nil else {
            throw WorkerFailure(code: "not_found", message: "Voice preview token is invalid.")
        }
        let root = support.appendingPathComponent("jobs")
        for directory in (try? FileManager.default.contentsOfDirectory(at: root, includingPropertiesForKeys: nil)) ?? [] where UUID(uuidString: directory.lastPathComponent) != nil
        {
            let file = try SafePath.resolve(".design-preview.json", inside: directory)
            guard let bytes = try? Data(contentsOf: file), let record = try? WireJSON.decode(bytes), record["id"]?.string == previewID, record["model"]?.string == model else {
                continue
            }
            guard let created = record["created"]?.number, now.timeIntervalSince1970 - created < 3600,
                let path = record["path"]?.string, FileManager.default.fileExists(atPath: path),
                let description = record["description"]?.string
            else { throw WorkerFailure(code: "not_found", message: "Voice preview expired or its audio was removed; design a voice again.") }
            let audio = try SafePath.resolve(URL(fileURLWithPath: path).lastPathComponent, inside: directory)
            guard audio.resolvingSymlinksInPath().path == URL(fileURLWithPath: path).resolvingSymlinksInPath().path else { throw missing() }
            return try SavedVoiceStore.saveDesigned(model: model, name: name, description: description, design: record["design"]?.object ?? [:], preview: audio, support: support)
        }
        throw missing()
    }
    private static func missing() -> WorkerFailure { WorkerFailure(code: "not_found", message: "Voice preview expired or was removed; design a voice again.") }
}
