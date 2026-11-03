// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalAPI

@main enum SavedDesignProof {
    static func main() async {
        var live: RealModelsService?
        var server: ArsenalAPIServer?
        do {
            guard ProcessInfo.processInfo.environment["VELLA_LOCK_HELD"] == "1" else { throw Failure("Use scripts/mlock gpulock with an alarm") }
            let args = CommandLine.arguments
            guard args.count == 5 else { throw Failure("Usage: saved-design-proof REPO HELPERS MODEL EVIDENCE") }
            let repo = URL(fileURLWithPath: args[1]), model = args[3], paths = ArsenalPaths()
            guard paths.support.path.contains("/Arsenal-dev/weights/voices/mb-v2-fix-f/") else { throw Failure("Use isolated persistent fix-f support") }
            var client = HuggingFaceClient(baseURL: URL(string: "http://127.0.0.1:9")!)
            client.maximumAttempts = 1; client.token = { nil }
            // Reuse the existing signed catalog and its pinned Git folders, never re-sign it for the probe.
            let service = try await RealModelsService.open(
                catalogData: Data(contentsOf: repo.appendingPathComponent("catalog/catalog.json")),
                signature: Data(contentsOf: repo.appendingPathComponent("catalog/catalog.json.sig")),
                publicKey: Data(contentsOf: repo.appendingPathComponent("Resources/catalog-public-key")),
                source: LocalGitCatalogSource(repo: repo), kinds: KindSpec.load(directory: repo.appendingPathComponent("kinds")), paths: paths,
                artifacts: BundledWorkerArtifact(helpers: URL(fileURLWithPath: args[2]), uv: UvEnvironment(paths: paths)), client: client,
                kindsDirectory: repo.appendingPathComponent("kinds"))
            live = service
            try await service.add(id: model)
            try await service.get(id: model, precision: "bf16")
            try await service.load(id: model, precision: "bf16", fast: true)
            let listener = ArsenalAPIServer(service: service, supportDirectory: paths.support, port: 0)
            server = listener
            let port = try await listener.start()
            func post(_ path: String, _ body: JSONObject) async throws -> JSONObject {
                var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/providers/elevenlabs" + path)!)
                request.httpMethod = "POST"; request.httpBody = try WireJSON.encode(body); request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                let (bytes, response) = try await URLSession.shared.data(for: request)
                guard (response as? HTTPURLResponse)?.statusCode == 200 else { throw Failure(String(decoding: bytes, as: UTF8.self)) }
                return try WireJSON.decode(bytes)
            }
            let text = "Hello world."
            let previews = try await post(
                "/v1/text-to-voice/create-previews?output_format=wav_24000",
                ["model_id": .string(model), "voice_description": "A calm adult narrator with a clear conversational voice.", "text": .string(text)])
            guard let candidates = previews["previews"]?.array, candidates.count == 3,
                let token = candidates[1]["generated_voice_id"]?.string, let encoded = candidates[1]["audio_base_64"]?.string,
                let chosen = Data(base64Encoded: encoded)
            else { throw Failure("Three voice previews were not returned") }
            let saved = try await post("/v1/text-to-voice", ["generated_voice_id": .string(token), "voice_name": "Chosen second preview"])
            guard let voice = saved["voice_id"]?.string else { throw Failure("No saved voice") }
            let manifest = try WireJSON.decode(Data(contentsOf: paths.support.appendingPathComponent("voices/\(model)/\(voice).json")))
            guard let retained = manifest["preview"]?.string, manifest["seed"]?.number != nil, manifest["modelRevision"]?.string != nil else {
                throw Failure("The saved voice lost its recipe")
            }
            let retainedBytes = try Data(contentsOf: URL(fileURLWithPath: retained))
            // Native rates differ; compare the exact retained native WAV against native saved speech.
            let speak = try await service.run(id: model, op: "speak", params: ["text": .string(text), "voice": .string(voice), "stream": false], sink: nil)
            guard let output = speak["path"]?.string else { throw Failure("No saved speech audio") }
            let actual = try Data(contentsOf: URL(fileURLWithPath: output))
            guard actual == retainedBytes else { throw Failure("Saved speech differs from its chosen preview") }
            try await service.unload(id: model)
            try await service.load(id: model, precision: "bf16", fast: true)
            let reopened = try await service.run(id: model, op: "speak", params: ["text": .string(text), "voice": .string(voice), "stream": false], sink: nil)
            guard let reopenedPath = reopened["path"]?.string, try Data(contentsOf: URL(fileURLWithPath: reopenedPath)) == retainedBytes else {
                throw Failure("Worker restart changed the saved voice")
            }
            try await service.unload(id: model)
            listener.stop(); server = nil
            await service.shutdown(); live = nil
            let memory = await service.memory()
            guard memory.models.isEmpty else { throw Failure("Unload left a worker") }
            let evidence: JSONObject = [
                "model": .string(model), "dtype": "bf16", "fast": true, "chosen_preview": 2, "preview_sha256": .string(try FileDigest.sha256(URL(fileURLWithPath: retained))),
                "saved_speech_sha256": speak["sha256"] ?? .null, "restarted_speech_sha256": reopened["sha256"] ?? .null, "exact_native_wav_match": true,
                "retained_preview_bytes": .number(Double(retainedBytes.count)), "provider_preview_bytes": .number(Double(chosen.count)), "recipe": .object(manifest),
                "worker_memory_after_unload_mb": .number(memory.workersMB),
                "network_fallback": "disabled (loopback discard)", "conditions": ["measurement_lock": false, "note": "gpulock functional proof; not a benchmark"]
            ]
            try WireJSON.encode(evidence).write(to: URL(fileURLWithPath: args[4]), options: .atomic)
            print("voices: \(model) selected preview → save → speak → restart → speak matched exactly; unloaded")
        } catch {
            server?.stop(); await live?.shutdown()
            FileHandle.standardError.write(Data(("voices: " + error.localizedDescription + "\n").utf8)); exit(1)
        }
    }
}
private struct Failure: LocalizedError {
    let message: String
    init(_ message: String) { self.message = message }
    var errorDescription: String? { message }
}
