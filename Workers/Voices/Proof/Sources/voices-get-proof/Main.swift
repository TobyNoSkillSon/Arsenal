// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import ArsenalWire

@main enum CatalogCycleProof {
    static func main() async {
        var live: RealModelsService?
        do {
            guard ProcessInfo.processInfo.environment["VELLA_LOCK_HELD"] == "1" else { throw ProofError("Run the real cycle under mlock gpulock with an alarm") }
            let a = CommandLine.arguments
            guard a.count == 6 else { throw ProofError("Usage: voices-get-proof REPOSITORY HELPERS MODEL DTYPE fast|exact") }
            let repository = URL(fileURLWithPath: a[1]), helpers = URL(fileURLWithPath: a[2]), modelID = a[3], dtype = a[4], fast = a[5] == "fast"
            let paths = ArsenalPaths()
            let persistent = FileManager.default.homeDirectoryForCurrentUser.appendingPathComponent("Library/Application Support/Arsenal-dev/weights/voices/").path
            guard paths.support.path.hasPrefix(persistent), !paths.support.path.contains("..") else {
                throw ProofError("Use isolated persistent Arsenal-dev/weights/voices support, without symlinked weight paths")
            }
            let folder = repository.appendingPathComponent("models/" + modelID)
            let model = try ModelManifest.load(folder)
            let files = try SafePath.files(in: folder).map { relative in
                let file = folder.appendingPathComponent(relative)
                return CatalogFile(path: relative, sha256: try FileDigest.sha256(file), bytes: try FileDigest.bytes(file))
            }
            let process = Process(); process.executableURL = URL(fileURLWithPath: "/usr/bin/git"); process.arguments = ["-C", repository.path, "rev-parse", "HEAD"]
            let pipe = Pipe(); process.standardOutput = pipe; try process.run(); process.waitUntilExit()
            guard process.terminationStatus == 0 else { throw ProofError("Cannot pin catalog commit") }
            let commit = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
            let document = CatalogDocument(sequence: 1, sourceCommit: commit, models: [CatalogModel(manifest: model, files: files)])
            let data = try document.encoded()
            let signed = try CatalogTrust.sign(data: data, keyFile: paths.support.appendingPathComponent("proof-signing/key"))
            // Network fallback is disabled. A successful Get proves zero weight-download bytes.
            let allowsDownload = modelID == "kokoro-82m"
            var client = HuggingFaceClient(baseURL: URL(string: allowsDownload ? "https://huggingface.co" : "http://127.0.0.1:9")!)
            client.maximumAttempts = allowsDownload ? 4 : 1; client.stallTimeout = allowsDownload ? 60 : 1; client.token = { nil }
            let service = try await RealModelsService.open(
                catalogData: data, signature: signed.signature, publicKey: signed.publicKey,
                source: LocalGitCatalogSource(repo: repository), sourceName: "signed local catalog proof",
                kinds: KindSpec.load(directory: repository.appendingPathComponent("kinds")),
                paths: paths, artifacts: BundledWorkerArtifact(helpers: helpers, uv: UvEnvironment(paths: paths)), client: client,
                kindsDirectory: repository.appendingPathComponent("kinds"))
            live = service
            try await service.add(id: modelID)
            let start = Date()
            try await service.get(id: modelID, precision: dtype)
            let getSeconds = Date().timeIntervalSince(start)
            try await service.load(id: modelID, precision: dtype, fast: fast)
            var extras: JSONObject = [:]
            var params: JSONObject = ["text": "Hello world.", "seed": 24301, "stream": false]
            if modelID == "kokoro-82m" { params["voice"] = "af_heart" }
            if modelID == "pocket-tts" { params["voice"] = "alba" }
            if modelID == "qwen3-tts-voicedesign" || modelID == "breeze-tts-2" {
                let design = try await service.run(
                    id: modelID, op: "design_voice",
                    params: ["description": "A calm adult narrator with a clear conversational voice.", "preview_text": "Hello world.", "seed": 24301], sink: nil)
                guard let voice = design["voice_id"]?.string, let path = design["preview"]?["path"]?.string,
                    FileManager.default.fileExists(atPath: path)
                else { throw ProofError("Design returned no saved voice or preview") }
                extras["design_voice"] = .object(design); params["voice"] = .string(voice)
            }
            if modelID == "qwen3-tts-base" || modelID == "pocket-tts" || modelID == "breeze-tts-2" {
                let reference = ProcessInfo.processInfo.environment["ARSENAL_VOICE_REFERENCE"] ?? "/tmp/arsenal-v2-voices/reference.wav"
                let clone = try await service.run(
                    id: modelID, op: "clone_voice", params: ["audio": ["path": .string(reference)], "transcript": "Hello world.", "name": "Cycle proof"], sink: nil)
                guard let voice = clone["voice_id"]?.string else { throw ProofError("Clone returned no saved voice") }
                extras["clone_voice"] = .object(clone); params["voice"] = .string(voice)
            }
            if ProcessInfo.processInfo.environment["ARSENAL_PROVE_VOICE_SETTING"] == "1", let voice = params["voice"] {
                try await service.setSetting(id: modelID, key: "voice", value: voice)
                params.removeValue(forKey: "voice")
                guard await service.rows().first(where: { $0.id == modelID })?.needsReload == false else {
                    throw ProofError("Request-time voice setting incorrectly requires reload")
                }
                extras["voice_from_setting"] = voice
            }
            let result = try await service.run(id: modelID, op: "speak", params: params, sink: nil)
            extras["status"] = .object(try await service.supervisor.request(id: modelID, op: "status"))
            let resident = await service.rows().first { $0.id == modelID }
            extras["kernels"] = .array((resident?.kernels ?? []).map(JSONValue.object))
            guard let path = result["path"]?.string, let duration = result["duration_s"]?.number, duration > 0,
                let hash = result["sha256"]?.string, try FileDigest.sha256(URL(fileURLWithPath: path)) == hash
            else { throw ProofError("Speech returned no valid WAV") }
            if modelID == "kokoro-82m", dtype == "fp16", fast {
                let reference = URL(fileURLWithPath: "/tmp/arsenal-v2-voices/reference.wav")
                if !FileManager.default.fileExists(atPath: reference.path) { try FileManager.default.copyItem(at: URL(fileURLWithPath: path), to: reference) }
            }
            let voices = try await service.run(id: modelID, op: "voices", params: [:], sink: nil)
            extras["voices"] = .object(voices)
            try await service.unload(id: modelID)
            let memory = await service.memory()
            guard memory.models.isEmpty else { throw ProofError("Unload left a resident worker") }
            await service.shutdown()
            let evidence: JSONObject = [
                "model": .string(modelID), "dtype": .string(dtype), "fast": .bool(fast), "catalog_commit": .string(commit), "catalog_signature_verified": true,
                "get_s": .number(getSeconds), "cycle_s": .number(Date().timeIntervalSince(start)), "network_download_bytes": allowsDownload ? .null : 0,
                "network_fallback": .string(allowsDownload ? "Kokoro HF Get authorized; bytes see downloader receipts" : "disabled (loopback discard port)"),
                "speak": .object(result), "extra_ops": .object(extras), "worker_memory_after_unload_mb": .number(memory.workersMB),
                "conditions": ["measurement_lock": false, "note": "gpulock functional proof; not benchmark"]
            ]
            try JSONEncoder().encode(JSONValue.object(evidence)).write(to: paths.support.appendingPathComponent("catalog-cycle-proof.json"))
            print("voices: \(modelID) \(dtype) \(fast ? "fast" : "exact") Get → Load → ops → Unload passed; \(allowsDownload ? "HF Get" : "0 network bytes")")
        } catch { await live?.shutdown(); FileHandle.standardError.write(Data(("voices: " + error.localizedDescription + "\n").utf8)); exit(1) }
    }
}
struct ProofError: LocalizedError { let message: String; init(_ message: String) { self.message = message }; var errorDescription: String? { message } }
