// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

struct VoicePreview: Sendable {
    let model: String
    let voiceID: String
    let path: String
    let text: String
    let directory: URL
    let created = Date()
}

extension APIRouter {
    func elevenlabsVoiceDesign(_ request: HTTPRequest, _ connection: HTTPConnection, path: String) async throws {
        if request.method == "GET" {
            let parts = path.split(separator: "/")
            guard parts.count == 4, parts.last == "stream", let preview = previews[String(parts[2])] else {
                throw HTTPFailure(404, "Voice preview not found; design a voice again.")
            }
            let format = try audioFormat(request, default: "wav_24000")
            guard Date().timeIntervalSince(preview.created) < 3600, FileManager.default.fileExists(atPath: preview.path) else {
                throw HTTPFailure(404, "Voice preview expired; design a voice again.")
            }
            let data = try Data(contentsOf: outputPath(preview.path))
            let (pcm, rate, channels) = try PCM.readWAV(data)
            let converted = try PCM.convert(pcm, inputRate: rate, inputChannels: channels, outputRate: format.rate)
            try await connection.respondBytes(format.wav ? PCM.wav(converted, rate: format.rate) : converted, type: format.type)
            return
        }
        let body = try object(request)
        if path == "/v1/text-to-voice" {
            let generated = try required(body, "generated_voice_id"), name = try required(body, "voice_name")
            guard let preview = previews[generated] else { throw HTTPFailure(404, "Voice preview not found; design a voice again.") }
            guard Date().timeIntervalSince(preview.created) < 3600, FileManager.default.fileExists(atPath: preview.path) else {
                previews[generated] = nil; removeTransientJob(preview.directory); throw HTTPFailure(404, "Voice preview expired; design a voice again.")
            }
            _ = try await resolve(preview.model, kind: "voices")
            let result = try await service.saveDesignedVoice(id: preview.model, previewID: preview.voiceID, name: name)
            guard let voiceID = result["voice_id"]?.string else { throw HTTPFailure(502, "No saved voice ID was returned.") }
            previews[generated] = nil; previewOrder.removeAll { $0 == generated }; removeTransientJob(preview.directory)
            try await connection.respond(voiceObject(id: voiceID, name: name, model: preview.model, category: "generated"), headers: audioHeaders(model: preview.model)); return
        }
        let description = try required(body, "voice_description")
        let text = body["text"]?.string ?? "Hello! This is a sample of my voice. I can read your scripts and narrate your stories in this tone."
        let model = try await audioModel(body["model_id"]?.string, kind: "voices", op: "design_voice", aliases: ["eleven_multilingual_ttv_v2", "eleven_ttv_v3"])
        let format = try audioFormat(request, default: "wav_24000")
        let supportsSeed = await service.operations(id: model).first(where: { $0.name == "design_voice" })?.inputSchema["properties"]?["seed"] != nil
        let firstSeed: UInt32
        if supportsSeed, let supplied = body["seed"] {
            guard let seed = supplied.number, seed.isFinite, seed.rounded() == seed, seed >= 0, seed <= Double(UInt32.max) else {
                throw HTTPFailure(422, "seed must be an integer between 0 and 4294967295.")
            }
            firstSeed = UInt32(seed)
        } else {
            firstSeed = UInt32.random(in: 0...UInt32.max)
        }
        var output: [JSONValue] = []
        for index in 0..<3 {
            let job = try newJob()
            var keep = false
            defer { if !keep { removeTransientJob(job) } }
            var params: JSONObject = ["description": .string(description), "preview_text": .string(text), "out_dir": .string(job.path)]
            params["language"] = body["language"]
            if supportsSeed {
                // Stay in the declared UInt32 range at its upper boundary; the worker records each seed.
                params["seed"] = .number(Double(firstSeed &+ UInt32(index)))
            }
            let result = try await run(model, op: "design_voice", params: params)
            guard let voiceID = result["voice_id"]?.string, let path = result["preview"]?["path"]?.string else { throw HTTPFailure(502, "Voice design returned no preview.") }
            let file = try outputPath(path)
            guard let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= PCM.maxBytes else { throw HTTPFailure(502, "Voice preview exceeds 64 MiB.") }
            let (pcm, rate, channels) = try PCM.readWAV(Data(contentsOf: file))
            let converted = try PCM.convert(pcm, inputRate: rate, inputChannels: channels, outputRate: format.rate)
            let generated = UUID().uuidString.replacingOccurrences(of: "-", with: "").lowercased()
            previews[generated] = VoicePreview(model: model, voiceID: voiceID, path: path, text: text, directory: job)
            previewOrder.append(generated); keep = true
            while previewOrder.count > 32 {
                let expired = previewOrder.removeFirst()
                if let old = previews.removeValue(forKey: expired) { removeTransientJob(old.directory) }
            }
            let audio = body["stream_previews"]?.bool == true ? Data() : format.wav ? PCM.wav(converted, rate: format.rate) : converted
            output.append([
                "audio_base_64": .string(audio.base64EncodedString()), "generated_voice_id": .string(generated), "media_type": .string(format.type),
                "duration_secs": .number(Double(pcm.count) / Double(rate * channels * 2)), "language": body["language"] ?? .null
            ])
        }
        let ignored = (["loudness", "guidance_scale", "quality", "prompt_strength", "should_enhance", "reference_audio_base64"] + (supportsSeed ? [] : ["seed"])).filter {
            body[$0] != nil
        }.map { $0 + " ignored: not declared by this model" }
        try await connection.respond(["previews": output.map(plain), "text": text], headers: audioHeaders(model: model, warnings: ignored))
    }
    func elevenlabsClone(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        guard request.method == "POST" else { throw HTTPFailure(405, "Use POST.") }
        let parts = try Multipart.parse(request.body, contentType: request.headers["content-type"] ?? "")
        func text(_ name: String) -> String? { parts.first { $0.name == name }.flatMap { String(data: request.body[$0.range], encoding: .utf8) } }
        guard let name = text("name"), !name.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw HTTPFailure(422, "name is required.") }
        let files = parts.filter { $0.name == "files" || $0.name == "files[]" }
        guard let file = files.first, !file.range.isEmpty else { throw HTTPFailure(422, "Upload a reference in files.") }
        var transcript = text("transcript")
        if transcript == nil, let labels = text("labels"), let object = try? WireJSON.decode(Data(labels.utf8)) { transcript = object["transcript"]?.string }
        let preferred = await preferredCloner(transcript: transcript)
        let model = try await audioModel(text("model_id") ?? preferred, kind: "voices", op: "clone_voice", aliases: [])
        let declaration = await service.operations(id: model).first { $0.name == "clone_voice" }
        if (declaration?.inputSchema["properties"]?["transcript"]?["minLength"]?.number ?? 0) > 0, transcript?.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty != false {
            throw HTTPFailure(422, "This model requires the reference transcript; supply transcript or labels containing transcript.")
        }
        let job = try newJob(); defer { removeTransientJob(job) }
        let audio = job.appendingPathComponent("reference." + audioExtension(file.filename, contentType: file.contentType))
        try Data(request.body[file.range]).write(to: audio)
        var params: JSONObject = ["audio": ["path": .string(audio.path)], "name": .string(name)]
        if declaration?.inputSchema["properties"]?["transcript"] != nil { params["transcript"] = .string(transcript ?? "") }
        let result = try await run(model, op: "clone_voice", params: params)
        guard let id = result["voice_id"]?.string else { throw HTTPFailure(502, "Runner returned no cloned voice ID.") }
        var warnings = files.count > 1 ? ["Only the first reference clip was used"] : []
        if transcript == nil { warnings.append("No transcript supplied; use a cloning model that supports audio-only reference") }
        if text("remove_background_noise") == "true" { warnings.append("remove_background_noise ignored; clean the reference with audio-isolation first") }
        try await connection.respond(["voice_id": id, "requires_verification": false], headers: audioHeaders(model: model, warnings: warnings))
    }
    func deleteElevenLabsVoice(_ voiceID: String, model requested: String?) async throws {
        guard voiceID.hasPrefix("voice-"), UUID(uuidString: String(voiceID.dropFirst(6))) != nil else { throw HTTPFailure(422, "Preset voices cannot be deleted.") }
        if let requested {
            _ = try await resolve(requested, kind: "voices")
            try await service.deleteSavedVoice(id: requested, voiceID: voiceID); return
        }
        for row in await installedModels().filter({ $0["kind"] as? String == "voices" }) {
            guard let model = row["id"] as? String else { continue }
            do { try await service.deleteSavedVoice(id: model, voiceID: voiceID); return } catch let failure as WorkerFailure
                where failure.code == "unknown_voice" || failure.code == "not_found"
            { continue }
        }
        throw HTTPFailure(404, "Saved voice does not exist in your model table.")
    }
    func preferredCloner(transcript: String?) async -> String? {
        let preferred = transcript?.isEmpty == false ? ["qwen3-tts-base", "pocket-tts", "breeze-tts-2"] : ["pocket-tts", "breeze-tts-2"]
        let installed = await installedModels()
        for id in preferred where installed.contains(where: { $0["id"] as? String == id }) {
            if await service.operations(id: id).contains(where: { $0.name == "clone_voice" }) { return id }
        }
        return nil
    }
    func voiceObject(id: String, name: String, model: String, category: String = "premade") -> [String: Any] {
        [
            "voice_id": id, "name": name, "category": category, "labels": ["language": "en"], "description": "Local voice", "preview_url": NSNull(), "settings": NSNull(),
            "sharing": NSNull(), "available_for_tiers": [], "high_quality_base_model_ids": [model], "is_owner": true
        ]
    }
}
