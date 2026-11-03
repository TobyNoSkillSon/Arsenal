// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func elevenlabsAudio(_ request: HTTPRequest, _ connection: HTTPConnection, path: String) async throws {
        guard request.method == "POST" else { throw HTTPFailure(405, "Use POST.") }
        let isolation = path.hasPrefix("/v1/audio-isolation")
        let music = path.hasPrefix("/v1/music")
        let op = isolation ? "enhance" : music ? "generate_music" : "generate_sound"
        var params: JSONObject = [:], warnings: [String] = []
        let model: String
        var upload: URL?
        defer { if let upload { removeTransientJob(upload) } }
        if isolation {
            let form = try Multipart.parse(request.body, contentType: request.headers["content-type"] ?? "")
            func text(_ name: String) -> String? { form.first { $0.name == name }.flatMap { String(data: request.body[$0.range], encoding: .utf8) } }
            guard let audio = form.first(where: { $0.name == "audio" }), !audio.range.isEmpty else { throw HTTPFailure(422, "audio is required.") }
            let requested = text("model_id")
            if requested?.hasPrefix("sam-") == true { throw HTTPFailure(409, "No qualified prompted-isolation model; RE-USE cleans speech.") }
            model = try await audioModel(requested, kind: "enhancement", op: op, aliases: [])
            let directory = try newJob(); upload = directory
            let ext = text("file_format") == "pcm_s16le_16" ? "wav" : audioExtension(audio.filename, contentType: audio.contentType)
            let file = directory.appendingPathComponent("input." + ext)
            let data = Data(request.body[audio.range])
            try (text("file_format") == "pcm_s16le_16" ? PCM.wav(data, rate: 16000) : data).write(to: file)
            params["audio"] = .object(["path": .string(file.path)])
            if text("preview_b64") != nil { warnings.append("preview_b64 ignored") }
        } else {
            let body = try object(request)
            model = try await audioModel(body["model_id"]?.string, kind: "sound", op: op, aliases: music ? ["music_v1", "music_v2", "music_v2_5"] : ["eleven_text_to_sound_v2"])
            if music {
                (params, warnings) = try musicParams(body)
            } else {
                params["prompt"] = .string(try required(body, "text"))
                params["duration"] = body["duration_seconds"]
                if body["loop"]?.bool == true { warnings.append("loop ignored: output is not seamless") }
                if body["prompt_influence"] != nil { warnings.append("prompt_influence ignored") }
            }
            params["seed"] = body["seed"]
            params["steps"] = body["steps"]
            params["guidance"] = body["guidance"]
            if let negative = body["negative_prompt"] { params["negative_prompt"] = negative }
            if let value = params["duration"] {
                guard let duration = value.number, duration.isFinite, duration > 0, duration <= 380 else {
                    throw HTTPFailure(422, "Duration must be greater than zero and at most 380 seconds.")
                }
            }
            if let seed = params["seed"] { _ = try jsonInteger(seed, name: "seed", range: 0...Int(UInt32.max)) }
            if let steps = params["steps"] { _ = try jsonInteger(steps, name: "steps", range: 1...100) }
        }
        let format = try audioFormat(request, default: isolation ? "wav_24000" : "wav_44100")
        let (bytes, rate, channels) = try await renderAudio(id: model, op: op, params: params)
        let encoded: Data
        if format.wav && format.rate == rate {
            encoded = PCM.wav(bytes, rate: rate, channels: channels)
        } else {
            let pcm = try PCM.convert(bytes, inputRate: rate, inputChannels: channels, outputRate: format.rate)
            encoded = format.wav ? PCM.wav(pcm, rate: format.rate) : pcm
            if channels > 1 { warnings.append("output downmixed to mono") }
        }
        let headers = audioHeaders(model: model, warnings: warnings)
        if path.hasSuffix("/stream") {
            try await connection.beginStream(type: format.type, headers: headers)
            for i in stride(from: 0, to: encoded.count, by: 4096) { try await connection.chunk(Data(encoded[i..<min(encoded.count, i + 4096)])) }
            try await connection.finishStream()
        } else {
            try await connection.respondBytes(encoded, type: format.type, headers: headers)
        }
    }
    func audioModel(_ requested: String?, kind: String, op: String, aliases: [String]) async throws -> String {
        if let requested, !aliases.contains(requested) {
            _ = try await resolve(requested, kind: kind)
            guard await service.operations(id: requested).contains(where: { $0.name == op }) else { throw HTTPFailure(400, "Model \(requested) does not offer \(op).") }
            return requested
        }
        for row in await installedModels().filter({ $0["kind"] as? String == kind }) {
            guard let id = row["id"] as? String else { continue }
            if await service.operations(id: id).contains(where: { $0.name == op }) { return id }
        }
        throw HTTPFailure(404, "No installed model offers \(op); call \(MCPNames.prefix)find then \(MCPNames.prefix)download(model: ID).")
    }
    func musicParams(_ body: JSONObject) throws -> (JSONObject, [String]) {
        var prompt = body["prompt"]?.string, warnings: [String] = []
        var duration = body["music_length_ms"]?.number.map { $0 / 1000 }
        var negative: String?
        if let planValue = body["composition_plan"] ?? body["music_prompt"], planValue != .null {
            guard prompt == nil || prompt?.isEmpty == true, let plan = planValue.object else { throw HTTPFailure(422, "Send prompt or a composition_plan object, not both.") }
            let sections = plan["sections"]?.array ?? []
            var styles = plan["positive_global_styles"]?.array?.compactMap(\.string) ?? []
            for section in sections { styles += section["positive_local_styles"]?.array?.compactMap(\.string) ?? [] }
            var seen: Set<String> = []
            prompt = styles.filter { seen.insert($0.lowercased()).inserted }.joined(separator: ", ")
            negative = plan["negative_global_styles"]?.array?.compactMap(\.string).joined(separator: ", ")
            if duration == nil {
                let total = sections.compactMap { $0["duration_ms"]?.number }.reduce(0, +)
                if total > 0 { duration = total / 1000 }
            }
            if sections.count > 1 { warnings.append("composition plan flattened into one prompt") }
            if sections.contains(where: { !($0["lines"]?.array ?? []).isEmpty }) { warnings.append("lyrics ignored: instrumental only") }
        }
        guard let prompt, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw HTTPFailure(422, "Send a prompt or composition_plan with styles.") }
        if body["music_length_ms"] != nil && body["music_length_ms"]?.number == nil { throw HTTPFailure(422, "music_length_ms must be a number.") }

        if body["finetune_id"] != nil { warnings.append("finetune_id ignored") }
        var params: JSONObject = ["prompt": .string(prompt)]
        params["lyrics"] = body["lyrics"] ?? body["lyrics_text"]
        params["language"] = body["language"]
        params["instrumental"] = body["force_instrumental"] ?? body["instrumental"]
        if let duration { params["duration"] = .number(duration) }
        if let negative, !negative.isEmpty { params["negative_prompt"] = .string(negative) }
        return (params, warnings)
    }
    func audioFormat(_ request: HTTPRequest, default fallback: String) throws -> (wav: Bool, rate: Int, type: String) {
        let raw = request.query["output_format"] ?? fallback
        let parts = raw.split(separator: "_")
        guard parts.count == 2, ["pcm", "wav"].contains(String(parts[0])), let rate = Int(parts[1]), [8000, 16000, 22050, 24000, 32000, 44100, 48000].contains(rate) else {
            throw HTTPFailure(400, "Request wav_44100 or pcm_24000; MP3, Opus and G.711 are not supported.")
        }
        return (parts[0] == "wav", rate, parts[0] == "wav" ? "audio/wav" : "audio/pcm")
    }
    func audioHeaders(model: String, warnings: [String] = []) -> [String: String] {
        var headers = ["X-Arsenal-Model": model]
        if !warnings.isEmpty { headers["X-Arsenal-Warnings"] = warnings.joined(separator: "; ") }
        return headers
    }
    func audioExtension(_ name: String?, contentType: String?) -> String {
        let ext = name.map { URL(fileURLWithPath: $0).pathExtension.lowercased() } ?? ""
        if ["wav", "mp3", "m4a", "flac", "aiff", "caf"].contains(ext) { return ext }
        return contentType?.contains("mpeg") == true ? "mp3" : "wav"
    }
}
