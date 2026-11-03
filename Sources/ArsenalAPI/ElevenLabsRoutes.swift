// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func elevenlabs(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let path = String(request.path.dropFirst("/providers/elevenlabs".count))
        if ["/v1/sound-generation", "/v1/music", "/v1/music/stream", "/v1/audio-isolation", "/v1/audio-isolation/stream"].contains(path) {
            try await elevenlabsAudio(request, connection, path: path); return
        }
        if ["/v1/text-to-voice", "/v1/text-to-voice/design", "/v1/text-to-voice/create-previews"].contains(path) || path.hasPrefix("/v1/text-to-voice/") {
            try await elevenlabsVoiceDesign(request, connection, path: path); return
        }
        if request.method == "DELETE", path.hasPrefix("/v1/voices/") {
            let voiceID = String(path.dropFirst("/v1/voices/".count))
            guard !voiceID.isEmpty, !voiceID.contains("/") else { throw HTTPFailure(404, "Voice endpoint not found.") }
            try await deleteElevenLabsVoice(voiceID, model: request.query["model_id"])
            try await connection.respond(["status": "ok"]); return
        }
        if path == "/v1/voices/add" { try await elevenlabsClone(request, connection); return }
        if request.method == "GET" {
            switch path {
            case "/v1/models":
                let models = await rows().filter { ["voices", "sound", "enhancement"].contains($0["kind"] as? String ?? "") }.enumerated().sorted { a, b in
                    let ranks = ["voices": 0, "sound": 1, "enhancement": 2]
                    let left = ranks[a.element["kind"] as? String ?? ""] ?? 3, right = ranks[b.element["kind"] as? String ?? ""] ?? 3
                    return left == right ? a.offset < b.offset : left < right
                }.map(\.element).map { row -> [String: Any] in
                    [
                        "model_id": row["id"] ?? "", "name": row["name"] ?? "", "description": row["summary"] ?? "", "can_be_finetuned": false,
                        "can_do_text_to_speech": row["kind"] as? String == "voices", "can_do_voice_conversion": false, "can_use_style": false, "can_use_speaker_boost": false,
                        "serves_pro_voices": false, "token_cost_factor": 1.0, "max_characters_request_free_user": 10000, "max_characters_request_subscribed_user": 10000,
                        "maximum_text_length_per_request": 10000, "languages": [["language_id": "en", "name": "English"]], "model_rates": [:], "concurrency_group": "local"
                    ]
                }
                try await connection.respond(models)
            case "/v2/voices", "/v1/voices":
                let voices = try await localVoices()
                try await connection.respond(["voices": voices, "has_more": false, "total_count": voices.count, "next_page_token": NSNull()])
            case let path where path.hasPrefix("/v1/voices/"):
                let id = String(path.dropFirst("/v1/voices/".count))
                guard let voice = try await localVoices().first(where: { $0["voice_id"] as? String == id }) else { throw HTTPFailure(404, "Unknown voice \(id).") }
                try await connection.respond(voice)
            default: throw HTTPFailure(404, "ElevenLabs endpoint not supported.")
            }
            return
        }
        guard path.hasPrefix("/v1/text-to-speech/") else { throw HTTPFailure(404, "ElevenLabs endpoint not supported.") }
        let suffix = path.dropFirst("/v1/text-to-speech/".count).split(separator: "/")
        guard suffix.count == 1 || suffix.count == 2 && suffix[1] == "stream" else { throw HTTPFailure(404, "TTS endpoint not supported.") }
        let voice = String(suffix[0]), body = try object(request)
        let model: String
        if let provided = body["model_id"]?.string { model = provided } else { model = try await defaultModel(kind: "voices") }
        _ = try await resolve(model, kind: "voices", operation: "speak")
        let text = try required(body, "text")
        guard text.count <= 10_000 else { throw HTTPFailure(413, "Speech input exceeds 10,000 characters.") }
        try reject(
            body,
            [
                "pronunciation_dictionary_locators", "previous_text", "next_text", "previous_request_ids", "next_request_ids", "apply_text_normalization",
                "apply_language_text_normalization", "seed"
            ])
        if let settings = body["voice_settings"]?.object { try reject(settings, ["stability", "similarity_boost", "style", "use_speaker_boost"]) }
        let format = request.query["output_format"] ?? "mp3_44100_128", parts = format.split(separator: "_")
        guard parts.count == 2, ["pcm", "wav"].contains(String(parts[0])), let rate = Int(parts[1]), [8000, 16000, 22050, 24000, 32000, 44100, 48000].contains(rate) else {
            throw HTTPFailure(400, "Request pcm_24000 or wav_24000; MP3, Opus and G.711 are not supported.")
        }
        var params: JSONObject = ["text": .string(text), "voice": .string(voice), "stream": false]
        params["speed"] = body["voice_settings"]?["speed"]; params["language"] = body["language_code"]
        params["reference"] = body["reference"]; params["instructions"] = body["instructions"]
        let (bytes, inputRate, channels) = try await renderSpeech(id: model, params: params)
        let pcm = try PCM.convert(bytes, inputRate: inputRate, inputChannels: channels, outputRate: rate)
        let output = parts[0] == "wav" ? PCM.wav(pcm, rate: rate) : pcm
        let type = parts[0] == "wav" ? "audio/wav" : "audio/pcm"
        if suffix.count == 2 {
            try await connection.beginStream(type: type)
            for offset in stride(from: 0, to: output.count, by: 4096) { try await connection.chunk(Data(output[offset..<min(output.count, offset + 4096)])) }
            try await connection.finishStream()
        } else {
            try await connection.respondBytes(output, type: type)
        }
    }
    func localVoices() async throws -> [[String: Any]] {
        var voices: [[String: Any]] = [], seen: Set<String> = []
        for model in await installedModels().filter({ $0["kind"] as? String == "voices" }) {
            guard let id = model["id"] as? String else { continue }
            let result = try await run(id, op: "voices", params: [:])
            for voice in result["voices"]?.array ?? [] {
                guard let id = voice["id"]?.string, seen.insert(id).inserted else { continue }
                voices.append([
                    "voice_id": id, "name": voice["name"]?.string ?? id, "category": "premade", "labels": ["language": voice["language"]?.string ?? "en"],
                    "description": voice["description"]?.string ?? "Local voice", "preview_url": NSNull(), "settings": NSNull(), "sharing": NSNull(), "available_for_tiers": [],
                    "high_quality_base_model_ids": [model["id"] ?? ""], "is_owner": true
                ])
            }
        }
        return voices
    }
}
