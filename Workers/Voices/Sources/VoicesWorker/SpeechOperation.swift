// SPDX-License-Identifier: MIT
import ArsenalWire

/// Named operations use the same qualified speech engine, with explicit capability guards.
enum SpeechOperation {
    static let designModels: Set<String> = ["qwen3-tts-voicedesign", "breeze-tts-2"]
    static let cloneModels: Set<String> = ["pocket-tts", "qwen3-tts-base", "breeze-tts-2", "fish-drama-3", "voxcpm2"]
    /// Models whose sampler takes request controls, with the controls each accepts.
    static let samplingControls: [String: Set<String>] = [
        "fish-drama-3": ["max_tokens", "temperature", "top_p", "top_k"],
        // VoxCPM2's own controls (cfg_value, inference_timesteps, clone_mode, retry_badcase) are checked by its runner.
        "voxcpm2": ["max_tokens"]
    ]

    static func check(_ request: JSONObject, modelID: String?) throws {
        guard let modelID else { throw WorkerFailure(code: "not_loaded", message: "Load a voice model before speaking") }
        for (key, minimum, maximum, integer) in [
            ("max_tokens", 1.0, modelID == "breeze-tts-2" ? 750.0 : modelID == "fish-drama-3" || modelID == "voxcpm2" ? 4096.0 : 8192.0, true),
            ("temperature", 0.0, 2.0, false), ("top_p", 0.000001, 1.0, false),
            ("top_k", 0.0, 4096.0, true), ("repetition_penalty", 0.01, 2.0, false), ("guidance", 0.000001, 20.0, false)
        ] {
            guard let value = request[key] else { continue }
            let accepted =
                samplingControls[modelID].map { $0.contains(key) }
                ?? ((modelID == "breeze-tts-2" || modelID.hasPrefix("qwen3-tts")) && (key != "guidance" || modelID == "breeze-tts-2"))
            guard let number = value.number, number.isFinite, (minimum...maximum).contains(number), !integer || number.rounded() == number, accepted,
                key != "top_k" || modelID != "fish-drama-3" || number >= 1
            else { throw WorkerFailure(code: "invalid_request", message: "Unsupported or invalid \(key) for \(modelID)") }
        }
        if request["op"]?.string == "design_voice" {
            guard designModels.contains(modelID) else { throw WorkerFailure(code: "unsupported_op", message: "This model does not support voice design") }
            _ = try string(request, "description")
            if request["name"] != nil { _ = try string(request, "name") }
        }
        if request["op"]?.string == "clone_voice" {
            guard cloneModels.contains(modelID) else { throw WorkerFailure(code: "unsupported_op", message: "This model does not support voice cloning") }
            _ = try string(request["audio"]?.object ?? [:], "path")
            _ = try string(request, "name")
            // Pocket and VoxCPM2 clone from the clip alone; a VoxCPM2 transcript enables continuation cloning.
            if modelID != "pocket-tts" && modelID != "voxcpm2" { _ = try string(request, "transcript") }
        }
        if !cloneModels.contains(modelID), request["reference"]?["path"]?.string?.isEmpty == false {
            throw WorkerFailure(code: "unsupported_op", message: "This model does not support reference voice cloning")
        }
    }

    /// The engine's codec language map uses full names; accept standard ISO language codes too.
    static func language(_ value: String?) -> String? {
        guard let value else { return nil }
        return [
            "en": "english", "zh": "chinese", "ja": "japanese", "ko": "korean", "de": "german", "fr": "french", "ru": "russian", "pt": "portuguese", "es": "spanish",
            "it": "italian"
        ][value.lowercased()] ?? value.lowercased()
    }
}
