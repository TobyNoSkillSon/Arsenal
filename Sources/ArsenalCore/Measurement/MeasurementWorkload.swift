// SPDX-License-Identifier: MIT
import Foundation

public enum MeasurementWorkload {
    public static let prefix = "The small observatory stood above the town."
    public static let continuation =
        " Each evening the astronomer opened the dome and checked the instruments. The sky was clear, and the first bright star appeared above the eastern hills."
    public static let speech = "The sky is clear above the quiet town."
    public static let items = ["The delivery arrived early.", "The parcel was damaged.", "Please cancel this order.", "Thank you for your help."]
    public static func identity(kind: ModelStatKind) -> String { "arsenal-served-" + kind.rawValue + "-v3-seed0-defaults" }
    public static func request(kind: ModelStatKind, model: String, batched: Bool, count: Int, directory: URL) throws -> (op: String, params: JSONObject) {
        switch kind {
        case .decisions:
            let questions: JSONObject = [
                "tone": ["type": "choice", "instructions": "Is the message positive or negative?", "criteria": ["positive": "positive", "negative": "negative"]]
            ]
            return batched
                ? ("judge", ["model": .string(model), "items": .array((0..<count).map { .string(items[$0 % items.count]) }), "questions": .object(questions)])
                : ("decide", ["model": .string(model), "state": .string(items[0]), "questions": .object(questions)])
        case .chat:
            return ("chat", ["messages": [["role": "user", "content": .string(prefix)]], "max_tokens": 32, "seed": 0, "stream": true, "thinking": false])
        case .images:
            return ("generate", ["prompt": "A quiet observatory on a hill beneath a clear evening sky.", "width": 1024, "height": 1024, "seed": 0])
        case .voices:
            var params: JSONObject = ["text": .string(speech), "language": "English", "seed": 0, "stream": true]
            if model == "qwen3-tts-voicedesign" || model == "breeze-tts-2" { params["instructions"] = "A calm, clear voice." }
            if model == "qwen3-tts-base" {
                let file = try MeasurementMedia.tone(directory: directory)
                params["reference"] = ["path": .string(file.path), "text": "A steady tone."]
            }
            return ("speak", params)
        case .sound:
            return ("generate_sound", ["prompt": "Gentle rain on a window.", "duration": 3, "seed": 0])
        case .enhancement:
            return ("enhance", ["audio": ["path": .string(try MeasurementMedia.tone(directory: directory).path)], "stream": false])
        case .audio, .embeddings, .perception, .restoration, .geometry: throw ModelStatKind.noWorkload(kind)
        }
    }
}
