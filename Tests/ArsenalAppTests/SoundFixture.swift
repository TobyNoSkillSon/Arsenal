// SPDX-License-Identifier: MIT
import AppKit
import Foundation
@testable import Arsenal
import ArsenalCore

/// The sound window's fixtures: table rows for a sound model, two voice models (one not on this Mac), their audio
/// operations with the input schemas read from the repository's model folders (so the window is checked against what
/// the folders declare today), and a synthesized take for previews. No worker, no weights.
enum SoundFixture {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    static func row(_ id: String, _ name: String, kind: String, tools: [String], state: ModelState = .ready, loaded: Bool = false) -> ModelRow {
        var row = ModelRow(
            id: id, name: name, kind: kind, params: "", licence: "MIT", precisions: ["fp16"], states: ["fp16": PrecisionState(state: loaded ? .loaded : state)],
            selectedPrecision: "fp16", fast: false, runner: "native", summary: "", runnerLanguage: "swift", fastExact: true)
        row.tools = tools
        return row
    }
    static var stableAudio: ModelRow { row("stable-audio-3-medium", "Stable Audio 3 Medium", kind: "sound", tools: ["sound"]) }
    static var kokoro: ModelRow { row("kokoro-82m", "Kokoro 82M", kind: "voices", tools: ["voice", "sound"], loaded: true) }
    static var qwenBase: ModelRow { row("qwen3-tts-base", "Qwen3 Base 1.7B", kind: "voices", tools: ["voice", "sound"], state: .notInstalled) }
    static var rows: [ModelRow] { [stableAudio, kokoro, qwenBase] }
    static var metadata: [String: JSONObject] {
        Dictionary(uniqueKeysWithValues: rows.map { ($0.id, ["tools": .array($0.tools.map { .string($0) })]) })
    }

    /// Operations each fixture model declares, as its model.toml lists them.
    static let declared: [String: [(name: String, title: String, output: String)]] = [
        "stable-audio-3-medium": [("generate_sound", "Generate Sound", "audio"), ("generate_music", "Generate Music", "audio")],
        "stable-audio-3-small-sfx": [("generate_sound", "Generate Sound", "audio")],
        "kokoro-82m": [("speak", "Speak", "audio"), ("voices", "Voices", "json")],
        "pocket-tts": [("speak", "Speak", "audio"), ("voices", "Voices", "json"), ("clone_voice", "Clone Voice", "json")],
        "breeze-tts-2": [("speak", "Speak", "audio"), ("voices", "Voices", "json"), ("design_voice", "Design Voice", "json"), ("clone_voice", "Clone Voice", "json")],
        "qwen3-tts-base": [("speak", "Speak", "audio"), ("voices", "Voices", "json"), ("clone_voice", "Clone Voice", "json")],
        "qwen3-tts-voicedesign": [("speak", "Speak", "audio"), ("voices", "Voices", "json"), ("design_voice", "Design Voice", "json")]
    ]

    /// The operation's input schema from `models/<id>/ops/<op>.input.json`, as the folder declares it now.
    static func schema(_ model: String, _ op: String) -> JSONObject {
        let url = shippedOrRetiredFolder(repo, model).appendingPathComponent("ops/\(op).input.json")
        guard let data = try? Data(contentsOf: url), case .object(let object)? = try? JSONDecoder().decode(JSONValue.self, from: data) else {
            return ["type": "object", "properties": [:]]
        }
        return object
    }
    static func operations(_ model: String) -> [ModelOperation] {
        (declared[model] ?? []).map { ModelOperation(name: $0.name, title: $0.title, inputSchema: schema(model, $0.name), output: $0.output, stream: $0.output == "audio") }
    }
    static func descriptors(_ model: String) -> [OperationDescriptor] {
        operations(model).filter { $0.output == "audio" }.map {
            OperationDescriptor(name: $0.name, title: $0.title, inputSchema: $0.inputSchema, output: $0.output, stream: $0.stream)
        }
    }
    static var allOperations: [String: [ModelOperation]] { Dictionary(uniqueKeysWithValues: declared.keys.map { ($0, operations($0)) }) }

    static func service() -> FixtureService {
        FixtureService(rows: Fixtures.rows(.loaded).filter { $0.id != "kokoro-82m" } + rows, metadata: metadata, operations: allOperations)
    }

    static func model(_ row: ModelRow, unavailable: OperationProblem? = nil) -> ModelChoice {
        ModelChoice(model: ToolModel(row), resident: row.residentDType != nil, unavailable: unavailable)
    }
    static var models: [ModelChoice] {
        [
            model(stableAudio), model(kokoro),
            model(qwenBase, unavailable: .init(message: "Qwen3 Base 1.7B isn\u{2019}t on this Mac yet.", action: .download, actionTitle: "Download 3.4 GB"))
        ]
    }

    static let kokoroVoices: [VoiceChoice] = [
        VoiceChoice(id: "af_heart", name: "Heart", language: "en-US"), VoiceChoice(id: "af_bella", name: "Bella", language: "en-US"),
        VoiceChoice(id: "am_adam", name: "Adam", language: "en-US"), VoiceChoice(id: "bf_emma", name: "Emma", language: "en-GB")
    ]

    // MARK: Audio

    /// A 16-bit PCM WAV with a shaped envelope (a few swells and decays over noise and a tone), so a waveform shows form.
    static func wav(seconds: Double, rate: Int = 44_100, channels: Int = 2, seed: Int = 1) -> Data {
        let frames = Int(seconds * Double(rate))
        var pcm = Data(capacity: frames * channels * 2)
        var state = UInt64(truncatingIfNeeded: seed) &* 0x9E37_79B9_7F4A_7C15 | 1
        func noise() -> Double {
            state ^= state << 13; state ^= state >> 7; state ^= state << 17
            return Double(state % 20_000) / 10_000 - 1
        }
        for i in 0..<frames {
            let t = Double(i) / Double(rate), x = t / max(0.01, seconds)
            let swells = 0.55 * pow(sin(.pi * x * Double(3 + seed % 3)), 2) + 0.35 * exp(-pow((x - 0.18) * 9, 2)) + 0.25 * exp(-pow((x - 0.72) * 14, 2))
            let fade = min(1, x * 20) * min(1, (1 - x) * 12)
            let value = fade * swells * (0.6 * noise() + 0.4 * sin(2 * .pi * 220 * t))
            let sample = Int16(max(-1, min(1, value)) * 30_000)
            for _ in 0..<channels { withUnsafeBytes(of: sample.littleEndian) { pcm.append(contentsOf: $0) } }
        }
        var wav = Data("RIFF".utf8)
        func u32(_ n: Int) { withUnsafeBytes(of: UInt32(n).littleEndian) { wav.append(contentsOf: $0) } }
        func u16(_ n: Int) { withUnsafeBytes(of: UInt16(n).littleEndian) { wav.append(contentsOf: $0) } }
        u32(pcm.count + 36); wav += Data("WAVEfmt ".utf8); u32(16); u16(1); u16(channels); u32(rate); u32(rate * channels * 2); u16(channels * 2); u16(16)
        wav += Data("data".utf8); u32(pcm.count); wav += pcm
        return wav
    }

    /// A take saved in `folder`, read as the window reads one.
    static func take(
        in folder: URL, seconds: Double, text: String, model: ModelRow = stableAudio, operation: String = "Generate Sound", seed: Int? = 1_234_567,
        voice: String? = nil, rate: Int = 44_100, channels: Int = 2, made: Double = 0.62, wave: Int = 1
    ) throws -> SoundTake {
        try FileManager.default.createDirectory(at: folder, withIntermediateDirectories: true)
        let url = folder.appendingPathComponent("take-\(UUID().uuidString.prefix(6)).wav")
        try wav(seconds: seconds, rate: rate, channels: channels, seed: wave).write(to: url)
        guard let shape = AudioShape.read(url) else { throw QAFailure("the fixture take is not readable audio") }
        return SoundTake(url: url, model: model.id, modelName: model.name, operation: operation, text: text, seed: seed, voice: voice, shape: shape, seconds: made)
    }
}
