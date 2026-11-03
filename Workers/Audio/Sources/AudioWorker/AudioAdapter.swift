// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Tokenizers

func audioFailure(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }

func audioInput(_ input: JSONObject, rate: Int, seconds: Int = 600, maximumFrames: Int? = nil) throws -> NativeMedia.Audio {
    guard let path = input["audio"]?["path"]?.string else { throw audioFailure("An audio file is required.") }
    return try NativeMedia.audio(URL(fileURLWithPath: path), sampleRate: rate, maximumFrames: maximumFrames ?? rate * seconds)
}

func audioInteger(_ input: JSONObject, key: String, fallback: Int, range: ClosedRange<Int>) throws -> Int {
    guard let value = input[key] else { return fallback }
    guard let number = value.number, number.isFinite, number.rounded() == number, number >= Double(range.lowerBound), number <= Double(range.upperBound) else {
        throw audioFailure("\(key) must be an integer in \(range.lowerBound)...\(range.upperBound).")
    }
    return Int(number)
}

func audioKeys(_ input: JSONObject, allowed: Set<String>) throws {
    guard Set(input.keys).isSubset(of: allowed.union(["out_dir"])) else { throw audioFailure("Unsupported audio option.") }
}

/// Transcription `language` is a BCP-47 code for every audio model. Qwen-family checkpoints name the detected language in
/// English ("language English<asr_text>…"); those names map to their codes. A code passes through; a name outside the
/// table is kept as the model wrote it rather than guessed.
let audioLanguageCodes: [String: String] = [
    "chinese": "zh", "english": "en", "cantonese": "yue", "arabic": "ar", "german": "de", "french": "fr", "spanish": "es", "portuguese": "pt",
    "indonesian": "id", "italian": "it", "korean": "ko", "russian": "ru", "thai": "th", "vietnamese": "vi", "japanese": "ja", "turkish": "tr",
    "hindi": "hi", "malay": "ms", "dutch": "nl", "swedish": "sv", "danish": "da", "finnish": "fi", "polish": "pl", "czech": "cs", "filipino": "fil",
    "persian": "fa", "greek": "el", "hungarian": "hu", "macedonian": "mk", "romanian": "ro"
]
func audioLanguageCode(_ language: String?) -> String? {
    guard let raw = language?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty, raw.lowercased() != "none" else { return nil }
    return audioLanguageCodes[raw.lowercased()] ?? raw
}
/// The language most chunks of one transcription were in (the earliest on a tie), as a BCP-47 code.
func dominantAudioLanguage(_ languages: [String?]) -> String? {
    let codes = languages.compactMap(audioLanguageCode)
    var counts: [String: Int] = [:]
    for code in codes { counts[code, default: 0] += 1 }
    return codes.first { counts[$0] == counts.values.max() }
}

func audioEligible(_ name: String, _ shape: [Int]) -> Bool {
    name.hasSuffix(".weight") && shape.count == 2 && shape[1] % 64 == 0
}

/// Local assets only. The tokenizer library has an async constructor; the engine owns its serialized thread.
func localTokenizer(_ directory: URL) throws -> any Tokenizer {
    let semaphore = DispatchSemaphore(value: 0)
    let box = TokenizerResult()
    Task.detached {
        do { box.result = .success(try await AutoTokenizer.from(modelFolder: directory)) } catch { box.result = .failure(error) }
        semaphore.signal()
    }
    guard semaphore.wait(timeout: .now() + 60) == .success, let result = box.result else {
        throw audioFailure("Local tokenizer initialization timed out.")
    }
    return try result.get()
}
private final class TokenizerResult: @unchecked Sendable {
    var result: Result<any Tokenizer, Error>?
}

class AudioAdapter: NativeModel {
    var architecture: String { fatalError("Adapter architecture") }
    var operations: Set<String> { [] }
    var graph: Module?
    var random = false
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { graph.map { Dictionary(uniqueKeysWithValues: $0.parameters().flattened()) } ?? [:] }
    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws { fatalError("Adapter load") }
    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        throw audioFailure("Unsupported audio operation: " + operation)
    }
    func unload() { graph = nil }
    func prepare(
        _ module: Module, _ context: NativeLoadContext, cancellation: NativeCancellation,
        map: @escaping (String, MLXArray) throws -> (String, MLXArray)? = { ($0, $1) },
        subdirectories: [String] = [],
        eligible: @escaping (String, [Int]) -> Bool = audioEligible
    ) throws {
        try cancellation.check()
        random = context.randomWeights
        graph = module
        if random {
            try NativeParameters.initializeRandom(module)
            if context.precision != "fp16" {
                // Exercise the production streamed tier installer, including immediate-owner replacement.
                let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
                defer { try? FileManager.default.removeItem(at: file) }
                try save(arrays: Dictionary(uniqueKeysWithValues: module.parameters().flattened()), url: file)
                _ = try NativePrecisionTier(context.precision).load(module, archives: [file], map: { ($0, $1) }, eligible: eligible)
            }
        } else {
            // Safetensors at the snapshot root, plus those of named subdirectories (an asset converted `into` one).
            let archives = try ([context.weights] + subdirectories.map { context.weights.appendingPathComponent($0) }).flatMap {
                try FileManager.default.contentsOfDirectory(at: $0, includingPropertiesForKeys: nil).filter { $0.pathExtension == "safetensors" }
            }
            guard !archives.isEmpty else { throw audioFailure("Checkpoint has no safetensors.") }
            _ = try NativePrecisionTier(context.precision).load(module, archives: archives, map: map, eligible: eligible)
        }
        module.train(false)
        try NativeParameters.requireHalf(parameters)
        try cancellation.check()
    }
    func config<T: Decodable>(_ context: NativeLoadContext, as type: T.Type) throws -> T {
        try JSONDecoder().decode(type, from: Data(contentsOf: context.weights.appendingPathComponent("config.json")))
    }
}
