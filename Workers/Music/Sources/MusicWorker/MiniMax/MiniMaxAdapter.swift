// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Tokenizers

/// The five component directories of the maker's modular checkpoint, in load order.
enum MiniMaxComponents {
    static let all = ["language_model", "rvq_depth_decoder", "condition_encoder", "transformer", "vocoder"]
}

final class MiniMaxAdapter: NativeModel {
    let architecture = "minimax_music3"
    let operations: Set<String> = ["generate_music"]
    private(set) var pipeline: MiniMaxPipeline?
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] {
        guard let p = pipeline else { return [:] }
        return MusicSupport.parameters([
            ("language_model.", p.lm), ("rvq_depth_decoder.", p.depth), ("condition_encoder.", p.condition), ("transformer.", p.transformer),
            ("vocoder.", p.vocoder)
        ])
    }

    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let configuration = context.randomWeights ? MiniMaxConfiguration.tiny : try MiniMaxConfiguration.load(context.weights)
        let lm = MiniMaxLanguageModel(configuration), depth = MiniMaxDepthDecoder(configuration), condition = MiniMaxConditionEncoder(configuration)
        let transformer = MiniMaxFlowTransformer(configuration), vocoder = MiniMaxVocoder(configuration)
        let modules: [(String, Module)] = [
            ("language_model", lm), ("rvq_depth_decoder", depth), ("condition_encoder", condition), ("transformer", transformer), ("vocoder", vocoder)
        ]
        let tokenizer: (any Tokenizers.Tokenizer)?
        if context.randomWeights {
            for (_, module) in modules { try MusicSupport.initialize(module, precision: "fp16"); try cancellation.check() }
            tokenizer = try MusicSupport.fixtureTokenizer()
        } else {
            guard context.precision == "fp16" else { throw MusicSupport.failure("MiniMax Music 3 ships 16-bit only (fp16).") }
            for (directory, module) in modules {
                try cancellation.check()
                try Self.load(module, directory: context.weights.appendingPathComponent(directory))
            }
            tokenizer = try MusicSupport.wait { try await AutoTokenizer.from(modelFolder: context.weights.appendingPathComponent("tokenizer")) }
        }
        try Self.computeInBF16(lm)
        pipeline = MiniMaxPipeline(
            configuration: configuration, lm: lm, depth: depth, condition: condition, transformer: transformer, vocoder: vocoder, tokenizer: tokenizer)
    }

    /// The global language model computes in BF16, as the maker trained it: on the maker's own example prompt its residual
    /// stream reaches 65,024 (FP16's limit is 65,504) and grows with prompt length (log/002-bf16-global-lm.md). Get stores
    /// FP16; casting back is exact for every BF16 value in FP16's normal range. One tensor at a time keeps the peak flat.
    static func computeInBF16(_ module: Module) throws {
        for (key, value) in module.parameters().flattened() where value.dtype == .float16 {
            let converted = value.asType(.bfloat16)
            eval(converted)
            try module.update(parameters: ModuleParameters.unflattened([(key, converted)]), verify: .none)
        }
    }

    static func load(_ module: Module, directory: URL) throws {
        guard
            FileManager.default.fileExists(atPath: directory.appendingPathComponent(MusicConversion.layoutMarker).path)
                || FileManager.default.fileExists(atPath: directory.deletingLastPathComponent().appendingPathComponent(MusicConversion.layoutMarker).path)
        else { throw MusicSupport.failure("Prepare the MiniMax Music 3 half snapshot (Get) before Load.") }
        let archives = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !archives.isEmpty else { throw MusicSupport.failure("Missing MiniMax Music 3 weights in " + directory.lastPathComponent) }
        _ = try NativePrecisionTier("fp16").load(module, archives: archives, map: { ($0, $1) }, eligible: { _, _ in false })
    }

    struct Options {
        var request: MiniMaxPipeline.Request
        var batch: Int
        var format: AudioFileFormat
        var sampleRate: Double?
        var metadata: Bool
    }

    static let maxSeconds = 360.0
    /// One frame (1/25 s): a shorter budget cannot emit any audio.
    static let minSeconds = 0.04
    /// Songs in one request run one after another inside the operation's one-hour limit; their total requested length is
    /// capped so a full batch cannot outlast it.
    static let maxTotalSeconds = 720.0

    /// A present field of the wrong JSON type is an error, never silently replaced by the default.
    static func field<T>(_ input: JSONObject, _ key: String, _ kind: String, _ read: (JSONValue) -> T?) throws -> T? {
        guard let value = input[key] else { return nil }
        guard let typed = read(value) else { throw MusicSupport.failure("\(key) must be \(kind).") }
        return typed
    }

    static func options(_ input: JSONObject) throws -> Options {
        let prompt = try field(input, "prompt", "a string") { $0.string } ?? ""
        guard !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty, prompt.count <= 10_000 else {
            throw MusicSupport.failure("A nonempty music description (prompt) is required.")
        }
        let instrumental = try field(input, "instrumental", "true or false") { $0.bool } ?? false
        var lyrics = try field(input, "lyrics", "a string") { $0.string } ?? ""
        // The checkpoint needs lyrics; none (or `instrumental`) is the maker's documented [instrumental] tag.
        if instrumental || lyrics.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty { lyrics = "[instrumental]" }
        guard lyrics.count <= 100_000 else { throw MusicSupport.failure("Lyrics are limited to 100,000 characters.") }
        let duration = try field(input, "duration", "a number") { $0.number } ?? 60
        guard duration.isFinite, duration >= minSeconds, duration <= maxSeconds else {
            throw MusicSupport.failure("Duration must be from 0.04 to 360 seconds.")
        }
        let steps = try field(input, "steps", "a whole number") { $0.number } ?? 30
        guard steps.rounded() == steps, (1...200).contains(steps) else { throw MusicSupport.failure("Steps must be a whole number from 1 to 200.") }
        let guidance = try field(input, "guidance", "a number") { $0.number } ?? 1.7
        guard guidance.isFinite, (0...20).contains(guidance) else { throw MusicSupport.failure("Guidance must be from 0 to 20.") }
        let batch = try field(input, "batch_size", "a whole number") { $0.number } ?? 1
        guard batch.rounded() == batch, (1...16).contains(batch) else { throw MusicSupport.failure("Batch size must be from 1 to 16.") }
        guard duration * batch <= maxTotalSeconds else {
            throw MusicSupport.failure("A request makes at most 720 seconds of music in total (batch size times duration); ask for fewer or shorter songs.")
        }
        let seed: UInt64
        if let value = try field(input, "seed", "a whole number") { $0.number } {
            guard value.rounded() == value, value >= 0, value <= Double(UInt32.max) else { throw MusicSupport.failure("Seed must be a whole number from 0 to 4294967295.") }
            seed = UInt64(value)
        } else {
            seed = UInt64(UInt32.random(in: 0...UInt32.max))
        }
        guard let format = AudioFileFormat(rawValue: try field(input, "format", "a string") { $0.string } ?? "wav") else {
            throw MusicSupport.failure("Unsupported music audio format.")
        }
        var rate: Double?
        if let value = try field(input, "sample_rate", "a number") { $0.number } {
            guard value == 44_100 || value == 32_000 else { throw MusicSupport.failure("Sample rate must be 44100 (native) or 32000.") }
            rate = value == 44_100 ? nil : value
        }
        let metadata = try field(input, "include_metadata", "true or false") { $0.bool } ?? false
        // Stereo 16-bit (32-bit for wav32) PCM for the longest song, plus container overhead, against the job limit.
        let perSecond: Double = (format == .wav32 ? 4 : 2) * 2 * (rate ?? 44_100)
        let pcm = Int64((duration * perSecond).rounded(.up))
        guard (pcm + pcm / 100 + 65_536) * Int64(batch) <= MusicRequests.artifactLimit else { throw MusicSupport.failure("Requested songs exceed the per-job output limit.") }
        return Options(
            request: .init(caption: prompt, lyrics: lyrics, duration: duration, seed: seed, steps: Int(steps), guidance: Float(guidance)),
            batch: Int(batch), format: format, sampleRate: rate, metadata: metadata)
    }

    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "generate_music", let pipeline else { throw MusicSupport.failure("MiniMax Music 3 is not loaded or the operation is unsupported.") }
        try cancellation.check()
        let options = try Self.options(input)
        guard let directory = input["out_dir"]?.string else { throw MusicSupport.failure("The host must supply out_dir.") }
        let root = URL(fileURLWithPath: directory)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        var files: [JSONValue] = []
        do {
            for index in 0..<options.batch {
                try cancellation.check()
                var request = options.request
                request.seed = (options.request.seed &+ UInt64(index)) & 0xFFFF_FFFF
                let output = try pipeline.generate(
                    request, cancelled: { (try? cancellation.check()) == nil },
                    progress: { progress in
                        try? emit(MusicSupport.progress(SongProgress(stage: progress.stage, fraction: progress.fraction, item: index), batch: options.batch))
                    })
                try cancellation.check()
                let audio = SongAudio(channels: [output.left, output.right], sampleRate: Double(pipeline.c.sampleRate))
                guard audio.frameCount > 0, audio.channels.allSatisfy({ $0.count == audio.frameCount && $0.allSatisfy(\.isFinite) }) else {
                    throw MusicSupport.failure("MiniMax Music 3 returned empty or non-finite stereo audio.")
                }
                let name = "song-\(index)-" + UUID().uuidString
                let url = root.appendingPathComponent(name + "." + options.format.fileExtension)
                try AudioFiles.write(audio, to: url, format: options.format, sampleRate: options.sampleRate)
                let mime = options.format == .flac ? "audio/flac" : (options.format == .m4a || options.format == .alac ? "audio/mp4" : "audio/wav")
                files.append(.object(try NativeMedia.file(url, root: root, mime: mime, id: "audio-\(index)")))
                if options.metadata {
                    let receipt: JSONObject = [
                        "seed": .number(Double(request.seed)), "steps": .number(Double(request.steps)), "guidance": .number((Double(request.guidance) * 1000).rounded() / 1000),
                        "duration_limit": .number(request.duration), "frames": .number(Double(output.frames)), "ended_by_model": .bool(output.ended),
                        "prompt_tokens": .number(Double(output.promptTokens)), "seconds": .number(audio.seconds),
                        "sample_rate": .number(options.sampleRate ?? Double(pipeline.c.sampleRate))
                    ]
                    let metadata = root.appendingPathComponent(name + ".json")
                    try WireJSON.encode(receipt).write(to: metadata)
                    files.append(.object(try NativeMedia.file(metadata, root: root, mime: "application/json", id: "metadata-\(index)")))
                }
            }
        } catch {
            try cancellation.check()
            throw error
        }
        return ["files": .array(files)]
    }

    func unload() { pipeline = nil }
}
