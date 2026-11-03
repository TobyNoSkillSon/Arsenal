// SPDX-License-Identifier: MIT
import Foundation
import Hub
import IntegrationKit
import MLX
import MLXNN
import ModelPrecision
import ArsenalWire
import Tokenizers

final class SAMAdapter: AudioAdapter {
    override var architecture: String { "sam_audio" }
    override var operations: Set<String> { ["separate_audio"] }
    var textGraph: T5Encoder?
    private var chunkSeconds: Float = 10
    var tokenizer: (any Tokenizer)?
    override var parameters: [String: MLXArray] {
        var result = super.parameters
        for (name, value) in textGraph?.parameters().flattened() ?? [] { result["text_encoder." + name] = value }
        return result
    }
    override func unload() { textGraph = nil; tokenizer = nil; super.unload() }
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        chunkSeconds = context.randomWeights ? Float(context.config["lab_chunk_seconds"]?.number ?? 10) : 10
        guard chunkSeconds > 0, chunkSeconds <= 10 else { throw audioFailure("Invalid SAM chunk size.") }
        let configuration = try config(context, as: SAMAudioConfig.self)
        let model = SAMAudio(config: configuration)
        let expected = Dictionary(uniqueKeysWithValues: model.parameters().flattened().map { ($0.0, $0.1.shape) })
        try prepare(
            model, context, cancellation: cancellation,
            map: { sourceName, tensor in
                let name = audioLSTMKey(sourceName)
                let weightName = name.hasSuffix(".scales") ? String(name.dropLast(7)) + ".weight" : name.hasSuffix(".biases") ? String(name.dropLast(7)) + ".weight" : name
                guard let shape = expected[weightName] else { throw audioFailure("Unknown SAM checkpoint parameter: " + name) }
                if name == weightName && tensor.dtype != .uint32 && tensor.shape != shape { throw audioFailure("SAM checkpoint shape differs: " + name) }
                return (name, tensor)
            })
        let textDirectory: URL
        let textConfiguration: T5ModelConfig
        if context.randomWeights {
            textDirectory = context.folder.appendingPathComponent("gate/tokenizer")
            textConfiguration =
                context.config["lab_text"]?.bool == true
                ? T5ModelConfig(vocabSize: 512, dModel: configuration.textEncoder.dim, dKV: 16, dFF: 128, numLayers: 1, numHeads: 4) : T5ModelConfig()
        } else {
            // Get prepared the encoder half of the pinned T5-base asset inside the selected snapshot.
            textDirectory = context.weights.appendingPathComponent(SAMConversion.textDirectory)
            textConfiguration = try JSONDecoder().decode(T5ModelConfig.self, from: Data(contentsOf: textDirectory.appendingPathComponent("config.json")))
        }
        let text = T5Encoder(config: textConfiguration)
        if context.randomWeights {
            let helper = AudioAdapter()
            try helper.prepare(text, context, cancellation: cancellation)
            tokenizer = try localTokenizer(textDirectory)
        } else {
            _ = try NativePrecisionTier(context.precision).load(
                text, archives: [textDirectory.appendingPathComponent("model.safetensors")],
                map: { name, tensor in
                    guard let pair = T5Encoder.sanitize(weights: [name: tensor]).first else { return nil }
                    return (pair.key, pair.value)
                }, eligible: audioEligible)
            let config = Data("{\"tokenizer_class\":\"T5Tokenizer\",\"eos_token\":\"</s>\",\"pad_token\":\"<pad>\",\"unk_token\":\"<unk>\",\"model_max_length\":512}".utf8)
            tokenizer = try AutoTokenizer.from(
                tokenizerConfig: JSONDecoder().decode(Config.self, from: config),
                tokenizerData: JSONDecoder().decode(Config.self, from: Data(contentsOf: textDirectory.appendingPathComponent("tokenizer.json"))), strict: true)
        }
        textGraph = text
        try NativeParameters.requireHalf(parameters)
    }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "prompt", "mode", "seed", "steps", "anchors"])
        guard operation == "separate_audio", let model = graph as? SAMAudio, let textGraph, let tokenizer,
            let prompt = input["prompt"]?.string, !prompt.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
        else { throw audioFailure("SAM Audio requires a nonempty text prompt.") }
        guard input["stems"] == nil else { throw audioFailure("SAM Audio separates a prompted sound rather than named stems.") }
        let mode = input["mode"]?.string ?? "extract", steps = try audioInteger(input, key: "steps", fallback: 16, range: 2...100)
        guard ["extract", "remove"].contains(mode), (2...100).contains(steps) else { throw audioFailure("Use extract/remove and 2...100 separation steps.") }
        if input["seed"] != nil { MLXRandom.seed(UInt64(try audioInteger(input, key: "seed", fallback: 0, range: 0...Int(UInt32.max)))) }
        try cancellation.check()
        let pcm = try audioInput(input, rate: model.sampleRate, maximumFrames: min(model.sampleRate * 600, (NativeFiles.maxBytes - 88) / 4))
        let ids = tokenizer.encode(text: prompt, addSpecialTokens: true)
        guard !ids.isEmpty else { throw audioFailure("The text prompt tokenized to no tokens.") }
        let clipped = Array(ids.prefix(model.config.textEncoder.maxLength ?? 512))
        let tokenIDs = MLXArray(clipped.map(Int32.init)).reshaped(1, -1)
        let mask = MLXArray.ones(tokenIDs.shape, dtype: .bool)
        let text = textGraph(inputIDs: tokenIDs, attentionMask: mask).asType(.float16)
        model.cancellationCheck = { try cancellation.check() }
        defer { model.cancellationCheck = nil }
        let result: SAMAudioSeparationResult
        let audio = MLXArray(pcm.samples).reshaped(1, 1, -1).asType(.float16)
        if let entries = input["anchors"]?.array {
            guard !entries.isEmpty, entries.count <= 32, pcm.samples.count <= 10 * pcm.sampleRate else {
                throw audioFailure("Temporal anchors support 1...32 spans on at most ten seconds of audio.")
            }
            let duration = Double(pcm.samples.count) / Double(pcm.sampleRate)
            var anchors: [SAMAudioAnchor] = []
            for entry in entries {
                guard let object = entry.object, Set(object.keys).isSubset(of: ["start", "end", "positive"]),
                    let start = entry["start"]?.number, let end = entry["end"]?.number, start.isFinite, end.isFinite, start >= 0, end > start, end <= duration
                else { throw audioFailure("Anchor times must be ordered seconds within the audio duration.") }
                anchors.append((entry["positive"]?.bool == false ? "-" : "+", Float(start), Float(end)))
            }
            let sizes = MLXArray([model.audioCodec.wavIdxToFeatureIdx(pcm.samples.count)]).asType(.int32)
            let audioMask = SAMAudioProcessor.maskFromSizes(sizes)
            let alignment = model.processor.processAnchors([anchors], audioPadMask: audioMask, batchSize: 1)
            result = try model.separate(
                audios: audio, descriptions: [prompt], sizes: sizes, anchorIDs: alignment.anchorIDs, anchorAlignment: alignment.anchorAlignment, audioPadMask: audioMask,
                ode: SAMAudioODEOptions(stepSize: 1 / Float(steps)), _textFeatures: text, _textMask: mask)
        } else {
            result = try model.separateLong(
                audios: audio, descriptions: [prompt], chunkSeconds: chunkSeconds, overlapSeconds: chunkSeconds * 0.3, ode: SAMAudioODEOptions(stepSize: 1 / Float(steps)),
                _textFeatures: text, _textMask: mask)
        }
        guard let target = result.target.first, let residual = result.residual.first, target.size >= pcm.samples.count, residual.size >= pcm.samples.count else {
            throw audioFailure("SAM Audio omitted its output pair.")
        }
        let first = try audioWave(
            Array(target.reshaped(-1).asArray(Float.self).prefix(pcm.samples.count)), rate: pcm.sampleRate, channels: 1, id: "target", input: input, cancellation: cancellation)
        let second = try audioWave(
            Array(residual.reshaped(-1).asArray(Float.self).prefix(pcm.samples.count)), rate: pcm.sampleRate, channels: 1, id: "residual", input: input, cancellation: cancellation)
        return ["files": .array(mode == "extract" ? [first, second] : [second, first])]
    }
}
