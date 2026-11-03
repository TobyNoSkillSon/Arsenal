// SPDX-License-Identifier: MIT
// Fish Audio Drama 3: a Qwen3.5-MoE backbone, a temporal codebook tower and the Fish S1 DAC codec. Reference: mlx-audio
// dad2087f `fish_drama3/drama3.py`. Each decode step samples the frame's main token (EOS or a semantic token) from the
// backbone, lets the tower sample codebooks 1…9, and feeds the frame back into the backbone. Frames stream through the
// causal codec as they are produced.
import CryptoKit
import Foundation
import MLX
import MLXLMCommon
import MLXNN
import MLXRandom

public enum Drama3Defaults {
    /// Repetition-aware sampling from the training recipe: a semantic main token that repeats one of the last `rasWindow`
    /// main tokens is replaced by a second draw.
    public static let rasWindow = 16
    public static let rasTemperature: Float = 1.0
    public static let rasTopK = 30
    public static let rasTopP: Float = 0.95
    public static let minTemperature: Float = 1e-5
    /// The reference recipe's sampling defaults.
    public static let maxFrames = 1024
    public static let temperature: Float = 0.7
    public static let topK = 30
    public static let topP: Float = 0.7
    public static let prefillChunk = 256
    public static let voiceCacheSize = 4
    /// Streamed audio pieces, in frames (21.5 frames per second): a short first piece, then steady ones.
    public static let firstChunkFrames = 4
    public static let chunkFrames = 8
    static let keptPrefixes = ["backbone.", "semantic_head.", "tower."]
}

/// One id per row of logits [B, V], as the reference samples (`sample_top_k_top_p`). Temperature applies before top-p;
/// the cumulative probability runs over the full vocabulary; the token that crosses `topP` is dropped unless it ranks
/// first. Only the `topK` candidates are sorted, since all others are dropped. Returns uint32 ids [B].
public func drama3SampleTopKTopP(_ logits: MLXArray, temperature: Float, topK: Int, topP: Float, key: MLXArray? = nil) -> MLXArray {
    var x = logits.asType(.float32) / max(temperature, Drama3Defaults.minTemperature)
    // At low temperature the logits share a huge offset that would swamp the log-sum-exp in float32.
    x -= x.max(axis: -1, keepDims: true)
    let k = min(topK, x.dim(-1))
    var ids = argPartition(-x, kth: k - 1, axis: -1)[.ellipsis, ..<k]
    var values = takeAlong(x, ids, axis: -1)
    let order = argSort(-values, axis: -1)
    values = takeAlong(values, order, axis: -1)
    ids = takeAlong(ids, order, axis: -1)
    let probs = exp(values - x.logSumExp(axis: -1, keepDims: true))
    let dropped = (probs.cumsum(axis: -1) .> topP) .&& (MLXArray((0..<k).map { Int32($0) }) .> Int32(0))
    let masked = MLX.where(dropped, MLXArray(-Float.infinity), values)
    let choice = key.map { MLXRandom.categorical(masked, key: $0) } ?? MLXRandom.categorical(masked)
    return takeAlong(ids, expandedDimensions(choice, axis: -1), axis: -1)[.ellipsis, 0]
}

/// What Drama 3 needs from the codec (Fish S1 DAC, 44.1 kHz, 2048 samples per frame, 10 codebooks).
public protocol Drama3Codec: AnyObject {
    var sampleRate: Int { get }
    var frameLength: Int { get }
    /// Mono samples at `sampleRate` → codes [codebooks, frames] int32.
    func encodeCodes(_ samples: [Float]) throws -> MLXArray
    func resetStreaming()
    /// New frames [1, codebooks, n] → their samples [n · frameLength], continuing the stream.
    func decodeStep(_ codes: MLXArray) -> MLXArray
}

/// The model tree the checkpoint stores: `backbone.*`, `semantic_head.*`, `tower.*`.
final class Drama3Network: Module {
    @ModuleInfo(key: "backbone") var backbone: Drama3Backbone
    /// Rows [semantic band…, EOS]: the only tokens a step may emit.
    @ModuleInfo(key: "semantic_head") var semanticHead: Linear
    @ModuleInfo(key: "tower") var tower: Drama3Tower

    init(_ config: Drama3Config) {
        _backbone.wrappedValue = Drama3Backbone(config.text)
        _semanticHead.wrappedValue = Linear(config.text.hiddenSize, config.semanticEnd - config.semanticBegin + 2, bias: false)
        _tower.wrappedValue = Drama3Tower(config.tower, semanticBegin: config.semanticBegin, semanticEnd: config.semanticEnd)
        super.init()
    }
}

/// Encoded references and, once built, the caches of the system block.
final class Drama3Voice {
    let references: [(String, Drama3Codes)]
    var prefix: (backbone: [KVCache], tower: [KVCache])?
    init(references: [(String, Drama3Codes)]) { self.references = references }
}

public struct Drama3Sampling: Equatable, Sendable {
    public var temperature: Float = Drama3Defaults.temperature
    public var topK: Int = Drama3Defaults.topK
    public var topP: Float = Drama3Defaults.topP
    public var maxFrames: Int = Drama3Defaults.maxFrames
    public init(temperature: Float? = nil, topK: Int? = nil, topP: Float? = nil, maxFrames: Int? = nil) {
        if let temperature { self.temperature = temperature }
        if let topK { self.topK = topK }
        if let topP { self.topP = topP }
        if let maxFrames { self.maxFrames = maxFrames }
    }
}

public final class Drama3Model {
    public let config: Drama3Config
    let network: Drama3Network
    let candidateIDs: MLXArray
    var tokenizer: Drama3Tokenizing?
    var codec: Drama3Codec?
    private var voices: [(key: String, voice: Drama3Voice)] = []

    public init(config: Drama3Config) {
        self.config = config
        network = Drama3Network(config)
        candidateIDs = MLXArray(Array(Int32(config.semanticBegin)...Int32(config.semanticEnd)) + [Int32(config.eosID)])
    }

    var codebooks: Int { config.tower.codebooks }
    var tower: Drama3Tower { network.tower }
    var backbone: Drama3TextModel { network.backbone.model }

    // MARK: - Weights

    /// Quantizes the modules the checkpoint stores quantized (they have `.scales`), then loads every tensor; the key sets
    /// must match exactly.
    func load(weights: [String: MLXArray]) throws {
        let kept = weights.filter { key, _ in Drama3Defaults.keptPrefixes.contains { key.hasPrefix($0) } }
        if let quantization = config.quantization {
            quantize(model: network) { path, _ in
                guard kept["\(path).scales"] != nil else { return nil }
                let spec = quantization.spec(path)
                return (spec.groupSize, spec.bits, .affine)
            }
        } else if kept.keys.contains(where: { $0.hasSuffix(".scales") }) {
            throw Drama3Error("checkpoint has quantized tensors but config.json declares no quantization")
        }
        try network.update(parameters: ModuleParameters.unflattened(kept), verify: .all)
    }

    /// Loads the backbone and tower from the `model*.safetensors` shards of a snapshot directory.
    static func shardWeights(_ directory: URL) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(atPath: directory.path)
            .filter { $0.hasPrefix("model") && $0.hasSuffix(".safetensors") }.sorted()
        guard !files.isEmpty else { throw Drama3Error("no model*.safetensors shards in \(directory.path)") }
        var weights: [String: MLXArray] = [:]
        for file in files {
            for (key, value) in try MLX.loadArrays(url: directory.appendingPathComponent(file)) { weights[key] = value }
        }
        return weights
    }

    // MARK: - Voices

    /// Encoded references for a voice. The few most recent voices are kept, keyed by audio and transcript content, so
    /// repeated requests skip encoding and the system-block prefill.
    func voice(references: [([Float], String)]) throws -> Drama3Voice {
        var hash = SHA256()
        for (samples, text) in references {
            for part in [samples.withUnsafeBytes { Data($0) }, Data(text.utf8)] {
                var length = UInt64(part.count).littleEndian
                hash.update(data: Data(bytes: &length, count: 8)); hash.update(data: part)
            }
        }
        let key = hash.finalize().map { String(format: "%02x", $0) }.joined()
        if let index = voices.firstIndex(where: { $0.key == key }) {
            let entry = voices.remove(at: index); voices.append(entry); return entry.voice
        }
        let encoded: [(String, Drama3Codes)] = try references.map { samples, text in
            guard let codec else { throw Drama3Error("codec is not loaded") }
            return (text, Self.hostCodes(try codec.encodeCodes(samples)))
        }
        let voice = Drama3Voice(references: encoded)
        voices.append((key, voice))
        while voices.count > Drama3Defaults.voiceCacheSize { voices.removeFirst() }
        return voice
    }

    static func hostCodes(_ codes: MLXArray) -> Drama3Codes {
        let codes = codes.asType(.int32)
        eval(codes)
        let books = codes.dim(0), frames = codes.dim(1), flat = codes.asArray(Int32.self)
        return Drama3Codes(rows: (0..<books).map { Array(flat[($0 * frames)..<(($0 + 1) * frames)]) })
    }

    // MARK: - Decoding

    func embed(_ frames: MLXArray) -> MLXArray {
        tower.embedBackbone(frames, text: backbone.embedTokens(frames[.ellipsis, 0]))
    }

    /// Feeds rows [start, stop) into both caches; returns row stop-1's backbone hidden state [1, 1, D]. The tower is
    /// teacher-forced with the following row, so it covers [start, min(stop, T - 1)).
    func prefill(_ frames: MLXArray, caches: (backbone: [KVCache], tower: [KVCache]), start: Int, stop: Int) -> MLXArray {
        let frames = frames.expandedDimensions(axis: 0)
        let last = frames.dim(1) - 1
        var hidden: MLXArray?
        for begin in stride(from: start, to: stop, by: Drama3Defaults.prefillChunk) {
            let end = min(begin + Drama3Defaults.prefillChunk, stop)
            let h = backbone(embeddings: embed(frames[0..., begin..<end]), cache: caches.backbone)
            let labelled = min(end, last)
            if labelled > begin {
                tower.prefill(
                    h[0..., ..<(labelled - begin)], frames: frames[0..., begin..<labelled], nextFrames: frames[0..., (begin + 1)..<(labelled + 1)],
                    caches: caches.tower)
            }
            eval([h] + caches.backbone.flatMap(\.state) + caches.tower.flatMap(\.state))
            hidden = h
        }
        return hidden![0..., (hidden!.dim(1) - 1)...]
    }

    /// Fresh caches holding the system block, copied from the voice's snapshot.
    func startCaches(_ frames: MLXArray, systemRows: Int, voice: Drama3Voice) -> (backbone: [KVCache], tower: [KVCache]) {
        if voice.prefix == nil {
            let caches = (backbone: backbone.makeCache(), tower: tower.makeCache())
            _ = prefill(frames, caches: caches, start: 0, stop: systemRows)
            voice.prefix = (caches.backbone.map { $0.copy() }, caches.tower.map { $0.copy() })
        }
        return (voice.prefix!.backbone.map { $0.copy() }, voice.prefix!.tower.map { $0.copy() })
    }

    struct Step {
        var frame: MLXArray, hidden: MLXArray, history: MLXArray, main: MLXArray
        var arrays: [MLXArray] { [frame, hidden, history, main] }
    }

    /// One frame: (frame [1, 1, 11], next hidden, history, main token [1]), all lazy.
    func step(_ hidden: MLXArray, frame: MLXArray, history: MLXArray, caches: (backbone: [KVCache], tower: [KVCache]), sampling: Drama3Sampling)
        -> Step
    {
        let logits = network.semanticHead(hidden[0..., 0]).asType(.float32)
        let first = candidateIDs[drama3SampleTopKTopP(logits, temperature: sampling.temperature, topK: sampling.topK, topP: sampling.topP)]
        let retry = candidateIDs[
            drama3SampleTopKTopP(logits, temperature: Drama3Defaults.rasTemperature, topK: Drama3Defaults.rasTopK, topP: Drama3Defaults.rasTopP)]
        let repeated = (history.expandedDimensions(axis: 0) .== first.expandedDimensions(axis: 1)).any(axis: -1)
        let main = MLX.where(repeated .&& tower.isSemantic(first), retry, first)
        let c0 = clip(main - Int32(config.semanticBegin), min: 0, max: config.tower.vocabSize - 1)
        let semantic = tower.isSemantic(main).expandedDimensions(axis: 1)
        let codes = tower.residualCodes(
            hidden, c0: c0.expandedDimensions(axis: 1), semantic: semantic, frame: frame, caches: caches.tower,
            sample: { headLogits in
                drama3SampleTopKTopP(headLogits[0..., 0], temperature: sampling.temperature, topK: sampling.topK, topP: sampling.topP)
                    .asType(.int32).expandedDimensions(axis: 1)
            })
        let newFrame = concatenated([main.expandedDimensions(axis: 1).asType(.int32)] + codes.map { $0.asType(.int32) }, axis: -1)
            .expandedDimensions(axis: 1)
        let nextHidden = backbone(embeddings: embed(newFrame), cache: caches.backbone)
        let newHistory = concatenated([history[1...], main.asType(history.dtype)])
        return Step(frame: newFrame, hidden: nextHidden, history: newHistory, main: main)
    }

    /// Codes [codebooks] of each generated frame until EOS. Step n + 1 is queued before step n's EOS is read, so the GPU
    /// never waits for the host; the one wasted step after EOS only touches this call's caches.
    func frames(
        _ prompt: Drama3Prompt, voice: Drama3Voice, sampling: Drama3Sampling, isCancelled: () -> Bool, emit: (MLXArray) async throws -> Void
    ) async throws {
        let frames = MLXArray(prompt.values, [prompt.count, prompt.width])
        let caches = startCaches(frames, systemRows: prompt.systemRows, voice: voice)
        let hidden = prefill(frames, caches: caches, start: prompt.systemRows, stop: prompt.count)
        let window = min(Drama3Defaults.rasWindow, prompt.count)
        var pending = step(
            hidden, frame: frames[(prompt.count - 1)...].expandedDimensions(axis: 0), history: frames[(prompt.count - window)..., 0], caches: caches,
            sampling: sampling)
        asyncEval(pending.arrays)
        for n in 0..<sampling.maxFrames {
            if isCancelled() { return }
            let current = pending
            if n + 1 < sampling.maxFrames {
                pending = step(current.hidden, frame: current.frame, history: current.history, caches: caches, sampling: sampling)
                asyncEval(pending.arrays)
            }
            if current.main.item(Int32.self) == Int32(config.eosID) { return }
            try await emit(current.frame[0, 0, 1...])
        }
    }

    /// Speech for `text`, in the voice of the references (each a mono clip at the codec rate with its exact transcript).
    /// Long text is generated chunk by chunk; each chunk sees the first and the previous chunk as history, and
    /// `sampling.maxFrames` caps the frames of one chunk. Audio is emitted as it streams through the codec.
    public func generate(
        text: String, references: [([Float], String)], sampling: Drama3Sampling = Drama3Sampling(), seed: UInt64? = nil,
        isCancelled: () -> Bool = { false }, emit: ([Float]) async throws -> Void
    ) async throws {
        guard let tokenizer, let codec else { throw Drama3Error("Drama 3 is not loaded") }
        try Drama3Text.validate(text, location: "text")
        guard sampling.maxFrames > 0, sampling.topK > 0, sampling.topP > 0, sampling.temperature >= 0 else { throw Drama3Error("invalid sampling settings") }
        if let seed {
            // The request's own random state, scoped to this task: draws made elsewhere in the process (another task, a
            // test suite) can't land between the seed and the samples. Same sequence as MLXRandom.seed(seed).
            return try await withRandomState(MLXRandom.RandomState(seed: seed)) {
                try await generate(text: text, references: references, sampling: sampling, seed: nil, isCancelled: isCancelled, emit: emit)
            }
        }
        let voice = try voice(references: references)
        codec.resetStreaming()
        // The codec's stream buffers stay on the device otherwise until the next request.
        defer { codec.resetStreaming() }
        var pinned: (String, Drama3Codes)?, latest: (String, Drama3Codes)?
        let chunks = Drama3Text.split(text)
        for (index, chunk) in chunks.enumerated() {
            let history = [pinned, latest].compactMap { $0 }
            let prompt = try Drama3PromptBuilder.build(
                references: voice.references, history: history, target: chunk, tokenizer: tokenizer, codebooks: codebooks,
                vocab: config.tower.vocabSize, semanticBegin: config.semanticBegin)
            var codes: [MLXArray] = [], emitted = 0
            func flush() async throws {
                guard codes.count > emitted else { return }
                let audio = codec.decodeStep(stacked(Array(codes[emitted...]), axis: 1).expandedDimensions(axis: 0)).asType(.float32)
                eval(audio)
                emitted = codes.count
                try await emit(audio.asArray(Float.self))
            }
            try await frames(prompt, voice: voice, sampling: sampling, isCancelled: isCancelled) { frame in
                codes.append(frame)
                let due = index == 0 && emitted == 0 ? Drama3Defaults.firstChunkFrames : Drama3Defaults.chunkFrames
                if codes.count - emitted >= due { try await flush() }
            }
            if isCancelled() { return }
            try await flush()
            if !codes.isEmpty {
                let turn = (chunk, Self.hostCodes(stacked(codes, axis: 1)))
                if pinned == nil { pinned = turn } else { latest = turn }
            }
        }
    }

    /// Drops every cached voice (their caches hold device memory).
    public func clearVoices() { voices.removeAll() }
}
