//
//  YuE2Pipeline.swift
//  YuE2Music
//
//  Created by David Sherlock on 2026.
//
//  yue2_infer's YuE2Pipeline on MLX: plan a score (ABC) on the AR path,
//  write 25 Hz song tokens after it (with the optional guidance branch), then
//  re-read prefix and tokens on the AR path once per context chunk and solve
//  the 64-channel acoustic latents with 32 midpoint steps on the NAR path,
//  and decode them in tiles with the Oobleck decoder. Arsenal runs the
//  transformer and decoder in the selected half (or packed) precision.
//

import Foundation
import MLX
import MLXFast
import MLXLMCommon
import MLXNN
import MLXRandom

final class YuE2Pipeline {
    let network: YuE2Network
    let tokenizer: Tiktoken
    private(set) var decoders: [YuE2Decoder: OobleckDecoder] = [:]
    let folder: URL

    static let latentChannels = 64
    static let hop = 1920
    static let halo = 16

    init(network: YuE2Network, decoder: OobleckDecoder, tokenizer: Tiktoken, folder: URL) {
        self.network = network; self.tokenizer = tokenizer; self.folder = folder
        decoders[.standard] = decoder
    }

    // The decoder's Oobleck configuration (config.json `decoder_config`).
    static let decoderConfiguration = OobleckConfiguration(
        audioChannels: 2, channelMultiples: [1, 2, 4, 8, 16, 32], decoderChannels: 64, decoderInputChannels: 64,
        downsamplingRatios: [2, 2, 4, 4, 5, 6], encoderHiddenSize: 64, samplingRate: 48_000)

    // A decoder, loaded on first use.
    func decoder(_ which: YuE2Decoder) throws(YuE2Error) -> OobleckDecoder {
        if let loaded = decoders[which] { return loaded }
        throw .modelMissing("The selected snapshot contains only the standard YuE2 decoder.")
    }

    // `decoder.layers.N…` (an nn.Sequential) → the shared Oobleck decoder's names, weight norm folded.
    // `layers.0` conv1 · `layers.1…6` block.0…5 (`layers.0` snake1, `.1` conv_t1, `.2…4` res_unit1…3,
    // each `layers.0` snake1, `.1` conv1, `.2` snake2, `.3` conv2) · `layers.7` snake1 · `layers.8` conv2.
    static func decoderName(_ key: String) -> String? {
        let parts = key.split(separator: ".").map(String.init)
        guard parts.count >= 3, parts[0] == "layers", let top = Int(parts[1]) else { return nil }
        let leaf = parts.dropFirst(2).joined(separator: ".")
        switch top {
        case 0: return "conv1." + leaf
        case 7: return "snake1." + leaf
        case 8: return "conv2." + leaf
        case 1...6:
            guard parts.count >= 5, parts[2] == "layers", let inner = Int(parts[3]) else { return nil }
            let rest = Array(parts.dropFirst(4))
            let block = "block.\(top - 1)."
            switch inner {
            case 0: return block + "snake1." + rest.joined(separator: ".")
            case 1: return block + "conv_t1." + rest.joined(separator: ".")
            case 2...4:
                guard rest.count >= 3, rest[0] == "layers", let unit = Int(rest[1]) else { return nil }
                let names = ["snake1", "conv1", "snake2", "conv2"]
                guard unit < names.count else { return nil }
                return block + "res_unit\(inner - 1).\(names[unit])." + rest.dropFirst(2).joined(separator: ".")
            default: return nil
            }
        default: return nil
        }
    }

    // MARK: AR — the score and the song tokens

    struct Tokens {
        var ids: [Int]
        var truncated: Bool
    }

    func newCaches() -> [KVCacheSimple] { network.backbone.layers.map { _ in KVCacheSimple() } }

    // Logits `[1, V]` (bfloat16) for the last of `ids`, appended at `offset`.
    func arLogits(_ ids: [Int], caches: [KVCacheSimple], offset: Int) -> MLXArray {
        let rope = YuERotary(positions: offset..<offset + ids.count, headDim: 128, base: 1_000_000)
        var x = network.backbone.embedTokens(MLXArray(ids.map(Int32.init)).reshaped(1, -1))
        let mask: MLXFast.ScaledDotProductAttentionMaskMode = ids.count > 1 ? .causal : .none
        for (layer, cache) in zip(network.backbone.layers, caches) {
            x = layer.autoregressive(x, rope: rope, cache: cache, mask: mask)
        }
        return network.lmHead(network.backbone.norm(x[0..., (ids.count - 1)...]))[0..., 0, 0...]
    }

    // `generate_tokens`: one stage of sampling, guided when `negative` is given.
    func generate(
        prefix: [Int], negative: [Int]?, guidance: Float, settings: YuE2SamplingSettings, seed: UInt64,
        phase: YuE2Phase, legacy: Bool, cancelled: () -> Bool, onToken: (Int) -> Void
    ) throws(YuE2Error) -> Tokens {
        guard prefix.count + settings.maxTokens <= YuE2Protocol.context else {
            throw .invalidRequest("prompt (\(prefix.count) tokens) + \(settings.maxTokens) tokens exceeds YuE2's 24,576-token context")
        }
        if let negative, negative.count + settings.maxTokens > YuE2Protocol.context {
            throw .invalidRequest("the guidance prompt + generation budget exceeds YuE2's context")
        }
        let end = phase == .abc ? YuE2Protocol.abcEnd : YuE2Protocol.musicEnd
        let mask = YuE2Sampling.phaseMask(phase)
        let random = MLXRandom.RandomState(seed: seed)
        let positive = newCaches()
        let unconditionalCaches = guidance != 1 && negative != nil ? newCaches() : nil
        var conditional = arLogits(prefix, caches: positive, offset: 0)
        var unconditional = unconditionalCaches.map { arLogits(negative!, caches: $0, offset: 0) }
        var positiveOffset = prefix.count, negativeOffset = negative?.count ?? 0
        var history: [Int] = []
        for step in 0..<settings.maxTokens {
            if cancelled() { throw .cancelled }
            var logits = conditional
            if let unconditional {
                let g = MLXArray(guidance).asType(.float16)
                logits = unconditional + g * (conditional - unconditional)
            }
            let scores = YuE2Sampling.scores(
                logits, mask: mask, settings: settings, history: history, step: step,
                end: end, legacy: legacy)
            let token = YuE2Sampling.draw(scores, settings: settings, legacy: legacy, random: random)
            onToken(token)
            if token == end { return Tokens(ids: history, truncated: false) }
            history.append(token)
            if step + 1 < settings.maxTokens {
                conditional = arLogits([token], caches: positive, offset: positiveOffset)
                positiveOffset += 1
                if let caches = unconditionalCaches {
                    unconditional = arLogits([token], caches: caches, offset: negativeOffset)
                    negativeOffset += 1
                }
            }
        }
        return Tokens(ids: history, truncated: true)
    }

    // MARK: NAR — acoustic latents

    // One context chunk: the AR path's keys and values over prefix + codec + MUSIC_END, fixed through the ODE.
    struct AcousticChunk {
        let keys: [MLXArray]
        let values: [MLXArray]
        let rope: YuERotary
        let positions: MLXArray
        let frames: Int
    }

    func prefillAcoustic(_ arTokens: [Int], frames: Int) -> AcousticChunk {
        let rope = YuERotary(positions: 0..<arTokens.count, headDim: 128, base: 1_000_000)
        var x = network.backbone.embedTokens(MLXArray(arTokens.map(Int32.init)).reshaped(1, -1))
        var keys: [MLXArray] = [], values: [MLXArray] = []
        for layer in network.backbone.layers {
            let (out, k, v) = layer.prefill(x, rope: rope)
            x = out
            keys.append(k)
            values.append(v)
            MLX.eval(x, k, v)
        }
        let narLength = frames + 2
        let narRope = YuERotary(positions: arTokens.count..<arTokens.count + narLength, headDim: 128, base: 1_000_000)
        let positions = network.latentPositions.pe[0..<narLength].expandedDimensions(axis: 0)
        return AcousticChunk(keys: keys, values: values, rope: narRope, positions: positions, frames: frames)
    }

    // `CachedNAR.velocity`: `state` `[frames, 64]` bfloat16 → velocity `[frames, 64]`.
    func velocity(_ state: MLXArray, rawT: Double, chunk: AcousticChunk) -> MLXArray {
        let n = chunk.frames + 2
        let padded = MLX.padded(state, widths: [IntOrPair((1, 1)), IntOrPair((0, 0))])
        // `_shift_t_value` with shift 1: sigmoid in the model's dtype.
        let shifted = MLX.sigmoid(MLXArray(Float(rawT)).asType(.float16))
        var x = network.vae2llm(padded.expandedDimensions(axis: 0))
        // swiftlint:disable:next shorthand_operator
        x = x + network.timeEmbedder(MLX.broadcast(shifted, to: [n])).expandedDimensions(axis: 0)
        // swiftlint:disable:next shorthand_operator
        x = x + chunk.positions
        for (i, layer) in network.backbone.layers.enumerated() {
            x = layer.acoustic(x, rope: chunk.rope, arKeys: chunk.keys[i], arValues: chunk.values[i])
        }
        return network.llm2vae(network.backbone.norm(x))[0, 1..<(n - 1)]
    }

    // `torch.logit(t).clamp(-20, 20)` in double.
    static func rawTime(_ t: Double) -> Double { min(20, max(-20, Foundation.log(t / (1 - t)))) }

    // The midpoint solve of one chunk from `noise` `[frames, 64]` float32; returns float32 latents.
    func solve(
        _ chunk: AcousticChunk, noise: MLXArray, steps: Int, cancelled: () -> Bool,
        progress: (Int) -> Void
    ) throws(YuE2Error) -> MLXArray {
        var state = noise.asType(.float16)
        let dt = 1.0 / Double(steps)
        for step in 0..<steps {
            if cancelled() { throw .cancelled }
            let t = 1.0 - Double(step) * dt
            let first = velocity(state, rawT: Self.rawTime(t), chunk: chunk)
            let mid = state - first * Float(dt / 2)
            // swiftlint:disable:next shorthand_operator
            state = state - velocity(mid, rawT: Self.rawTime(t - dt / 2), chunk: chunk) * Float(dt)
            MLX.eval(state)
            progress(step + 1)
        }
        return state.asType(.float16)
    }

    // `nar.synthesize`: chunks at upstream's cuts, one noise draw for the whole song.
    func synthesize(
        prefix: [Int], codec: [Int], noise: MLXArray, steps: Int, cancelled: () -> Bool,
        progress: (Double) -> Void
    ) throws(YuE2Error) -> MLXArray {
        let ranges = YuE2Protocol.chunks(frames: codec.count, prefixTokens: prefix.count)
        guard !ranges.isEmpty else { throw .generationFailed("no song tokens to render") }
        var output: [MLXArray] = []
        for (index, range) in ranges.enumerated() {
            let ar = prefix + codec[range].map { $0 + YuE2Protocol.codecOffset } + [YuE2Protocol.musicEnd]
            let chunk = prefillAcoustic(ar, frames: range.count)
            let latents = try solve(
                chunk, noise: noise[range.lowerBound..<range.upperBound], steps: steps,
                cancelled: cancelled
            ) { done in
                progress((Double(index) + Double(done) / Double(steps)) / Double(ranges.count))
            }
            output.append(latents)
            MLX.Memory.clearCache()
        }
        let latents = MLX.concatenated(output, axis: 0)
        guard MLX.isNaN(latents).any().item(Bool.self) == false, MLX.isInf(latents).any().item(Bool.self) == false else {
            throw .generationFailed("acoustic flow matching produced non-finite latents")
        }
        return latents
    }

    // MARK: Decoding

    // The decoder's natural output length for `frames` latent frames.
    static func outputLength(frames: Int) -> Int {
        var length = frames
        for stride in [6, 5, 4, 4, 2, 2] { length = (length - 1) * stride - 2 * ((stride + 1) / 2) + 2 * stride }
        return length
    }

    // `decode_tiled`: cores of `core` frames with 16 frames of context either side, cropped exactly; then clamp.
    func decode(
        _ latents: MLXArray, decoder which: YuE2Decoder, core: Int, cancelled: () -> Bool,
        progress: (Double) -> Void
    ) throws(YuE2Error) -> [[Float]] {
        let decoder = try decoder(which)
        let frames = latents.dim(0)
        let total = Self.outputLength(frames: frames)
        var pieces: [MLXArray] = []
        let starts = Array(stride(from: 0, to: frames, by: core))
        for (index, start) in starts.enumerated() {
            if cancelled() { throw .cancelled }
            let end = min(frames, start + core)
            let left = max(0, start - Self.halo), right = min(frames, end + Self.halo)
            let tile = decoder(latents[left..<right].asType(.float16).expandedDimensions(axis: 0))[0] // [samples, 2]
            let outStart = start * Self.hop, outEnd = min(end * Self.hop, total)
            let cropStart = (start - left) * Self.hop
            guard cropStart + (outEnd - outStart) <= tile.dim(0) else {
                throw .generationFailed("a decoder tile did not cover its core")
            }
            let crop = tile[cropStart..<cropStart + (outEnd - outStart)]
            MLX.eval(crop)
            pieces.append(crop)
            MLX.Memory.clearCache()
            progress(Double(index + 1) / Double(starts.count))
        }
        let audio = MLX.clip(MLX.concatenated(pieces, axis: 0), min: -1, max: 1).transposed(1, 0)
        MLX.eval(audio)
        return (0..<audio.dim(0)).map { audio[$0].asArray(Float.self) }
    }
}
