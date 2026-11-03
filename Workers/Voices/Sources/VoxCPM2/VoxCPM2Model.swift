// SPDX-License-Identifier: MIT
// VoxCPM2: MiniCPM-4 text-semantic LM + residual acoustic LM, local encoder, local DiT (flow matching) and AudioVAE V2.
// Structure from mlx-audio-swift e8450402 (MIT), Sources/MLXAudioTTS/Models/VoxCPM2/VoxCPM2Model.swift; the generation
// follows the maker's voxcpm/model/voxcpm2.py (OpenBMB/VoxCPM, Apache-2.0): all four prompt layouts (zero-shot,
// reference, continuation, reference + continuation), the context patches of continuation decoding, the length cap
// min(6·target tokens + 10, max_len) and the bad-case retry with seed + 1, none of which the Swift port had.

import Foundation
import MLX
import MLXNN
import MLXRandom
import Tokenizers

/// A request VoxCPM2 cannot serve, or a snapshot it cannot load; `localizedDescription` carries the message.
public struct VoxCPM2Error: LocalizedError, CustomStringConvertible {
    public let message: String
    init(_ message: String) { self.message = message }
    public var description: String { message }
    public var errorDescription: String? { message }
}

final class VoxScalarQuantization: Module {
    let scale: Float
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(inDim: Int, outDim: Int, latentDim: Int, scale: Int) {
        self.scale = Float(scale)
        _inProj.wrappedValue = Linear(inDim, latentDim)
        _outProj.wrappedValue = Linear(latentDim, outDim)
        super.init()
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { outProj(MLX.round(MLX.tanh(inProj(x)) * scale) / scale) }
}

/// Network inputs for one request, unbatched: tokens (S,), masks (S,) float, feats (S, P, D).
public struct VoxCPM2Prompt {
    public var tokens: [Int32]
    public var textMask: [Float]
    public var audioMask: [Float]
    public var feats: MLXArray
    /// Trailing prompt-audio patches that seed decoding in continuation modes (the maker's streaming_prefix_len − 1 = 3).
    public var contextPatches: Int
}

/// How a reference clip is used. `reference` isolates it between ref-audio tokens (timbre only); `continuation` makes
/// it the start of the utterance (needs its transcript); `both` is the maker's "ultimate cloning".
public enum VoxCPM2CloneMode: String, Sendable, CaseIterable {
    case reference, continuation, both
}

/// Decoding state after a prefill or a step.
struct VoxCPM2State {
    var lmCache: [VoxKV]
    var resCache: [VoxKV]
    var lmHidden: MLXArray // (1, H)
    var resHidden: MLXArray // (1, H)
    var prefixCond: MLXArray // (1, P, D)
}

public struct VoxCPM2Generation: Sendable {
    public var cfgValue: Float = 2
    public var inferenceTimesteps = 10
    public var minLength = 2
    public var maxLength = 4096
    public var retryBadcase = true
    public var retryMaxTimes = 3
    public var retryRatio: Float = 6
    public init() {}
}

final class VoxCPM2Model: Module {
    let config: VoxCPM2Configuration
    let patchSize: Int
    let featDim: Int

    @ModuleInfo(key: "base_lm") var baseLM: VoxMiniCPMModel
    @ModuleInfo(key: "residual_lm") var residualLM: VoxMiniCPMModel
    @ModuleInfo(key: "feat_encoder") var featEncoder: VoxCPMLocEnc
    @ModuleInfo(key: "feat_decoder") var featDecoder: VoxUnifiedCFM
    @ModuleInfo(key: "fsq_layer") var fsqLayer: VoxScalarQuantization
    @ModuleInfo(key: "enc_to_lm_proj") var encToLmProj: Linear
    @ModuleInfo(key: "lm_to_dit_proj") var lmToDitProj: Linear
    @ModuleInfo(key: "res_to_dit_proj") var resToDitProj: Linear
    @ModuleInfo(key: "fusion_concat_proj") var fusionConcatProj: Linear
    @ModuleInfo(key: "stop_proj") var stopProj: Linear
    @ModuleInfo(key: "stop_head") var stopHead: Linear
    @ModuleInfo(key: "audio_vae") var audioVAE: VoxAudioVAE

    static let audioStartToken: Int32 = 101
    static let refAudioStartToken: Int32 = 103
    static let refAudioEndToken: Int32 = 104
    static let contextPatches = 3

    var sampleRate: Int { config.audioVaeConfig.outSampleRate }
    var encodeRate: Int { config.audioVaeConfig.sampleRate }
    var dtype: DType { encToLmProj.weight.dtype }

    init(_ config: VoxCPM2Configuration) {
        self.config = config
        patchSize = config.patchSize
        featDim = config.featDim
        let lm = config.lmConfig
        _baseLM.wrappedValue = VoxMiniCPMModel(lm)
        var res = lm
        res.numHiddenLayers = config.residualLmNumLayers; res.vocabSize = 0; res.noRope = config.residualLmNoRope
        _residualLM.wrappedValue = VoxMiniCPMModel(res)
        var enc = lm
        enc.hiddenSize = config.encoderConfig.hiddenDim; enc.intermediateSize = config.encoderConfig.ffnDim
        enc.numAttentionHeads = config.encoderConfig.numHeads; enc.numHiddenLayers = config.encoderConfig.numLayers
        enc.kvChannels = config.encoderConfig.kvChannels; enc.vocabSize = 0
        _featEncoder.wrappedValue = VoxCPMLocEnc(enc, inputDim: config.featDim)
        var dit = lm
        dit.hiddenSize = config.ditConfig.hiddenDim; dit.intermediateSize = config.ditConfig.ffnDim
        dit.numAttentionHeads = config.ditConfig.numHeads; dit.numHiddenLayers = config.ditConfig.numLayers
        dit.kvChannels = config.ditConfig.kvChannels; dit.vocabSize = 0
        _featDecoder.wrappedValue = VoxUnifiedCFM(
            inChannels: config.featDim, estimator: VoxCPMLocDiT(dit, inChannels: config.featDim), meanMode: config.ditConfig.ditMeanMode)
        _fsqLayer.wrappedValue = VoxScalarQuantization(
            inDim: lm.hiddenSize, outDim: lm.hiddenSize, latentDim: config.scalarQuantizationLatentDim, scale: config.scalarQuantizationScale)
        _encToLmProj.wrappedValue = Linear(config.encoderConfig.hiddenDim, lm.hiddenSize)
        _lmToDitProj.wrappedValue = Linear(lm.hiddenSize, config.ditConfig.hiddenDim)
        _resToDitProj.wrappedValue = Linear(lm.hiddenSize, config.ditConfig.hiddenDim)
        _fusionConcatProj.wrappedValue = Linear(lm.hiddenSize * 2, lm.hiddenSize)
        _stopProj.wrappedValue = Linear(lm.hiddenSize, lm.hiddenSize)
        _stopHead.wrappedValue = Linear(lm.hiddenSize, 2, bias: false)
        _audioVAE.wrappedValue = VoxAudioVAE(config.audioVaeConfig)
        super.init()
    }

    // MARK: Weights

    /// Every shard of a snapshot; floating tensors cast to `dtype` in memory, integer buffers kept.
    static func loadWeights(_ directory: URL, dtype: DType) throws -> [String: MLXArray] {
        let files = try FileManager.default.contentsOfDirectory(at: directory, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw VoxCPM2Error("No .safetensors files in \(directory.path)") }
        var all: [String: MLXArray] = [:]
        for file in files {
            for (key, value) in try MLX.loadArrays(url: file) {
                all[key] = value.dtype.isFloatingPoint && value.dtype != dtype ? value.asType(dtype) : value
            }
        }
        return all
    }

    /// Maps the release's names onto this module tree and applies them. Throws when a parameter is left without a tensor.
    func apply(weights raw: [String: MLXArray]) throws {
        var vae: [String: MLXArray] = [:], rest: [String: MLXArray] = [:]
        for (key, value) in raw {
            // Rotary tables are recomputed (MiniCPM.swift); the LocEnc CLS token keeps its file name.
            if key.contains(".rope.") { continue }
            if key.hasPrefix("audio_vae.") { vae[String(key.dropFirst("audio_vae.".count))] = value } else { rest[key] = value }
        }
        for (key, value) in try audioVAE.sanitize(weights: vae) { rest["audio_vae." + key] = value }
        for key in ["audio_vae.decoder.srBoundaries", "audio_vae.decoder._sr_boundaries"] {
            if let bounds = rest.removeValue(forKey: key) { audioVAE.decoder.srBoundaries = bounds.asType(.int32) }
        }
        let expected = Set(parameters().flattened().map(\.0)).subtracting(["audio_vae.decoder.srBoundaries"])
        let missing = expected.subtracting(rest.keys)
        guard missing.isEmpty else { throw VoxCPM2Error("VoxCPM2 weights miss \(missing.count) tensors, e.g. \(missing.sorted().prefix(4))") }
        let unused = Set(rest.keys).subtracting(expected)
        guard unused.isEmpty else { throw VoxCPM2Error("VoxCPM2 weights carry \(unused.count) unknown tensors, e.g. \(unused.sorted().prefix(4))") }
        try update(parameters: ModuleParameters.unflattened(rest), verify: [.shapeMismatch])
    }

    // MARK: Text

    /// The maker's tokenizer: no special tokens; a multi-character all-CJK vocabulary token is fed one character at a
    /// time (mask_multichar_chinese_tokens).
    static func tokenize(_ text: String, tokenizer: Tokenizer) -> [Int] {
        tokenizer.encode(text: text, addSpecialTokens: false).flatMap { id -> [Int] in
            guard let token = tokenizer.convertIdToToken(id) else { return [id] }
            let word = token.replacingOccurrences(of: "▁", with: "")
            let chars = word.unicodeScalars
            guard chars.count >= 2, chars.allSatisfy({ (0x4E00...0x9FFF).contains($0.value) }), tokenizer.convertTokenToId(word) != nil else { return [id] }
            let ids = chars.map { tokenizer.convertTokenToId(String($0)) ?? tokenizer.unknownTokenId ?? id }
            return ids
        }
    }

    /// The maker's core-level cleanup: newlines become spaces, whitespace runs collapse.
    static func cleanText(_ text: String) -> String {
        text.replacingOccurrences(of: "\n", with: " ").replacingOccurrences(of: #"\s+"#, with: " ", options: .regularExpression)
    }

    // MARK: Audio prompt

    /// 16 kHz mono samples → latent patches (T, P, D), padded to whole patches on the right (reference) or left (prompt).
    func encodeAudio(_ samples: [Float], padLeft: Bool) -> MLXArray {
        let patchLength = patchSize * audioVAE.chunkSize
        let pad = (patchLength - samples.count % patchLength) % patchLength
        let padded = padLeft ? [Float](repeating: 0, count: pad) + samples : samples + [Float](repeating: 0, count: pad)
        return encodePadded(MLXArray(padded).reshaped(1, -1, 1))
    }

    /// (1, N, 1) channel-last samples, N a whole number of patches → (T, P, D).
    func encodePadded(_ audio: MLXArray) -> MLXArray {
        let z = audioVAE.encode(audio, sampleRate: encodeRate) // (1, T', D)
        let frames = z.dim(1) / patchSize * patchSize
        return z[0, ..<frames, 0...].reshaped(-1, patchSize, featDim)
    }

    /// The maker's four layouts (`_generate_with_prompt_cache`). `text` is the tokenized text the LM reads: in the
    /// continuation layouts the maker tokenizes prompt text and target as one string, so the join is not two token lists.
    func prompt(text ids: [Int], promptFeat: MLXArray? = nil, refFeat: MLXArray? = nil) -> VoxCPM2Prompt {
        var tokens: [Int32] = [], textMask: [Float] = [], audioMask: [Float] = [], feats: [MLXArray] = []
        let zero = MLXArray.zeros([1, patchSize, featDim], dtype: dtype)
        if let refFeat {
            let n = refFeat.dim(0)
            tokens += [Self.refAudioStartToken] + [Int32](repeating: 0, count: n) + [Self.refAudioEndToken]
            textMask += [1] + [Float](repeating: 0, count: n) + [1]
            audioMask += [0] + [Float](repeating: 1, count: n) + [0]
            feats += [zero, refFeat.asType(dtype), zero]
        }
        let text = ids.map(Int32.init) + [Self.audioStartToken]
        tokens += text
        textMask += [Float](repeating: 1, count: text.count)
        audioMask += [Float](repeating: 0, count: text.count)
        feats.append(MLXArray.zeros([text.count, patchSize, featDim], dtype: dtype))
        var context = 0
        if let promptFeat {
            let n = promptFeat.dim(0)
            tokens += [Int32](repeating: 0, count: n)
            textMask += [Float](repeating: 0, count: n)
            audioMask += [Float](repeating: 1, count: n)
            feats.append(promptFeat.asType(dtype))
            context = min(Self.contextPatches, n)
        }
        return VoxCPM2Prompt(tokens: tokens, textMask: textMask, audioMask: audioMask, feats: MLX.concatenated(feats, axis: 0), contextPatches: context)
    }

    // MARK: Decoding

    func embedFeatures(_ feats: MLXArray) -> MLXArray { encToLmProj(featEncoder(feats)) }

    /// Prefill: (1) LocEnc of every patch, (2) text embeddings (scale_emb only with μP), (3) both LMs over the prompt.
    func prefill(_ p: VoxCPM2Prompt) -> VoxCPM2State {
        let feats = p.feats.expandedDimensions(axis: 0)
        let textMask = MLXArray(p.textMask).asType(dtype).reshaped(1, -1, 1)
        let audioMask = MLXArray(p.audioMask).asType(dtype).reshaped(1, -1, 1)
        let featEmbed = embedFeatures(feats)
        var textEmbed = baseLM.embedTokens!(MLXArray(p.tokens).reshaped(1, -1))
        if config.lmConfig.useMup { textEmbed *= Float(config.lmConfig.scaleEmb) }
        let combined = textMask * textEmbed + audioMask * featEmbed
        let (raw, lmCache) = baseLM(combined, causal: true)
        let enc = fsqLayer(raw) * audioMask + raw * textMask
        let resIn = fusionConcatProj(MLX.concatenated([enc, audioMask * featEmbed], axis: -1))
        let (resOut, resCache) = residualLM(resIn, causal: true)
        return VoxCPM2State(
            lmCache: lmCache, resCache: resCache, lmHidden: enc[0..., -1, 0...], resHidden: resOut[0..., -1, 0...], prefixCond: feats[0..., -1, 0..., 0...])
    }

    /// DiT conditioning for the next patch: (1, 2·H_dit).
    func ditConditioning(_ s: VoxCPM2State) -> MLXArray {
        MLX.concatenated([lmToDitProj(s.lmHidden), resToDitProj(s.resHidden)], axis: -1)
    }

    func stopLogits(_ s: VoxCPM2State) -> MLXArray { stopHead(silu(stopProj(s.lmHidden))) }

    /// Feeds one generated patch (1, P, D) back through LocEnc and both LMs.
    func step(_ s: VoxCPM2State, patch: MLXArray) -> VoxCPM2State {
        let embed = embedFeatures(patch.expandedDimensions(axis: 1)) // (1, 1, H)
        let (lmOut, lmCache) = baseLM(embed, causal: true, cache: s.lmCache)
        let lmHidden = fsqLayer(lmOut[0..., -1, 0...])
        let resIn = fusionConcatProj(MLX.concatenated([lmHidden.expandedDimensions(axis: 1), embed], axis: -1))
        let (resOut, resCache) = residualLM(resIn, causal: true, cache: s.resCache)
        return VoxCPM2State(lmCache: lmCache, resCache: resCache, lmHidden: lmHidden, resHidden: resOut[0..., -1, 0...], prefixCond: patch)
    }

    func samplePatch(_ s: VoxCPM2State, generation: VoxCPM2Generation, noise: MLXArray? = nil) -> MLXArray {
        featDecoder.sample(
            mu: ditConditioning(s), steps: generation.inferenceTimesteps, patchSize: patchSize, cond: s.prefixCond.transposed(0, 2, 1),
            cfgValue: generation.cfgValue, noise: noise
        ).transposed(0, 2, 1)
    }

    /// The maker's `_inference` loop. Returns the generated patches (without the prompt context) and whether the stop
    /// head ended it.
    func generatePatches(_ p: VoxCPM2Prompt, maxLength: Int, generation: VoxCPM2Generation, isCancelled: () -> Bool) throws -> (patches: [MLXArray], stopped: Bool) {
        var state = prefill(p)
        var patches: [MLXArray] = []
        for i in 0..<maxLength {
            if isCancelled() { throw CancellationError() }
            let patch = samplePatch(state, generation: generation)
            let stop = stopLogits(state).argMax(axis: -1)
            eval(patch, stop)
            patches.append(patch)
            if i > generation.minLength && stop.item(Int.self) == 1 { return (patches, true) }
            state = step(state, patch: patch)
            eval(state.lmHidden, state.resHidden)
        }
        return (patches, false)
    }

    /// Latent patches (prompt context first) → 48 kHz samples, with the context's samples cut off.
    func decode(context: MLXArray?, patches: [MLXArray]) -> [Float] {
        var all = patches
        if let context { all.insert(context.expandedDimensions(axis: 0).asType(dtype), at: 0) }
        let latent = MLX.concatenated(all.map { $0.reshaped(1, -1, featDim) }, axis: 1) // (1, T·P, D) channel-last
        let audio = audioVAE.decode(latent).reshaped(-1).asType(.float32)
        eval(audio)
        let skip = (context?.dim(0) ?? 0) * patchSize * audioVAE.decodeChunkSize
        let samples = audio.asArray(Float.self)
        return skip < samples.count ? Array(samples[skip...]) : []
    }
}
