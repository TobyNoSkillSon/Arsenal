//
//  AudioTokens.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The 5 Hz semantic codes and the modules either side of them: the audio
//  tokenizer (25 Hz VAE latents → attention-pooled → FSQ code, used for
//  covers) and the detokenizer (codes → 25 Hz "LM hints" the DiT starts
//  from). The quantizer is vector-quantize-pytorch's ResidualFSQ with one
//  layer and `preserve_symmetry`, levels [8, 8, 8, 5, 5, 5] = 64,000 codes.
//  Adapted from mere.run (MIT, see THIRD_PARTY_NOTICES.md).
//

import Foundation
import MLX
import MLXFast
import MLXNN

// One-layer residual FSQ with symmetry-preserving bounds and a hard clamp.
final class ScalarQuantizer: Module {
    @ModuleInfo(key: "project_in") var projectIn: Linear
    @ModuleInfo(key: "project_out") var projectOut: Linear

    let levels: [Int]
    private let _levelsMinusOne: MLXArray
    private let _clampValue: MLXArray
    private let _basis: MLXArray

    init(_ c: ACEStepConfiguration) {
        levels = c.fsqInputLevels
        let d = levels.count
        _projectIn.wrappedValue = Linear(c.hiddenSize, d, bias: true)
        _projectOut.wrappedValue = Linear(d, c.hiddenSize, bias: true)
        let minusOne = levels.map { Float($0 - 1) }
        _levelsMinusOne = MLXArray(minusOne, [1, 1, d])
        _clampValue = MLXArray(minusOne.map { 1 + 1 / $0 }, [1, 1, d])
        var running: Int32 = 1
        var values: [Int32] = []
        for level in levels {
            values.append(running)
            running *= Int32(level)
        }
        _basis = MLXArray(values, [1, 1, d])
    }

    // Codes `[B, T]` for hidden states `[B, T, 2048]`, and their quantized embedding.
    func callAsFunction(_ x: MLXArray) -> (MLXArray, MLXArray) {
        var y = projectIn(x).asType(.float32)
        y = MLX.tanh(y / _clampValue) * _clampValue
        y = MLX.clip(y, min: -1, max: 1)
        let level = MLX.floor(_levelsMinusOne * ((y + 1) / 2) + 0.5)
        let codes = (level.asType(.int32) * _basis).sum(axis: -1)
        let quantized = projectOut((level * (2 / _levelsMinusOne) - 1).asType(x.dtype))
        return (codes, quantized)
    }

    // The quantized embedding `[B, T, 2048]` for codes `[B, T]`.
    func embedding(codes: MLXArray, dtype: DType) -> MLXArray {
        var remaining = codes.asType(.int32)
        var digits: [MLXArray] = []
        for level in levels {
            let l = MLXArray(Int32(level))
            digits.append(MLX.remainder(remaining, l).expandedDimensions(axis: -1))
            remaining = MLX.floorDivide(remaining, l)
        }
        let scalars = MLX.concatenated(digits, axis: -1).asType(.float32) * (2 / _levelsMinusOne) - 1
        return projectOut(scalars.asType(dtype))
    }
}

// Pools each 5-frame patch to one vector with a learned leading token.
final class AttentionPooler: Module {
    let configuration: ACEStepConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_token") var specialToken: MLXArray
    @ModuleInfo(key: "layers") var layers: [ACEEncoderLayer]

    init(_ c: ACEStepConfiguration) {
        configuration = c
        _embedTokens.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _specialToken.wrappedValue = MLXArray.zeros([1, 1, c.hiddenSize])
        _layers.wrappedValue = (0..<c.numAttentionPoolerHiddenLayers).map { ACEEncoderLayer(c, index: $0) }
    }

    // `[B, T, P, D]` → `[B, T, D]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), p = x.dim(2), d = configuration.hiddenSize
        var h = embedTokens(x)
        h = MLX.concatenated([MLX.broadcast(specialToken.asType(h.dtype), to: [b, t, 1, d]), h], axis: 2)
        h = h.reshaped(b * t, p + 1, d)
        let rope = RotaryTable(length: p + 1, headDim: configuration.headDim, theta: configuration.ropeTheta)
        for layer in layers { h = layer(h, full: .none, slidingMask: .none, rope: rope) }
        return norm(h)[0..., 0, 0...].reshaped(b, t, d)
    }
}

// 25 Hz VAE latents → 5 Hz codes (covers).
final class AudioTokenizer: Module {
    @ModuleInfo(key: "audio_acoustic_proj") var audioAcousticProj: Linear
    @ModuleInfo(key: "attention_pooler") var attentionPooler: AttentionPooler
    @ModuleInfo(key: "quantizer") var quantizer: ScalarQuantizer

    init(_ c: ACEStepConfiguration) {
        _audioAcousticProj.wrappedValue = Linear(c.audioAcousticHiddenDim, c.hiddenSize, bias: true)
        _attentionPooler.wrappedValue = AttentionPooler(c)
        _quantizer.wrappedValue = ScalarQuantizer(c)
    }

    // `[B, T5, 5, 64]` → codes `[B, T5]` and quantized `[B, T5, 2048]`.
    func callAsFunction(_ patches: MLXArray) -> (MLXArray, MLXArray) {
        quantizer(attentionPooler(audioAcousticProj(patches)))
    }
}

// 5 Hz quantized embeddings → 25 Hz latent-shaped hints.
final class AudioDetokenizer: Module {
    let configuration: ACEStepConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_tokens") var specialTokens: MLXArray
    @ModuleInfo(key: "layers") var layers: [ACEEncoderLayer]
    @ModuleInfo(key: "proj_out") var projOut: Linear

    init(_ c: ACEStepConfiguration) {
        configuration = c
        _embedTokens.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _specialTokens.wrappedValue = MLXArray.zeros([1, c.poolWindowSize, c.hiddenSize])
        _layers.wrappedValue = (0..<c.numAttentionPoolerHiddenLayers).map { ACEEncoderLayer(c, index: $0) }
        _projOut.wrappedValue = Linear(c.hiddenSize, c.audioAcousticHiddenDim, bias: true)
    }

    // `[B, T5, 2048]` → `[B, T5 * 5, 64]`.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let b = x.dim(0), t = x.dim(1), p = configuration.poolWindowSize, d = configuration.hiddenSize
        var h = embedTokens(x).expandedDimensions(axis: 2)
        h = MLX.broadcast(h, to: [b, t, p, d]) + specialTokens.asType(h.dtype).reshaped(1, 1, p, d)
        h = h.reshaped(b * t, p, d)
        let rope = RotaryTable(length: p, headDim: configuration.headDim, theta: configuration.ropeTheta)
        for layer in layers { h = layer(h, full: .none, slidingMask: .none, rope: rope) }
        h = projOut(norm(h))
        return h.reshaped(b, t * p, configuration.audioAcousticHiddenDim)
    }
}
