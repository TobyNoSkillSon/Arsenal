//
//  Diffusion.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The DiT decoder: patchify (Conv1d, stride 2) the noisy latents beside
//  their context, 24 layers of AdaLN-modulated self-attention (sliding /
//  full alternating), cross-attention to the packed conditions and a
//  SwiGLU MLP, then un-patchify. Mirrors upstream's MLX decoder
//  (acestep/models/mlx/dit_model.py) line for line; structure adapted from
//  mere.run (MIT, see THIRD_PARTY_NOTICES.md).
//

import Foundation
import MLX
import MLXFast
import MLXNN

// Sinusoidal timestep features → MLP → 6 modulation vectors.
final class TimestepEmbedding: Module {
    let inChannels: Int
    let dimension: Int

    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    @ModuleInfo(key: "time_proj") var timeProj: Linear

    init(inChannels: Int = 256, dimension: Int) {
        self.inChannels = inChannels
        self.dimension = dimension
        _linear1.wrappedValue = Linear(inChannels, dimension, bias: true)
        _linear2.wrappedValue = Linear(dimension, dimension, bias: true)
        _timeProj.wrappedValue = Linear(dimension, dimension * 6, bias: true)
    }

    // `t` `[B]` → (`temb` `[B, D]`, `proj` `[B, 6, D]`).
    func callAsFunction(_ t: MLXArray) -> (MLXArray, MLXArray) {
        let half = inChannels / 2
        let freqs = MLX.exp(-Float(Foundation.log(10_000.0)) * MLXArray(0..<half).asType(.float32) / Float(half))
        let args = (t.asType(.float32) * 1000).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let features = MLX.concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
        let temb = linear2(silu(linear1(features.asType(linear1.weight.dtype))))
        let proj = timeProj(silu(temb)).reshaped(t.dim(0), 6, dimension)
        return (temb, proj)
    }
}

// One DiT layer.
final class DiTLayer: Module {
    let sliding: Bool

    @ModuleInfo(key: "self_attn_norm") var selfAttnNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var selfAttn: ACEAttention
    @ModuleInfo(key: "cross_attn_norm") var crossAttnNorm: RMSNorm
    @ModuleInfo(key: "cross_attn") var crossAttn: ACEAttention
    @ModuleInfo(key: "mlp_norm") var mlpNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: ACEMLP
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(_ c: ACEStepConfiguration, index: Int) {
        sliding = c.isSliding(layer: index)
        _selfAttnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _selfAttn.wrappedValue = ACEAttention(c)
        _crossAttnNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _crossAttn.wrappedValue = ACEAttention(c)
        _mlpNorm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _mlp.wrappedValue = ACEMLP(hiddenSize: c.hiddenSize, intermediateSize: c.intermediateSize)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 6, c.hiddenSize])
    }

    func callAsFunction(
        _ x: MLXArray, modulation temb: MLXArray, selfMask: MLXFast.ScaledDotProductAttentionMaskMode,
        rope: RotaryTable, cross: (MLXArray, MLXArray)
    ) -> MLXArray {
        let parts = MLX.split(scaleShiftTable + temb, parts: 6, axis: 1)
        var h = x
        let attended = selfAttn(selfAttnNorm(h) * (1 + parts[1]) + parts[0], mask: sliding ? selfMask : .none, rope: rope)
        // swiftlint:disable:next shorthand_operator
        h = h + attended * parts[2]
        // swiftlint:disable:next shorthand_operator
        h = h + crossAttn(crossAttnNorm(h), mask: .none, keysValues: cross)
        // swiftlint:disable:next shorthand_operator
        h = h + mlp(mlpNorm(h) * (1 + parts[4]) + parts[3]) * parts[5]
        return h
    }
}

// Cross-attention keys and values per layer, computed once per condition and reused every step.
struct CrossAttentionCache {
    var layers: [(MLXArray, MLXArray)]
}

// The diffusion transformer.
final class DiTDecoder: Module {
    let configuration: ACEStepConfiguration

    @ModuleInfo(key: "layers") var layers: [DiTLayer]
    @ModuleInfo(key: "proj_in") var projIn: Conv1d
    @ModuleInfo(key: "proj_out") var projOut: ConvTransposed1d
    @ModuleInfo(key: "time_embed") var timeEmbed: TimestepEmbedding
    @ModuleInfo(key: "time_embed_r") var timeEmbedR: TimestepEmbedding
    @ModuleInfo(key: "condition_embedder") var conditionEmbedder: Linear
    @ModuleInfo(key: "norm_out") var normOut: RMSNorm
    @ParameterInfo(key: "scale_shift_table") var scaleShiftTable: MLXArray

    init(_ c: ACEStepConfiguration) {
        configuration = c
        _layers.wrappedValue = (0..<c.numHiddenLayers).map { DiTLayer(c, index: $0) }
        _projIn.wrappedValue = Conv1d(
            inputChannels: c.inChannels, outputChannels: c.hiddenSize,
            kernelSize: c.patchSize, stride: c.patchSize)
        _projOut.wrappedValue = ConvTransposed1d(
            inputChannels: c.hiddenSize, outputChannels: c.audioAcousticHiddenDim,
            kernelSize: c.patchSize, stride: c.patchSize)
        _timeEmbed.wrappedValue = TimestepEmbedding(dimension: c.hiddenSize)
        _timeEmbedR.wrappedValue = TimestepEmbedding(dimension: c.hiddenSize)
        _conditionEmbedder.wrappedValue = Linear(c.hiddenSize, c.hiddenSize, bias: true)
        _normOut.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _scaleShiftTable.wrappedValue = MLXArray.zeros([1, 2, c.hiddenSize])
    }

    // The cross-attention keys and values for an encoder sequence `[B, L, 2048]`.
    func crossCache(for encoder: MLXArray) -> CrossAttentionCache {
        let projected = conditionEmbedder(encoder)
        return CrossAttentionCache(layers: layers.map { $0.crossAttn.crossKeysValues(projected) })
    }

    // Velocity `[B, T, 64]` for noisy latents `x` `[B, T, 64]` at time `t` `[B]`.
    func callAsFunction(
        _ x: MLXArray, t: MLXArray, tr: MLXArray, context: MLXArray,
        cache: CrossAttentionCache
    ) -> MLXArray {
        let (tembT, projT) = timeEmbed(t)
        let (tembR, projR) = timeEmbedR(t - tr)
        let temb = tembT + tembR
        let modulation = projT + projR
        var h = MLX.concatenated([context, x], axis: -1)
        let length = h.dim(1)
        let pad = (configuration.patchSize - length % configuration.patchSize) % configuration.patchSize
        if pad > 0 {
            h = MLX.concatenated([h, MLXArray.zeros([h.dim(0), pad, h.dim(2)], dtype: h.dtype)], axis: 1)
        }
        h = projIn(h)
        let patches = h.dim(1)
        let rope = RotaryTable(length: patches, headDim: configuration.headDim, theta: configuration.ropeTheta)
        let mask = AttentionMasks.bidirectional(length: patches, window: configuration.slidingWindow, dtype: h.dtype)
        for (index, layer) in layers.enumerated() {
            h = layer(h, modulation: modulation, selfMask: mask, rope: rope, cross: cache.layers[index])
        }
        let out = MLX.split(scaleShiftTable + temb.expandedDimensions(axis: 1), parts: 2, axis: 1)
        h = normOut(h) * (1 + out[1]) + out[0]
        return projOut(h)[0..., ..<length, 0...]
    }
}

// The DiT checkpoint as one module tree: decoder, condition encoder, audio
// tokenizer, detokenizer and the null condition used for guidance.
final class ACEStepNetwork: Module {
    @ModuleInfo(key: "decoder") var decoder: DiTDecoder
    @ModuleInfo(key: "encoder") var encoder: ConditionEncoder
    @ModuleInfo(key: "tokenizer") var tokenizer: AudioTokenizer
    @ModuleInfo(key: "detokenizer") var detokenizer: AudioDetokenizer
    @ParameterInfo(key: "null_condition_emb") var nullCondition: MLXArray

    init(_ c: ACEStepConfiguration) {
        _decoder.wrappedValue = DiTDecoder(c)
        _encoder.wrappedValue = ConditionEncoder(c)
        _tokenizer.wrappedValue = AudioTokenizer(c)
        _detokenizer.wrappedValue = AudioDetokenizer(c)
        _nullCondition.wrappedValue = MLXArray.zeros([1, 1, c.hiddenSize])
    }
}
