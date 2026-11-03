//
//  Encoders.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The condition encoder's parts: the lyric encoder (text-encoder token
//  embeddings → 8 bidirectional layers), the timbre encoder (reference-audio
//  latents → 4 layers, first token kept) and the condition encoder that
//  packs lyrics, timbre and the projected caption into one sequence for
//  the DiT's cross-attention. Adapted from mere.run (MIT, see
//  THIRD_PARTY_NOTICES.md); checked against upstream's PyTorch modules.
//

import Foundation
import MLX
import MLXFast
import MLXNN

// Lyric token embeddings `[B, L, 1024]` → `[B, L, 2048]`.
final class LyricEncoder: Module {
    let configuration: ACEStepConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "layers") var layers: [ACEEncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm

    init(_ c: ACEStepConfiguration) {
        configuration = c
        _embedTokens.wrappedValue = Linear(c.textHiddenDim, c.hiddenSize, bias: true)
        _layers.wrappedValue = (0..<c.numLyricEncoderHiddenLayers).map { ACEEncoderLayer(c, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
    }

    // Runs in BF16, as upstream does, and returns the input's dtype (FP16 overflows here; nothing runs in FP32); with the trained weights the last layer's
    // residual stream reaches about 48,000 on "[Instrumental]" and passes FP16's 65,504 from four lyric lines on,
    // which made the encoder output NaN and failed every such song in the diffusion (31 Oct 2026). The normed
    // output stays under 100, so it returns to half precision for the DiT.
    func callAsFunction(_ embeddings: MLXArray, keep: MLXArray) -> MLXArray {
        var x = embedTokens(embeddings.asType(.bfloat16))
        let length = x.dim(1)
        let padded = !AttentionMasks.isAllKept(keep)
        let full = AttentionMasks.bidirectional(length: length, window: nil, keep: padded ? keep : nil, dtype: x.dtype)
        let window = configuration.useSlidingWindow ? configuration.slidingWindow : nil
        let sliding = AttentionMasks.bidirectional(length: length, window: window, keep: padded ? keep : nil, dtype: x.dtype)
        let rope = RotaryTable(length: length, headDim: configuration.headDim, theta: configuration.ropeTheta)
        for layer in layers { x = layer(x, full: full, slidingMask: sliding, rope: rope) }
        return norm(x).asType(embeddings.dtype)
    }
}

// Reference-audio latents `[N, T, 64]` → one embedding per reference, unpacked per batch item.
final class TimbreEncoder: Module {
    let configuration: ACEStepConfiguration

    @ModuleInfo(key: "embed_tokens") var embedTokens: Linear
    @ModuleInfo(key: "layers") var layers: [ACEEncoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ParameterInfo(key: "special_token") var specialToken: MLXArray

    init(_ c: ACEStepConfiguration) {
        configuration = c
        _embedTokens.wrappedValue = Linear(c.timbreHiddenDim, c.hiddenSize, bias: true)
        _layers.wrappedValue = (0..<c.numTimbreEncoderHiddenLayers).map { ACEEncoderLayer(c, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hiddenSize, eps: c.rmsNormEps)
        _specialToken.wrappedValue = MLXArray.zeros([1, 1, c.hiddenSize])
    }

    // - Parameters:
    //   - packed: every reference's latents `[N, T, 64]`.
    //   - order: which batch item each reference belongs to, `N` values.
    // - Returns: `[B, maxRefs, D]` and its keep mask `[B, maxRefs]`.
    func callAsFunction(_ packed: MLXArray, order: [Int]) -> (MLXArray, MLXArray) {
        var x = embedTokens(packed)
        let length = x.dim(1)
        let window = configuration.useSlidingWindow ? configuration.slidingWindow : nil
        let sliding = AttentionMasks.bidirectional(length: length, window: window, dtype: x.dtype)
        let rope = RotaryTable(length: length, headDim: configuration.headDim, theta: configuration.ropeTheta)
        for layer in layers { x = layer(x, full: .none, slidingMask: sliding, rope: rope) }
        let first = norm(x)[0..., 0, 0...] // [N, D]
        let batch = (order.max() ?? 0) + 1
        var groups = Array(repeating: [Int](), count: batch)
        for (index, item) in order.enumerated() { groups[item].append(index) }
        let width = max(1, groups.map(\.count).max() ?? 1)
        var rows: [MLXArray] = []
        var keep: [Int32] = []
        for group in groups {
            var row =
                group.isEmpty
                ? MLXArray.zeros([0, first.dim(1)], dtype: first.dtype)
                : first[MLXArray(group.map(Int32.init))]
            if group.count < width {
                row = MLX.concatenated([row, MLXArray.zeros([width - group.count, first.dim(1)], dtype: first.dtype)], axis: 0)
            }
            rows.append(row.expandedDimensions(axis: 0))
            keep += Array(repeating: 1, count: group.count) + Array(repeating: 0, count: width - group.count)
        }
        return (MLX.concatenated(rows, axis: 0), MLXArray(keep, [batch, width]))
    }
}

// Packs lyrics, timbre and caption into the DiT's cross-attention sequence.
final class ConditionEncoder: Module {
    @ModuleInfo(key: "text_projector") var textProjector: Linear
    @ModuleInfo(key: "lyric_encoder") var lyricEncoder: LyricEncoder
    @ModuleInfo(key: "timbre_encoder") var timbreEncoder: TimbreEncoder

    init(_ c: ACEStepConfiguration) {
        _textProjector.wrappedValue = Linear(c.textHiddenDim, c.hiddenSize, bias: false)
        _lyricEncoder.wrappedValue = LyricEncoder(c)
        _timbreEncoder.wrappedValue = TimbreEncoder(c)
    }

    func callAsFunction(
        text: MLXArray, textKeep: MLXArray, lyrics: MLXArray, lyricKeep: MLXArray,
        references: MLXArray, referenceOrder: [Int]
    ) -> (MLXArray, MLXArray) {
        let projected = textProjector(text)
        let encodedLyrics = lyricEncoder(lyrics, keep: lyricKeep)
        let (timbre, timbreKeep) = timbreEncoder(references, order: referenceOrder)
        let first = SequencePacking.pack(encodedLyrics, timbre, lyricKeep, timbreKeep)
        return SequencePacking.pack(first.0, projected, first.1, textKeep)
    }
}
