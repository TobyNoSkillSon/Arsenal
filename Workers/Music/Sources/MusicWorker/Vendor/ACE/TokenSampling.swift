//
//  TokenSampling.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  mlx_lm's `make_sampler` as upstream calls it: log-probabilities, then
//  top-p (sorted cumulative mass, ascending), then top-k, then
//  `categorical(logprobs / temperature)`; temperature 0 is argmax. The
//  same operations in the same order, so a seeded run draws the same
//  Gumbel noise as Python's.
//

import Foundation
import MLX
import MLXRandom

enum TokenSampling {

    // The deterministic part — log-softmax, top-p, top-k, divide by temperature —
    // compiled once per setting, as mlx_lm compiles it; the draw stays outside.
    static func filter(temperature: Float, topP: Float?, topK: Int?) -> @Sendable (MLXArray) -> MLXArray {
        compile { logits in
            var logprobs = logits - MLX.logSumExp(logits, keepDims: true)
            if let topP, topP > 0, topP < 1 { logprobs = applyTopP(logprobs, topP) }
            if let topK, topK > 0, topK < logprobs.dim(-1) { logprobs = applyTopK(logprobs, topK) }
            return logprobs * (1 / temperature)
        }
    }

    // One token through a compiled filter; temperature 0 is argmax.
    static func sample(
        _ logits: MLXArray, filter: @Sendable (MLXArray) -> MLXArray, temperature: Float,
        random: MLXRandom.RandomState
    ) -> MLXArray {
        guard temperature > 0 else { return MLX.argMax(logits, axis: -1) }
        return MLXRandom.categorical(filter(logits), key: random)
    }

    // One token from `logits` `[1, V]`.
    static func sample(
        _ logits: MLXArray, temperature: Float, topP: Float?, topK: Int?,
        random: MLXRandom.RandomState
    ) -> MLXArray {
        var logprobs = logits - MLX.logSumExp(logits, keepDims: true)
        guard temperature > 0 else { return MLX.argMax(logprobs, axis: -1) }
        if let topP, topP > 0, topP < 1 { logprobs = applyTopP(logprobs, topP) }
        if let topK, topK > 0, topK < logprobs.dim(-1) { logprobs = applyTopK(logprobs, topK) }
        return MLXRandom.categorical(logprobs * (1 / temperature), key: random)
    }

    static func applyTopP(_ logprobs: MLXArray, _ topP: Float) -> MLXArray {
        let probs = MLX.exp(logprobs)
        let order = MLX.argSort(logprobs, axis: -1)
        let sorted = MLX.takeAlong(probs, order, axis: -1)
        let cumulative = MLX.cumsum(sorted, axis: -1)
        let positions = MLXArray(Int32(0)..<Int32(order.dim(-1))).reshaped(1, -1).asType(order.dtype)
        let inverse = MLX.putAlong(MLX.zeros(like: order), order, values: positions, axis: -1)
        let original = MLX.takeAlong(cumulative, inverse, axis: -1)
        return MLX.where(original .> (1 - topP), logprobs, MLXArray(-Float.infinity))
    }

    static func applyTopK(_ logprobs: MLXArray, _ topK: Int) -> MLXArray {
        let masked = MLX.argPartition(-logprobs, kth: topK - 1, axis: -1)[.ellipsis, topK...]
        return MLX.putAlong(logprobs, masked, values: MLXArray(-Float.infinity).asType(logprobs.dtype), axis: -1)
    }
}
