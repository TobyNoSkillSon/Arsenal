//
//  YuE2Sampling.swift
//  YuE2Music
//
//  Created by David Sherlock on 2026.
//
//  yue2/sampling.py's `distribution`: only the phase's vocabulary (ordinary
//  text for the score, codec tokens for the song) plus its end token; the
//  end blocked before the minimum; a windowed frequency penalty (penalty^n
//  for a token seen n times in the last `window`); temperature; top-k; top-p
//  keeping the first token. Upstream then draws with torch.multinomial; here
//  the same distribution is drawn with MLX's categorical over the top-k
//  survivors (a seeded Swift run replays; it cannot replay torch's draws).
//

import Foundation
import MLX
import MLXRandom

// One stage's sampling (`Sampling` upstream).
struct YuE2SamplingSettings: Equatable {
    var temperature: Float
    var topP: Float
    var topK: Int
    var repetitionPenalty: Float
    var penaltyWindow: Int
    var minTokens: Int
    var maxTokens: Int
}

enum YuE2Phase { case abc, semantic }

enum YuE2Sampling {

    // `[1, V]` additive masks: 0 where the phase may write, -inf elsewhere.
    static func phaseMask(_ phase: YuE2Phase) -> MLXArray {
        var mask = [Float](repeating: -.infinity, count: YuE2Protocol.vocabularySize)
        switch phase {
        case .abc:
            for i in 0..<YuE2Protocol.eod { mask[i] = 0 }
            mask[YuE2Protocol.abcEnd] = 0
        case .semantic:
            for i in YuE2Protocol.codecOffset..<(YuE2Protocol.codecOffset + YuE2Protocol.codecSize) { mask[i] = 0 }
            mask[YuE2Protocol.musicEnd] = 0
        }
        return MLXArray(mask, [1, YuE2Protocol.vocabularySize])
    }

    // The scores upstream's `distribution` returns (before its draw), as `[1, V]`.
    static func scores(
        _ logits: MLXArray, mask: MLXArray, settings s: YuE2SamplingSettings, history: [Int], step: Int,
        end: Int, legacy: Bool
    ) -> MLXArray {
        var scores = legacy ? logits : logits.asType(.float32)
        // swiftlint:disable:next shorthand_operator
        scores = scores + mask.asType(scores.dtype)
        if step < s.minTokens { scores[0..., end] = MLXArray(-Float.infinity).asType(scores.dtype) }
        if s.repetitionPenalty != 1, !history.isEmpty {
            var counts: [Int: Int] = [:]
            for id in history.suffix(s.penaltyWindow) { counts[id, default: 0] += 1 }
            let ids = counts.keys.sorted()
            let index = MLXArray(ids.map(Int32.init))
            let alpha = MLXArray(ids.map { pow(s.repetitionPenalty, Float(counts[$0]!)) }).reshaped(1, -1).asType(scores.dtype)
            let selected = scores[0..., index]
            scores[0..., index] = MLX.where(selected .< 0, selected * alpha, selected / alpha)
        }
        return scores
    }

    // Temperature, top-k and top-p over the sorted probabilities keeping the first (three under
    // legacy) — the rest of upstream's `distribution` — on the best few hundred only (everything
    // else is -inf upstream anyway). Upstream's top-k keeps every score equal to the k-th, and
    // bfloat16 logits tie often, so the candidates are the k best plus a margin, cut at the k-th
    // score; if even the margin ties, the whole vocabulary is used. Returns the candidate ids,
    // best first, and their filtered scores.
    static func topCandidates(_ scores: MLXArray, settings s: YuE2SamplingSettings, legacy: Bool) -> (ids: MLXArray, scores: MLXArray) {
        var x = scores
        // swiftlint:disable:next shorthand_operator
        if s.temperature != 1 { x = x / MLXArray(s.temperature).asType(x.dtype) }
        let vocabulary = x.dim(-1)
        let k = min(s.topK, vocabulary)
        var m = min(vocabulary, k + 256)
        var top = MLX.argPartition(-x, kth: m - 1, axis: -1)[0..., ..<m]
        var unsorted = MLX.takeAlong(x, top, axis: -1)
        var order = MLX.argSort(-unsorted, axis: -1)
        var values = MLX.takeAlong(unsorted, order, axis: -1)
        if m < vocabulary, values[0, k - 1].item(Float.self) == values[0, m - 1].item(Float.self) {
            m = vocabulary
            top = MLXArray(Int32(0)..<Int32(vocabulary)).reshaped(1, -1)
            unsorted = x
            order = MLX.argSort(-unsorted, axis: -1)
            values = MLX.takeAlong(unsorted, order, axis: -1)
        }
        let ids = MLX.takeAlong(top, order, axis: -1)
        let negInf = MLXArray(-Float.infinity).asType(values.dtype)
        values = MLX.where(values .< values[0..., (k - 1)..<k], negInf, values)
        if s.topP < 1 {
            let probabilities = MLX.softmax(values, axis: -1)
            var removed = (MLX.cumsum(probabilities, axis: -1) - probabilities) .> MLXArray(s.topP).asType(probabilities.dtype)
            removed = MLX.logicalAnd(removed, MLXArray(0..<m).reshaped(1, -1) .>= (legacy ? 3 : 1))
            values = MLX.where(removed, negInf, values)
        }
        return (ids, values)
    }

    // The same as a full `[1, V]` score vector (-inf outside the candidates), as upstream returns it.
    static func filtered(_ scores: MLXArray, settings s: YuE2SamplingSettings, legacy: Bool) -> MLXArray {
        guard s.temperature != 0 else { return scores }
        let (ids, values) = topCandidates(scores, settings: s, legacy: legacy)
        let full = MLX.full(scores.shape, values: MLXArray(-Float.infinity).asType(values.dtype))
        return MLX.putAlong(full, ids, values: values, axis: -1)
    }

    // One token: argmax at temperature 0, else a draw from softmax of the filtered candidates.
    static func draw(_ scores: MLXArray, settings s: YuE2SamplingSettings, legacy: Bool, random: MLXRandom.RandomState) -> Int {
        if s.temperature == 0 { return MLX.argMax(scores, axis: -1).item(Int.self) }
        let (ids, values) = topCandidates(scores, settings: s, legacy: legacy)
        let choice = MLXRandom.categorical(values.asType(.float32), key: random).reshaped(1, 1)
        return MLX.takeAlong(ids, choice, axis: -1).item(Int.self)
    }
}
