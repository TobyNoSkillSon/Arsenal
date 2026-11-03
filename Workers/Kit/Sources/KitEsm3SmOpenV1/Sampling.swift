// SPDX-License-Identifier: MIT
// Token sampling and the iterative unmasking order, ported from esm/utils/sampling.py and esm/utils/generation.py
// (esm commit 8f0834c). Foundation only, on the host in Double over float32 logits.
import Foundation

enum Sampling {
    /// log_softmax of one row; -inf entries stay -inf.
    static func logSoftmax(_ row: [Double]) -> [Double] {
        let m = row.max() ?? 0
        guard m.isFinite else { return row.map { _ in -Double.infinity } }
        let lse = m + Foundation.log(row.reduce(0) { $0 + Foundation.exp($1 - m) })
        return row.map { $0 - lse }
    }

    /// Categorical entropy of a log-probability row (0 log 0 = 0).
    static func entropy(_ logp: [Double]) -> Double {
        let normalised = logSoftmax(logp)
        return -normalised.reduce(0) { $0 + ($1.isFinite ? Foundation.exp($1) * $1 : 0) }
    }

    /// top_p_logits: keep the smallest prefix (by descending logit) whose softmax mass is <= top_p, at least one;
    /// the rest become float32's most negative value.
    static func topP(_ row: [Double], _ p: Double) -> [Double] {
        guard p < 1 else { return row }
        let order = row.indices.sorted { row[$0] != row[$1] ? row[$0] > row[$1] : $0 < $1 }
        let probs = logSoftmax(order.map { row[$0] }).map { Foundation.exp($0) }
        var out = row, mass = 0.0
        for (rank, index) in order.enumerated() {
            mass += probs[rank]
            if rank > 0 && mass > p { out[index] = -Double(Float.greatestFiniteMagnitude) }
        }
        return out
    }

    /// sample_logits for one row: invalid ids masked, then top-p, then argmax at temperature 0 or a draw from
    /// softmax(logits / T). Returns the token and the log-probabilities the entropy uses (masked, before temperature).
    static func sample(_ logits: [Double], valid: Set<Int>, temperature: Double, topP p: Double, rng: inout SeededGenerator) -> (Int, [Double]) {
        var row = logits.indices.map { valid.contains($0) ? logits[$0] : -Double.infinity }
        row = topP(row, p)
        let logp = logSoftmax(row)
        if temperature == 0 {
            let best = row.indices.max { row[$0] != row[$1] ? row[$0] < row[$1] : $0 > $1 } ?? 0
            return (best, logp)
        }
        let scaled = logSoftmax(row.map { $0 / temperature })
        let u = Double.random(in: 0 ..< 1, using: &rng)
        var acc = 0.0
        for (i, lp) in scaled.enumerated() where lp.isFinite {
            acc += Foundation.exp(lp)
            if u < acc { return (i, logp) }
        }
        return (scaled.indices.last { scaled[$0].isFinite } ?? 0, logp)
    }

    /// sample_function_logits for one position's [8][260] logits: log_softmax(logits / T); <none> when its mean
    /// probability over the 8 depths is above 0.05, else the argmax of the other tokens at each depth. T = 0 is taken
    /// as its limit (one-hot probabilities); the maker's code divides by zero there and returns <pad>.
    static func function(_ logits: [[Double]], temperature: Double, topP p: Double) -> (ids: [Int], logp: [[Double]]) {
        let none = ESM3Vocab.functionNone
        var logp = logits.map { row -> [Double] in
            let r = topP(row, p)
            if temperature == 0 {
                let best = r.indices.max { r[$0] != r[$1] ? r[$0] < r[$1] : $0 > $1 } ?? 0
                return r.indices.map { $0 == best ? 0 : -Double.infinity }
            }
            return logSoftmax(r.map { $0 / temperature })
        }
        let pNone = logp.reduce(0) { $0 + Foundation.exp($1[none]) } / Double(logp.count)
        if pNone > 0.05 { return ([Int](repeating: none, count: logp.count), logp) }
        for d in logp.indices { logp[d][none] = -Double.infinity }
        let ids = logp.map { row in row.indices.max { row[$0] != row[$1] ? row[$0] < row[$1] : $0 > $1 } ?? 0 }
        return (ids, logp)
    }

    /// _get_iterative_sampling_mask_for_prompt_and_step: which masked positions to fill at this step. `candidates` are
    /// the positions still masked (BOS/EOS excluded), `entropy` per position (strategy "entropy": lowest first, ties
    /// by position), else a seeded random choice.
    static func choose(candidates: [Int], count: Int, strategy: String, entropy: [Double], rng: inout SeededGenerator) -> [Int] {
        let n = Swift.min(Swift.max(count, 0), candidates.count)
        if strategy == "entropy" {
            return Array(candidates.sorted { entropy[$0] != entropy[$1] ? entropy[$0] < entropy[$1] : $0 < $1 }.prefix(n))
        }
        var shuffled = candidates
        shuffled.shuffle(using: &rng)
        return Array(shuffled.prefix(n))
    }
}
