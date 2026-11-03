//
//  SequencePacking.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  Upstream's `pack_sequences`: two sequences side by side, then a stable
//  sort so every kept token comes before every padding token.
//

import Foundation
import MLX

enum SequencePacking {

    // `[B, L1+L2, D]` with kept tokens first, and its keep mask.
    static func pack(_ a: MLXArray, _ b: MLXArray, _ keepA: MLXArray, _ keepB: MLXArray) -> (MLXArray, MLXArray) {
        let hidden = MLX.concatenated([a, b], axis: 1)
        let keep = MLX.concatenated([keepA.asType(.int32), keepB.asType(.int32)], axis: 1)
        let batch = keep.dim(0), length = keep.dim(1)
        let values = keep.asArray(Int32.self)
        var order: [Int32] = []
        var newKeep: [Int32] = []
        for row in 0..<batch {
            let slice = values[(row * length)..<((row + 1) * length)]
            let kept = slice.indices.filter { slice[$0] != 0 }.map { Int32($0 - row * length) }
            let dropped = slice.indices.filter { slice[$0] == 0 }.map { Int32($0 - row * length) }
            order += kept + dropped
            newKeep += Array(repeating: 1, count: kept.count) + Array(repeating: 0, count: dropped.count)
        }
        let index = MLX.broadcast(MLXArray(order, [batch, length, 1]), to: [batch, length, hidden.dim(2)])
        return (MLX.takeAlong(hidden, index, axis: 1), MLXArray(newKeep, [batch, length]))
    }
}
