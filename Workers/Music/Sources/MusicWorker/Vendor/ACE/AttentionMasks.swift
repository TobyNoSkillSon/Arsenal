//
//  AttentionMasks.swift
//  ACEStepMusic
//
//  Created by David Sherlock on 2026.
//
//  The additive masks upstream builds for its bidirectional encoders
//  (`create_4d_mask`) and the DiT's sliding window: 0 where a query may
//  look, a large negative number where it may not. Pure.
//

import Foundation
import MLX
import MLXFast

enum AttentionMasks {

    // The value a masked position gets (upstream's MLX decoder uses -1e9).
    static let blocked: Float = -1e9

    // A bidirectional mask `[B or 1, 1, L, L]`: keys within `window` of the
    // query (when given) and, with `keep` (`[B, L]`, 1 = real token), only
    // real keys. `.none` when nothing is masked.
    static func bidirectional(
        length: Int, window: Int?, keep: MLXArray? = nil,
        dtype: DType = .float32
    ) -> MLXFast.ScaledDotProductAttentionMaskMode {
        var allowed: MLXArray?
        if let window {
            let index = MLXArray(Int32(0)..<Int32(length))
            let distance = MLX.abs(index.reshaped(length, 1) - index.reshaped(1, length))
            allowed = (distance .<= MLXArray(Int32(window))).reshaped(1, 1, length, length)
        }
        if let keep {
            let keys = (keep .> 0).reshaped(keep.dim(0), 1, 1, length)
            allowed = allowed.map { MLX.logicalAnd($0, keys) } ?? MLX.broadcast(keys, to: [keep.dim(0), 1, length, length])
        }
        guard let allowed else { return .none }
        let zeros = MLXArray.zeros(allowed.shape, dtype: dtype)
        return .array(MLX.where(allowed, zeros, MLXArray(blocked).asType(dtype)))
    }

    // Whether every position of a `[B, L]` keep mask is real.
    static func isAllKept(_ keep: MLXArray) -> Bool {
        keep.min().item(Int32.self) > 0
    }
}
