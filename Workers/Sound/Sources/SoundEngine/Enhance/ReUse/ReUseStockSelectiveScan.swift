// SPDX-License-Identifier: MIT
// Vireo (new file). Plain-MLX reference for the fixed-16 selective scan that
// SpeechLens only ships as a custom Metal kernel. Used as the stock path of
// the family engine gate (kernel output must match this before the kernel
// serves) and as the fallback when the gate fails or VIREO_KERNELS=off.
// Sequential over L: correct and simple, not fast.

import Foundation
import MLX
import MLXNN

enum StockSelectiveScan {
    /// Same contract as `ReUseMetalSelectiveScan.run`: `deltaRaw` is the
    /// pre-softplus `dtProj(dt)` output, `A_log` is pre `-exp`.
    /// Shapes: deltaRaw/u/z [B, L, D], A_log [D, N], B/C [B, L, N], D [D].
    static func run(
        deltaRaw: MLXArray,
        A_log: MLXArray,
        B: MLXArray,
        C: MLXArray,
        u: MLXArray,
        D: MLXArray,
        z: MLXArray
    ) -> MLXArray {
        let batch = deltaRaw.shape[0]
        let seqLen = deltaRaw.shape[1]
        let dSize = deltaRaw.shape[2]
        let nSize = A_log.shape[1]

        let delta = MLXNN.softplus(deltaRaw.asType(.float32))            // [B, L, D]
        let a = -MLX.exp(A_log.asType(.float32))                          // [D, N]
        let deltaA = MLX.exp(delta.expandedDimensions(axis: -1) * a)      // [B, L, D, N]
        let deltaBu = (delta * u).expandedDimensions(axis: -1)
            * B.expandedDimensions(axis: 2)                                // [B, L, D, N]
        let cExpanded = C.expandedDimensions(axis: 2)                      // [B, L, 1, N]

        var state = MLXArray.zeros([batch, dSize, nSize], type: Float.self)
        var ys = [MLXArray]()
        ys.reserveCapacity(seqLen)
        for t in 0..<seqLen {
            state = deltaA[0..., t] * state + deltaBu[0..., t]
            ys.append((state * cExpanded[0..., t]).sum(axis: -1))          // [B, D]
            // Bound the lazy graph on long sequences.
            if t % 256 == 255 { MLX.eval(state) }
        }
        var y = MLX.stacked(ys, axis: 1)                                  // [B, L, D]
        y = y + u * D
        return y * MLXNN.silu(z)
    }
}
