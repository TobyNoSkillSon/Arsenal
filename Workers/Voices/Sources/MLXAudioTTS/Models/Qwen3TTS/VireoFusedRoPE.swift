// SPDX-License-Identifier: MIT
// Ported from Vireo 9366da5e14dcafe846fc41565c94f4f145eaf7cc; shader now belongs to the model folder.
import Foundation
import ArsenalKernels
@preconcurrency import MLX

/// Vireo fast path: the rotate-half RoPE of Qwen3-TTS (`q * cos + rotateHalf(q) * sin`, same for k) as one Metal
/// dispatch for q and k instead of ~12 MLX kernels (two slices, a negate, a concat, two multiplies and an add each).
/// It replays MLX's rounding: every product and the sum are rounded to the activation dtype exactly where the
/// separate MLX ops round (`T a = x * c` is what MLX's Multiply kernel computes), so the output is bit-identical;
/// the model's self-test checks that on this machine before the path serves.
///
/// Layout: q `[B, L, Hq, D]` and k `[B, L, Hk, D]` before the head transpose (elementwise, so the layout does not
/// change any value); cos/sin `[Bc, L, D]` with Bc = 1 or B, already in the activation dtype.
enum VireoFusedRoPE {
    /// Returns (q, k) rotated, same shapes and dtype as the inputs.
    static func apply(q: MLXArray, k: MLXArray, cos: MLXArray, sin: MLXArray, candidate: FolderMLXKernel? = nil) -> (MLXArray, MLXArray) {
        guard let kernel = candidate ?? VireoFolderKernels.current?.active(VireoFolderKernels.ropeSite) else {
            return (stock(q, cos: cos, sin: sin), stock(k, cos: cos, sin: sin))
        }
        let (B, L, Hq, D) = (q.dim(0), q.dim(1), q.dim(2), q.dim(3))
        let Hk = k.dim(2)
        let cosC = cos.reshaped([-1, L, D])
        let sinC = sin.reshaped([-1, L, D])
        let dims = MLXArray([Int32(B), Int32(L), Int32(Hq), Int32(Hk), Int32(D), Int32(cosC.dim(0))])
        let total = B * L * (Hq + Hk) * D
        guard let outputs = kernel(
            [q, k, cosC, sinC, dims],
            template: [("T", q.dtype)],
            grid: (total, 1, 1),
            threadGroup: (Swift.min(256, total), 1, 1),
            outputShapes: [q.shape, k.shape],
            outputDTypes: [q.dtype, k.dtype]
        ) else { return (stock(q, cos: cos, sin: sin), stock(k, cos: cos, sin: sin)) }
        return (outputs[0], outputs[1])
    }

    static func stock(_ x: MLXArray, cos c: MLXArray, sin s: MLXArray) -> MLXArray {
        let t = x.transposed(0, 2, 1, 3)
        let half = t.dim(-1) / 2
        let rot = concatenated([-t[.ellipsis, half...], t[.ellipsis, ..<half]], axis: -1)
        return (t * expandedDimensions(c, axis: 1) + rot * expandedDimensions(s, axis: 1)).transposed(0, 2, 1, 3)
    }

    /// Kernel check against the stock expression on fixed inputs (bf16 and fp32, B = 1, L = 1 and 3,
    /// Hq ≠ Hk): returns the number of differing elements (0 = bit-identical).
    static func unitCheck() -> Int {
        var differing = 0
        for dtype in [DType.bfloat16, .float32] {
            for L in [1, 3] {
                let q = MLXRandom.normal([1, L, 4, 128], key: MLXRandom.key(1)).asType(dtype)
                let k = MLXRandom.normal([1, L, 2, 128], key: MLXRandom.key(2)).asType(dtype)
                let angles = MLXRandom.uniform(low: -3.0, high: 3.0, [1, L, 128], key: MLXRandom.key(3))
                let c = MLX.cos(angles).asType(dtype), s = MLX.sin(angles).asType(dtype)
                let (fq, fk) = apply(q: q, k: k, cos: c, sin: s)
                for (a, b) in [(fq, stock(q, cos: c, sin: s)), (fk, stock(k, cos: c, sin: s))] {
                    differing += (a .!= b).asType(.int32).sum().item(Int.self)
                }
            }
        }
        return differing
    }
}
