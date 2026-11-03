// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
// The three sparse operators MoGe-3's refiner needs, expressed on top of the
// mappings built in `SparseNeighbor.swift`.
//
// These mirror FlexGEMM's *explicit GEMM* path, which is itself plain tensor
// code — `index_select` + `addmm` in
// `flex_gemm/ops/spconv/functions.py:SparseConvExplicitGemmFunction`. Only the
// index construction was ever CUDA-bound, and that now lives in the Metal
// kernel next door.

import Foundation
import MLX

public enum SparseOps {

    /// Rows per chunk in the submanifold convolution.
    ///
    /// The convolution materialises a `(rows, 27 * Ci)` im2col block. Doing the
    /// whole tensor at once would be ~3 GB at the refiner's finest level
    /// (~900k voxels), so it is chunked; doing one small matmul per tap instead
    /// would avoid the buffer but leaves a `K = Ci = 32` GEMM, which is far
    /// from the hardware's efficient shape. Chunking keeps `K = 27 * Ci` and
    /// caps the scratch at a few hundred MB.
    public static let defaultConvChunkRows = 1 << 16

    /// Submanifold 3×3×3 convolution: output coords equal input coords.
    ///
    /// `out[n] = bias + Σ_v weight[v]ᵀ · features[map[n, v]]`, with a missing
    /// neighbour (`map[n, v] == -1`) contributing zero.
    ///
    /// - Parameters:
    ///   - features: `(N, Ci)`.
    ///   - map: `(N, 27)` Int32 from ``SparseNeighbor/submanifoldMap(_:coords:)``.
    ///   - weight: `(27, Ci, Co)` — taps in the order of ``submanifoldTaps3x3x3``.
    ///   - bias: `(Co,)` or `nil`.
    ///   - chunkRows: rows per im2col block; see ``defaultConvChunkRows``.
    public static func submanifoldConv3d(
        features: MLXArray,
        map: MLXArray,
        weight: MLXArray,
        bias: MLXArray?,
        chunkRows: Int = defaultConvChunkRows
    ) -> MLXArray {
        precondition(chunkRows > 0, "chunkRows must be positive, got \(chunkRows)")
        let n = features.dim(0)
        let ci = features.dim(1)
        let taps = weight.dim(0)
        let co = weight.dim(2)
        precondition(
            map.dim(0) == n && map.dim(1) == taps,
            "neighbour map \(map.shape) does not match features \(features.shape) / weight \(weight.shape)")
        precondition(
            weight.dim(1) == ci,
            "weight in-channels \(weight.dim(1)) != feature channels \(ci)")

        // A zero row at index N absorbs the -1 sentinels, so the gather stays
        // branch-free and missing neighbours contribute nothing.
        let padded = MLX.concatenated(
            [features, MLXArray.zeros([1, ci], dtype: features.dtype)], axis: 0)
        let gatherIndex = MLX.where(map .< MLXArray(Int32(0)), MLXArray(Int32(n)), map)

        // (27, Ci, Co) → (27 * Ci, Co). Row-major, so tap varies slowest and
        // channel fastest — the same order the gathered block is flattened in.
        let rhs = weight.asType(features.dtype).reshaped([taps * ci, co])

        var blocks: [MLXArray] = []
        blocks.reserveCapacity((n + chunkRows - 1) / chunkRows)
        var start = 0
        while start < n {
            let end = min(start + chunkRows, n)
            let rows = end - start
            let gathered = padded[gatherIndex[start..<end, 0...].reshaped([rows * taps])]
            let lhs = gathered.reshaped([rows, taps * ci])
            var block = MLX.matmul(lhs, rhs)
            if let bias {
                block = block + bias.asType(features.dtype)
            }
            // Bound the scratch: without this the whole chunked loop stays
            // lazy and every im2col block is live at once, defeating the point.
            MLX.eval(block)
            blocks.append(block)
            start = end
        }
        if blocks.isEmpty {
            return MLXArray.zeros([0, co], type: Float.self)
        }
        return blocks.count == 1 ? blocks[0] : MLX.concatenated(blocks, axis: 0)
    }

    /// Mean-pool fine features into their coarse cells.
    ///
    /// A true `sum / count`, matching FlexGEMM's `reduce="mean"`. (Note for
    /// anyone cross-reading mlx-swift-TRELLIS: its `SparseDownsample`
    /// deliberately computes `sum / (count + 1)` to reproduce PyTorch's
    /// `scatter_reduce(include_self=True)`. That is a different operator.)
    public static func poolMean(
        features: MLXArray, mapping: SparseNeighbor.PoolMapping
    ) -> MLXArray {
        let n = mapping.fineCount
        let c = features.dim(1)
        let m = mapping.coarseCount
        let k = mapping.members.dim(1)
        let padded = MLX.concatenated(
            [features, MLXArray.zeros([1, c], dtype: features.dtype)], axis: 0)
        let gathered = padded[mapping.members.reshaped([m * k])]
        precondition(
            features.dim(0) == n,
            "feature rows \(features.dim(0)) != mapping fine count \(n)")
        let summed = gathered.reshaped([m, k, c]).sum(axis: 1)
        return summed / mapping.counts.asType(summed.dtype)
    }

    /// Nearest-neighbour upsample: every fine cell takes its coarse parent's
    /// feature. The inverse of ``poolMean(features:mapping:)`` over the same
    /// mapping — the reuse FlexGEMM expresses as `down_cache.T`.
    public static func upsampleNearest(
        coarseFeatures: MLXArray, mapping: SparseNeighbor.PoolMapping
    ) -> MLXArray {
        precondition(
            coarseFeatures.dim(0) == mapping.coarseCount,
            "coarse rows \(coarseFeatures.dim(0)) != mapping coarse count \(mapping.coarseCount)")
        return coarseFeatures[mapping.grouping]
    }
}

// swiftlint:enable all
