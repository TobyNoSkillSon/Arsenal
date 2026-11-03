// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
// Sparse 3D tensor for the MoGe-3 refiner.
//
// MoGe-3 refines log-depth in a voxelised point-coordinate space
// (`moge/model/v3.py:_voxelize`). Every pixel of the points-head output
// becomes one voxel at `(batch, i, j, round(logz * refiner_depth_resolution))`,
// and a sparse 3D UNet (`moge/model/modules/sparse_unet.py`) predicts a
// residual on `logz`.
//
// Upstream builds this on FlexGEMM (Triton + CUDA). Here it is MLX + a single
// custom Metal kernel; see `SparseNeighbor.swift`.
//
// **Coord convention.** Columns are `[b, i, j, z]` — batch, row, column,
// depth bin — matching `_voxelize`'s `torch.stack([batch, i, j, z_idx], -1)`.
// This is *not* the `[b, z, y, x]` order used by spconv / mlx-swift-TRELLIS.
// The kernel tap order below is tied to it, so the two must be changed
// together or not at all.

import Foundation
import MLX

/// A sparse 3D volume: `(N, 4)` int32 coords + `(N, C)` features, plus the
/// ambient sparse shape the coords index into.
public struct SparseTensor {
    /// `(N, 4)` Int32, columns `[b, i, j, z]`.
    public let coords: MLXArray
    /// `(N, C)` features.
    public let features: MLXArray
    /// Ambient extent `[B, S_i, S_j, S_z]` (the `shape[:4]` of
    /// `Sparse3DUNet.forward`'s `shape` argument — the channel entry is
    /// carried by `features` instead).
    public let shape: [Int]

    public init(coords: MLXArray, features: MLXArray, shape: [Int]) {
        precondition(
            coords.ndim == 2 && coords.dim(1) == 4,
            "SparseTensor.coords must be (N, 4), got \(coords.shape)")
        precondition(
            features.ndim == 2,
            "SparseTensor.features must be (N, C), got \(features.shape)")
        precondition(
            coords.dim(0) == features.dim(0),
            "coords/features row mismatch: \(coords.dim(0)) vs \(features.dim(0))")
        precondition(
            shape.count == 4,
            "SparseTensor.shape must be [B, S_i, S_j, S_z], got \(shape)")
        self.coords = coords
        self.features = features
        self.shape = shape
    }

    public var count: Int { coords.dim(0) }
    public var channels: Int { features.dim(1) }

    /// Same coords and shape, new features.
    public func withFeatures(_ f: MLXArray) -> SparseTensor {
        SparseTensor(coords: coords, features: f, shape: shape)
    }
}

/// Bit-field packing of `[b, i, j, z]` coords into a single sortable `Int64`.
///
/// Each spatial field is stored biased by `+1` in its own bit range, wide
/// enough to hold `extent + 1`. The bias matters: neighbour queries probe
/// `coord ± 1`, so `-1` must be representable, and bit-isolated fields mean an
/// out-of-range probe can never alias into an adjacent field the way a
/// mixed-radix packing would. An out-of-range key simply fails to match, which
/// is exactly the "no neighbour" answer we want.
public struct SparseKeyPacking {
    public let shiftB: Int
    public let shiftI: Int
    public let shiftJ: Int

    /// - Parameter shape: ambient extent `[B, S_i, S_j, S_z]`.
    public init(shape: [Int]) {
        precondition(shape.count == 4, "SparseKeyPacking needs [B, S_i, S_j, S_z], got \(shape)")
        // Width must hold the biased value `extent + 1` (max legal coord is
        // `extent - 1`, biased to `extent`), so size for `extent + 2`.
        let wz = Self.bitWidth(shape[3] + 2)
        let wj = Self.bitWidth(shape[2] + 2)
        let wi = Self.bitWidth(shape[1] + 2)
        let wb = Self.bitWidth(max(shape[0], 1) + 1)
        self.shiftJ = wz
        self.shiftI = wz + wj
        self.shiftB = wz + wj + wi
        precondition(
            shiftB + wb <= 62,
            "sparse coords do not fit in an Int64 key: shape \(shape) needs \(shiftB + wb) bits")
    }

    private static func bitWidth(_ value: Int) -> Int {
        var bits = 1
        while (1 << bits) < value { bits += 1 }
        return bits
    }

    /// Pack coords, optionally displaced by a spatial offset. The batch column
    /// is never displaced — neighbourhoods do not cross batch entries, matching
    /// FlexGEMM, which adds the kernel delta only to the trailing 3 columns.
    public func pack(_ coords: MLXArray, offset: (Int, Int, Int) = (0, 0, 0)) -> MLXArray {
        let c = coords.asType(.int64)
        let b = c[0..., 0]
        let i = c[0..., 1] + MLXArray(Int64(offset.0 + 1))
        let j = c[0..., 2] + MLXArray(Int64(offset.1 + 1))
        let z = c[0..., 3] + MLXArray(Int64(offset.2 + 1))
        return (b << shiftB) | (i << shiftI) | (j << shiftJ) | z
    }
}

/// The 27 kernel taps of a 3×3×3 submanifold convolution, in the order
/// FlexGEMM enumerates them.
///
/// `make_conv_kernel_delta` (`flex_gemm/ops/utils.py:73`) is
/// `itertools.product(range(-1, 2), ...)` over the spatial coord columns in
/// order, so with our `[b, i, j, z]` columns `i` varies slowest and `z`
/// fastest. That is the same row-major order as flattening the weight's
/// `(K1, K2, K3)` axes, which is what makes `weight.reshaped([Co, 27, Ci])`
/// line up tap-for-tap.
///
/// Sign is `+delta`: FlexGEMM builds `input_coords[:, None, :] + offsets`
/// (`flex_gemm/ops/neighbor_cache.py:1333`), so no kernel flip is needed.
public let submanifoldTaps3x3x3: [(Int, Int, Int)] = {
    var taps: [(Int, Int, Int)] = []
    taps.reserveCapacity(27)
    for di in -1...1 {
        for dj in -1...1 {
            for dz in -1...1 {
                taps.append((di, dj, dz))
            }
        }
    }
    return taps
}()

// swiftlint:enable all
