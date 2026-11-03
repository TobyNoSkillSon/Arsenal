// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
// Neighbour-map construction for the MoGe-3 sparse refiner.
//
// Everything here reduces to one primitive: *exact* lookup of a batch of keys
// in a sorted key set. That is the same strategy FlexGEMM's own device-agnostic
// reference takes (`torch.sort` + `torch.searchsorted` in
// `_compute_neighbor_cache_torch`) rather than the CUDA hashmap, so we stay
// aligned with upstream instead of inventing a third scheme.
//
// It runs as a single JIT'd Metal kernel because the map has to be rebuilt on
// every refinement step at every level — the refiner mutates `z`, so no map
// survives a step — and at resolution level 9 the finest level carries ~900k
// voxels. A host-side dictionary build was measured (in the sibling
// mlx-swift-TRELLIS port, on an M5 Max) at 48 ms for 97k voxels versus 6.6 ms
// for the convolution itself; at MoGe's scale it would be the entire runtime.

import Foundation
import MLX
import MLXFast

/// A coord set prepared for lookup: keys sorted ascending, plus the
/// permutation that maps a sorted position back to its original row.
public struct SortedCoordIndex {
    /// `(N,)` Int64 packed keys, ascending.
    public let sortedKeys: MLXArray
    /// `(N,)` Int32 — `order[k]` is the original row at sorted position `k`.
    public let order: MLXArray
    public let packing: SparseKeyPacking
    public let count: Int

    public init(coords: MLXArray, shape: [Int]) {
        let packing = SparseKeyPacking(shape: shape)
        let keys = packing.pack(coords)
        let order = MLX.argSort(keys, axis: 0).asType(.int32)
        self.sortedKeys = keys[order]
        self.order = order
        self.packing = packing
        self.count = coords.dim(0)
    }
}

public enum SparseNeighbor {

    // MARK: - The one kernel

    /// Batched exact binary search: for each query key, the position of an
    /// equal key in `sorted_keys`, or `-1`.
    ///
    /// Deliberately *only* a search — no submanifold-symmetry back-fill. That
    /// optimisation (writing the partner cell at tap `V-1-v`) is valid only
    /// when input and output coord sets coincide; the pool and upsample
    /// mappings below share this kernel and would be silently corrupted by it.
    private static let searchKernel: MLXFast.MLXFastKernel = MLXFast.metalKernel(
        name: "sparse_key_search",
        inputNames: ["sorted_keys", "queries", "sizes"],
        outputNames: ["out_pos"],
        source: """
            uint tid = thread_position_in_grid.x;
            int num_queries = sizes[1];
            if (tid >= (uint)num_queries) {
                return;
            }
            int lo = 0;
            int hi = sizes[0] - 1;
            int64_t key = queries[tid];
            int found = -1;
            while (lo <= hi) {
                int mid = (lo + hi) >> 1;
                int64_t probe = sorted_keys[mid];
                if (probe == key) {
                    found = mid;
                    break;
                } else if (probe < key) {
                    lo = mid + 1;
                } else {
                    hi = mid - 1;
                }
            }
            out_pos[tid] = found;
            """
    )

    /// Look `queries` up in `index`, returning `(Q,)` Int32 original row
    /// indices with `-1` for misses.
    public static func lookup(_ index: SortedCoordIndex, queries: MLXArray) -> MLXArray {
        let q = queries.size
        if q == 0 {
            return MLXArray.zeros([0], type: Int32.self)
        }
        let sizes = MLXArray([Int32(index.count), Int32(q)])
        let threadGroup = min(256, q)
        let positions = searchKernel(
            [index.sortedKeys, queries.reshaped([q]), sizes],
            grid: (q, 1, 1),
            threadGroup: (threadGroup, 1, 1),
            outputShapes: [[q]],
            outputDTypes: [.int32]
        )[0]

        // Sorted position → original row. `order` is only defined for hits, so
        // clamp the -1s before gathering and re-apply the sentinel after.
        let safe = MLX.maximum(positions, MLXArray(Int32(0)))
        let rows = index.order[safe]
        return MLX.where(positions .< 0, MLXArray(Int32(-1)), rows)
    }

    // MARK: - Submanifold neighbour map

    /// `(N, 27)` Int32 map for a 3×3×3 submanifold convolution: entry
    /// `[n, v]` is the row holding `coords[n] + tap[v]`, or `-1`.
    ///
    /// Output coords equal input coords, so the same index serves as both the
    /// key set and the query origin.
    public static func submanifoldMap(_ index: SortedCoordIndex, coords: MLXArray) -> MLXArray {
        // One tap at a time, rather than packing all 27 query sets and doing a
        // single launch. Same total work, but the Int64 query buffer is the
        // memory high-water mark here: at the refiner's finest level (~900k
        // voxels) holding all 27 at once is ~400 MB, versus ~7 MB per tap. That
        // matters — this package also targets iOS and visionOS.
        var columns: [MLXArray] = []
        columns.reserveCapacity(submanifoldTaps3x3x3.count)
        for tap in submanifoldTaps3x3x3 {
            let queries = index.packing.pack(coords, offset: tap)
            let hits = lookup(index, queries: queries)
            MLX.eval(hits) // let the query buffer go before building the next
            columns.append(hits)
        }
        // (N, 27), tap-major per row — matching the row-major flatten of the
        // weight's (K1, K2, K3) axes.
        return MLX.stacked(columns, axis: 1)
    }

    // MARK: - Pool / upsample mapping

    /// The correspondence between a fine coord set and its `factor`-pooled
    /// coarse set. Serves the downsample (scatter-mean) and, reused verbatim,
    /// the paired upsample (gather) — the same reuse FlexGEMM gets by passing
    /// `down_cache.T` to `SparseUpsample3d`.
    public struct PoolMapping {
        /// `(M, 4)` Int32 coarse coords, ascending in packed-key order.
        public let coarseCoords: MLXArray
        /// Coarse ambient extent `[B, ceil(S_i/f), ceil(S_j/f), ceil(S_z/f)]`.
        public let coarseShape: [Int]
        /// `(M, K)` Int32 gather matrix, `K = factor³`. Row `m` lists the fine
        /// rows pooled into coarse cell `m`, padded with `N` (the index of a
        /// zero row appended to the features).
        public let members: MLXArray
        /// `(M, 1)` Float member counts.
        public let counts: MLXArray
        /// `(N,)` Int32 — for each fine row, its coarse cell.
        public let grouping: MLXArray
        public let fineCount: Int
        public let coarseCount: Int
    }

    /// Build the pooling correspondence for `coords` at the given factor.
    ///
    /// A fixed-width `(M, factor³)` member matrix is used rather than a
    /// cumulative-sum segment reduction. The cumsum formulation is tempting —
    /// it is branch-free and needs no scatter — but summing ~900k rows in fp32
    /// and then differencing two ~1e6-magnitude partial sums to recover the sum
    /// of at most 8 values throws away most of the mantissa. A pooling group
    /// can hold at most `factor³` members by construction, so the dense gather
    /// is both exact and bounded.
    public static func poolMapping(
        coords: MLXArray, shape: [Int], factor: Int
    ) -> PoolMapping {
        precondition(factor > 1, "pool factor must be > 1, got \(factor)")
        let n = coords.dim(0)
        let coarseShape = [
            shape[0],
            (shape[1] + factor - 1) / factor,
            (shape[2] + factor - 1) / factor,
            (shape[3] + factor - 1) / factor
        ]

        // Floor-divide the spatial columns; the batch column passes through.
        let c = coords.asType(.int32)
        let coarse = MLX.concatenated(
            [
                c[0..., 0..<1],
                MLX.floorDivide(c[0..., 1...], MLXArray(Int32(factor)))
            ], axis: 1)

        let packing = SparseKeyPacking(shape: coarseShape)
        let keys = packing.pack(coarse)
        let order = MLX.argSort(keys, axis: 0).asType(.int32)
        let sortedKeys = keys[order]

        // Segment starts: a sorted position begins a new cell when its key
        // differs from its predecessor. Position 0 always does.
        let isNew: MLXArray
        if n <= 1 {
            isNew = MLXArray.ones([n], type: Int32.self)
        } else {
            let differs = (sortedKeys[1...] .!= sortedKeys[..<(n - 1)]).asType(.int32)
            isNew = MLX.concatenated([MLXArray.ones([1], type: Int32.self), differs], axis: 0)
        }
        let groupOfSorted = MLX.cumsum(isNew, axis: 0) - MLXArray(Int32(1)) // 0-based
        let coarseCount = n == 0 ? 0 : groupOfSorted[n - 1].item(Int.self) + 1

        // First sorted position of each cell. Taken from the boundary flags
        // rather than by scattering positions into a per-cell slot: advanced
        // index assignment gives no ordering guarantee among duplicate
        // addresses, so a scatter would pick an arbitrary member, not the first.
        let sortedPos = MLX.arange(n)
        let segStart = compact(sortedPos, keep: isNew, count: coarseCount)

        // Rank within the cell. The sorted positions of one cell are
        // contiguous, so rank is just the offset from the cell's start.
        let rank = sortedPos - segStart[groupOfSorted]

        // members[cell, rank] = original fine row; unused slots address the
        // zero row that `sparsePoolMean` appends to the features.
        let k = factor * factor * factor
        let membersFlat = MLXArray.full([coarseCount * k], values: MLXArray(Int32(n)), type: Int32.self)
        membersFlat[groupOfSorted * MLXArray(Int32(k)) + rank] = order
        let members = membersFlat.reshaped([coarseCount, k])

        // Counts from consecutive segment starts.
        let segEndExclusive = MLX.concatenated(
            [
                segStart[1...],
                MLXArray([Int32(n)])
            ], axis: 0)
        let counts = (segEndExclusive - segStart).asType(.float32).reshaped([coarseCount, 1])

        // Coarse coords, one per cell, in ascending key order.
        let coarseCoords = coarse[order][segStart]

        // Fine row → coarse cell, back in original row order.
        let grouping = MLXArray.zeros([n], type: Int32.self)
        grouping[order] = groupOfSorted

        return PoolMapping(
            coarseCoords: coarseCoords,
            coarseShape: coarseShape,
            members: members,
            counts: counts,
            grouping: grouping,
            fineCount: n,
            coarseCount: coarseCount
        )
    }

    /// Select the entries of `values` where `keep` is non-zero, given the known
    /// output length. `keep` doubles as its own scatter address via a prefix
    /// sum, so no host round-trip beyond `count` (already known) is needed.
    private static func compact(_ values: MLXArray, keep: MLXArray, count: Int) -> MLXArray {
        if count == 0 {
            return MLXArray.zeros([0], type: Int32.self)
        }
        // Prefix sum over the keep flags is each kept element's destination.
        // Dropped elements would otherwise collide with the preceding kept one,
        // so they are diverted to a scratch slot past the end.
        let destination = MLX.cumsum(keep, axis: 0) - MLXArray(Int32(1))
        let address = MLX.where(keep .!= MLXArray(Int32(0)), destination, MLXArray(Int32(count)))
        let scratchPadded = MLXArray.zeros([count + 1], type: Int32.self)
        scratchPadded[address] = values
        return scratchPadded[..<count]
    }
}

// swiftlint:enable all
