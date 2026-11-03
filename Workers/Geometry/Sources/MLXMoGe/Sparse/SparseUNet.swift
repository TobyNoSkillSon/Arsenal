// swiftlint:disable all
// Vendored inference source; see PORTED-FROM.md.
// The MoGe-3 sparse 3D UNet refiner.
//
// A port of `moge/model/modules/sparse_unet.py` (`Sparse3DUNet`) and the blocks
// it is built from in `moge/model/modules/flex_sparse_blocks.py`. The module
// knows nothing about point maps or log-depth: it takes a sparse volume plus a
// dense conditioning map and returns a per-voxel residual. `MoGeModelV3` owns
// the interpretation.
//
// Parameter keys match the PyTorch module tree one-for-one, so a converted
// checkpoint loads without remapping — including `fuse_proj.0` / `fuse_proj.2`
// straddling the `nn.Sequential` activation at index 1.

import Foundation
import MLX
import MLXNN

// MARK: - Submanifold convolution

/// `flex_gemm.nn.SubmanifoldConv3d` with a 3×3×3 kernel.
///
/// Retain the selected checkpoint's `(Co, 3, 3, 3, Ci)` shape. The forward
/// rearranges it to `(27, Ci, Co)` for the sparse contraction.
public class SubmanifoldConv3d: Module {
    @ParameterInfo(key: "weight") public var weight: MLXArray
    @ParameterInfo(key: "bias") public var bias: MLXArray?

    public init(inChannels: Int, outChannels: Int, bias: Bool = true) {
        self._weight.wrappedValue = MLXArray.zeros([outChannels, 3, 3, 3, inChannels], type: Float.self)
        self._bias.wrappedValue = bias ? MLXArray.zeros([outChannels], type: Float.self) : nil
        super.init()
    }

    public func callAsFunction(_ x: MLXArray, map: MLXArray) -> MLXArray {
        SparseOps.submanifoldConv3d(features: x, map: map, weight: weight.reshaped([weight.dim(0), 27, weight.dim(4)]).transposed(1, 2, 0), bias: bias)
    }
}

// MARK: - Residual block

/// `SparseResBlock3d`: LayerNorm → SiLU → conv → SiLU → conv, plus a skip.
///
/// Both convolutions run on the same coords, so they share one neighbour map.
public class SparseResBlock3d: Module {
    @ModuleInfo(key: "norm1") public var norm1: LayerNorm
    @ModuleInfo(key: "conv1") public var conv1: SubmanifoldConv3d
    @ModuleInfo(key: "conv2") public var conv2: SubmanifoldConv3d
    @ModuleInfo(key: "skip_connection") public var skipConnection: Linear?

    public init(channels: Int, outChannels: Int? = nil) {
        let outCh = outChannels ?? channels
        self._norm1.wrappedValue = LayerNorm(dimensions: channels, eps: 1e-6, affine: true)
        self._conv1.wrappedValue = SubmanifoldConv3d(inChannels: channels, outChannels: outCh)
        self._conv2.wrappedValue = SubmanifoldConv3d(inChannels: outCh, outChannels: outCh)
        // PyTorch uses nn.Identity when the widths match, which contributes no
        // parameters — so does a nil here.
        self._skipConnection.wrappedValue =
            channels != outCh ? Linear(channels, outCh) : nil
        super.init()
    }

    public func callAsFunction(_ feats: MLXArray, map: MLXArray) -> MLXArray {
        var h = MLXNN.silu(norm1(feats))
        h = conv1(h, map: map)
        h = MLXNN.silu(h)
        h = conv2(h, map: map)
        let residual = skipConnection.map { $0(feats) } ?? feats
        return h + residual
    }
}

// MARK: - Resampling

/// `PoolDown`: mean-pool by `factor`, then a linear width change.
public class SparsePoolDown: Module {
    public let factor: Int
    @ModuleInfo(key: "linear") public var linear: Linear

    public init(inChannels: Int, outChannels: Int, factor: Int) {
        self.factor = factor
        self._linear.wrappedValue = Linear(inChannels, outChannels)
        super.init()
    }

    public func callAsFunction(
        _ feats: MLXArray, mapping: SparseNeighbor.PoolMapping
    ) -> MLXArray {
        linear(SparseOps.poolMean(features: feats, mapping: mapping))
    }
}

/// `NearestUp`: a linear width change, *then* nearest-neighbour upsampling.
///
/// The order matters — projecting before the broadcast is cheaper and is what
/// upstream does; swapping them would still typecheck and give different
/// numbers only through the bias term, so it is worth stating.
public class SparseNearestUp: Module {
    @ModuleInfo(key: "linear") public var linear: Linear

    public init(inChannels: Int, outChannels: Int) {
        self._linear.wrappedValue = Linear(inChannels, outChannels)
        super.init()
    }

    public func callAsFunction(
        _ feats: MLXArray, mapping: SparseNeighbor.PoolMapping
    ) -> MLXArray {
        SparseOps.upsampleNearest(coarseFeatures: linear(feats), mapping: mapping)
    }
}

// MARK: - UNet

/// `Sparse3DUNet` — the refiner proper.
public class Sparse3DUNet: Module {
    public let modelChannels: [Int]
    public let downsampleFactors: [Int]
    public let encoderDownsample: Int

    @ModuleInfo(key: "input_proj") public var inputProj: Linear
    @ModuleInfo(key: "encoder_fuse") public var encoderFuse: Linear
    @ModuleInfo(key: "fuse_proj") public var fuseProj: [UnaryLayer]

    @ModuleInfo(key: "down_stages") public var downStages: [[SparseResBlock3d]]
    @ModuleInfo(key: "downsample_blocks") public var downsampleBlocks: [SparsePoolDown]
    @ModuleInfo(key: "bottleneck_stage") public var bottleneckStage: [SparseResBlock3d]
    @ModuleInfo(key: "upsample_blocks") public var upsampleBlocks: [SparseNearestUp]
    @ModuleInfo(key: "up_stages") public var upStages: [[SparseResBlock3d]]
    @ModuleInfo(key: "out_proj") public var outProj: Linear

    public init(
        inChannels: Int,
        outChannels: Int,
        encoderChannels: Int,
        modelChannels: [Int],
        encoderBlocksPerLevel: [Int],
        decoderBlocksPerLevel: [Int],
        bottleneckBlocks: Int = 1,
        downsampleFactors: [Int]? = nil,
        encoderDownsample: Int = 16
    ) {
        precondition(
            modelChannels.count >= 2,
            "model_channels must have at least 2 levels, got \(modelChannels)")
        let factors = downsampleFactors ?? Array(repeating: 2, count: modelChannels.count - 1)
        precondition(
            factors.count == modelChannels.count - 1,
            "downsample_factors must have length \(modelChannels.count - 1), got \(factors)")
        precondition(
            factors.allSatisfy { $0 > 1 },
            "all downsample_factors must be > 1, got \(factors)")
        precondition(
            factors.reduce(1, *) == encoderDownsample,
            "encoder_downsample (\(encoderDownsample)) must equal the product of downsample_factors (\(factors))")

        self.modelChannels = modelChannels
        self.downsampleFactors = factors
        self.encoderDownsample = encoderDownsample

        let numLevels = modelChannels.count
        let bottleneckChannels = modelChannels[numLevels - 1]

        self._inputProj.wrappedValue = Linear(inChannels, modelChannels[0])
        self._encoderFuse.wrappedValue = Linear(encoderChannels, bottleneckChannels)
        self._fuseProj.wrappedValue = [
            Linear(bottleneckChannels * 2, bottleneckChannels),
            SiLU(),
            Linear(bottleneckChannels, bottleneckChannels)
        ]

        precondition(
            encoderBlocksPerLevel.count == numLevels,
            "encoder_blocks_per_level must have length \(numLevels), got \(encoderBlocksPerLevel)")
        precondition(
            decoderBlocksPerLevel.count == numLevels - 1,
            "decoder_blocks_per_level must have length \(numLevels - 1), got \(decoderBlocksPerLevel)")

        var down: [[SparseResBlock3d]] = []
        var pools: [SparsePoolDown] = []
        for i in 0..<numLevels {
            down.append(
                (0..<encoderBlocksPerLevel[i]).map { _ in
                    SparseResBlock3d(channels: modelChannels[i])
                })
            if i < numLevels - 1 {
                pools.append(
                    SparsePoolDown(
                        inChannels: modelChannels[i],
                        outChannels: modelChannels[i + 1],
                        factor: factors[i]))
            }
        }
        self._downStages.wrappedValue = down
        self._downsampleBlocks.wrappedValue = pools

        self._bottleneckStage.wrappedValue = (0..<bottleneckBlocks).map { _ in
            SparseResBlock3d(channels: bottleneckChannels)
        }

        var ups: [SparseNearestUp] = []
        var upBlocks: [[SparseResBlock3d]] = []
        for i in 0..<(numLevels - 1) {
            let sourceLevel = numLevels - 1 - i
            let targetLevel = sourceLevel - 1
            ups.append(
                SparseNearestUp(
                    inChannels: modelChannels[sourceLevel],
                    outChannels: modelChannels[targetLevel]))
            upBlocks.append(
                (0..<decoderBlocksPerLevel[i]).map { _ in
                    SparseResBlock3d(channels: modelChannels[targetLevel])
                })
        }
        self._upsampleBlocks.wrappedValue = ups
        self._upStages.wrappedValue = upBlocks

        self._outProj.wrappedValue = Linear(modelChannels[0], outChannels)
        super.init()
    }

    /// Sample the dense conditioning map at the bottleneck's voxel columns.
    ///
    /// `encoderFeature` is `(B, H/encoderDownsample, W/encoderDownsample, C)`
    /// in NHWC; the bottleneck's coords have been floor-divided by exactly that
    /// factor, so `(b, i, j)` indexes it directly.
    private func sampleEncoderFeature(
        _ encoderFeature: MLXArray, coords: MLXArray
    ) -> MLXArray {
        precondition(
            encoderFeature.ndim == 4,
            "encoder feature must be (B, H, W, C) NHWC, got \(encoderFeature.shape)")
        let b = coords[0..., 0]
        let i = coords[0..., 1]
        let j = coords[0..., 2]
        return encoderFeature[b, i, j]
    }

    /// - Parameters:
    ///   - feats: `(N, inChannels)` voxel features.
    ///   - coords: `(N, 4)` Int32 `[b, i, j, z]`.
    ///   - shape: ambient extent `[B, S_i, S_j, S_z]`.
    ///   - encoderFeature: `(B, S_i / encoderDownsample, S_j / encoderDownsample, C)` NHWC.
    /// - Returns: `(N, outChannels)`.
    public func callAsFunction(
        feats: MLXArray, coords: MLXArray, shape: [Int], encoderFeature: MLXArray,
        checkpoint: () throws -> Void = {}
    ) rethrows -> MLXArray {
        let encoderH = encoderFeature.dim(1)
        let encoderW = encoderFeature.dim(2)
        precondition(
            shape[1] == encoderH * encoderDownsample && shape[2] == encoderW * encoderDownsample,
            "voxel grid \(shape[1])x\(shape[2]) is not \(encoderDownsample)x the encoder feature \(encoderH)x\(encoderW)")

        let numLevels = downStages.count
        let numTransitions = downsampleBlocks.count

        var h = inputProj(feats)
        var currentCoords = coords
        var currentShape = shape

        // Neighbour maps are keyed by level, not by block. Coords at level k are
        // identical on the way down and on the way back up (the upsample
        // restores the matching skip's coords), so one map per level serves the
        // down stage, the symmetric up stage and — at the deepest level — the
        // bottleneck. Pool mappings are likewise reused by the paired upsample,
        // which is what upstream expresses as `down_cache.T`.
        var levelConvMaps = [MLXArray?](repeating: nil, count: numLevels)
        var levelPoolMappings = [SparseNeighbor.PoolMapping?](repeating: nil, count: numTransitions)
        var skips = [(feats: MLXArray, coords: MLXArray, shape: [Int])?](
            repeating: nil, count: numTransitions)

        for level in 0..<numLevels {
            try checkpoint()
            let map = convMap(coords: currentCoords, shape: currentShape)
            levelConvMaps[level] = map
            for block in downStages[level] {
                h = block(h, map: map)
            }
            if level < numTransitions {
                skips[level] = (h, currentCoords, currentShape)
                let mapping = SparseNeighbor.poolMapping(
                    coords: currentCoords, shape: currentShape,
                    factor: downsampleFactors[level])
                h = downsampleBlocks[level](h, mapping: mapping)
                levelPoolMappings[level] = mapping
                currentCoords = mapping.coarseCoords
                currentShape = mapping.coarseShape
            }
            MLX.eval(h)
        }
        try checkpoint()

        var encoded = sampleEncoderFeature(encoderFeature, coords: currentCoords)
        encoded = encoderFuse(encoded.asType(h.dtype))
        h = MLX.concatenated([h, encoded], axis: -1)
        for layer in fuseProj {
            h = layer(h)
        }

        let bottleneckMap = levelConvMaps[numLevels - 1]!
        for block in bottleneckStage {
            h = block(h, map: bottleneckMap)
        }

        MLX.eval(h)
        for i in 0..<numTransitions {
            try checkpoint()
            let targetLevel = numLevels - 2 - i
            guard let skip = skips[targetLevel], let mapping = levelPoolMappings[targetLevel] else {
                fatalError("missing skip/pool mapping for level \(targetLevel)")
            }
            h = upsampleBlocks[i](h, mapping: mapping)
            currentCoords = skip.coords
            currentShape = skip.shape
            h = h + skip.feats
            let map = levelConvMaps[targetLevel]!
            for block in upStages[i] {
                h = block(h, map: map)
            }
            MLX.eval(h)
        }
        try checkpoint()

        return outProj(h)
    }

    private func convMap(coords: MLXArray, shape: [Int]) -> MLXArray {
        let index = SortedCoordIndex(coords: coords, shape: shape)
        return SparseNeighbor.submanifoldMap(index, coords: coords)
    }
}

// swiftlint:enable all
