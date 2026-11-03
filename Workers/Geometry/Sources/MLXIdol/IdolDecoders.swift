// SPDX-License-Identifier: MIT
// IDOL's two decoders (idol/models/decoder.py): the octree decoder places anchor points by descending an octree
// with predicted child probabilities; the Gaussian decoder turns each anchor into 32 Gaussians.
import Foundation
import MLX
import MLXNN
import MLXRandom

public struct IdolDecoderConfig: Sendable {
    public var channels = 1024, context = 16, heads = 16
    public var octreeBlocks = 4, gaussianBlocks = 16
    public var gaussiansPerAnchor = 32
    public init() {}
}

public final class IdolOctreeDecoder: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo(key: "l_embedder") var lEmbedder: SinusoidalEmbedder
    @ModuleInfo(key: "adaLN_modulation") var adaLN: SiLULinear
    @ModuleInfo var blocks: [ModulatedCrossOnlyBlock]
    @ModuleInfo(key: "out_proj") var outProj: Linear
    let channels: Int
    public init(_ c: IdolDecoderConfig = IdolDecoderConfig()) {
        channels = c.channels
        _inProj.wrappedValue = Linear(3, c.channels)
        _inputLayer.wrappedValue = Linear(c.channels, c.channels)
        _lEmbedder.wrappedValue = SinusoidalEmbedder(c.channels, maxPeriod: 1024, twoPi: true)
        _adaLN.wrappedValue = SiLULinear(c.channels, 6 * c.channels)
        _blocks.wrappedValue = (0..<c.octreeBlocks).map { _ in ModulatedCrossOnlyBlock(c.channels, context: c.context, heads: c.heads) }
        _outProj.wrappedValue = Linear(c.channels, 8)
    }
    /// cells [B, L, 3] in [0, 1] float32; resolution [B] (2^level); latent [B, Lc, 16]. Returns logits [B, L, 8] float32.
    public func callAsFunction(_ cells: MLXArray, resolution: MLXArray, latent: MLXArray) -> MLXArray {
        let (b, l) = (cells.dim(0), cells.dim(1))
        var h = inProj(cells.asType(.float32)) + PositionEmbedding.points(cells.reshaped([-1, 3]), channels: channels).reshaped([b, l, -1])
        h = inputLayer(h)
        let mod = adaLN(lEmbedder(resolution.asType(.float32))).asType(.float16)
        var out = h.asType(.float16)
        let context = latent.asType(.float16)
        for block in blocks { out = block(out, mod: mod, context: context) }
        return outProj(layerNorm32(out.asType(.float32), eps: 1e-5)).asType(.float32)
    }
}

public final class IdolGaussianDecoder: Module {
    @ModuleInfo(key: "in_proj") var inProj: Linear
    @ModuleInfo(key: "input_layer") var inputLayer: Linear
    @ModuleInfo var blocks: [TransformerCrossBlock]
    @ModuleInfo(key: "out_proj") var outProj: Linear
    /// Fixed per-Gaussian offset pattern [32, 3] and base offset scale, stored with the weights (cast to FP16 at Get).
    @ParameterInfo(key: "points_offset_perturbation") var offsetPattern: MLXArray
    @ParameterInfo(key: "base_offset_scale") var baseOffsetScale: MLXArray
    let channels: Int, perAnchor: Int
    public init(_ c: IdolDecoderConfig = IdolDecoderConfig()) {
        channels = c.channels; perAnchor = c.gaussiansPerAnchor
        _inProj.wrappedValue = Linear(3, c.channels)
        _inputLayer.wrappedValue = Linear(c.channels, c.channels)
        _blocks.wrappedValue = (0..<c.gaussianBlocks).map { _ in TransformerCrossBlock(c.channels, context: c.context, heads: c.heads) }
        _outProj.wrappedValue = Linear(c.channels, c.gaussiansPerAnchor * 15)
        _offsetPattern.wrappedValue = MLXArray.zeros([c.gaussiansPerAnchor, 3])
        _baseOffsetScale.wrappedValue = MLXArray.zeros([])
    }
    /// anchors [B, L, 3] float32; latent [B, Lc, 16] -> per-anchor features [B, L, 32·15] float32.
    public func callAsFunction(_ anchors: MLXArray, latent: MLXArray) -> MLXArray {
        let (b, l) = (anchors.dim(0), anchors.dim(1))
        var h = inProj(anchors.asType(.float32)) + PositionEmbedding.points(anchors.reshaped([-1, 3]), channels: channels).reshaped([b, l, -1])
        h = inputLayer(h)
        var out = h.asType(.float16)
        let context = latent.asType(.float16)
        for block in blocks { out = block(out, context: context) }
        return outProj(layerNorm32(out.asType(.float32), eps: 1e-5)).asType(.float32)
    }

    /// The decoder outputs of one object (batch 1) as activated Gaussians (GaussianFixedlenDecoder.to_representation).
    public func gaussians(anchors: MLXArray, features: MLXArray, representation r: IdolRepresentation = IdolRepresentation()) -> IdolGaussians {
        let l = features.dim(1), g = perAnchor
        var start = 0
        func slice(_ width: Int) -> MLXArray {
            defer { start += g * width }
            return features[0, 0..., start..<(start + g * width)].reshaped([l, g, width])
        }
        let xyz = slice(3), dc = slice(3), scaling = slice(3), rotation = slice(4), opacity = slice(1), offsetScale = slice(1)
        let scale = softplus(offsetScale + baseOffsetScale.asType(.float32))
        let offset = MLX.tanh(xyz * r.lrXYZ + offsetPattern.asType(.float32)) * (0.5 * r.perturbSize) * scale
        let rawXYZ = (offset + anchors[0].expandedDimensions(axis: 1)).reshaped([-1, 3])
        return IdolGaussians(
            rawXYZ: rawXYZ, featuresDC: (dc * r.lrDC).reshaped([-1, 3]), rawScaling: (scaling * r.lrScaling).reshaped([-1, 3]),
            rawRotation: (rotation * r.lrRotation).reshaped([-1, 4]), rawOpacity: (opacity * r.lrOpacity).reshaped([-1, 1]), representation: r)
    }
}

/// config.json decoder_gs.representation_config.
public struct IdolRepresentation: Sendable {
    public var lrXYZ: Float = 1, lrDC: Float = 1, lrOpacity: Float = 1, lrScaling: Float = 1, lrRotation: Float = 0.1
    public var perturbSize: Float = 1.5, filterKernel: Float = 0.0009, scalingBias: Float = 0.004, opacityBias: Float = 0.1
    public init() {}
}

/// Octree descent (OctreeProbabilityFixedlenDecoder.sample) for batch 1: the integer bookkeeping runs on the host,
/// each level's probabilities on the GPU.
public enum IdolOctree {
    /// Systematic resampling: splits `count` items among the bins of `probabilities` (one row), with the stratified
    /// grid offset `u0` in [0, 1/count). Same arithmetic as the reference, float32 cumulative sums and a left search.
    public static func systematic(_ probabilities: ArraySlice<Float>, count: Int, u0: Float) -> [Int] {
        let p = probabilities.count
        var out = [Int](repeating: 0, count: p)
        guard count > 0 else { return out }
        var probs = probabilities.map { max(0, $0) }
        let sum = probs.reduce(0, +)
        if sum == 0 { probs = [Float](repeating: 1 / Float(p), count: p) } else { probs = probs.map { $0 / max(sum, 1) } }
        var cdf = [Float](repeating: 0, count: p), running: Float = 0
        for i in 0..<p { running += probs[i]; cdf[i] = min(running, Float(1.0 - 1e-12)) }
        let n = Float(count)
        for k in 0..<count {
            let u = min(u0 + Float(k) / n, Float(1.0 - 1e-12))
            var index = 0
            while index < p && cdf[index] < u { index += 1 }
            out[min(index, p - 1)] += 1
        }
        return out
    }

    /// Places `points` anchors [1, points, 3] in [0, 1]^3 at octree depth `level`.
    public static func sample(
        decoder: IdolOctreeDecoder, latent: MLXArray, points: Int, level: Int, key: MLXArray, cancellation: () throws -> Void = {}
    ) throws -> MLXArray {
        var cells: [(Int, Int, Int)] = [(0, 0, 0)]
        var counts = [points]
        let keys = MLXRandom.split(key: key, into: level + 1)
        for l in 1...level {
            try cancellation()
            let parent = Float(1 << (l - 1))
            let centres = MLXArray(cells.flatMap { [(Float($0.0) + 0.5) / parent, (Float($0.1) + 0.5) / parent, (Float($0.2) + 0.5) / parent] }, [1, cells.count, 3])
            let logits = decoder(centres, resolution: MLXArray([Float(1 << l)]), latent: latent)
            let probs = softmax(logits, axis: -1)[0].asArray(Float.self)
            let offsets = MLXRandom.uniform(0..<1, [cells.count], key: keys[l - 1]).asArray(Float.self)
            var next: [(Int, Int, Int)] = [], nextCounts: [Int] = []
            for (i, cell) in cells.enumerated() {
                let split = systematic(probs[(i * 8)..<(i * 8 + 8)], count: counts[i], u0: offsets[i] / Float(max(counts[i], 1)))
                for child in 0..<8 where split[child] > 0 {
                    // child order: i fastest, then j, then k (the reference's child_offset)
                    next.append((cell.0 * 2 + (child & 1), cell.1 * 2 + ((child >> 1) & 1), cell.2 * 2 + ((child >> 2) & 1)))
                    nextCounts.append(split[child])
                }
            }
            cells = next; counts = nextCounts
        }
        var coords: [Float] = []
        coords.reserveCapacity(points * 3)
        for (cell, count) in zip(cells, counts) {
            for _ in 0..<count { coords += [Float(cell.0), Float(cell.1), Float(cell.2)] }
        }
        let jitter = MLXRandom.uniform(0..<1, [points, 3], key: keys[level])
        return ((MLXArray(coords, [points, 3]) + jitter) / Float(1 << level)).reshaped([1, points, 3])
    }
}
