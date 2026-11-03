// SPDX-License-Identifier: LicenseRef-S-Lab-1.0 (Swift port of GSeanCDAT/GIMM-VFI; see LICENSE-upstream-GIMMNative)
//
//  RAFT as GIMM-VFI-R embeds it (raft/raft.py, extractor.py, update.py, corr.py; weights `flow_estimator.*`, fine-tuned
//  with the rest of the model): BasicEncoder feature (instance norm) and context (batch norm, also returning its
//  intermediate features) networks, the all-pairs correlation pyramid with radius-4 lookups, 20 SepConvGRU iterations and
//  convex 8x upsampling. Both directions of a frame pair run as one batch of two: the encoders are per-sample, so this
//  equals the maker's two calls.
//

import Foundation
import MLX
import MLXNN

/// Correlation pyramid of all-pairs dot products, one [rows, H, W] map per level (FP16, as autocast computes it).
struct CorrPyramid {
    let levels: [MLXArray]

    /// corr.py `corr`: fmap1 · fmap2 / sqrt(dim) for every pixel pair; rows are fmap1's pixels (batch-major).
    static func volume(_ fmap1: MLXArray, _ fmap2: MLXArray) -> MLXArray {
        let n = fmap1.dim(0), h = fmap1.dim(1), w = fmap1.dim(2), c = fmap1.dim(3)
        let a = fmap1.asType(.float16).reshaped([n, h * w, c]), b = fmap2.asType(.float16).reshaped([n, h * w, c])
        return (matmul(a, b.transposed(0, 2, 1)) / Float(c).squareRoot()).reshaped([n * h * w, h, w])
    }

    init(_ volume: MLXArray, levels count: Int = 4) {
        var levels = [volume]
        for _ in 1..<count {
            let v = levels.last!, h = v.dim(1) / 2, w = v.dim(2) / 2
            // F.avg_pool2d(2, stride=2): floor sizes, the odd last row/column dropped.
            let even = v[0..., 0..<(2 * h), 0..<(2 * w)].reshaped([v.dim(0), h, 2, w, 2])
            levels.append(MLX.mean(even.asType(.float32), axes: [2, 4]).asType(v.dtype))
        }
        self.levels = levels
    }

    /// corr.py lookup: for each row's centroid (x, y) at [N, h, w, 2], bilinear samples (zeros padding, align_corners)
    /// of the row's own map at (x + a, y + b) / 2^level for a, b in -4...4, the x offset outer (RAFT's meshgrid order).
    /// Returns float32 [N, h, w, 4 * 81].
    func lookup(_ coords: MLXArray, radius: Int = 4) -> MLXArray {
        let n = coords.dim(0), h = coords.dim(1), w = coords.dim(2), rows = n * h * w, k = 2 * radius + 1
        let offsets = MLXArray((0..<k).map { Float($0 - radius) })
        let ox = broadcast(offsets.reshaped([1, k, 1]), to: [1, k, k]).reshaped([1, k * k])
        let oy = broadcast(offsets.reshaped([1, 1, k]), to: [1, k, k]).reshaped([1, k * k])
        let c = coords.asType(.float32).reshaped([rows, 2])
        var outputs: [MLXArray] = []
        for (level, volume) in levels.enumerated() {
            let lh = volume.dim(1), lw = volume.dim(2), scale = Float(1 << level)
            let ix = c[0..., 0..<1] / scale + ox, iy = c[0..., 1..<2] / scale + oy
            let flat = volume.reshaped([rows * lh * lw])
            let rowBase = MLXArray((0..<rows).map { Int32($0 * lh * lw) }).reshaped([rows, 1])
            let x0 = MLX.floor(ix), y0 = MLX.floor(iy), fx = ix - x0, fy = iy - y0
            var acc: MLXArray?
            for (dx, dy) in [(0, 0), (1, 0), (0, 1), (1, 1)] {
                let cx = x0 + Float(dx), cy = y0 + Float(dy)
                let inside = (cx .>= 0) .&& (cx .<= Float(lw - 1)) .&& (cy .>= 0) .&& (cy .<= Float(lh - 1))
                let weight = MLX.where(inside, (dx == 1 ? fx : 1 - fx) * (dy == 1 ? fy : 1 - fy), MLXArray(Float(0)))
                let xi = MLX.clip(cx, min: 0, max: Float(lw - 1)).asType(.int32), yi = MLX.clip(cy, min: 0, max: Float(lh - 1)).asType(.int32)
                let term = take(flat, rowBase + yi * Int32(lw) + xi).asType(.float32) * weight
                acc = acc.map { $0 + term } ?? term
            }
            outputs.append(acc!.reshaped([n, h, w, k * k]))
        }
        return concatenated(outputs, axis: -1)
    }
}

enum RAFT {
    static let prefix = "flow_estimator."
    static let iterations = 20

    static func residual(_ x: MLXArray, _ p: GIMMParams, _ key: String, stride: Int, batch: Bool) -> MLXArray {
        func norm(_ v: MLXArray, _ name: String) -> MLXArray { batch ? batchNorm(v, p, key + "." + name) : instanceNorm(v) }
        var y = relu(norm(conv(x, p, key + ".conv1", stride: stride), "norm1"))
        y = relu(norm(conv(y, p, key + ".conv2"), "norm2"))
        var shortcut = x
        if stride != 1 {
            let down = conv(x, p, key + ".downsample.0", stride: stride)
            shortcut = batch ? batchNorm(down, p, key + ".downsample.1") : instanceNorm(down)
        }
        return relu(shortcut + y)
    }

    /// BasicEncoder: output (conv2) and the features after layer1, layer2 and layer3.
    static func encoder(_ x: MLXArray, _ p: GIMMParams, _ name: String, batch: Bool) -> (MLXArray, [MLXArray]) {
        let key = prefix + name
        var h = conv(x, p, key + ".conv1", stride: 2)
        h = relu(batch ? batchNorm(h, p, key + ".norm1") : instanceNorm(h))
        var features: [MLXArray] = []
        for (layer, stride) in [(1, 1), (2, 2), (3, 2)] {
            h = residual(h, p, key + ".layer\(layer).0", stride: stride, batch: batch)
            h = residual(h, p, key + ".layer\(layer).1", stride: 1, batch: batch)
            features.append(h)
        }
        return (conv(h, p, key + ".conv2"), features)
    }

    /// Result of both directions (batch index 0: frame 0 → 1, index 1: frame 1 → 0).
    struct Output {
        /// [2, H, W, 2] float32 flow at full input resolution.
        let flow: MLXArray
        /// Context features after layer2 (1/4) and layer3 (1/8), float32, for each direction's first frame.
        let features: [MLXArray]
        /// Feature-network output of each frame, [2, H/8, W/8, 256] float32 (RAFT returns fmap1.float()).
        let fmap: MLXArray
    }

    /// `images`: [2, H, W, 3] FP16 holding 255 * frame for frame 0 and frame 1.
    static func run(_ images: MLXArray, _ p: GIMMParams, cancellation: () throws -> Void) rethrows -> Output {
        let normalized = 2 * (images / 255) - 1
        let (fmap, _) = encoder(normalized, p, "fnet", batch: false)
        let fmap1 = fmap, fmap2 = concatenated([fmap[1..<2], fmap[0..<1]], axis: 0)
        let pyramid = CorrPyramid(CorrPyramid.volume(fmap1, fmap2))
        let (context, features) = encoder(normalized, p, "cnet", batch: true)
        var net = MLX.tanh(context[0..., 0..., 0..., 0..<128])
        let inp = relu(context[0..., 0..., 0..., 128..<256])
        let n = images.dim(0), h = fmap.dim(1), w = fmap.dim(2)
        let coords0 = coordsGrid(n, h, w)
        var coords1 = coords0
        eval(pyramid.levels, net, inp)
        let u = prefix + "update_block."
        for _ in 0..<iterations {
            try cancellation()
            let corr = pyramid.lookup(coords1)
            let flow = coords1 - coords0
            var cor = relu(conv(corr, p, u + "encoder.convc1"))
            cor = relu(conv(cor, p, u + "encoder.convc2"))
            var flo = relu(conv(flow, p, u + "encoder.convf1"))
            flo = relu(conv(flo, p, u + "encoder.convf2"))
            let motion = relu(conv(concatenated([cor, flo], axis: -1), p, u + "encoder.conv"))
            let x = concatenated([inp.asType(.float32), motion.asType(.float32), flow], axis: -1)
            net = sepConvGRU(net, x, p, u + "gru")
            let delta = conv(relu(conv(net, p, u + "flow_head.conv1")), p, u + "flow_head.conv2")
            // coords1 starts as an alias of coords0; MLX += would mutate both.
            // swiftlint:disable:next shorthand_operator
            coords1 = coords1 + delta
            eval(coords1, net)
        }
        let mask = 0.25 * conv(relu(conv(net, p, u + "mask.0")), p, u + "mask.2")
        let flow = upsample(coords1 - coords0, mask: mask)
        return Output(flow: flow, features: [features[1], features[2]], fmap: fmap.asType(.float32))
    }

    static func sepConvGRU(_ h0: MLXArray, _ x: MLXArray, _ p: GIMMParams, _ key: String) -> MLXArray {
        var h = h0
        for pass in ["1", "2"] {
            let hx = concatenated([h.asType(.float32), x], axis: -1)
            let z = MLX.sigmoid(conv(hx, p, key + ".convz" + pass)), r = MLX.sigmoid(conv(hx, p, key + ".convr" + pass))
            let q = MLX.tanh(conv(concatenated([(r * h).asType(.float32), x], axis: -1), p, key + ".convq" + pass))
            h = (1 - z) * h + z * q
        }
        return h
    }

    /// Convex upsampling: a softmax (float32) over each pixel's 3x3 neighbourhood of 8 * flow, per 8x8 sub-pixel.
    static func upsample(_ flow: MLXArray, mask: MLXArray) -> MLXArray {
        let n = flow.dim(0), h = flow.dim(1), w = flow.dim(2)
        let weights = softmax(mask.asType(.float32).reshaped([n, h, w, 9, 8, 8]), axis: 3)
        let padded = MLX.padded(8 * flow, widths: [IntOrPair(0), IntOrPair(1), IntOrPair(1), IntOrPair(0)])
        var patches: [MLXArray] = []
        for ky in 0..<3 { for kx in 0..<3 { patches.append(padded[0..., ky..<(ky + h), kx..<(kx + w)]) } }
        let unfolded = stacked(patches, axis: 3) // [n, h, w, 9, 2]
        let up = MLX.sum(weights.expandedDimensions(axis: -1) * unfolded.reshaped([n, h, w, 9, 1, 1, 2]), axis: 3) // [n, h, w, 8, 8, 2]
        return up.transposed(0, 1, 3, 2, 4, 5).reshaped([n, 8 * h, 8 * w, 2])
    }
}
