// SPDX-License-Identifier: MIT
// d1-3B's picture path on MLX at the checkpoint's bf16: SigLIP2 NaFlex (transformers 5.14 `Siglip2VisionModel`, its
// post-layernorm included, no head) and the LFM2-VL projector (2x2 pixel unshuffle, linear, exact GELU, linear), as
// `Lfm2VlModel.get_image_features`. Each piece runs unpadded, which equals the processor's padded, masked batch.
import Foundation
import MLX
import MLXFast

// `a += b` on an MLXArray rewrites the shared array object in place; residual updates stay `a = a + b`
// (marked per line for SwiftLint's shorthand rule).

public final class VisionTower: @unchecked Sendable {
    struct Block {
        let ln1w: MLXArray, ln1b: MLXArray
        let qkvW: D1Matrix
        let qkvB: MLXArray
        let oW: D1Matrix
        let oB: MLXArray
        let ln2w: MLXArray, ln2b: MLXArray
        let fc1W: D1Matrix
        let fc1B: MLXArray
        let fc2W: D1Matrix
        let fc2B: MLXArray
    }
    var intake: D1ComponentSession?
    let dim = 1152
    let heads = 16
    let eps: Float = 1e-6
    let patchW: D1Matrix
    let patchB: MLXArray
    let position: MLXArray // (16, 16, dim) BF16; interpolation uses transient FP32
    let blocks: [Block]
    let postW: MLXArray, postB: MLXArray
    let l1W: D1Matrix, l2W: D1Matrix
    let l1B: MLXArray, l2B: MLXArray
    let processor: PictureProcessor

    public init(store: SafetensorsFile, processor: PictureProcessor) throws {
        self.processor = processor
        let v = "model.vision_tower.vision_model."
        func w(_ name: String) throws -> MLXArray { try store.take(name) }
        func m(_ name: String) throws -> D1Matrix { try store.matrix(name) }
        patchW = try m(v + "embeddings.patch_embedding.weight")
        patchB = try w(v + "embeddings.patch_embedding.bias")
        let pos = try w(v + "embeddings.position_embedding.weight")
        guard pos.shape == [256, dim], patchW.shape == [dim, 768] else { throw JSONError("model.safetensors: unexpected SigLIP2 shapes") }
        position = pos.reshaped([16, 16, dim])
        var blocks: [Block] = []
        var i = 0
        while store.contains(v + "encoder.layers.\(i).layer_norm1.weight") {
            let p = v + "encoder.layers.\(i)."
            let a = p + "self_attn."
            blocks.append(
                Block(
                    ln1w: try w(p + "layer_norm1.weight"), ln1b: try w(p + "layer_norm1.bias"),
                    qkvW: try D1Matrix.concatenate([m(a + "q_proj.weight"), m(a + "k_proj.weight"), m(a + "v_proj.weight")]),
                    qkvB: concatenated([try w(a + "q_proj.bias"), try w(a + "k_proj.bias"), try w(a + "v_proj.bias")], axis: 0),
                    oW: try m(a + "out_proj.weight"), oB: try w(a + "out_proj.bias"),
                    ln2w: try w(p + "layer_norm2.weight"), ln2b: try w(p + "layer_norm2.bias"),
                    fc1W: try m(p + "mlp.fc1.weight"), fc1B: try w(p + "mlp.fc1.bias"),
                    fc2W: try m(p + "mlp.fc2.weight"), fc2B: try w(p + "mlp.fc2.bias")))
            i += 1
        }
        guard blocks.count == 27 else { throw JSONError("model.safetensors: expected 27 SigLIP2 layers, found \(blocks.count)") }
        self.blocks = blocks
        postW = try w(v + "post_layernorm.weight")
        postB = try w(v + "post_layernorm.bias")
        let projector = "model.multi_modal_projector."
        l1W = try m(projector + "linear_1.weight")
        l1B = try w(projector + "linear_1.bias")
        l2W = try m(projector + "linear_2.weight")
        l2B = try w(projector + "linear_2.bias")
        eval(
            [patchB, position, postW, postB, l1B, l2B] + [patchW, l1W, l2W].flatMap(\.arrays)
                + blocks.flatMap { [$0.ln1w, $0.ln1b, $0.qkvB, $0.oB, $0.ln2w, $0.ln2b, $0.fc1B, $0.fc2B] + [$0.qkvW, $0.oW, $0.fc1W, $0.fc2W].flatMap(\.arrays) })
    }

    /// `F.interpolate(mode="bilinear", align_corners=False, antialias=True)` weights, (output, input).
    static func bilinear(_ input: Int, _ output: Int) -> MLXArray {
        let table = Resample.weights(input, output, filter: { max(0, 1 - abs($0)) }, support: 1)
        var m = [Float](repeating: 0, count: output * input)
        for (o, (lo, w)) in table.enumerated() { for (j, x) in w.enumerated() { m[o * input + lo + j] = Float(x) } }
        return MLXArray(m, [output, input])
    }

    /// The position grid resized to (h, w) in fp32, flattened row-major, cast to the weights' dtype.
    private func positions(_ h: Int, _ w: Int) -> MLXArray {
        let wh = Self.bilinear(16, h)
        let ww = Self.bilinear(16, w)
        let across = matmul(position.asType(.float32).transposed(0, 2, 1), ww.T) // (16 rows, dim, w)
        let both = matmul(across.transposed(2, 1, 0), wh.T) // (w, dim, h)
        return both.transposed(2, 0, 1).reshaped([h * w, dim]).asType(patchW.dtype)
    }

    private func linear(_ x: MLXArray, _ w: D1Matrix, _ b: MLXArray) -> MLXArray {
        if w.bits == 16 { return addMM(b, x, w.weight.T) }
        return w(x) + b
    }

    /// torch's bf16 `gelu_pytorch_tanh` / exact `gelu`: computed in fp32, rounded once.
    private func geluTanh(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        let inner = Float((2 / Double.pi).squareRoot()) * (f + 0.044715 * f * f * f)
        return (0.5 * f * (1 + tanh(inner))).asType(x.dtype)
    }
    private func geluErf(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (0.5 * f * (1 + erf(f / Float(2.0.squareRoot())))).asType(x.dtype)
    }

    /// Encoder output for pieces of one shape, (B, h*w, dim).
    private func encode(_ patches: MLXArray, _ h: Int, _ w: Int) -> MLXArray {
        let b = patches.dim(0)
        let n = h * w
        let headDim = dim / heads
        // bf16 throughout, as the checkpoint (an fp32 tower was 10x closer to fp32 features but 25-40% slower per
        // picture and kept the same answers; log/007).
        var x = linear(patches.asType(patchW.dtype), patchW, patchB) + positions(h, w).expandedDimensions(axis: 0)
        intake?.output("vision/embedding", x)
        for (i, blk) in blocks.enumerated() {
            x = intake?.input("vision/layer-\(i)", x) ?? x
            var y = MLXFast.layerNorm(x, weight: blk.ln1w, bias: blk.ln1b, eps: eps)
            let qkv = linear(y, blk.qkvW, blk.qkvB)
            let q = qkv[.ellipsis, ..<dim].reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
            let k = qkv[.ellipsis, dim..<(2 * dim)].reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
            let v = qkv[.ellipsis, (2 * dim)...].reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
            let o = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: Float(1 / Double(headDim).squareRoot()), mask: .none)
            x = x + linear(o.transposed(0, 2, 1, 3).reshaped([b, n, dim]), blk.oW, blk.oB) // swiftlint:disable:this shorthand_operator
            y = MLXFast.layerNorm(x, weight: blk.ln2w, bias: blk.ln2b, eps: eps)
            x = x + linear(geluTanh(linear(y, blk.fc1W, blk.fc1B)), blk.fc2W, blk.fc2B) // swiftlint:disable:this shorthand_operator
            intake?.output("vision/layer-\(i)", x)
        }
        return MLXFast.layerNorm(x, weight: postW, bias: postB, eps: eps)
    }

    /// The projector over one piece's (h, w, dim) grid, (h/2 * w/2, 2048).
    private func project(_ grid: MLXArray, _ h: Int, _ w: Int) -> MLXArray {
        // Lfm2VlMultiModalProjector.pixel_unshuffle, its axis names kept: (b, "width"=h, "height"=w, c).
        var x = grid.reshaped([1, h, w / 2, dim * 2]).transposed(0, 2, 1, 3)
        x = x.reshaped([1, w / 2, h / 2, dim * 4]).transposed(0, 2, 1, 3).reshaped([(h / 2) * (w / 2), dim * 4])
        return linear(geluErf(linear(x, l1W, l1B)), l2W, l2B)
    }

    func intakeForward(_ patches: MLXArray) -> MLXArray {
        let encoded = encode(patches, 4, 4)
        let grid = intake?.input("projector/input", encoded[0]) ?? encoded[0]
        let output = project(grid, 4, 4)
        intake?.output("projector", output)
        return output
    }

    /// Features of `pictures` (already capped) in prompt order: per picture, its pieces in feature order.
    public func features(_ pictures: [(Picture, PictureProcessor.Layout)]) -> MLXArray {
        var pieces: [(values: [Float], h: Int, w: Int)] = []
        for (p, layout) in pictures {
            for piece in processor.pieces(p, layout) {
                let (values, h, w) = processor.patches(piece)
                pieces.append((values, h, w))
            }
        }
        // Pieces of one shape (the tiles) run as one batch; outputs go back in order.
        var out = [MLXArray?](repeating: nil, count: pieces.count)
        var groups: [String: [Int]] = [:]
        var order: [String] = []
        for (i, p) in pieces.enumerated() {
            let key = "\(p.h)x\(p.w)"
            if groups[key] == nil { order.append(key) }
            groups[key, default: []].append(i)
        }
        for key in order {
            let members = groups[key]!
            let h = pieces[members[0]].h
            let w = pieces[members[0]].w
            let flat = members.flatMap { pieces[$0].values }
            let encoded = encode(MLXArray(flat, [members.count, h * w, 768]), h, w)
            for (j, i) in members.enumerated() { out[i] = project(encoded[j].reshaped([h, w, dim]), h, w) }
        }
        return concatenated(out.map { $0! }, axis: 0)
    }
}
