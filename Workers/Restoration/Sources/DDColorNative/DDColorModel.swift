// SPDX-License-Identifier: MIT
// Ported from xocialize/mlx-ddcolor-swift (MIT, see LICENSE-upstream-DDColorNative and PORTED-FROM.md), itself a port of
// piddnad/DDColor (Apache-2.0). Functional forward over the flat mlx-community/DDColor-modelscope-fp16 tensors.
import Foundation
import MLX
import MLXFast

/// The ConvNeXt-L DDColor (`ddcolor_modelscope`): gray-RGB in [0,1], NHWC `(1,S,S,3)`, to raw ab `(1,S,S,2)`.
/// Activations follow the weights' dtype; norms use MLX's fused layer norm, which accumulates in float32.
public final class DDColorModel: @unchecked Sendable {
    public static let depths = [3, 3, 27, 3]
    public static let dims = [192, 384, 768, 1536]
    public static let numQueries = 100, decLayers = 9, numScales = 3, hiddenDim = 256, numHeads = 8, nf = 512

    let w: [String: MLXArray]
    public let dtype: DType
    /// Component outputs of the last forward, kept only when `capture` is set (parity tests).
    public var capture = false
    public private(set) var captured: [String: MLXArray] = [:]

    public init(weights: [String: MLXArray]) throws {
        try DDColorWeights.validate(weights)
        w = weights; dtype = weights["encoder.arch.norm0.weight"]!.dtype
    }

    private func a(_ k: String) -> MLXArray { w[k]! }
    private func conv(_ x: MLXArray, _ k: String, bias: Bool = true, stride: Int = 1, pad: Int = 0, groups: Int = 1) -> MLXArray {
        let y = MLX.conv2d(x, a(k + ".weight"), stride: .init(stride), padding: .init(pad), groups: groups)
        return bias ? y + a(k + ".bias") : y
    }
    private func ln(_ x: MLXArray, _ k: String, eps: Float) -> MLXArray {
        MLXFast.layerNorm(x, weight: a(k + ".weight"), bias: a(k + ".bias"), eps: eps)
    }
    private func bn(_ x: MLXArray, _ k: String) -> MLXArray {
        DDColorOps.batchNorm(x, mean: a(k + ".running_mean"), variance: a(k + ".running_var"), weight: a(k + ".weight"), bias: a(k + ".bias"))
    }
    private func linear(_ x: MLXArray, _ k: String) -> MLXArray { MLX.matmul(x, a(k + ".weight").transposed()) + a(k + ".bias") }
    private func keep(_ name: String, _ x: MLXArray) { if capture { captured[name] = x } }

    private func encoder(_ input: MLXArray) -> [MLXArray] {
        var x = input, features: [MLXArray] = []
        for i in 0..<4 {
            let ds = "encoder.arch.downsample_layers.\(i)"
            if i == 0 {
                x = ln(conv(x, ds + ".0", stride: 4), ds + ".1", eps: 1e-6)
            } else {
                x = conv(ln(x, ds + ".0", eps: 1e-6), ds + ".1", stride: 2)
            }
            for j in 0..<Self.depths[i] {
                let p = "encoder.arch.stages.\(i).\(j)"
                var h = conv(x, p + ".dwconv", pad: 3, groups: Self.dims[i])
                h = ln(h, p + ".norm", eps: 1e-6)
                h = DDColorOps.gelu(linear(h, p + ".pwconv1"))
                h = linear(h, p + ".pwconv2") * a(p + ".gamma")
                // swiftlint:disable:next shorthand_operator
                x = x + h
            }
            let f = ln(x, "encoder.arch.norm\(i)", eps: 1e-6)
            keep("encoder.norm\(i)", f); features.append(f)
        }
        return features
    }

    /// UnetBlockWide: CustomPixelShuffle_ICNR (conv, BN, ReLU, shuffle 2, blur), BN of the skip, ReLU, conv 3x3, ReLU, BN.
    private func unetBlock(_ up: MLXArray, skip: MLXArray, index: Int) -> MLXArray {
        let p = "decoder.layers.\(index)"
        var x = DDColorOps.relu(bn(conv(up, p + ".shuf.conv.0", bias: false), p + ".shuf.conv.1"))
        x = DDColorOps.blur(DDColorOps.pixelShuffle(x, 2))
        let cat = DDColorOps.relu(MLX.concatenated([x, bn(skip, p + ".bn")], axis: -1))
        let y = bn(DDColorOps.relu(conv(cat, p + ".conv.0", bias: false, pad: 1)), p + ".conv.2")
        keep("decoder.layers.\(index)", y); return y
    }

    private func colorDecoder(_ scales: [MLXArray], image: MLXArray) -> MLXArray {
        let cd = "decoder.color_decoder"
        var src: [MLXArray] = [], pos: [MLXArray] = []
        for i in 0..<Self.numScales {
            let f = scales[i], h = f.dim(1), wd = f.dim(2)
            pos.append(DDColorOps.positionEmbedding(h: h, w: wd).asType(dtype))
            let projected = conv(f, "\(cd).input_proj.\(i)").reshaped([h * wd, Self.hiddenDim])
            src.append(projected + a("\(cd).level_embed.weight")[i])
        }
        let queryEmbed = a("\(cd).query_embed.weight")
        var output = a("\(cd).query_feat.weight")
        for i in 0..<Self.decLayers {
            let level = i % Self.numScales
            let cb = "\(cd).transformer_cross_attention_layers.\(i)", sb = "\(cd).transformer_self_attention_layers.\(i)"
            let fb = "\(cd).transformer_ffn_layers.\(i)"
            var t = attention(output + queryEmbed, src[level] + pos[level], src[level], cb + ".multihead_attn")
            output = ln(output + t, cb + ".norm", eps: 1e-5)
            let qk = output + queryEmbed
            t = attention(qk, qk, output, sb + ".self_attn")
            output = ln(output + t, sb + ".norm", eps: 1e-5)
            let ff = linear(DDColorOps.relu(linear(output, fb + ".linear1")), fb + ".linear2")
            output = ln(output + ff, fb + ".norm", eps: 1e-5)
        }
        output = ln(output, "\(cd).decoder_norm", eps: 1e-5)
        var embedding = output
        for layer in 0..<3 {
            embedding = linear(embedding, "\(cd).color_embed.layers.\(layer)")
            if layer < 2 { embedding = DDColorOps.relu(embedding) }
        }
        let h = image.dim(1), wd = image.dim(2), c = image.dim(3)
        let out = MLX.matmul(embedding, image.reshaped([h * wd, c]).transposed())
        let color = out.reshaped([Self.numQueries, h, wd]).transposed(1, 2, 0).reshaped([1, h, wd, Self.numQueries])
        keep("decoder.color_decoder", color); return color
    }

    private func attention(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, _ p: String) -> MLXArray {
        DDColorOps.attention(
            q: q, k: k, v: v, inWeight: a(p + ".in_proj_weight"), inBias: a(p + ".in_proj_bias"), outWeight: a(p + ".out_proj.weight"),
            outBias: a(p + ".out_proj.bias"), heads: Self.numHeads)
    }

    /// Gray-RGB in [0,1] (any float dtype) to raw ab, both NHWC. The ImageNet normalization uses the checkpoint's own
    /// `mean`/`std` buffers, and the normalized input is what the refine head concatenates, as in the provider's forward.
    public func callAsFunction(_ grayRGB: MLXArray) -> MLXArray {
        captured = [:]
        let x = (grayRGB.asType(dtype) - a("mean")) / a("std")
        let features = encoder(x)
        let out0 = unetBlock(features[3], skip: features[2], index: 0)
        let out1 = unetBlock(out0, skip: features[1], index: 1)
        let out2 = unetBlock(out1, skip: features[0], index: 2)
        let out3 = DDColorOps.blur(DDColorOps.pixelShuffle(DDColorOps.relu(conv(out2, "decoder.last_shuf.conv.0")), 4))
        keep("decoder.last_shuf", out3)
        let color = colorDecoder([out0, out1, out2], image: out3)
        let ab = conv(MLX.concatenated([color, x], axis: -1), "refine_net.0.0")
        keep("ab", ab); return ab
    }
}
