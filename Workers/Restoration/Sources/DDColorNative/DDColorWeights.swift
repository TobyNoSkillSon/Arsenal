// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXRandom

public struct DDColorWeightError: Error, LocalizedError {
    public let message: String
    public var errorDescription: String? { message }
}

/// The exact tensor inventory of mlx-community/DDColor-modelscope-fp16 (575 tensors, NHWC kernels, spectral norm resolved),
/// so Load refuses a checkpoint that is missing, adds or reshapes anything, and random-weight smokes build the real graph.
public enum DDColorWeights {
    public static func shapes() -> [String: [Int]] {
        var s: [String: [Int]] = ["mean": [1, 1, 1, 3], "std": [1, 1, 1, 3]]
        let dims = DDColorModel.dims, hd = DDColorModel.hiddenDim
        func vec(_ k: String, _ n: Int, _ names: [String] = ["weight", "bias"]) { for name in names { s[k + "." + name] = [n] } }
        s["encoder.arch.downsample_layers.0.0.weight"] = [dims[0], 4, 4, 3]; vec("encoder.arch.downsample_layers.0.0", dims[0], ["bias"])
        vec("encoder.arch.downsample_layers.0.1", dims[0])
        for i in 1..<4 {
            vec("encoder.arch.downsample_layers.\(i).0", dims[i - 1])
            s["encoder.arch.downsample_layers.\(i).1.weight"] = [dims[i], 2, 2, dims[i - 1]]; vec("encoder.arch.downsample_layers.\(i).1", dims[i], ["bias"])
        }
        for i in 0..<4 {
            let d = dims[i]
            for j in 0..<DDColorModel.depths[i] {
                let p = "encoder.arch.stages.\(i).\(j)"
                s[p + ".dwconv.weight"] = [d, 7, 7, 1]; vec(p + ".dwconv", d, ["bias"]); vec(p + ".norm", d); s[p + ".gamma"] = [d]
                s[p + ".pwconv1.weight"] = [4 * d, d]; vec(p + ".pwconv1", 4 * d, ["bias"])
                s[p + ".pwconv2.weight"] = [d, 4 * d]; vec(p + ".pwconv2", d, ["bias"])
            }
            vec("encoder.arch.norm\(i)", d)
        }
        // UnetBlockWide i: input channels (up), skip channels, shuffle output channels, block output channels.
        let blocks = [(dims[3], dims[2], 2048, 512), (512, dims[1], 2048, 512), (512, dims[0], 1024, 256)]
        let bnNames = ["weight", "bias", "running_mean", "running_var"]
        for (i, (up, skip, shuf, out)) in blocks.enumerated() {
            let p = "decoder.layers.\(i)"
            s[p + ".shuf.conv.0.weight"] = [shuf, 1, 1, up]; vec(p + ".shuf.conv.1", shuf, bnNames)
            vec(p + ".bn", skip, bnNames)
            s[p + ".conv.0.weight"] = [out, 3, 3, shuf / 4 + skip]; vec(p + ".conv.2", out, bnNames)
        }
        s["decoder.last_shuf.conv.0.weight"] = [4096, 1, 1, 256]; vec("decoder.last_shuf.conv.0", 4096, ["bias"])
        let cd = "decoder.color_decoder"
        for (i, c) in [512, 512, 256].enumerated() { s["\(cd).input_proj.\(i).weight"] = [hd, 1, 1, c]; vec("\(cd).input_proj.\(i)", hd, ["bias"]) }
        s["\(cd).level_embed.weight"] = [3, hd]; s["\(cd).query_embed.weight"] = [100, hd]; s["\(cd).query_feat.weight"] = [100, hd]
        vec("\(cd).decoder_norm", hd)
        for i in 0..<3 { s["\(cd).color_embed.layers.\(i).weight"] = [hd, hd]; vec("\(cd).color_embed.layers.\(i)", hd, ["bias"]) }
        for i in 0..<DDColorModel.decLayers {
            for (layer, attn) in [("transformer_cross_attention_layers", "multihead_attn"), ("transformer_self_attention_layers", "self_attn")] {
                let p = "\(cd).\(layer).\(i)"
                s[p + ".\(attn).in_proj_weight"] = [3 * hd, hd]; s[p + ".\(attn).in_proj_bias"] = [3 * hd]
                s[p + ".\(attn).out_proj.weight"] = [hd, hd]; vec(p + ".\(attn).out_proj", hd, ["bias"]); vec(p + ".norm", hd)
            }
            let f = "\(cd).transformer_ffn_layers.\(i)"
            s[f + ".linear1.weight"] = [2048, hd]; vec(f + ".linear1", 2048, ["bias"])
            s[f + ".linear2.weight"] = [hd, 2048]; vec(f + ".linear2", hd, ["bias"]); vec(f + ".norm", hd)
        }
        s["refine_net.0.0.weight"] = [2, 1, 1, 103]; vec("refine_net.0.0", 2, ["bias"])
        return s
    }

    public static func validate(_ weights: [String: MLXArray]) throws {
        let expected = shapes()
        let missing = Set(expected.keys).subtracting(weights.keys).sorted(), extra = Set(weights.keys).subtracting(expected.keys).sorted()
        guard missing.isEmpty, extra.isEmpty else {
            throw DDColorWeightError(message: "DDColor checkpoint tensors differ: missing \(missing.prefix(4)), unexpected \(extra.prefix(4))")
        }
        let dtypes = Set(weights.values.map(\.dtype))
        guard dtypes.count == 1, let dtype = dtypes.first, dtype == .float16 || dtype == .bfloat16 else {
            throw DDColorWeightError(message: "DDColor weights must share one half dtype, found \(dtypes)")
        }
        for (key, shape) in expected where weights[key]!.shape != shape {
            throw DDColorWeightError(message: "DDColor tensor \(key) has shape \(weights[key]!.shape), expected \(shape)")
        }
    }

    /// Random half weights over the real inventory (lab smokes only). Variances stay positive and norms near one, so the
    /// random graph runs finite end to end.
    public static func random(dtype: DType, seed: UInt64 = 0) -> [String: MLXArray] {
        MLXRandom.seed(seed)
        var out: [String: MLXArray] = [:]
        for (key, shape) in shapes() {
            let value: MLXArray
            if key.hasSuffix("running_var") || key == "std" || key.hasSuffix("norm.weight") || key.hasSuffix(".1.weight") && shape.count == 1 {
                value = MLX.ones(shape)
            } else if key.hasSuffix("running_mean") || key == "mean" || key.hasSuffix("bias") {
                value = MLX.zeros(shape)
            } else {
                value = MLXRandom.normal(shape) * 0.02
            }
            out[key] = value.asType(dtype)
        }
        return out
    }
}
