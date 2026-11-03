// SPDX-License-Identifier: MIT
// The four maker checkpoints of UniPASE (Xiaobin-Rong/unipase: DeWavLM-Omni.pt, Adapter.pt, Vocoder_DWO-L1.pt,
// PostNet.pt; PyTorch state dicts under "model") mapped onto this port's one parameter table, in MLX layouts.
// One table of names and shapes serves conversion, loading and the no-weights check.
import Foundation
import MLX
import MLXRandom

public struct UniPASEError: Error, LocalizedError, Sendable {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

public enum UniPASEWeights {
    public static let fileName = "model.safetensors"
    /// Checkpoint file → parameter prefix in this port.
    public static let checkpoints: [(file: String, prefix: String)] = [
        ("DeWavLM-Omni.pt", "wavlm."), ("Adapter.pt", "adapter."), ("Vocoder_DWO-L1.pt", "vocoder."), ("PostNet.pt", "postnet.")
    ]
    /// Tensors per checkpoint and in total: 488 + 162 + 161 + 226 = 1037 maker tensors become 1033 parameters
    /// (the pos_conv weight norm folds weight_g and weight_v into one weight; WavLM's unused final encoder LayerNorm
    /// and the vocoder's Hann window buffer are checked and dropped).
    public static let sourceTensorCounts = ["DeWavLM-Omni.pt": 488, "Adapter.pt": 162, "Vocoder_DWO-L1.pt": 161, "PostNet.pt": 226]
    public static let parameterCount = 1033

    public enum Layout: Equatable, Sendable {
        /// As stored (vectors, matrices, embeddings).
        case keep
        /// PyTorch Conv1d [out, in, k] → MLX [out, k, in].
        case conv1d
        /// A 1×1 Conv1d [out, in, 1] → a Linear weight [out, in].
        case pointwise1d
        /// PyTorch Conv2d [out, in, kh, kw] → MLX [out, kh, kw, in].
        case conv2d
        /// A 1×1 Conv2d [out, in, 1, 1] → a Linear weight [out, in].
        case pointwise2d
        /// PyTorch ConvTranspose1d [in, out, k], stride 1 → the equivalent Conv1d weight [out, k, in], k reversed.
        case transposed1d
        /// PyTorch ConvTranspose2d [in, out, kh, kw], stride 1 → the equivalent Conv2d weight [out, kh, kw, in], both reversed.
        case transposed2d
        /// Any shape → a flat vector (per-channel affines and slopes stored with broadcast axes).
        case flat
    }

    /// Port key and layout for a maker key of `file`, or nil for a tensor the conversion handles itself.
    public static func map(_ key: String, file: String) -> (key: String, layout: Layout)? {
        guard let prefix = checkpoints.first(where: { $0.file == file })?.prefix else { return nil }
        let name = prefix + key
        switch prefix {
        case "wavlm.":
            if key.hasPrefix("encoder.pos_conv.0.weight_") || key.hasPrefix("encoder.layer_norm.") { return nil }
            if key.hasPrefix("feature_extractor.") && key.hasSuffix(".0.weight") { return (name, .conv1d) }
            if key.hasSuffix(".grep_a") { return (name, .flat) }
            return (name, .keep)
        case "adapter.", "vocoder.":
            if key == "head.istft.window" { return nil }
            if key.hasSuffix(".q.weight") || key.hasSuffix(".k.weight") || key.hasSuffix(".v.weight") || key.hasSuffix(".proj_out.weight") {
                return (name, .pointwise1d)
            }
            if key.hasSuffix(".weight") && (key.contains(".embed.") || key.contains(".dwconv.") || key.contains(".conv1.") || key.contains(".conv2.")) {
                return (name, .conv1d)
            }
            return (name, .keep)
        default:
            if key == "conv.0.weight" { return (name, .conv2d) }
            if key == "deconv.weight" { return (name, .transposed2d) }
            if key.hasSuffix("_linear.weight") { return (name, .transposed1d) }
            if (key.contains(".attn_conv_") && key.hasSuffix(".weight")) || key.hasSuffix("attn_concat_proj.0.weight") { return (name, .pointwise2d) }
            if key.hasSuffix(".gamma") || key.hasSuffix(".beta") { return (name, .flat) }
            return (name, .keep)
        }
    }

    public static func arrange(_ a: MLXArray, layout: Layout) -> MLXArray {
        switch layout {
        case .keep: return a
        case .conv1d: return a.transposed(0, 2, 1)
        case .pointwise1d: return a.reshaped(a.dim(0), a.dim(1))
        case .conv2d: return a.transposed(0, 2, 3, 1)
        case .pointwise2d: return a.reshaped(a.dim(0), a.dim(1))
        case .transposed1d: return a.transposed(1, 2, 0)[0..., .stride(by: -1), 0...]
        case .transposed2d: return a.transposed(1, 2, 3, 0)[0..., .stride(by: -1), .stride(by: -1), 0...]
        case .flat: return a.reshaped(-1)
        }
    }

    /// WavLM's convolutional position embedding is weight-normalised over every axis but the kernel axis (dim=2):
    /// weight = g · v / ‖v‖, folded once in float32 → MLX [out, k, in/groups].
    public static func foldWeightNorm(g: MLXArray, v: MLXArray) -> MLXArray {
        let vf = v.asType(.float32), norm = sqrt(vf.square().sum(axes: [0, 1], keepDims: true))
        return (g.asType(.float32) * vf / norm).transposed(0, 2, 1)
    }

    /// Every parameter this port reads, with its MLX shape.
    public static let shapes: [String: [Int]] = {
        var s: [String: [Int]] = [:]
        // DeWavLM-Omni (WavLM Large: layer_norm extractor, 24 pre-norm layers, gated relative position bias).
        let cnn: [(Int, Int)] = [(10, 1)] + Array(repeating: (3, 512), count: 4) + Array(repeating: (2, 512), count: 2)
        for (i, (k, cin)) in cnn.enumerated() {
            s["wavlm.feature_extractor.conv_layers.\(i).0.weight"] = [512, k, cin]
            s["wavlm.feature_extractor.conv_layers.\(i).2.1.weight"] = [512]
            s["wavlm.feature_extractor.conv_layers.\(i).2.1.bias"] = [512]
        }
        s["wavlm.layer_norm.weight"] = [512]; s["wavlm.layer_norm.bias"] = [512]
        s["wavlm.post_extract_proj.weight"] = [1024, 512]; s["wavlm.post_extract_proj.bias"] = [1024]
        s["wavlm.mask_emb"] = [1024]
        s["wavlm.encoder.pos_conv.0.weight"] = [1024, 128, 64]; s["wavlm.encoder.pos_conv.0.bias"] = [1024]
        for i in 0..<24 {
            let p = "wavlm.encoder.layers.\(i)."
            for proj in ["q_proj", "k_proj", "v_proj", "out_proj"] { s[p + "self_attn.\(proj).weight"] = [1024, 1024]; s[p + "self_attn.\(proj).bias"] = [1024] }
            s[p + "self_attn.grep_linear.weight"] = [8, 64]; s[p + "self_attn.grep_linear.bias"] = [8]; s[p + "self_attn.grep_a"] = [16]
            if i == 0 { s[p + "self_attn.relative_attention_bias.weight"] = [320, 16] }
            for norm in ["self_attn_layer_norm", "final_layer_norm"] { s[p + norm + ".weight"] = [1024]; s[p + norm + ".bias"] = [1024] }
            s[p + "fc1.weight"] = [4096, 1024]; s[p + "fc1.bias"] = [4096]; s[p + "fc2.weight"] = [1024, 4096]; s[p + "fc2.bias"] = [1024]
        }
        // Vocos backbones (adapter and vocoder share the layout: dim 1024, ConvNeXt 3072, 12 layers, 4 ResNet + 1 attention).
        for net in ["adapter.", "vocoder."] {
            let d = net + "decoder."
            s[d + "embed.weight"] = [1024, 7, 1024]; s[d + "embed.bias"] = [1024]
            for n in ["norm", "final_layer_norm", "pos_net.5"] { s[d + n + ".weight"] = [1024]; s[d + n + ".bias"] = [1024] }
            for r in [0, 1, 3, 4] {
                let p = d + "pos_net.\(r)."
                for n in ["norm1", "norm2"] { s[p + n + ".weight"] = [1024]; s[p + n + ".bias"] = [1024] }
                for c in ["conv1", "conv2"] { s[p + c + ".weight"] = [1024, 3, 1024]; s[p + c + ".bias"] = [1024] }
            }
            s[d + "pos_net.2.norm.weight"] = [1024]; s[d + "pos_net.2.norm.bias"] = [1024]
            for c in ["q", "k", "v", "proj_out"] { s[d + "pos_net.2.\(c).weight"] = [1024, 1024]; s[d + "pos_net.2.\(c).bias"] = [1024] }
            for i in 0..<12 {
                let p = d + "convnext.\(i)."
                s[p + "gamma"] = [1024]; s[p + "dwconv.weight"] = [1024, 7, 1]; s[p + "dwconv.bias"] = [1024]
                s[p + "norm.weight"] = [1024]; s[p + "norm.bias"] = [1024]
                s[p + "pwconv1.weight"] = [3072, 1024]; s[p + "pwconv1.bias"] = [3072]; s[p + "pwconv2.weight"] = [1024, 3072]; s[p + "pwconv2.bias"] = [1024]
            }
        }
        s["adapter.proj.weight"] = [1024, 1024]; s["adapter.proj.bias"] = [1024]
        s["adapter.head.weight"] = [1024, 1024]; s["adapter.head.bias"] = [1024]
        s["vocoder.head.out.weight"] = [1282, 1024]; s["vocoder.head.out.bias"] = [1282]
        // TFGridNet PostNet: 48 channels, 5 blocks, BiLSTM 100, 4 heads with E = 2, unfold 4 at hop 1.
        s["postnet.conv.0.weight"] = [48, 3, 3, 6]; s["postnet.conv.0.bias"] = [48]
        s["postnet.conv.1.weight"] = [48]; s["postnet.conv.1.bias"] = [48]
        s["postnet.deconv.weight"] = [6, 3, 3, 48]; s["postnet.deconv.bias"] = [6]
        for b in 0..<5 {
            let p = "postnet.blocks.\(b)."
            for side in ["intra", "inter"] {
                s[p + side + "_norm.weight"] = [48]; s[p + side + "_norm.bias"] = [48]
                for suffix in ["", "_reverse"] {
                    s[p + side + "_rnn.weight_ih_l0" + suffix] = [400, 192]; s[p + side + "_rnn.weight_hh_l0" + suffix] = [400, 100]
                    s[p + side + "_rnn.bias_ih_l0" + suffix] = [400]; s[p + side + "_rnn.bias_hh_l0" + suffix] = [400]
                }
                s[p + side + "_linear.weight"] = [48, 4, 200]; s[p + side + "_linear.bias"] = [48]
            }
            for (qkv, out, per) in [("Q", 8, 2), ("K", 8, 2), ("V", 48, 12)] {
                s[p + "attn_conv_\(qkv).weight"] = [out, 48]; s[p + "attn_conv_\(qkv).bias"] = [out]
                s[p + "attn_norm_\(qkv).gamma"] = [4 * per]; s[p + "attn_norm_\(qkv).beta"] = [4 * per]; s[p + "attn_norm_\(qkv).act.weight"] = [4]
            }
            s[p + "attn_concat_proj.0.weight"] = [48, 48]; s[p + "attn_concat_proj.0.bias"] = [48]
            s[p + "attn_concat_proj.1.weight"] = [1]
            s[p + "attn_concat_proj.2.gamma"] = [48]; s[p + "attn_concat_proj.2.beta"] = [48]
        }
        return s
    }()

    /// Loads the converted file; every parameter must be present with its exact shape and nothing else.
    public static func load(_ url: URL, dtype: DType) throws -> [String: MLXArray] {
        let arrays = try MLX.loadArrays(url: url)
        try check(arrays.mapValues(\.shape))
        let cast = arrays.mapValues { $0.dtype == dtype ? $0 : $0.asType(dtype) }
        eval(Array(cast.values))
        return cast
    }

    /// Names and shapes against the table.
    public static func check(_ found: [String: [Int]]) throws {
        guard found.count == parameterCount, shapes.count == parameterCount else {
            throw UniPASEError("UniPASE weights hold \(found.count) tensors, expected \(parameterCount)")
        }
        for (name, shape) in shapes where found[name] != shape {
            throw UniPASEError("UniPASE weight \(name) is \(found[name].map { "\($0)" } ?? "missing"), expected \(shape)")
        }
    }

    /// The no-weights contract check: small random weights with unit norm scales, in the serving dtype.
    public static func random(dtype: DType, seed: UInt64 = 20_261_102) -> [String: MLXArray] {
        let keys = MLXRandom.split(key: MLXRandom.key(seed), into: shapes.count)
        var out: [String: MLXArray] = [:]
        for (i, (name, shape)) in shapes.sorted(by: { $0.key < $1.key }).enumerated() {
            let unit =
                name.hasSuffix("norm.weight") || name.hasSuffix(".gamma") || name.contains("norm1.weight") || name.contains("norm2.weight")
                || name.hasSuffix("pos_net.5.weight") || name.hasSuffix(".2.1.weight") || name.hasSuffix("conv.1.weight")
            out[name] = unit ? MLXArray.ones(shape, dtype: dtype) : (MLXRandom.normal(shape, key: keys[i]) * 0.02).asType(dtype)
        }
        eval(Array(out.values))
        return out
    }
}
