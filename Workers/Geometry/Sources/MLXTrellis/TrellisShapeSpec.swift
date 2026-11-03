// SPDX-License-Identifier: MIT
import Foundation

/// Independent architecture inventory: dimensions come from the pinned model configs, not checkpoint headers.
public struct TrellisShapeSpec {
    public let lab: Bool
    public var channels: Int { lab ? 64 : 1536 }
    public var condition: Int { lab ? 64 : 1024 }
    public var heads: Int { lab ? 4 : 12 }
    public var blocks: Int { lab ? 1 : 30 }
    public var ssResolution: Int { lab ? 1 : 16 }
    public var decoderChannels: [Int] { lab ? [64, 64, 64, 64, 64] : [1024, 512, 256, 128, 64] }
    public var decoderBlocks: [Int] { lab ? [1, 1, 1, 1, 0] : [4, 16, 8, 4, 0] }
    public var ssDecoderChannels: [Int] { lab ? [16, 8, 8] : [512, 128, 32] }
    public init(lab: Bool) { self.lab = lab }

    public func flow(input: Int, output: Int) -> [String: [Int]] {
        var result: [String: [Int]] = [:]
        func linear(_ name: String, _ input: Int, _ output: Int) { result[name + ".weight"] = [output, input]; result[name + ".bias"] = [output] }
        let c = channels, h = lab ? 128 : 8192
        linear("input_layer", input, c); linear("out_layer", c, output)
        linear("adaLN_modulation.1", c, c * 6)
        linear("t_embedder.mlp.0", 256, c); linear("t_embedder.mlp.2", c, c)
        for i in 0..<blocks {
            let prefix = "blocks.\(i)."
            result[prefix + "modulation"] = [6 * c]
            result[prefix + "norm2.weight"] = [c]; result[prefix + "norm2.bias"] = [c]
            for kind in ["self_attn", "cross_attn"] {
                for qk in ["q", "k"] { result[prefix + kind + "." + qk + "_rms_norm.gamma"] = [heads, c / heads] }
                linear(prefix + kind + ".to_out", c, c)
            }
            linear(prefix + "self_attn.to_qkv", c, c * 3)
            linear(prefix + "cross_attn.to_q", c, c)
            linear(prefix + "cross_attn.to_kv", condition, c * 2)
            linear(prefix + "mlp.mlp.0", c, h); linear(prefix + "mlp.mlp.2", h, c)
        }
        return result
    }
    public func slatDecoder(texture: Bool) -> [String: [Int]] {
        var result: [String: [Int]] = [:]
        func pair(_ p: String, _ shape: [Int]) { result[p + ".weight"] = shape; result[p + ".bias"] = [shape[0]] }
        let channels = decoderChannels, counts = decoderBlocks
        pair("from_latent", [channels[0], 32]); pair("output_layer", [texture ? 6 : 7, channels.last!])
        for i in channels.indices {
            let c = channels[i]
            for j in 0..<counts[i] {
                let prefix = "blocks.\(i).\(j)."
                pair(prefix + "conv", [c, 3, 3, 3, c]); pair(prefix + "norm", [c])
                pair(prefix + "mlp.0", [c * 4, c]); pair(prefix + "mlp.2", [c, c * 4])
            }
            if i + 1 < channels.count {
                let prefix = "blocks.\(i).\(counts[i]).", output = channels[i + 1]
                pair(prefix + "norm1", [c]); pair(prefix + "conv1", [output * 8, 3, 3, 3, c])
                pair(prefix + "conv2", [output, 3, 3, 3, output])
                if !texture { pair(prefix + "to_subdiv", [8, c]) }
            }
        }
        return result
    }
    public func sparseStructureDecoder() -> [String: [Int]] {
        var result: [String: [Int]] = [:]
        func pair(_ p: String, _ shape: [Int]) { result[p + ".weight"] = shape; result[p + ".bias"] = [shape[0]] }
        func residual(_ p: String, _ c: Int) {
            for i in [1, 2] { pair(p + ".norm\(i)", [c]); pair(p + ".conv\(i)", [c, c, 3, 3, 3]) }
        }
        let c = ssDecoderChannels
        pair("input_layer", [c[0], 8, 3, 3, 3])
        for i in [0, 1] { residual("middle_block.\(i)", c[0]) }
        for (i, ch) in [(0, c[0]), (1, c[0]), (3, c[1]), (4, c[1]), (6, c[2]), (7, c[2])] { residual("blocks.\(i)", ch) }
        pair("blocks.2.conv", [c[1] * 8, c[0], 3, 3, 3]); pair("blocks.5.conv", [c[2] * 8, c[1], 3, 3, 3])
        pair("out_layer.0", [c[2]]); pair("out_layer.2", [1, c[2], 3, 3, 3])
        return result
    }
    public func dino() -> [String: [Int]] {
        var result: [String: [Int]] = [:]
        let c = condition, count = lab ? 1 : 24, mlp = c * 4
        func pair(_ p: String, _ shape: [Int], bias: Bool = true) { result[p + ".weight"] = shape; if bias { result[p + ".bias"] = [shape[0]] } }
        pair("embeddings.patch_embeddings", [c, 3, 16, 16]); result["embeddings.cls_token"] = [1, 1, c]; result["embeddings.register_tokens"] = [1, 4, c]
        result["embeddings.mask_token"] = [1, 1, c]
        pair("norm", [c])
        for i in 0..<count {
            let p = "layer.\(i)."
            pair(p + "norm1", [c]); pair(p + "norm2", [c])
            for projection in ["q", "k", "v", "o"] { pair(p + "attention." + projection + "_proj", [c, c], bias: projection != "k") }
            pair(p + "mlp.up_proj", [mlp, c]); pair(p + "mlp.down_proj", [c, mlp])
            result[p + "layer_scale1.lambda1"] = [c]; result[p + "layer_scale2.lambda1"] = [c]
        }
        return result
    }
    public var components: [String: [String: [Int]]] {
        [
            "struct_flow": flow(input: 8, output: 8), "shape_flow_512": flow(input: 32, output: 32), "shape_flow_1024": flow(input: 32, output: 32),
            "tex_flow_512": flow(input: 64, output: 32), "tex_flow_1024": flow(input: 64, output: 32), "shape_dec": slatDecoder(texture: false),
            "tex_dec": slatDecoder(texture: true), "struct_dec": sparseStructureDecoder(), "dino": dino()
        ]
    }
}
