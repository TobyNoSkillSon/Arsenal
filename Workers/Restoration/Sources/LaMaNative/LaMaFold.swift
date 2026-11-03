// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXRandom

public struct LaMaWeightError: Error, LocalizedError {
    public let message: String
    public var errorDescription: String? { message }
}

/// Big-LaMa's eval BatchNorms folded into the convolutions before them. The original checkpoint carries running
/// variances up to 4.8e5 (past FP16's 65,504) and pre-norm activations in the thousands; folded, every block is
/// conv + bias over normalized values and the generator runs in FP16. The fold is exact algebra on the float32
/// source (scale = gamma / sqrt(var + eps), shift = beta - mean * scale), done once at Get.
public enum LaMaFold {
    static let eps: Float = 1e-5
    static let blocks = 5...22

    /// FFC_BN_ACT prefixes with (global input, global output, kernel, in, out local/global channels).
    struct FFCSpec { let prefix: String; let global: Bool; let toGlobal: Bool; let k: Int; let inL: Int; let inG: Int; let outL: Int; let outG: Int }
    static var ffcs: [FFCSpec] {
        var list = [
            FFCSpec(prefix: "model.1", global: false, toGlobal: false, k: 7, inL: 4, inG: 0, outL: 64, outG: 0),
            FFCSpec(prefix: "model.2", global: false, toGlobal: false, k: 3, inL: 64, inG: 0, outL: 128, outG: 0),
            FFCSpec(prefix: "model.3", global: false, toGlobal: false, k: 3, inL: 128, inG: 0, outL: 256, outG: 0),
            FFCSpec(prefix: "model.4", global: false, toGlobal: true, k: 3, inL: 256, inG: 0, outL: 128, outG: 384)
        ]
        for i in blocks {
            for c in ["conv1", "conv2"] {
                list.append(FFCSpec(prefix: "model.\(i).\(c)", global: true, toGlobal: true, k: 3, inL: 128, inG: 384, outL: 128, outG: 384))
            }
        }
        return list
    }
    static let ups = [(24, 25, 512, 256), (27, 28, 256, 128), (30, 31, 128, 64)]

    /// The folded inventory the Swift generator reads (NHWC / MLX ConvTranspose layout, as mlx-community/LaMa-fp32).
    public static func foldedShapes() -> [String: [Int]] {
        var s: [String: [Int]] = [:]
        for f in ffcs {
            let p = f.prefix + ".ffc."
            s[p + "convl2l.weight"] = [f.outL, f.k, f.k, f.inL]; s[p + "convl2l.bias"] = [f.outL]
            if f.global { s[p + "convg2l.weight"] = [f.outL, f.k, f.k, f.inG] }
            if f.toGlobal { s[p + "convl2g.weight"] = [f.outG, f.k, f.k, f.inL]; s[p + "convl2g.bias"] = [f.outG] }
            if f.global {
                let g = p + "convg2g."
                s[g + "conv1.0.weight"] = [f.outG / 2, 1, 1, f.inG]; s[g + "conv1.0.bias"] = [f.outG / 2]
                s[g + "fu.conv_layer.weight"] = [f.outG, 1, 1, f.outG]; s[g + "fu.conv_layer.bias"] = [f.outG]
                s[g + "conv2.weight"] = [f.outG, 1, 1, f.outG / 2]
            }
        }
        for (t, _, cin, cout) in ups { s["model.\(t).weight"] = [cout, 3, 3, cin]; s["model.\(t).bias"] = [cout] }
        s["model.34.weight"] = [3, 7, 7, 64]; s["model.34.bias"] = [3]
        return s
    }

    public static func validate(_ weights: [String: MLXArray]) throws {
        let expected = foldedShapes()
        let missing = Set(expected.keys).subtracting(weights.keys).sorted(), extra = Set(weights.keys).subtracting(expected.keys).sorted()
        guard missing.isEmpty, extra.isEmpty else {
            throw LaMaWeightError(message: "LaMa folded checkpoint differs: missing \(missing.prefix(4)), unexpected \(extra.prefix(4))")
        }
        let dtypes = Set(weights.values.map(\.dtype))
        guard dtypes.count == 1, let dtype = dtypes.first, dtype == .float16 || dtype == .bfloat16 else {
            throw LaMaWeightError(message: "LaMa weights must share one half dtype, found \(dtypes)")
        }
        for (key, shape) in expected where weights[key]!.shape != shape {
            throw LaMaWeightError(message: "LaMa tensor \(key) has shape \(weights[key]!.shape), expected \(shape)")
        }
    }

    /// Fold the original float32 generator (mlx-community/LaMa-fp32 names; BatchNorm buffers present, no
    /// num_batches_tracked) into the folded inventory, still float32. Every source tensor is consumed exactly once.
    public static func fold(_ source: [String: MLXArray]) throws -> [String: MLXArray] {
        // Weight preparation, not inference: the whole fold runs on the CPU device.
        try Device.withDefaultDevice(.cpu) { try foldOnDefaultDevice(source) }
    }

    private static func foldOnDefaultDevice(_ source: [String: MLXArray]) throws -> [String: MLXArray] {
        var used = Set<String>()
        func get(_ k: String) throws -> MLXArray {
            guard let v = source[k] else { throw LaMaWeightError(message: "LaMa source lacks \(k)") }
            used.insert(k); return v.asType(.float32, stream: .cpu)
        }
        func bn(_ p: String) throws -> (MLXArray, MLXArray) {
            let scale = try get(p + ".weight") * MLX.rsqrt(try get(p + ".running_var") + eps, stream: .cpu)
            return (scale, try get(p + ".bias") - (try get(p + ".running_mean")) * scale)
        }
        func scaled(_ k: String, _ s: MLXArray) throws -> MLXArray { try get(k) * s.reshaped([-1, 1, 1, 1]) }
        var out: [String: MLXArray] = [:]
        for f in ffcs {
            let p = f.prefix + ".ffc."
            let (sl, tl) = try bn(f.prefix + ".bn_l")
            out[p + "convl2l.weight"] = try scaled(p + "convl2l.weight", sl); out[p + "convl2l.bias"] = tl
            if f.global { out[p + "convg2l.weight"] = try scaled(p + "convg2l.weight", sl) }
            if f.toGlobal {
                let (sg, tg) = try bn(f.prefix + ".bn_g")
                out[p + "convl2g.weight"] = try scaled(p + "convl2g.weight", sg); out[p + "convl2g.bias"] = tg
                if f.global {
                    let g = p + "convg2g."
                    out[g + "conv2.weight"] = try scaled(g + "conv2.weight", sg)
                    let (s1, t1) = try bn(g + "conv1.1")
                    out[g + "conv1.0.weight"] = try scaled(g + "conv1.0.weight", s1); out[g + "conv1.0.bias"] = t1
                    let (sf, tf) = try bn(g + "fu.bn")
                    out[g + "fu.conv_layer.weight"] = try scaled(g + "fu.conv_layer.weight", sf); out[g + "fu.conv_layer.bias"] = tf
                }
            }
        }
        for (t, b, _, _) in ups {
            let (s, shift) = try bn("model.\(b)")
            out["model.\(t).weight"] = try scaled("model.\(t).weight", s)
            out["model.\(t).bias"] = try get("model.\(t).bias") * s + shift
        }
        out["model.34.weight"] = try get("model.34.weight"); out["model.34.bias"] = try get("model.34.bias")
        let orphans = Set(source.keys).subtracting(used).filter { !$0.hasSuffix("num_batches_tracked") }
        guard orphans.isEmpty else { throw LaMaWeightError(message: "LaMa source has unfolded tensors: \(orphans.sorted().prefix(4))") }
        return out
    }

    /// Random folded half weights over the real inventory (lab smokes only).
    public static func random(dtype: DType, seed: UInt64 = 0) -> [String: MLXArray] {
        MLXRandom.seed(seed)
        var out: [String: MLXArray] = [:]
        for (key, shape) in foldedShapes() {
            out[key] = (key.hasSuffix(".bias") ? MLX.zeros(shape) : MLXRandom.normal(shape) * 0.02).asType(dtype)
        }
        return out
    }
}
