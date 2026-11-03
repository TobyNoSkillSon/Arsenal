// SPDX-License-Identifier: MIT
// Audiobox Aesthetics: a WavLM Base+ encoder (12 layers, 768 wide, gated relative position bias)
// with four small MLP heads, one per axis. Ported from the maker's audiobox_aesthetics/model/{aes,wavlm}.py
// and infer.py (10-second windows, zero-padded with a mask, weighted by their length).
import Foundation
import MLX
import MLXNN

let modelKind = "local.audio-quality"
let modelOps = ["score_quality"]

let axes = ["CE", "CU", "PC", "PQ"]
let sampleRate = 16_000
let windowSamples = 10 * sampleRate

final class Model {
    let net: Aesthetics
    let transform: [String: (mean: Double, std: Double)]

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        var t: [String: (Double, Double)] = [:]
        let tt = config["target_transform"] as? [String: Any] ?? [:]
        for a in axes {
            guard let d = tt[a] as? [String: Any], let m = (d["mean"] as? NSNumber)?.doubleValue,
                let s = (d["std"] as? NSNumber)?.doubleValue
            else { throw RunnerError.failed("config.json has no target_transform for \(a).") }
            t[a] = (m, s)
        }
        transform = t
        net = Aesthetics()
        var w = try Weights.load(weightsDir)
        // Conv1d: PyTorch [out, in, k] -> MLX [out, k, in].
        for i in 0..<7 {
            let key = "wavlm_model.feature_extractor.conv_layers.\(i).0.weight"
            w[key] = w[key]!.transposed(0, 2, 1)
        }
        // Weight-normalised positional conv (dim=2): w = g * v / ||v|| over dims 0 and 1.
        let g = w.removeValue(forKey: "wavlm_model.encoder.pos_conv.0.weight_g")!.asType(.float32)
        let v = w.removeValue(forKey: "wavlm_model.encoder.pos_conv.0.weight_v")!.asType(.float32)
        let norm = sqrt((v * v).sum(axes: [0, 1], keepDims: true))
        w["wavlm_model.encoder.pos_conv.0.weight"] = (g * v / norm).asType(.float16).transposed(0, 2, 1)
        w.removeValue(forKey: "wavlm_model.mask_emb")  // training only
        try Weights.apply(w, to: net, rename: Model.rename)
    }

    /// proj_layer.<axis>.<i>: nn.Sequential of [Linear, LayerNorm, GELU] x 4 + Linear (indices 0..12).
    static func rename(_ key: String) -> String? {
        let parts = key.split(separator: ".").map(String.init)
        // conv_layers.<i>.0 (Conv1d) and conv_layers.0.2 (GroupNorm): numeric keys would read as list indices.
        if parts.count == 6, parts[2] == "conv_layers" {
            return (parts[0...3] + [parts[4] == "0" ? "conv" : "norm", parts[5]]).joined(separator: ".")
        }
        if parts.first == "proj_layer", parts.count == 4, let i = Int(parts[2]) {
            if i % 3 == 0 { return "proj_layer.\(parts[1]).linear.\(i / 3).\(parts[3])" }
            return "proj_layer.\(parts[1]).norm.\((i - 1) / 3).\(parts[3])"
        }
        return key
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        let url = try params.file("audio")
        let wav = try AudioInput.mono(url, sampleRate: Double(sampleRate))
        let n = wav.dim(0)
        guard n > 0 else { throw RunnerError.invalid("The audio file holds no samples.") }
        // infer.py make_inference_batch: hop 10 s, window 10 s, zero-pad the last window, weight = its length share.
        var sums = [String: Double](uniqueKeysWithValues: axes.map { ($0, 0) })
        var weightSum = 0.0
        var start = 0
        while start < n {
            try params.checkCancelled()
            let len = min(windowSamples, n - start)
            var chunk = wav[start..<(start + len)]
            if len < windowSamples { chunk = concatenated([chunk, MLXArray.zeros([windowSamples - len])]) }
            let preds = net(chunk.asType(.float16).expandedDimensions(axis: 0), validSamples: len)
            let weight = Double(len) / Double(windowSamples)
            for a in axes {
                let raw = Double(preds[a]!.asType(.float32).item(Float.self))
                sums[a]! += (raw * transform[a]!.std + transform[a]!.mean) * weight
            }
            weightSum += weight
            start += windowSamples
        }
        var out: [String: Any] = [:]
        for a in axes { out[a] = sums[a]! / weightSum }
        out["seconds"] = Double(n) / Double(sampleRate)
        return out
    }
}

// MARK: - Network

final class Aesthetics: Module {
    @ModuleInfo(key: "wavlm_model") var wavlm: WavLM
    @ModuleInfo(key: "proj_layer") var proj: AxisHeads
    @ModuleInfo(key: "layer_weights") var layerWeights: AxisWeights

    override init() {
        _wavlm.wrappedValue = WavLM()
        _proj.wrappedValue = AxisHeads()
        _layerWeights.wrappedValue = AxisWeights()
    }

    /// wav [1, 160000] float16, the first `validSamples` real. Returns the normalised prediction per axis.
    func callAsFunction(_ wav: MLXArray, validSamples: Int) -> [String: MLXArray] {
        let (layers, valid) = wavlm(wav, validSamples: validSamples)  // 13 x [1, T, 768], valid [1, T] float32
        let stack = stacked(layers, axis: -1).asType(.float32)  // [1, T, 768, 13]
        let mask = valid.expandedDimensions(axis: -1)  // [1, T, 1]
        let count = maximum(mask.sum(axis: 1), MLXArray(Float(1)))
        var out: [String: MLXArray] = [:]
        for a in axes {
            let weights = softmax(layerWeights[a].asType(.float32), axis: -1)
            var embed = (stack * weights).sum(axis: -1)  // [1, T, 768]
            embed = (embed * mask).sum(axis: 1) / count  // [1, 768]
            // F.normalize: x / max(||x||, 1e-12)
            embed = embed / maximum(sqrt((embed * embed).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
            out[a] = proj[a](embed.asType(.float16)).squeezed()
        }
        return out
    }
}

final class AxisWeights: Module {
    @ParameterInfo(key: "CE") var ce: MLXArray
    @ParameterInfo(key: "CU") var cu: MLXArray
    @ParameterInfo(key: "PC") var pc: MLXArray
    @ParameterInfo(key: "PQ") var pq: MLXArray
    override init() {
        _ce.wrappedValue = MLXArray.ones([13]); _cu.wrappedValue = MLXArray.ones([13])
        _pc.wrappedValue = MLXArray.ones([13]); _pq.wrappedValue = MLXArray.ones([13])
    }
    subscript(_ a: String) -> MLXArray { a == "CE" ? ce : a == "CU" ? cu : a == "PC" ? pc : pq }
}

final class AxisHeads: Module {
    @ModuleInfo(key: "CE") var ce: Head
    @ModuleInfo(key: "CU") var cu: Head
    @ModuleInfo(key: "PC") var pc: Head
    @ModuleInfo(key: "PQ") var pq: Head
    override init() {
        _ce.wrappedValue = Head(); _cu.wrappedValue = Head(); _pc.wrappedValue = Head(); _pq.wrappedValue = Head()
    }
    subscript(_ a: String) -> Head { a == "CE" ? ce : a == "CU" ? cu : a == "PC" ? pc : pq }
}

/// create_mlp_block(768, 1, 5, gelu, layer_norm=True)
final class Head: Module, UnaryLayer {
    let linear: [Linear]
    let norm: [LayerNorm]
    override init() {
        linear = (0..<4).map { _ in Linear(768, 768) } + [Linear(768, 1)]
        norm = (0..<4).map { _ in LayerNorm(dimensions: 768, eps: 1e-5) }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var x = x
        for i in 0..<4 { x = gelu(norm[i](linear[i](x))) }
        return linear[4](x)
    }
}

final class WavLM: Module {
    @ModuleInfo(key: "feature_extractor") var featureExtractor: FeatureExtractor
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "post_extract_proj") var postExtractProj: Linear
    @ModuleInfo(key: "encoder") var encoder: Encoder

    override init() {
        _featureExtractor.wrappedValue = FeatureExtractor()
        _layerNorm.wrappedValue = LayerNorm(dimensions: 512, eps: 1e-5)
        _postExtractProj.wrappedValue = Linear(512, 768)
        _encoder.wrappedValue = Encoder()
    }

    /// Returns the 13 layer results (input to layer 1, then each layer's output) and the valid-frame mask.
    func callAsFunction(_ wav: MLXArray, validSamples: Int) -> ([MLXArray], MLXArray) {
        var x = featureExtractor(wav)  // [1, T, 512]
        x = layerNorm(x)
        let frames = x.dim(1)
        // forward_padding_mask: drop the tail remainder, a frame is padding when all its samples are padding.
        let perFrame = (wav.dim(1) - wav.dim(1) % frames) / frames
        var keep = [Float](repeating: 0, count: frames)
        for t in 0..<frames where t * perFrame < validSamples { keep[t] = 1 }
        let valid = MLXArray(keep).reshaped([1, frames])
        x = postExtractProj(x)
        return (encoder(x, valid: valid), valid)
    }
}

final class FeatureExtractor: Module {
    @ModuleInfo(key: "conv_layers") var convLayers: [ConvBlock]
    override init() {
        let spec: [(Int, Int, Int)] = [(1, 10, 5)] + Array(repeating: (512, 3, 2), count: 4) + Array(repeating: (512, 2, 2), count: 2)
        _convLayers.wrappedValue = spec.enumerated().map { i, s in ConvBlock(inC: s.0, k: s.1, stride: s.2, groupNorm: i == 0) }
    }
    func callAsFunction(_ wav: MLXArray) -> MLXArray {
        var x = wav.expandedDimensions(axis: -1)  // [1, L, 1]
        for block in convLayers { x = block(x) }
        return x
    }
}

final class ConvBlock: Module {
    @ModuleInfo(key: "conv") var conv: Conv1d
    @ModuleInfo(key: "norm") var norm: GroupNormPerChannel?
    init(inC: Int, k: Int, stride: Int, groupNorm: Bool) {
        _conv.wrappedValue = Conv1d(inputChannels: inC, outputChannels: 512, kernelSize: k, stride: stride, bias: false)
        _norm.wrappedValue = groupNorm ? GroupNormPerChannel() : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv(x)
        if let norm { y = norm(y) }
        return gelu(y)
    }
}

/// Fp32GroupNorm(512, 512): each channel normalised over time, in float32.
final class GroupNormPerChannel: Module {
    var weight = MLXArray.ones([512])
    var bias = MLXArray.zeros([512])
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let mean = x32.mean(axis: 1, keepDims: true)
        let variance = ((x32 - mean) * (x32 - mean)).mean(axis: 1, keepDims: true)
        let y = (x32 - mean) * rsqrt(variance + 1e-5) * weight.asType(.float32) + bias.asType(.float32)
        return y.asType(x.dtype)
    }
}

final class PosConv: Module {
    var weight = MLXArray.zeros([768, 128, 48])
    var bias = MLXArray.zeros([768])
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = conv1d(x, weight, stride: 1, padding: 64, groups: 16) + bias
        y = y[0..., 0..<(y.dim(1) - 1), 0...]  // SamePad(128) removes one frame
        return gelu(y)
    }
}

final class Encoder: Module {
    @ModuleInfo(key: "pos_conv") var posConv: [PosConv]
    @ModuleInfo(key: "layer_norm") var layerNorm: LayerNorm
    @ModuleInfo(key: "layers") var layers: [EncoderLayer]

    override init() {
        _posConv.wrappedValue = [PosConv()]
        _layerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
        _layers.wrappedValue = (0..<12).map { EncoderLayer(relativeBias: $0 == 0) }
    }

    func callAsFunction(_ input: MLXArray, valid: MLXArray) -> [MLXArray] {
        var x = input * valid.expandedDimensions(axis: -1).asType(input.dtype)  // x[padding_mask] = 0
        x = x + posConv[0](x)
        x = layerNorm(x)
        var results = [x]
        let t = x.dim(1)
        let bias = layers[0].selfAttn.positionBias(length: t)  // [1, 12, T, T], shared by every layer
        // key_padding_mask -> -inf on padded keys
        let keyMask = MLX.where(valid .> 0, MLXArray(Float(0)), MLXArray(-Float.infinity)).reshaped([1, 1, 1, t])
        for layer in layers {
            x = layer(x, positionBias: bias, keyMask: keyMask)
            results.append(x)
        }
        return results
    }
}

final class EncoderLayer: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: Attention
    @ModuleInfo(key: "self_attn_layer_norm") var attnNorm: LayerNorm
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    @ModuleInfo(key: "final_layer_norm") var finalNorm: LayerNorm

    init(relativeBias: Bool) {
        _selfAttn.wrappedValue = Attention(relativeBias: relativeBias)
        _attnNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
        _fc1.wrappedValue = Linear(768, 3072)
        _fc2.wrappedValue = Linear(3072, 768)
        _finalNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
    }

    /// Post-norm (layer_norm_first = False).
    func callAsFunction(_ x: MLXArray, positionBias: MLXArray, keyMask: MLXArray) -> MLXArray {
        var h = attnNorm(x + selfAttn(x, positionBias: positionBias, keyMask: keyMask))
        h = finalNorm(h + fc2(gelu(fc1(h))))
        return h
    }
}

final class Attention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "grep_linear") var grepLinear: Linear
    @ParameterInfo(key: "grep_a") var grepA: MLXArray
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding?

    let heads = 12, headDim = 64

    init(relativeBias: Bool) {
        _qProj.wrappedValue = Linear(768, 768)
        _kProj.wrappedValue = Linear(768, 768)
        _vProj.wrappedValue = Linear(768, 768)
        _outProj.wrappedValue = Linear(768, 768)
        _grepLinear.wrappedValue = Linear(64, 8)
        _grepA.wrappedValue = MLXArray.ones([1, 12, 1, 1])
        _relativeAttentionBias.wrappedValue = relativeBias ? Embedding(embeddingCount: 320, dimensions: 12) : nil
    }

    /// compute_bias with _relative_positions_bucket(bidirectional=True), num_buckets 320, max_distance 800.
    func positionBias(length t: Int) -> MLXArray {
        let numBuckets = 160, maxExact = 80
        let scale = Float(log(Double(800) / Double(maxExact)))
        var buckets = [Int32](repeating: 0, count: t * t)
        for i in 0..<t {
            for j in 0..<t {
                let rel = j - i
                var bucket = rel > 0 ? numBuckets : 0
                let a = abs(rel)
                if a < maxExact {
                    bucket += a
                } else {
                    let large = maxExact + Int(log(Float(a) / Float(maxExact)) / scale * Float(numBuckets - maxExact))
                    bucket += min(large, numBuckets - 1)
                }
                buckets[i * t + j] = Int32(bucket)
            }
        }
        let values = relativeAttentionBias!(MLXArray(buckets).reshaped([t, t]))  // [T, T, 12]
        return values.transposed(2, 0, 1).expandedDimensions(axis: 0)  // [1, 12, T, T]
    }

    func callAsFunction(_ x: MLXArray, positionBias: MLXArray, keyMask: MLXArray) -> MLXArray {
        let (b, t) = (x.dim(0), x.dim(1))
        // Gated relative position bias, computed from the layer input split into heads (not from q_proj).
        let xh = x.reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)  // [B, H, T, 64]
        let g = sigmoid(grepLinear(xh).reshaped([b, heads, t, 2, 4]).sum(axis: -1))  // [B, H, T, 2]
        let gateA = g[.ellipsis, 0..<1], gateB = g[.ellipsis, 1..<2]
        let gate = gateA * (gateB * grepA - 1.0) + 2.0  // [B, H, T, 1]
        let mask = (gate * positionBias + keyMask).asType(x.dtype)

        let q = qProj(x).reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        let k = kProj(x).reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped([b, t, heads, headDim]).transposed(0, 2, 1, 3)
        let o = scaledDotProductAttention(
            queries: q, keys: k, values: v, scale: 1.0 / Float(headDim).squareRoot(), mask: mask)
        return outProj(o.transposed(0, 2, 1, 3).reshaped([b, t, heads * headDim]))
    }
}
