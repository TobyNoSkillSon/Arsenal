// SPDX-License-Identifier: MIT
// Chronos-2 (Amazon, Apache-2.0) ported to MLX from the maker's chronos-forecasting code
// (src/chronos/chronos2/model.py, layers.py and pipeline.py, release 2.3.3.dev0 at git 1224faf):
// Chronos2Model.forward and Chronos2Pipeline.predict for one univariate series without covariates.
// Instance scaling (mean, population std, arcsinh) runs on the host in Double; the patch embedding,
// the encoder (time attention with RoPE, group attention, ReLU feed-forward) and the quantile head run
// on the float16 weights. One pass predicts up to max_output_patches × 16 = 1024 steps, the kind's limit,
// so the pipeline's long-horizon unrolling is never needed.
import Foundation
import MLX
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.forecast"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["forecast"]

struct Chronos2Config {
    var dModel = 768
    var dFF = 3072
    var heads = 12
    var dKV = 64
    var layers = 12
    var eps: Float = 1e-6
    var ropeTheta = 10_000.0
    var contextLength = 8192
    var patchSize = 16
    var maxOutputPatches = 64
    var timeEncodingScale = 8192.0
    var quantiles: [Double] = []
    var useArcsinh = true
    var useRegToken = true
    var regTokenId = 1

    init(_ json: [String: Any]) throws {
        func int(_ key: String) -> Int? { (json[key] as? NSNumber)?.intValue }
        dModel = int("d_model") ?? dModel
        dFF = int("d_ff") ?? dFF
        heads = int("num_heads") ?? heads
        dKV = int("d_kv") ?? dKV
        layers = int("num_layers") ?? layers
        eps = (json["layer_norm_epsilon"] as? NSNumber)?.floatValue ?? eps
        ropeTheta = (json["rope_theta"] as? NSNumber)?.doubleValue ?? ropeTheta
        guard (json["dense_act_fn"] as? String ?? "relu") == "relu", json["is_gated_act"] as? Bool != true else {
            throw RunnerError.failed("Only the ReLU feed-forward of Chronos-2 is ported.")
        }
        guard let chronos = json["chronos_config"] as? [String: Any] else { throw RunnerError.failed("config.json has no chronos_config.") }
        func cint(_ key: String) -> Int? { (chronos[key] as? NSNumber)?.intValue }
        contextLength = cint("context_length") ?? contextLength
        patchSize = cint("input_patch_size") ?? patchSize
        guard cint("input_patch_stride") ?? patchSize == patchSize, cint("output_patch_size") ?? patchSize == patchSize else {
            throw RunnerError.failed("Only equal input patch size, stride and output patch size are ported.")
        }
        maxOutputPatches = cint("max_output_patches") ?? maxOutputPatches
        timeEncodingScale = (chronos["time_encoding_scale"] as? NSNumber)?.doubleValue ?? Double(contextLength)
        quantiles = (chronos["quantiles"] as? [NSNumber])?.map { $0.doubleValue } ?? []
        guard !quantiles.isEmpty else { throw RunnerError.failed("config.json lists no quantiles.") }
        useArcsinh = chronos["use_arcsinh"] as? Bool ?? false
        useRegToken = chronos["use_reg_token"] as? Bool ?? false
        regTokenId = int("reg_token_id") ?? 1
    }
}

/// Linear → ReLU → Linear plus a linear skip (ResidualBlock in chronos2/layers.py, no layer norm).
final class ResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear
    init(_ input: Int, _ hidden: Int, _ output: Int) {
        _hiddenLayer.wrappedValue = Linear(input, hidden)
        _outputLayer.wrappedValue = Linear(hidden, output)
        _residualLayer.wrappedValue = Linear(input, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { outputLayer(relu(hiddenLayer(x))) + residualLayer(x) }
}

/// Chronos2LayerNorm (T5 style): RMS over the last axis with the variance in float32, then cast and scale.
final class T5LayerNorm: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    init(_ dims: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dims])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let wide = x.asType(.float32)
        let variance = (wide * wide).mean(axis: -1, keepDims: true)
        return weight * (wide * rsqrt(variance + eps)).asType(weight.dtype)
    }
}

/// MHA in chronos2/layers.py: bias-free projections, no 1/sqrt(d) scaling, softmax in float32.
final class MHA: Module {
    let heads: Int
    let dKV: Int
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear

    init(_ c: Chronos2Config) {
        heads = c.heads
        dKV = c.dKV
        let inner = c.heads * c.dKV
        _q.wrappedValue = Linear(c.dModel, inner, bias: false)
        _k.wrappedValue = Linear(c.dModel, inner, bias: false)
        _v.wrappedValue = Linear(c.dModel, inner, bias: false)
        _o.wrappedValue = Linear(inner, c.dModel, bias: false)
    }

    /// Self-attention over the time axis with rotary positions (cos and sin [n, dKV] in the input's dtype).
    func time(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let b = x.dim(0), n = x.dim(1)
        func shape(_ t: MLXArray) -> MLXArray { t.reshaped([b, n, heads, dKV]).transposed(0, 2, 1, 3) }
        let queries = rotate(shape(q(x)), cos: cos, sin: sin)
        let keys = rotate(shape(k(x)), cos: cos, sin: sin)
        let values = shape(v(x))
        let scores = matmul(queries, keys.transposed(0, 1, 3, 2))
        let weights = softmax(scores.asType(.float32), axis: -1).asType(values.dtype)
        return o(matmul(weights, values).transposed(0, 2, 1, 3).reshaped([b, n, heads * dKV]))
    }

    /// Group attention runs across the series of a batch at each time step. With one series every query
    /// has one key, its softmax weight is 1 whatever the score, so the output is o(v(x)).
    func groupOfOne(_ x: MLXArray) -> MLXArray { o(v(x)) }

    /// apply_rotary_pos_emb with rotate_half: the two halves of the head dimension form the pairs.
    private func rotate(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let half = dKV / 2
        let first = x[.ellipsis, ..<half], second = x[.ellipsis, half...]
        let rotated = concatenated([-second, first], axis: -1)
        return x * cos + rotated * sin
    }
}

/// Pre-norm residual attention sublayer (TimeSelfAttention or GroupSelfAttention).
final class AttentionLayer: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: T5LayerNorm
    @ModuleInfo(key: "self_attention") var attention: MHA
    init(_ c: Chronos2Config) {
        _layerNorm.wrappedValue = T5LayerNorm(c.dModel, eps: c.eps)
        _attention.wrappedValue = MHA(c)
    }
}

/// MLP with ReLU (wi, wo).
final class MLP: Module {
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    init(_ c: Chronos2Config) {
        _wi.wrappedValue = Linear(c.dModel, c.dFF, bias: false)
        _wo.wrappedValue = Linear(c.dFF, c.dModel, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { wo(relu(wi(x))) }
}

/// Pre-norm residual feed-forward sublayer (FeedForward).
final class FFLayer: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: T5LayerNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    init(_ c: Chronos2Config) {
        _layerNorm.wrappedValue = T5LayerNorm(c.dModel, eps: c.eps)
        _mlp.wrappedValue = MLP(c)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + mlp(layerNorm(x)) }
}

/// Chronos2EncoderBlock: time attention, group attention, feed-forward (checkpoint layer.0, layer.1, layer.2).
final class EncoderBlock: Module {
    @ModuleInfo(key: "time") var time: AttentionLayer
    @ModuleInfo(key: "group") var group: AttentionLayer
    @ModuleInfo(key: "ff") var ff: FFLayer
    init(_ c: Chronos2Config) {
        _time.wrappedValue = AttentionLayer(c)
        _group.wrappedValue = AttentionLayer(c)
        _ff.wrappedValue = FFLayer(c)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        var hidden = x + time.attention.time(time.layerNorm(x), cos: cos, sin: sin)
        hidden = hidden + group.attention.groupOfOne(group.layerNorm(hidden))
        return ff(hidden)
    }
}

final class Encoder: Module {
    @ModuleInfo(key: "block") var block: [EncoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: T5LayerNorm
    init(_ c: Chronos2Config) {
        _block.wrappedValue = (0..<c.layers).map { _ in EncoderBlock(c) }
        _finalLayerNorm.wrappedValue = T5LayerNorm(c.dModel, eps: c.eps)
    }
}

final class Chronos2Network: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "input_patch_embedding") var inputPatchEmbedding: ResidualBlock
    @ModuleInfo(key: "encoder") var encoder: Encoder
    @ModuleInfo(key: "output_patch_embedding") var outputPatchEmbedding: ResidualBlock
    init(_ c: Chronos2Config) {
        _shared.wrappedValue = Embedding(embeddingCount: c.useRegToken ? 2 : 1, dimensions: c.dModel)
        _inputPatchEmbedding.wrappedValue = ResidualBlock(3 * c.patchSize, c.dFF, c.dModel)
        _encoder.wrappedValue = Encoder(c)
        _outputPatchEmbedding.wrappedValue = ResidualBlock(c.dModel, c.dFF, c.quantiles.count * c.patchSize)
    }
}

final class Model {
    let config: Chronos2Config
    let net: Chronos2Network

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, model.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = try Chronos2Config(try Files.json(weightsDir))
        net = Chronos2Network(config)
        // The checkpoint numbers each block's sublayers (layer.0, layer.1, layer.2); the module names say what they are.
        let names = ["time", "group", "ff"]
        try Weights.apply(try Weights.load(weightsDir), to: net, rename: { key in
            var parts = key.split(separator: ".").map(String.init)
            guard parts.count > 4, parts[1] == "block", parts[3] == "layer", let index = Int(parts[4]), index < names.count else { return key }
            parts.replaceSubrange(3...4, with: [names[index]])
            return parts.joined(separator: ".")
        })
    }

    var maxHorizon: Int { config.maxOutputPatches * config.patchSize }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let values = try params.numbers("values")
        let horizon = params.int("horizon", default: 0)
        guard !values.isEmpty, values.count <= 8192 else { throw RunnerError.invalid("values holds 1 to 8192 numbers.") }
        guard values.allSatisfy({ $0.isFinite }) else { throw RunnerError.invalid("values must be finite numbers.") }
        guard horizon >= 1, horizon <= maxHorizon else { throw RunnerError.invalid("horizon must be from 1 to \(maxHorizon) steps.") }

        let quantiles = try forecast(Array(values.suffix(config.contextLength)), horizon: horizon, params: params)
        let medianIndex = config.quantiles.firstIndex { abs($0 - 0.5) < 1e-9 }
        return [
            "horizon": horizon,
            "quantile_levels": config.quantiles,
            "median": medianIndex.map { index in quantiles.map { $0[index] } } ?? [],
            "quantiles": quantiles,
        ]
    }

    /// Chronos2Model.forward for one series → [horizon][quantile] in the data's units.
    private func forecast(_ series: [Double], horizon: Int, params: Params) throws -> [[Double]] {
        let c = config
        let p = c.patchSize
        let length = series.count
        let contextPatches = (length + p - 1) / p
        let padded = contextPatches * p
        let pad = padded - length
        let outputPatches = min((horizon + p - 1) / p, c.maxOutputPatches)

        // InstanceNorm (float32 in the maker's code): mean and population std; a zero std becomes 1e-5.
        let loc = series.reduce(0, +) / Double(length)
        var scale = (series.reduce(0) { $0 + ($1 - loc) * ($1 - loc) } / Double(length)).squareRoot()
        if scale == 0 { scale = 1e-5 }

        // Patch rows [time encoding ‖ values ‖ mask]. Context: left-padded with NaN to whole patches (value 0,
        // mask 0), time index -padded..<0 over time_encoding_scale. Future: time 0..<steps, values and mask 0.
        let rows = contextPatches + outputPatches
        var input = [Float](repeating: 0, count: rows * 3 * p)
        for i in 0..<padded {
            let base = (i / p) * 3 * p, k = i % p
            input[base + k] = Float(Double(i - padded) / c.timeEncodingScale)
            if i >= pad {
                let standardized = (series[i - pad] - loc) / scale
                input[base + p + k] = Float(c.useArcsinh ? asinh(standardized) : standardized)
                input[base + 2 * p + k] = 1
            }
        }
        for i in 0..<(outputPatches * p) {
            input[(contextPatches + i / p) * 3 * p + i % p] = Float(Double(i) / c.timeEncodingScale)
        }
        let embedded = net.inputPatchEmbedding(MLXArray(input, [1, rows, 3 * p]).asType(.float16))
        let contextEmbeds = embedded[0..., ..<contextPatches]
        let futureEmbeds = embedded[0..., contextPatches...]
        var parts = [contextEmbeds]
        if c.useRegToken { parts.append(net.shared(MLXArray([Int32(c.regTokenId)], [1, 1]))) }
        parts.append(futureEmbeds)
        var hidden = concatenated(parts, axis: 1)

        // Every token is attended: each context patch holds an observation, and [REG] and future patches are
        // always unmasked. RoPE positions 0..<n, computed in float32 and cast as Chronos2RotaryEmbedding does.
        let n = hidden.dim(1)
        let half = c.dKV / 2
        var cosValues = [Float](repeating: 0, count: n * c.dKV), sinValues = cosValues
        for i in 0..<half {
            let inverseFrequency = Float(1.0 / pow(c.ropeTheta, Double(2 * i) / Double(c.dKV)))
            for position in 0..<n {
                let angle = Float(position) * inverseFrequency
                for column in [i, i + half] {
                    cosValues[position * c.dKV + column] = Foundation.cos(angle)
                    sinValues[position * c.dKV + column] = Foundation.sin(angle)
                }
            }
        }
        let cos = MLXArray(cosValues, [n, c.dKV]).asType(hidden.dtype)
        let sin = MLXArray(sinValues, [n, c.dKV]).asType(hidden.dtype)

        for block in net.encoder.block {
            hidden = block(hidden, cos: cos, sin: sin)
            try params.checkCancelled()
        }
        hidden = net.encoder.finalLayerNorm(hidden)
        let raw = net.outputPatchEmbedding(hidden[0..., (n - outputPatches)...]).asType(.float32)  // [1, N, Q * P]
        eval(raw)
        let flat = raw.asArray(Float.self)
        let levels = c.quantiles.count
        // "b n (q p) -> b q (n p)", then sinh and rescale.
        return (0..<horizon).map { step in
            let patch = step / p, k = step % p
            return (0..<levels).map { q in
                let value = Double(flat[patch * levels * p + q * p + k])
                return (c.useArcsinh ? sinh(value) : value) * scale + loc
            }
        }
    }
}
