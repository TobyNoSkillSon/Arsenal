// SPDX-License-Identifier: MIT
// TimesFM 2.5 200M (Google, Apache-2.0) ported to MLX from the maker's timesfm package 3.0.2:
// timesfm_2p5/timesfm_2p5_torch.py (TimesFM_2p5_200M_torch_module.decode and the compiled decode),
// timesfm_2p5/timesfm_2p5_base.py (forecast), torch/transformer.py, torch/dense.py, torch/util.py and
// torch/normalization.py. It runs the model card's forecast settings: normalize_inputs,
// use_continuous_quantile_head, force_flip_invariance, infer_is_positive and fix_quantile_crossing.
// Statistics, normalisation and the output post-processing run on the host in Double; the tokenizer,
// the 20 transformer layers (with a key/value cache for the autoregressive steps) and both output heads
// run on the float16 weights.
import Foundation
import MLX
import MLXFast
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.forecast"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["forecast"]

/// TimesFM_2p5_200M_Definition (its config.json does not carry the architecture).
enum TimesFM {
    static let patch = 32             // input_patch_len
    static let outputPatch = 128      // output_patch_len
    static let quantileLength = 1024  // output_quantile_len
    static let outputs = 10           // mean, then quantiles 0.1 ... 0.9
    static let median = 5             // decode_index
    static let layers = 20
    static let heads = 16
    static let dims = 1280
    static let headDim = 80
    static let tolerance = 1e-6
}

/// RMSNorm with a `scale` multiplied as is (not 1 + scale), eps 1e-6, computed in float32.
final class RMSNorm: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray
    init(_ dims: Int) { _scale.wrappedValue = MLXArray.zeros([dims]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let wide = x.asType(.float32)
        return (wide * rsqrt(mean(wide * wide, axis: -1, keepDims: true) + 1e-6) * scale.asType(.float32)).asType(x.dtype)
    }
}

/// Two linear layers with SiLU and a linear skip (dense.ResidualBlock).
final class ResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear
    init(_ input: Int, _ hidden: Int, _ output: Int, bias: Bool) {
        _hiddenLayer.wrappedValue = Linear(input, hidden, bias: bias)
        _outputLayer.wrappedValue = Linear(hidden, output, bias: bias)
        _residualLayer.wrappedValue = Linear(input, output, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { outputLayer(silu(hiddenLayer(x))) + residualLayer(x) }
}

final class PerDimScale: Module {
    @ParameterInfo(key: "per_dim_scale") var perDimScale: MLXArray
    init(_ dims: Int) { _perDimScale.wrappedValue = MLXArray.zeros([dims]) }
    /// 1.442695041 / sqrt(dims) × softplus(per_dim_scale), in float32.
    func factor() -> MLXArray {
        let s = perDimScale.asType(.float32)
        let softplus = logAddExp(s, MLXArray(Float(0)))
        return softplus * Float(1.442695041 / Double(perDimScale.dim(0)).squareRoot())
    }
}

/// MultiHeadAttention with fused QKV, RoPE before the per-head RMS norm, per-dimension query scaling and unscaled
/// dot-product attention.
final class Attention: Module {
    @ModuleInfo(key: "qkv_proj") var qkvProj: Linear
    @ModuleInfo(key: "out") var out: Linear
    @ModuleInfo(key: "query_ln") var queryLn: RMSNorm
    @ModuleInfo(key: "key_ln") var keyLn: RMSNorm
    @ModuleInfo(key: "per_dim_scale") var perDimScale: PerDimScale
    override init() {
        _qkvProj.wrappedValue = Linear(TimesFM.dims, 3 * TimesFM.dims, bias: false)
        _out.wrappedValue = Linear(TimesFM.dims, TimesFM.dims, bias: false)
        _queryLn.wrappedValue = RMSNorm(TimesFM.headDim)
        _keyLn.wrappedValue = RMSNorm(TimesFM.headDim)
        _perDimScale.wrappedValue = PerDimScale(TimesFM.headDim)
    }

    /// x [1, n, dims] at the given positions; `cache` holds earlier keys and values [1, heads, k, headDim].
    func callAsFunction(_ x: MLXArray, rope: (cos: MLXArray, sin: MLXArray), cache: inout (MLXArray, MLXArray)?) -> MLXArray {
        let n = x.dim(1)
        let parts = split(qkvProj(x), parts: 3, axis: -1)
        func heads(_ t: MLXArray) -> MLXArray { t.reshaped([1, n, TimesFM.heads, TimesFM.headDim]) }
        var query = queryLn(rotate(heads(parts[0]), rope))
        let key = keyLn(rotate(heads(parts[1]), rope))
        query = (query.asType(.float32) * perDimScale.factor()).asType(x.dtype)
        var keys = key.transposed(0, 2, 1, 3)
        var values = heads(parts[2]).transposed(0, 2, 1, 3)
        let earlier = cache?.0.dim(2) ?? 0
        if let (k, v) = cache {
            keys = concatenated([k, keys], axis: 2)
            values = concatenated([v, values], axis: 2)
        }
        cache = (keys, values)
        // Causal: new token i (at absolute index earlier + i) sees keys 0 ... earlier + i.
        let total = earlier + n
        let rows = MLXArray(Int32(earlier) ..< Int32(total)).reshaped([n, 1])
        let columns = MLXArray(Int32(0) ..< Int32(total)).reshaped([1, total])
        let mask = rows .>= columns
        let attended = MLXFast.scaledDotProductAttention(
            queries: query.transposed(0, 2, 1, 3), keys: keys, values: values, scale: 1, mask: .array(mask))
        return out(attended.transposed(0, 2, 1, 3).reshaped([1, n, TimesFM.dims]))
    }

    /// RotaryPositionalEmbedding: the two halves of the head dimension are rotated together, in float32.
    private func rotate(_ x: MLXArray, _ rope: (cos: MLXArray, sin: MLXArray)) -> MLXArray {
        let half = TimesFM.headDim / 2
        let wide = x.asType(.float32)
        let first = wide[.ellipsis, ..<half], second = wide[.ellipsis, half...]
        return concatenated([first * rope.cos - second * rope.sin, second * rope.cos + first * rope.sin], axis: -1)
            .asType(x.dtype)
    }
}

/// transformer.Transformer: normed attention and feed-forward, each with a post-norm before the residual add.
final class Layer: Module {
    @ModuleInfo(key: "pre_attn_ln") var preAttnLn: RMSNorm
    @ModuleInfo(key: "post_attn_ln") var postAttnLn: RMSNorm
    @ModuleInfo(key: "attn") var attn: Attention
    @ModuleInfo(key: "pre_ff_ln") var preFfLn: RMSNorm
    @ModuleInfo(key: "post_ff_ln") var postFfLn: RMSNorm
    @ModuleInfo(key: "ff0") var ff0: Linear
    @ModuleInfo(key: "ff1") var ff1: Linear
    override init() {
        _preAttnLn.wrappedValue = RMSNorm(TimesFM.dims)
        _postAttnLn.wrappedValue = RMSNorm(TimesFM.dims)
        _attn.wrappedValue = Attention()
        _preFfLn.wrappedValue = RMSNorm(TimesFM.dims)
        _postFfLn.wrappedValue = RMSNorm(TimesFM.dims)
        _ff0.wrappedValue = Linear(TimesFM.dims, TimesFM.dims, bias: false)
        _ff1.wrappedValue = Linear(TimesFM.dims, TimesFM.dims, bias: false)
    }
    func callAsFunction(_ x: MLXArray, rope: (cos: MLXArray, sin: MLXArray), cache: inout (MLXArray, MLXArray)?) -> MLXArray {
        let attended = postAttnLn(attn(preAttnLn(x), rope: rope, cache: &cache)) + x
        return postFfLn(ff1(silu(ff0(preFfLn(attended))))) + attended
    }
}

final class Network: Module {
    @ModuleInfo(key: "tokenizer") var tokenizer: ResidualBlock
    @ModuleInfo(key: "stacked_xf") var layers: [Layer]
    @ModuleInfo(key: "output_projection_point") var pointHead: ResidualBlock
    @ModuleInfo(key: "output_projection_quantiles") var quantileHead: ResidualBlock
    override init() {
        _tokenizer.wrappedValue = ResidualBlock(2 * TimesFM.patch, TimesFM.dims, TimesFM.dims, bias: true)
        _layers.wrappedValue = (0..<TimesFM.layers).map { _ in Layer() }
        _pointHead.wrappedValue = ResidualBlock(TimesFM.dims, TimesFM.dims, TimesFM.outputPatch * TimesFM.outputs, bias: false)
        _quantileHead.wrappedValue = ResidualBlock(TimesFM.dims, TimesFM.dims, TimesFM.quantileLength * TimesFM.outputs, bias: false)
    }
}

/// util.update_running_stats: merge a patch's mean and population variance into the running ones.
struct RunningStats {
    var n = 0.0, mu = 0.0, sigma = 0.0
    mutating func update(_ x: [Double], masked: [Bool]) {
        let legit = x.indices.filter { !masked[$0] }
        let incN = Double(legit.count)
        let incMu = incN == 0 ? 0 : legit.reduce(0.0) { $0 + x[$1] } / incN
        let incVar = incN == 0 ? 0 : legit.reduce(0.0) { $0 + (x[$1] - incMu) * (x[$1] - incMu) } / incN
        let newN = n + incN
        let newMu = newN == 0 ? 0 : (n * mu + incMu * incN) / newN
        let newVar = newN == 0 ? 0 : (n * sigma * sigma + incN * incVar + n * (mu - newMu) * (mu - newMu)
            + incN * (incMu - newMu) * (incMu - newMu)) / newN
        n = newN; mu = newMu; sigma = max(newVar, 0).squareRoot()
    }
    func normalize(_ x: Double) -> Double { (x - mu) / (sigma < TimesFM.tolerance ? 1 : sigma) }
    func restore(_ x: Double) -> Double { x * sigma + mu }
}

final class Model {
    let net = Network()
    let maxHorizon = 1024
    let maxContext = 8192

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, model.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        try Weights.apply(try Weights.load(weightsDir), to: net)
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let values = try params.numbers("values")
        let horizon = params.int("horizon", default: 0)
        guard values.count <= maxContext else { throw RunnerError.invalid("values holds at most \(maxContext) numbers.") }
        guard values.allSatisfy({ $0.isFinite }) else { throw RunnerError.invalid("values must be finite numbers.") }
        guard horizon >= 1, horizon <= maxHorizon else { throw RunnerError.invalid("horizon must be from 1 to \(maxHorizon) steps.") }

        // forecast(): left-pad with masked zeros to whole patches (max_context = length rounded up to 32).
        let p = TimesFM.patch
        let width = (values.count + p - 1) / p * p
        let pad = width - values.count
        let padded = [Double](repeating: 0, count: pad) + values
        let masked = (0..<width).map { $0 < pad }
        let decodeHorizon = (horizon + TimesFM.outputPatch - 1) / TimesFM.outputPatch * TimesFM.outputPatch

        // infer_is_positive and normalize_inputs look at the padded row, zeros included (torch.std is unbiased).
        let isPositive = padded.allSatisfy { $0 >= 0 }
        let mu = padded.reduce(0, +) / Double(width)
        let sigma = width > 1 ? (padded.reduce(0) { $0 + ($1 - mu) * ($1 - mu) } / Double(width - 1)).squareRoot() : 0
        let inputs = padded.map { ($0 - mu) / (sigma < TimesFM.tolerance ? 1 : sigma) }

        // force_flip_invariance: average f(x) with -f(-x), the quantile order mirrored.
        let (forward, forwardSpread) = try decode(inputs, masked: masked, horizon: decodeHorizon, params: params)
        let (flipped, flippedSpread) = try decode(inputs.map { -$0 }, masked: masked, horizon: decodeHorizon, params: params)
        func mirror(_ row: [Double]) -> [Double] { [row[0]] + row[1...].reversed() }
        var full = zip(forward, flipped).map { a, b in zip(a, mirror(b)).map { ($0 - $1) / 2 } }
        let spread = zip(forwardSpread, flippedSpread).map { a, b in zip(a, mirror(b)).map { ($0 - $1) / 2 } }

        // use_continuous_quantile_head: quantiles from the 1024-step head, re-centred on the decoded median.
        for t in 0..<decodeHorizon {
            for q in [1, 2, 3, 4, 6, 7, 8, 9] { full[t][q] = spread[t][q] - spread[t][TimesFM.median] + full[t][TimesFM.median] }
        }
        full = Array(full.prefix(horizon))
        // fix_quantile_crossing, then undo normalize_inputs, then infer_is_positive.
        for t in full.indices {
            for i in [4, 3, 2, 1] where !(full[t][i] < full[t][i + 1]) { full[t][i] = full[t][i + 1] }
            for i in [6, 7, 8, 9] where !(full[t][i] > full[t][i - 1]) { full[t][i] = full[t][i - 1] }
            full[t] = full[t].map { value in
                let restored = value * sigma + mu
                return isPositive ? max(restored, 0) : restored
            }
        }
        let levels = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        return [
            "horizon": horizon,
            "quantile_levels": levels,
            "median": full.map { $0[TimesFM.median] },
            "quantiles": full.map { Array($0[1...]) },
        ]
    }

    /// TimesFM_2p5_200M_torch_module.decode for one series: [decodeHorizon][10] forecasts and the [1024][10]
    /// quantile spread, both in the units of `inputs`.
    private func decode(_ inputs: [Double], masked: [Bool], horizon: Int, params: Params) throws -> ([[Double]], [[Double]]) {
        let p = TimesFM.patch, o = TimesFM.outputPatch, q = TimesFM.outputs
        let patches = inputs.count / p
        let steps = (horizon - 1) / o
        let perStep = o / p

        // Prefill: running statistics per patch, each patch normalised by the statistics up to its end.
        var stats = RunningStats()
        var patchStats: [RunningStats] = []
        var tokens = [Float](repeating: 0, count: patches * 2 * p)
        for i in 0..<patches {
            let x = Array(inputs[(i * p) ..< ((i + 1) * p)]), m = Array(masked[(i * p) ..< ((i + 1) * p)])
            stats.update(x, masked: m)
            patchStats.append(stats)
            for k in 0..<p {
                tokens[i * 2 * p + k] = m[k] ? 0 : Float(stats.normalize(x[k]))
                tokens[i * 2 * p + p + k] = m[k] ? 1 : 0
            }
        }
        var caches = [(MLXArray, MLXArray)?](repeating: nil, count: TimesFM.layers)
        let (point, spreadRaw) = try forward(tokens, count: patches, start: 0, caches: &caches, spread: true, params: params)
        let last = patchStats[patches - 1]
        var full: [[Double]] = (0..<o).map { t in (0..<q).map { last.restore(Double(point[((patches - 1) * o + t) * q + $0])) } }
        let spread: [[Double]] = (0..<TimesFM.quantileLength).map { t in (0..<q).map { last.restore(Double(spreadRaw![t * q + $0])) } }

        // Autoregressive steps: the last 128-step median becomes four new patches.
        var previous = full.map { $0[TimesFM.median] }
        var position = patches
        for _ in 0..<steps {
            var newStats: [RunningStats] = []
            var stepTokens = [Float](repeating: 0, count: perStep * 2 * p)
            for i in 0..<perStep {
                let x = Array(previous[(i * p) ..< ((i + 1) * p)])
                stats.update(x, masked: [Bool](repeating: false, count: p))
                newStats.append(stats)
                for k in 0..<p { stepTokens[i * 2 * p + k] = Float(stats.normalize(x[k])) }
            }
            let (stepPoint, _) = try forward(stepTokens, count: perStep, start: position, caches: &caches, spread: false, params: params)
            position += perStep
            let lastStats = newStats[perStep - 1]
            let block: [[Double]] = (0..<o).map { t in
                (0..<q).map { lastStats.restore(Double(stepPoint[((perStep - 1) * o + t) * q + $0])) }
            }
            full += block
            previous = block.map { $0[TimesFM.median] }
        }
        return (full, spread)
    }

    /// Runs `count` tokens at positions start ..< start + count through the stack; returns the point head for every
    /// token ([count × 128 × 10]) and, when asked, the quantile head of the last token ([1024 × 10]).
    private func forward(_ tokens: [Float], count: Int, start: Int, caches: inout [(MLXArray, MLXArray)?],
                         spread: Bool, params: Params) throws -> ([Float], [Float]?) {
        let half = TimesFM.headDim / 2
        var cosValues = [Float](repeating: 0, count: count * half), sinValues = cosValues
        for i in 0..<half {
            let timescale = pow(10_000.0, 2.0 * Double(i) / Double(TimesFM.headDim))
            for t in 0..<count {
                let angle = Float(Double(start + t) / timescale)
                cosValues[t * half + i] = Foundation.cos(angle)
                sinValues[t * half + i] = Foundation.sin(angle)
            }
        }
        // [1, n, 1, half] broadcasts over the heads of [1, n, heads, headDim / 2].
        let rope = (MLXArray(cosValues, [1, count, 1, half]), MLXArray(sinValues, [1, count, 1, half]))
        var hidden = net.tokenizer(MLXArray(tokens, [1, count, 2 * TimesFM.patch]).asType(.float16))
        for (index, layer) in net.layers.enumerated() {
            hidden = layer(hidden, rope: rope, cache: &caches[index])
            try params.checkCancelled()
        }
        let point = net.pointHead(hidden).asType(.float32)
        var quantiles: MLXArray? = nil
        if spread { quantiles = net.quantileHead(hidden[0..., (count - 1)...]).asType(.float32) }
        eval([point] + (quantiles.map { [$0] } ?? []))
        // Keep the cache evaluated so the next step does not rebuild the prefill graph.
        eval(caches.compactMap { $0 }.flatMap { [$0.0, $0.1] })
        return (point.asArray(Float.self), quantiles?.asArray(Float.self))
    }
}
