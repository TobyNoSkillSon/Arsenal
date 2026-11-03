// SPDX-License-Identifier: MIT
// t0-alpha (The Forecasting Company, Apache-2.0) ported to MLX from the maker's code in tfc-t0:
// t0/torch/model/model.py and layers/ (backbone), t0/patcher.py, t0/scaler.py, t0/mask.py,
// t0/rollout.py and t0/forecaster.py (single pass). Scope: one univariate target series,
// horizon up to 1024 steps (one pass, no autoregressive rollout). Scaling and rescaling run on
// the host in Double; the transformer runs on the float16 weights.
import Foundation
import MLX
import MLXFast
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.forecast"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["forecast"]

/// Hyperparameters from config.json; the t0-alpha defaults fill any key the file leaves out.
struct T0Config {
    var embedDim = 512
    var numLayers = 24
    var numHeads = 8
    var mlpHiddenDim = 2048
    var patchSize = 32
    var groupEveryN = 3
    var quantileLevels: [Double] = [0.1, 0.25, 0.5, 0.75, 0.9]
    var scalerEps = 0.1
    var scalerUseArcsinh = true
    /// DEFAULT_MAX_HORIZON in t0/forecaster.py.
    let maxHorizon = 1024

    init(_ json: [String: Any]) {
        embedDim = (json["embed_dim"] as? NSNumber)?.intValue ?? embedDim
        numLayers = (json["num_layers"] as? NSNumber)?.intValue ?? numLayers
        numHeads = (json["num_heads"] as? NSNumber)?.intValue ?? numHeads
        mlpHiddenDim = (json["mlp_hidden_dim"] as? NSNumber)?.intValue ?? mlpHiddenDim
        patchSize = (json["patch_size"] as? NSNumber)?.intValue ?? patchSize
        groupEveryN = (json["group_every_n"] as? NSNumber)?.intValue ?? groupEveryN
        if let levels = json["quantile_levels"] as? [NSNumber] { quantileLevels = levels.map { $0.doubleValue }.sorted() }
        scalerEps = (json["scaler_eps"] as? NSNumber)?.doubleValue ?? scalerEps
        scalerUseArcsinh = json["scaler_use_arcsinh"] as? Bool ?? scalerUseArcsinh
    }
}

/// Cos, sin and XPos scale tables for the patch positions 0..<count, each [count, dims] with interleaved pairs.
struct RopeTables {
    let cosTable: MLXArray
    let sinTable: MLXArray
    let scaleTable: MLXArray

    /// Temporal RoPE with XPos scaling (rotary_embedding_torch as used in t0/torch/model/layers/rope.py).
    static func make(count: Int, dims: Int, theta: Double = 10_000, scaleBase: Double = 512) -> RopeTables {
        var cosValues = [Float](repeating: 0, count: count * dims)
        var sinValues = cosValues
        var scaleValues = cosValues
        let center = floor(Double(count - 1) / 2)
        for i in 0..<(dims / 2) {
            let inverseFrequency = 1 / pow(theta, Double(2 * i) / Double(dims))
            let base = (Double(2 * i) + 0.4 * Double(dims)) / (1.4 * Double(dims))
            for p in 0..<count {
                let angle = Double(p) * inverseFrequency
                let scale = pow(base, (Double(p) - center) / scaleBase)
                for k in [2 * i, 2 * i + 1] {
                    cosValues[p * dims + k] = Float(cos(angle))
                    sinValues[p * dims + k] = Float(sin(angle))
                    scaleValues[p * dims + k] = Float(scale)
                }
            }
        }
        return RopeTables(
            cosTable: MLXArray(cosValues, [count, dims]),
            sinTable: MLXArray(sinValues, [count, dims]),
            scaleTable: MLXArray(scaleValues, [count, dims]))
    }

    /// Rotates queries (scale) or keys (inverse scale) in float32, then returns them in the input's dtype.
    func rotate(_ x: MLXArray, keys: Bool) -> MLXArray {
        let values = x.asType(.float32)
        let rotated = values * cosTable + RopeTables.rotateHalf(values) * sinTable
        return (keys ? rotated / scaleTable : rotated * scaleTable).asType(x.dtype)
    }

    private static func rotateHalf(_ x: MLXArray) -> MLXArray {
        let shape = x.shape
        let pairs = x.reshaped(Array(shape.dropLast()) + [shape.last! / 2, 2])
        return stacked([-pairs[.ellipsis, 1], pairs[.ellipsis, 0]], axis: -1).reshaped(shape)
    }
}

/// RMSNorm with the checkpoint's `scale` parameter and eps 1e-8; computed in float32.
final class RMSNorm: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray

    init(_ dims: Int) {
        _scale.wrappedValue = MLXArray.ones([dims])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let input = x.asType(.float32)
        let normed = input * rsqrt(mean(input * input, axis: -1, keepDims: true) + 1e-8)
        return (normed * scale.asType(.float32)).asType(x.dtype)
    }
}

/// Two projections with ReLU between them (t0/torch/model/layers/mlp.py).
final class MLP: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear

    init(_ input: Int, _ hidden: Int, _ output: Int) {
        _hiddenLayer.wrappedValue = Linear(input, hidden)
        _outputLayer.wrappedValue = Linear(hidden, output)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        outputLayer(relu(hiddenLayer(x)))
    }
}

/// MLP plus a linear skip connection (ResidualBlock in t0/torch/model/layers/mlp.py).
final class ResidualBlock: Module {
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear

    init(_ input: Int, _ hidden: Int, _ output: Int) {
        _mlp.wrappedValue = MLP(input, hidden, output)
        _residualLayer.wrappedValue = Linear(input, output)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        mlp(x) + residualLayer(x)
    }
}

/// Patch embedding: [values ‖ within-patch time ‖ validity] projection plus the target type embedding.
/// Univariate input: every patch is a target patch, so the type id is always 0.
final class PatchEncoder: Module {
    @ModuleInfo(key: "projection") var projection: ResidualBlock
    @ModuleInfo(key: "type_embeddings") var typeEmbeddings: Embedding

    init(width: Int, patch: Int) {
        _projection.wrappedValue = ResidualBlock(patch * 3, width, width)
        _typeEmbeddings.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let typeIds = MLXArray.zeros([x.dim(0), x.dim(1)], dtype: .int32)
        return projection(x) + typeEmbeddings(typeIds)
    }
}

/// Self-attention with shared QKV projections (time_attention.py and group_attention.py).
final class SelfAttention: Module {
    enum Kind {
        /// Causal attention over patches with RoPE.
        case time(mask: MLXArray, rope: RopeTables)
        /// Attention across variates. With one variate each patch attends only itself, so the output is its values.
        case variate
    }

    let width: Int
    let heads: Int
    let headDim: Int
    @ModuleInfo(key: "wQKV") var wQKV: Linear
    @ModuleInfo(key: "wO") var wO: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(width: Int, heads: Int) {
        self.width = width
        self.heads = heads
        headDim = width / heads
        _wQKV.wrappedValue = Linear(width, 3 * width)
        _wO.wrappedValue = Linear(width, width)
        _qNorm.wrappedValue = RMSNorm(headDim)
        _kNorm.wrappedValue = RMSNorm(headDim)
    }

    func callAsFunction(_ x: MLXArray, kind: Kind) -> MLXArray {
        let patches = x.dim(1)
        let qkv = wQKV(x).reshaped([1, patches, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
        switch kind {
        case .variate:
            return wO(qkv[2].transposed(0, 2, 1, 3).reshaped([1, patches, width]))
        case .time(let mask, let rope):
            let queries = rope.rotate(qNorm(qkv[0]), keys: false)
            let keys = rope.rotate(kNorm(qkv[1]), keys: true)
            let attended = MLXFast.scaledDotProductAttention(
                queries: queries, keys: keys, values: qkv[2],
                scale: 1 / Float(headDim).squareRoot(), mask: .array(mask))
            return wO(attended.transposed(0, 2, 1, 3).reshaped([1, patches, width]))
        }
    }
}

/// Pre-norm attention with a residual connection.
final class AttentionBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "attention") var attention: SelfAttention

    init(width: Int, heads: Int) {
        _norm.wrappedValue = RMSNorm(width)
        _attention.wrappedValue = SelfAttention(width: width, heads: heads)
    }

    func callAsFunction(_ x: MLXArray, kind: SelfAttention.Kind) -> MLXArray {
        x + attention(norm(x), kind: kind)
    }
}

/// SwiGLU feed-forward; the first projection's first half is the gate (t0/torch/model/layers/feed_forward.py).
final class FeedForward: Module {
    @ModuleInfo(key: "up") var up: Linear
    @ModuleInfo(key: "down") var down: Linear

    init(width: Int, hidden: Int) {
        _up.wrappedValue = Linear(width, 2 * hidden)
        _down.wrappedValue = Linear(hidden, width)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(up(x), parts: 2, axis: -1)
        return down(silu(parts[0]) * parts[1])
    }
}

/// One attention layer (time or variate) followed by a pre-norm SwiGLU feed-forward.
final class TransformerLayer: Module {
    let isGroup: Bool
    @ModuleInfo(key: "attention_block") var attentionBlock: AttentionBlock
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: FeedForward

    init(width: Int, heads: Int, hidden: Int, isGroup: Bool) {
        self.isGroup = isGroup
        _attentionBlock.wrappedValue = AttentionBlock(width: width, heads: heads)
        _norm.wrappedValue = RMSNorm(width)
        _mlp.wrappedValue = FeedForward(width: width, hidden: hidden)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, rope: RopeTables) -> MLXArray {
        let attended = attentionBlock(x, kind: isGroup ? .variate : .time(mask: mask, rope: rope))
        return attended + mlp(norm(attended))
    }
}

/// Alternating time and variate attention stack (t0/torch/model/layers/transformer.py).
final class Transformer: Module {
    @ModuleInfo(key: "layers") var layers: [TransformerLayer]
    @ModuleInfo(key: "out_norm") var outNorm: RMSNorm

    init(_ config: T0Config) {
        _layers.wrappedValue = (0..<config.numLayers).map { index in
            TransformerLayer(
                width: config.embedDim, heads: config.numHeads, hidden: config.mlpHiddenDim,
                isGroup: (index + 1) % config.groupEveryN == 0)
        }
        _outNorm.wrappedValue = RMSNorm(config.embedDim)
    }

    func callAsFunction(_ x: MLXArray, mask: MLXArray, rope: RopeTables) -> MLXArray {
        var hidden = x
        for layer in layers { hidden = layer(hidden, mask: mask, rope: rope) }
        return outNorm(hidden)
    }
}

/// The checkpoint's network. Property keys match the safetensors names (patch_encoder.*, transformer.*, decoder.*).
final class T0Network: Module {
    @ModuleInfo(key: "patch_encoder") var patchEncoder: PatchEncoder
    @ModuleInfo(key: "transformer") var transformer: Transformer
    @ModuleInfo(key: "decoder") var decoder: ResidualBlock

    init(_ config: T0Config) {
        let patch = config.patchSize
        _patchEncoder.wrappedValue = PatchEncoder(width: config.embedDim, patch: patch)
        _transformer.wrappedValue = Transformer(config)
        _decoder.wrappedValue = ResidualBlock(config.embedDim, config.embedDim, patch * config.quantileLevels.count)
    }
}

/// Smallest multiple of `multiple` that is at least `value` (t0/types.py round_up).
private func roundUp(_ value: Int, _ multiple: Int) -> Int { (value + multiple - 1) / multiple * multiple }

/// Welford causal mean and std per cell, with eps under the square root (CausalScaler, variance_offset).
private func causalStats(_ x: [Double], invalid: [Bool], eps: Double) -> (loc: [Double], scale: [Double]) {
    var loc = [Double](repeating: 0, count: x.count)
    var scale = loc
    var count = 0.0, sum = 0.0, m2 = 0.0, previousMean = 0.0
    for i in x.indices {
        let valid = invalid[i] ? 0.0 : 1.0
        let masked = invalid[i] ? 0.0 : x[i]
        count += valid
        sum += masked
        let mean = sum / max(count, 1)
        m2 += (masked - previousMean) * (masked - mean) * valid
        let variance = max(m2, 0) / max(count - 1, 1)
        loc[i] = mean
        scale[i] = (variance + eps).squareRoot()
        previousMean = mean
    }
    return (loc, scale)
}

/// Numerically stable softplus, as F.softplus in the maker's quantile head.
private func softplus(_ x: Double) -> Double { x > 0 ? x + log1p(exp(-x)) : log1p(exp(x)) }

final class Model {
    let config: T0Config
    let net: T0Network

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, model.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = T0Config(try Files.json(weightsDir))
        net = T0Network(config)
        // head.quantile_levels is a constant of the head, not a learned weight. The feed-forward's
        // checkpoint indices mlp.0 and mlp.2 are named up and down: numeric keys would load as an array.
        try Weights.apply(try Weights.load(weightsDir), to: net, rename: { key in
            if key == "head.quantile_levels" { return nil }
            return key.replacingOccurrences(of: ".mlp.0.", with: ".mlp.up.")
                .replacingOccurrences(of: ".mlp.2.", with: ".mlp.down.")
        })
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let values = try params.numbers("values")
        let horizon = params.int("horizon", default: 0)
        guard values.count <= 8192 else { throw RunnerError.invalid("values holds at most 8192 numbers.") }
        guard values.allSatisfy({ $0.isFinite }) else { throw RunnerError.invalid("values must be finite numbers.") }
        guard horizon >= 1, horizon <= config.maxHorizon else {
            throw RunnerError.invalid("horizon must be from 1 to \(config.maxHorizon) steps.")
        }
        let quantiles = try forecast(values, horizon: horizon)
        let medianIndex = config.quantileLevels.firstIndex { abs($0 - 0.5) < 1e-9 }
        return [
            "horizon": horizon,
            "quantile_levels": config.quantileLevels,
            "median": medianIndex.map { index in quantiles.map { $0[index] } } ?? [],
            "quantiles": quantiles,
        ]
    }

    /// Returns [horizon][quantile] in the data's units, at the model's trained quantile levels.
    private func forecast(_ values: [Double], horizon: Int) throws -> [[Double]] {
        let patch = config.patchSize
        let context = values.count
        let contextWidth = roundUp(context, patch)
        let padLeft = contextWidth - context
        let forecastWidth = roundUp(horizon, patch)
        let width = contextWidth + forecastWidth
        let patches = width / patch

        // Window as in t0/rollout.py prepare_rollout_buffer: [PAD ‖ context ‖ WITHHELD].
        // Mask codes as in t0/types.py: 0 VALID, 1 PAD, 4 WITHHELD.
        var raw = [Double](repeating: 0, count: width)
        var mask = [Int](repeating: 1, count: width)
        for i in 0..<context {
            raw[padLeft + i] = values[i]
            mask[padLeft + i] = 0
        }
        for i in contextWidth..<width { mask[i] = 4 }

        // CausalScaler (t0/scaler.py): standardize valid cells, arcsinh, zero every cell without an observation.
        let stats = causalStats(raw, invalid: mask.map { $0 != 0 }, eps: config.scalerEps)
        var scaled = [Double](repeating: 0, count: width)
        for i in 0..<width where mask[i] == 0 {
            let standardized = (raw[i] - stats.loc[i]) / stats.scale[i]
            scaled[i] = config.scalerUseArcsinh ? asinh(standardized) : standardized
        }

        // Per patch: [values ‖ time index k/patch ‖ validity] as in t0/torch/model/layers/patch_encoder.py.
        var input = [Float](repeating: 0, count: patches * 3 * patch)
        for p in 0..<patches {
            for k in 0..<patch {
                let i = p * patch + k
                input[p * 3 * patch + k] = Float(scaled[i])
                input[p * 3 * patch + patch + k] = Float(k) / Float(patch)
                input[p * 3 * patch + 2 * patch + k] = mask[i] == 0 ? 1 : 0
            }
        }
        let x = MLXArray(input, [1, patches, 3 * patch]).asType(.float16)

        // Causal over patches (every patch holds a real cell, so no key is blocked by padding).
        var causal = [Bool](repeating: false, count: patches * patches)
        for i in 0..<patches { for j in 0...i { causal[i * patches + j] = true } }
        let mask4 = MLXArray(causal, [1, 1, patches, patches])
        let rope = RopeTables.make(count: patches, dims: config.embedDim / config.numHeads)

        let hidden = net.transformer(net.patchEncoder(x), mask: mask4, rope: rope)
        let decoded = net.decoder(hidden).asType(.float32)
        eval(decoded)
        let levels = config.quantileLevels.count
        let flat = decoded.asArray(Float.self)

        // Quantile head (t0/torch/model/layers/head.py): q0 = raw0, q_i = q0 + sum softplus(raw_j).
        // Then rescale each patch with the statistics at its last cell (rescale_predictions).
        var rescaled = [[[Double]]](repeating: [], count: patches)
        for p in 0..<patches {
            let loc = stats.loc[p * patch + patch - 1]
            let scale = stats.scale[p * patch + patch - 1]
            rescaled[p] = (0..<patch).map { k in
                let base = (p * patch + k) * levels
                var quantiles = [Double](repeating: 0, count: levels)
                quantiles[0] = Double(flat[base])
                for q in 1..<levels {
                    quantiles[q] = quantiles[q - 1] + softplus(Double(flat[base + q]))
                }
                return quantiles.map { value in
                    let unscaled = config.scalerUseArcsinh ? sinh(value) : value
                    return unscaled * scale + loc
                }
            }
        }

        // Steps after the context: patch (contextPatches - 1) predicts the first forecast patch (_predict_step).
        let contextPatches = contextWidth / patch
        return (0..<horizon).map { step in
            rescaled[contextPatches - 1 + step / patch][step % patch]
        }
    }
}
