// SPDX-License-Identifier: MIT
// t0-gamma (The Forecasting Company, Apache-2.0) ported to MLX from the maker's code in tfc-t0 0.6.0, with its MLX
// runtime (t0/mlx/layers.py, t0/mlx/model.py) as the structural reference and its shared host code as the reference
// for everything around the network: t0/data.py (TimeSeries.from_array), t0/rollout.py (prepare_rollout_buffer and the
// single-pass branch of rollout), t0/patcher.py, t0/scaler.py, t0/mask.py, t0/levels.py, t0/quantile.py and the
// adjustments in t0/forecaster.py. Scope: one target series, optionally with past covariates (extra target rows
// forecast jointly, as the maker's [1, V, T] context) and known-future covariates ([1, F, T + H]); horizons up to 1024
// steps in one pass (no autoregressive rollout). Scaling, the quantile head and the quantile-level maths run on the
// host in Double; the transformer runs on the float16 weights.
import Foundation
import MLX
import MLXFast
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.forecast"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["forecast"]

// MARK: - Network (t0/mlx/layers.py)

/// Cos, sin and XPos scale tables for the patch positions 0..<count, each [count, dims].
struct RopeTables {
    let cosTable: MLXArray
    let sinTable: MLXArray
    let scaleTable: MLXArray

    /// TimeAwareRotaryEmbedding (theta 10,000, xPos scale base 512). The rotation pairs adjacent coordinates
    /// (2i, 2i+1); the xPos scale of pair i sits on (2i, 2i+1) when interleaved and on (i, i + dims/2) for "halves".
    static func make(count: Int, dims: Int, halves: Bool, theta: Double = 10_000, scaleBase: Double = 512) -> RopeTables {
        var cosValues = [Float](repeating: 0, count: count * dims)
        var sinValues = cosValues
        var scaleValues = cosValues
        let center = floor(Double(count - 1) / 2)
        let half = dims / 2
        for i in 0..<half {
            let inverseFrequency = 1 / pow(theta, Double(2 * i) / Double(dims))
            let base = (Double(2 * i) + 0.4 * Double(dims)) / (1.4 * Double(dims))
            for p in 0..<count {
                let angle = Double(p) * inverseFrequency
                let scale = Float(pow(base, (Double(p) - center) / scaleBase))
                for k in [2 * i, 2 * i + 1] {
                    cosValues[p * dims + k] = Float(cos(angle))
                    sinValues[p * dims + k] = Float(sin(angle))
                }
                if halves {
                    scaleValues[p * dims + i] = scale
                    scaleValues[p * dims + i + half] = scale
                } else {
                    scaleValues[p * dims + 2 * i] = scale
                    scaleValues[p * dims + 2 * i + 1] = scale
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

/// Two projections with ReLU between them.
final class MLP: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear

    init(_ input: Int, _ hidden: Int, _ output: Int) {
        _hiddenLayer.wrappedValue = Linear(input, hidden)
        _outputLayer.wrappedValue = Linear(hidden, output)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { outputLayer(relu(hiddenLayer(x))) }
}

/// MLP plus a linear skip connection.
final class ResidualBlock: Module {
    @ModuleInfo(key: "mlp") var mlp: MLP
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear

    init(_ input: Int, _ hidden: Int, _ output: Int) {
        _mlp.wrappedValue = MLP(input, hidden, output)
        _residualLayer.wrappedValue = Linear(input, output)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { mlp(x) + residualLayer(x) }
}

/// Patch embedding: [values ‖ within-patch time ‖ validity] projection plus the variate-role embedding.
final class PatchEncoder: Module {
    @ModuleInfo(key: "projection") var projection: ResidualBlock
    @ModuleInfo(key: "type_embeddings") var typeEmbeddings: Embedding

    init(width: Int, patch: Int) {
        _projection.wrappedValue = ResidualBlock(patch * 3, width, width)
        _typeEmbeddings.wrappedValue = Embedding(embeddingCount: 3, dimensions: width)
    }

    /// `x` is [rows, patches, 3 * patch]; `typeIds` [rows, patches] (0 target, 1 historical, 2 future).
    func callAsFunction(_ x: MLXArray, typeIds: MLXArray) -> MLXArray {
        projection(x) + typeEmbeddings(typeIds)
    }
}

/// Self-attention with shared QKV projections, used along time (with RoPE) and across variates (without).
final class SelfAttention: Module {
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

    /// `x` is [batch, length, width]; `mask` a boolean [batch, 1, length, length] (true = may attend).
    func callAsFunction(_ x: MLXArray, mask: MLXArray, rope: RopeTables?) -> MLXArray {
        let batch = x.dim(0), length = x.dim(1)
        let qkv = wQKV(x).reshaped([batch, length, 3, heads, headDim]).transposed(2, 0, 3, 1, 4)
        var queries = qNorm(qkv[0]), keys = kNorm(qkv[1])
        if let rope {
            queries = rope.rotate(queries, keys: false)
            keys = rope.rotate(keys, keys: true)
        }
        let attended = MLXFast.scaledDotProductAttention(
            queries: queries, keys: keys, values: qkv[2], scale: 1 / Float(headDim).squareRoot(), mask: .array(mask))
        return wO(attended.transposed(0, 2, 1, 3).reshaped([batch, length, width]))
    }
}

/// Pre-norm attention with a residual connection. Variate (group) attention runs on the [patches, rows] layout.
final class AttentionBlock: Module {
    @ModuleInfo(key: "norm") var norm: RMSNorm
    @ModuleInfo(key: "attention") var attention: SelfAttention

    init(width: Int, heads: Int) {
        _norm.wrappedValue = RMSNorm(width)
        _attention.wrappedValue = SelfAttention(width: width, heads: heads)
    }

    func time(_ x: MLXArray, mask: MLXArray, rope: RopeTables) -> MLXArray {
        x + attention(norm(x), mask: mask, rope: rope)
    }

    func group(_ x: MLXArray, mask: MLXArray) -> MLXArray {
        let flipped = x.transposed(1, 0, 2)
        return x + attention(norm(flipped), mask: mask, rope: nil).transposed(1, 0, 2)
    }
}

/// SwiGLU feed-forward; the first projection's first half is the gate.
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

/// One attention layer (time or group) followed by a pre-norm SwiGLU feed-forward.
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

    func callAsFunction(_ x: MLXArray, timeMask: MLXArray, groupMask: MLXArray, rope: RopeTables) -> MLXArray {
        let attended = isGroup ? attentionBlock.group(x, mask: groupMask) : attentionBlock.time(x, mask: timeMask, rope: rope)
        return attended + mlp(norm(attended))
    }
}

/// Alternating time and group attention stack.
final class Transformer: Module {
    @ModuleInfo(key: "layers") var layers: [TransformerLayer]
    @ModuleInfo(key: "out_norm") var outNorm: RMSNorm

    init(_ config: T0Config) {
        _layers.wrappedValue = (0..<config.numLayers).map { index in
            TransformerLayer(
                width: config.embedDim, heads: config.numHeads, hidden: config.mlpHiddenDim,
                isGroup: config.groupEveryN > 0 && (index + 1) % config.groupEveryN == 0)
        }
        _outNorm.wrappedValue = RMSNorm(config.embedDim)
    }

    func callAsFunction(
        _ x: MLXArray, timeMask: MLXArray, groupMask: MLXArray, rope: RopeTables, checkCancelled: () throws -> Void,
        capture: ((String, MLXArray) -> Void)? = nil
    ) rethrows -> MLXArray {
        var hidden = x
        for (index, layer) in layers.enumerated() {
            hidden = layer(hidden, timeMask: timeMask, groupMask: groupMask, rope: rope)
            capture?("layer.\(index)", hidden)
            try checkCancelled()
        }
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

// MARK: - Host side: input layout, scaling, masks (t0/data.py, rollout.py, patcher.py, scaler.py, mask.py)

/// Mask codes (t0/types.py MaskType) and variate roles (VariateType).
enum Cell {
    static let valid = 0, pad = 1, missing = 2, withheld = 4
    static let target = 0, future = 2
}

/// One forward window: rows (targets first, then known-future covariates) on one time axis, left-padded to the
/// patch grid, the forecast region rounded up to whole patches (prepare_rollout_buffer).
struct Window {
    var values: [[Double]]
    var mask: [[Int]]
    var group: [[Int]]
    var role: [[Int]]
    let targets: Int
    let contextWidth: Int
    let forecastWidth: Int
    var width: Int { contextWidth + forecastWidth }

    /// `targets` are past observations (nil = missing), all of one length T; `futures` span T + horizon.
    static func make(targets: [[Double?]], futures: [[Double?]], horizon: Int, patch: Int) -> Window {
        let context = targets[0].count
        let padLeft = (patch - context % patch) % patch
        let contextWidth = context + padLeft
        let forecastWidth = roundUp(horizon, patch)
        let width = contextWidth + forecastWidth
        var window = Window(values: [], mask: [], group: [], role: [], targets: targets.count, contextWidth: contextWidth, forecastWidth: forecastWidth)
        for (index, row) in (targets + futures).enumerated() {
            let isFuture = index >= targets.count
            var values = [Double](repeating: 0, count: width)
            var mask = [Int](repeating: Cell.pad, count: width)
            var group = [Int](repeating: -1, count: width)
            var role = [Int](repeating: -1, count: width)
            for i in padLeft..<width {
                group[i] = 0
                role[i] = isFuture ? Cell.future : Cell.target
            }
            for t in 0..<context {
                values[padLeft + t] = row[t] ?? 0
                mask[padLeft + t] = row[t] == nil ? Cell.missing : Cell.valid
            }
            for s in 0..<forecastWidth {
                let i = contextWidth + s
                if !isFuture {
                    mask[i] = Cell.withheld
                } else if s < horizon {
                    values[i] = row[context + s] ?? 0
                    mask[i] = row[context + s] == nil ? Cell.missing : Cell.valid
                }
            }
            window.values.append(values)
            window.mask.append(mask)
            window.group.append(group)
            window.role.append(role)
        }
        return window
    }
}

/// Per-cell location and scale (CausalScaler.scale_input) and the scaled values the network reads.
struct Scaled {
    var values: [[Double]]
    var loc: [[Double]]
    var scale: [[Double]]

    static func make(_ window: Window, config: T0Config) -> Scaled {
        var result = Scaled(values: [], loc: [], scale: [])
        for r in window.values.indices {
            let x = window.values[r], mask = window.mask[r]
            let causal = causalStats(x, invalid: mask.map { $0 != Cell.valid }, eps: config.scalerEps, stdClamp: config.stdClamp)
            let global = globalStats(x, invalid: mask.map { $0 != Cell.valid }, eps: config.scalerEps)
            var loc = [Double](repeating: 0, count: x.count), scale = [Double](repeating: 1, count: x.count)
            var scaled = loc
            for i in x.indices {
                let real = window.group[r][i] >= 0
                if real && window.role[r][i] == Cell.future {
                    loc[i] = global.loc
                    scale[i] = global.scale
                } else if real {
                    loc[i] = causal.loc[i]
                    scale[i] = causal.scale[i]
                }
                let standardized = (x[i] - loc[i]) / scale[i]
                scaled[i] = config.scalerUseArcsinh ? asinh(standardized) : standardized
                if config.zeroNoValue && [Cell.pad, Cell.missing, Cell.withheld].contains(mask[i]) { scaled[i] = 0 }
            }
            result.values.append(scaled)
            result.loc.append(loc)
            result.scale.append(scale)
        }
        return result
    }
}

/// Smallest multiple of `multiple` that is at least `value` (t0/types.py round_up).
func roundUp(_ value: Int, _ multiple: Int) -> Int { (value + multiple - 1) / multiple * multiple }

/// Welford causal mean and std per cell (compute_causal_stats): invalid cells contribute nothing.
func causalStats(_ x: [Double], invalid: [Bool], eps: Double, stdClamp: Bool) -> (loc: [Double], scale: [Double]) {
    var loc = [Double](repeating: 0, count: x.count)
    var scale = loc
    var count = 0.0, sum = 0.0, m2 = 0.0, previousMean = 0.0
    for i in x.indices {
        let valid = invalid[i] ? 0.0 : 1.0
        let masked = invalid[i] ? 0.0 : x[i]
        count += valid
        sum += masked
        let clipped = Swift.max(count, 1)
        let mean = sum / clipped
        m2 += (masked - previousMean) * (masked - mean) * valid
        let variance = Swift.max(m2, 0) / Swift.max(clipped - 1, 1)
        loc[i] = mean
        scale[i] = stdClamp ? Swift.max(variance.squareRoot(), eps) : (variance + eps).squareRoot()
        previousMean = mean
    }
    return (loc, scale)
}

/// Global mean and std of a row's valid cells (compute_global_stats): eps is always a lower bound on the std.
func globalStats(_ x: [Double], invalid: [Bool], eps: Double) -> (loc: Double, scale: Double) {
    var count = 0.0, sum = 0.0
    for i in x.indices where !invalid[i] { count += 1; sum += x[i] }
    let mean = sum / Swift.max(count, 1)
    var squared = 0.0
    for i in x.indices where !invalid[i] { squared += (x[i] - mean) * (x[i] - mean) }
    return (mean, Swift.max((squared / Swift.max(count, 2)).squareRoot(), eps))
}

/// Per-patch metadata (reduce_patch_metadata) and the two attention masks (build_attention_masks).
struct PatchLayout {
    let rows: Int
    let patches: Int
    /// [rows][patches]: the first non-PAD cell's value, or -1 when the patch is all PAD.
    var groupIds: [[Int]]
    var roles: [[Int]]
    var attendable: [[Bool]]

    init(_ window: Window, patch: Int) {
        rows = window.values.count
        patches = window.width / patch
        groupIds = []
        roles = []
        attendable = []
        for r in 0..<rows {
            var g = [Int](repeating: -1, count: patches), t = g, a = [Bool](repeating: false, count: patches)
            for p in 0..<patches {
                if let first = (p * patch..<(p + 1) * patch).first(where: { window.mask[r][$0] != Cell.pad }) {
                    g[p] = window.group[r][first]
                    t[p] = window.role[r][first]
                    a[p] = true
                }
            }
            groupIds.append(g)
            roles.append(t)
            attendable.append(a)
        }
    }

    /// [rows, 1, patches, patches]: causal for targets, bidirectional for futures, same series only, no all-PAD key.
    func timeMask() -> [Bool] {
        var mask = [Bool](repeating: false, count: rows * patches * patches)
        for r in 0..<rows {
            for i in 0..<patches {
                let future = roles[r][i] == Cell.future
                for j in 0..<patches {
                    let same = groupIds[r][i] == groupIds[r][j] && groupIds[r][i] >= 0 && groupIds[r][j] >= 0
                    mask[(r * patches + i) * patches + j] = same && (future || i >= j) && attendable[r][j]
                }
            }
        }
        return mask
    }

    /// [patches, 1, rows, rows]: rows of one series attend each other; a future query reads only future keys.
    func groupMask() -> [Bool] {
        var mask = [Bool](repeating: false, count: patches * rows * rows)
        for p in 0..<patches {
            for a in 0..<rows {
                for b in 0..<rows {
                    let same = groupIds[a][p] == groupIds[b][p] && groupIds[a][p] >= 0 && groupIds[b][p] >= 0
                    let futureReadsOther = roles[a][p] == Cell.future && roles[b][p] != Cell.future
                    mask[(p * rows + a) * rows + b] = same && !futureReadsOther
                }
            }
        }
        return mask
    }

    /// Role ids for the type embedding, the all-PAD sentinel clamped to 0.
    func typeIds() -> [Int32] { roles.flatMap { $0.map { Int32(Swift.max($0, 0)) } } }
}

// MARK: - Request and model

/// One request, parsed and checked.
final class Model {
    let config: T0Config
    let net: T0Network

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, model.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = try T0Config(try Files.json(weightsDir))
        net = T0Network(config)
        // head.quantile_levels is a constant of the head, not a learned weight. The feed-forward's checkpoint indices
        // mlp.0 and mlp.2 are named up and down: numeric keys would load as an array.
        try Weights.apply(try Weights.load(weightsDir), to: net, rename: { key in
            if key == "head.quantile_levels" { return nil }
            return key.replacingOccurrences(of: ".mlp.0.", with: ".mlp.up.").replacingOccurrences(of: ".mlp.2.", with: ".mlp.down.")
        })
    }

    /// One request. Returns the op's output fields; throws RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "forecast" else { throw RunnerError.invalid("unknown op \(op)") }
        let request = try ForecastRequest(params.raw, config: config)
        // The maker's median is its 0.5 level, computed in the same pass as the trained levels; it is never
        // re-interpolated from whatever levels the caller asked for.
        var pass = request
        pass.levels = Adjustments.withMedian(request.levels)
        let all = try predict(pass, check: params.checkCancelled)
        let medianColumn = pass.levels.firstIndex(of: 0.5)!
        let keep = request.levels.map { level in pass.levels.firstIndex(of: level)! }
        let quantiles = all.map { $0.map { row in keep.map { row[$0] } } }
        let medians = all.map { $0.map { $0[medianColumn] } }
        var reply: [String: Any] = [
            "horizon": request.horizon,
            "quantile_levels": request.levels,
            "median": medians[0],
            "quantiles": quantiles[0],
        ]
        if quantiles.count > 1 {
            reply["past_covariate_forecasts"] = (1..<quantiles.count).map { ["median": medians[$0], "quantiles": quantiles[$0]] }
        }
        return reply
    }

    /// [target row][step][requested level], in the data's units, after the requested adjustments (ForecasterBase.predict).
    func predict(_ request: ForecastRequest, check: () throws -> Void) throws -> [[[Double]]] {
        let levels = request.symmetricAveraging ? Adjustments.withMirrors(request.levels) : request.levels
        var values = try forecast(targets: request.targets, futures: request.futures, horizon: request.horizon, levels: levels, check: check)
        let history = request.targets.map { $0.compactMap { $0 } }
        if request.symmetricAveraging {
            let averaged = history.map { !$0.isEmpty && $0.min()! != 0 }
            var negated = values
            if averaged.contains(true) {
                negated = try forecast(
                    targets: request.targets.map(Self.negated), futures: request.futures.map(Self.negated),
                    horizon: request.horizon, levels: levels, check: check)
            }
            let mirror = levels.map { level in levels.firstIndex { abs($0 - (1 - level)) <= 1e-9 }! }
            let keep = request.levels.map { level in levels.firstIndex { abs($0 - level) <= 1e-9 }! }
            values = values.indices.map { row in
                values[row].indices.map { step in
                    keep.map { column in
                        averaged[row] ? (values[row][step][column] - negated[row][step][mirror[column]]) / 2 : values[row][step][column]
                    }
                }
            }
        }
        for (row, seen) in history.enumerated() where !seen.isEmpty {
            let adjusted = Adjustments.adjust(values[row], history: seen, clip: request.clipNonnegative, snap: request.snapIntegers)
            values[row] = adjusted
        }
        return values
    }

    /// One pass at `levels`: [target row][step][level] (rollout's single-pass branch, sanitize, extrapolate).
    func forecast(targets: [[Double?]], futures: [[Double?]], horizon: Int, levels: [Double], check: () throws -> Void) throws -> [[[Double]]] {
        let trained = config.quantileLevels
        let rollout = Levels.rolloutLevels(trained: trained, requested: levels)
        let block = try trainedQuantiles(targets: targets, futures: futures, horizon: horizon, check: check)
        return block.map { steps in
            steps.prefix(horizon).map { row in
                let interpolated = Levels.interpolate(row, source: trained, query: rollout).map { $0.isNaN ? 0 : $0 }
                return Levels.extrapolate(interpolated, original: rollout, query: levels, trained: trained)
            }
        }
    }

    /// The network's quantiles at the trained levels, [target row][forecast-width step][trained level], data units.
    func trainedQuantiles(targets: [[Double?]], futures: [[Double?]], horizon: Int, check: () throws -> Void) throws -> [[[Double]]] {
        let patch = config.patchSize
        let window = Window.make(targets: targets, futures: futures, horizon: horizon, patch: patch)
        let scaled = Scaled.make(window, config: config)
        let layout = PatchLayout(window, patch: patch)
        let rows = layout.rows, patches = layout.patches

        // Per patch: [values ‖ time index k/patch ‖ validity].
        var input = [Float](repeating: 0, count: rows * patches * 3 * patch)
        for r in 0..<rows {
            for p in 0..<patches {
                let base = (r * patches + p) * 3 * patch
                for k in 0..<patch {
                    let i = p * patch + k
                    input[base + k] = Float(scaled.values[r][i])
                    input[base + patch + k] = Float(k) / Float(patch)
                    input[base + 2 * patch + k] = window.mask[r][i] == Cell.valid ? 1 : 0
                }
            }
        }
        let x = MLXArray(input, [rows, patches, 3 * patch]).asType(.float16)
        let typeIds = MLXArray(layout.typeIds(), [rows, patches])
        let timeMask = MLXArray(layout.timeMask(), [rows, 1, patches, patches])
        let groupMask = MLXArray(layout.groupMask(), [patches, 1, rows, rows])
        let rope = RopeTables.make(count: patches, dims: config.embedDim / config.numHeads, halves: config.xposHalves)

        // Lab only: ARSENAL_T0_CAPTURE_DIR saves each component's output for the per-component parity check.
        var captured: [String: MLXArray] = [:]
        let captureDir = ProcessInfo.processInfo.environment["ARSENAL_T0_CAPTURE_DIR"]
        let capture: ((String, MLXArray) -> Void)? = captureDir == nil ? nil : { name, value in captured[name] = value.asType(.float32) }
        let encoded = net.patchEncoder(x, typeIds: typeIds)
        capture?("encoder", encoded)
        let hidden = try net.transformer(encoded, timeMask: timeMask, groupMask: groupMask, rope: rope, checkCancelled: check, capture: capture)
        let targetRows = window.targets
        let decoded = net.decoder(hidden[0..<targetRows]).asType(.float32)
        eval(decoded)
        let flat = decoded.asArray(Float.self)
        if let captureDir {
            capture?("input", x)
            capture?("hidden", hidden)
            capture?("decoder", decoded)
            capture?("loc", MLXArray(scaled.loc.flatMap { $0.map { Float($0) } }, [rows, window.width]))
            capture?("scale", MLXArray(scaled.scale.flatMap { $0.map { Float($0) } }, [rows, window.width]))
            Self.captureCount += 1
            try MLX.save(arrays: captured, url: URL(fileURLWithPath: captureDir).appendingPathComponent("pass-\(Self.captureCount).safetensors"))
        }
        return Self.finish(flat, window: window, scaled: scaled, config: config)
    }

    nonisolated(unsafe) static var captureCount = 0

    /// Quantile head (q0 = raw0, q_i = q0 + Σ softplus(raw_j)), rescaling with each patch's last-cell statistics
    /// (rescale_predictions), and the forecast steps after the context (_predict_step).
    static func finish(_ flat: [Float], window: Window, scaled: Scaled, config: T0Config) -> [[[Double]]] {
        let patch = config.patchSize, levels = config.quantileLevels.count
        let patches = window.width / patch, contextPatches = window.contextWidth / patch
        return (0..<window.targets).map { r in
            (0..<window.forecastWidth).map { step in
                let p = contextPatches - 1 + step / patch, k = step % patch
                let loc = scaled.loc[r][p * patch + patch - 1], scale = scaled.scale[r][p * patch + patch - 1]
                let base = ((r * patches + p) * patch + k) * levels
                var quantiles = [Double](repeating: 0, count: levels)
                quantiles[0] = Double(flat[base])
                for q in 1..<levels { quantiles[q] = quantiles[q - 1] + softplus(Double(flat[base + q])) }
                return quantiles.map { value in (config.scalerUseArcsinh ? sinh(value) : value) * scale + loc }
            }
        }
    }

    /// A row negated, its gaps after the first observation linearly interpolated, a gap past the last observation
    /// holding that value (_negated). Filled cells become observations of the negated copy.
    static func negated(_ row: [Double?]) -> [Double?] {
        let seen = row.indices.filter { row[$0] != nil }
        guard let first = seen.first else { return row }
        return row.indices.map { i in
            if let value = row[i] { return -value }
            if i < first { return nil }
            guard let next = seen.first(where: { $0 > i }) else { return -row[seen.last!]! }
            let previous = seen.last(where: { $0 < i })!
            let t = Double(i - previous) / Double(next - previous)
            return -(row[previous]! + t * (row[next]! - row[previous]!))
        }
    }
}
