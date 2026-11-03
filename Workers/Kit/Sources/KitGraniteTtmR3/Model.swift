// SPDX-License-Identifier: MIT
// Granite TimeSeries TTM R3 (IBM, Apache-2.0), the 512-30-dec-90 checkpoint, ported to MLX from the maker's
// granite-tsfm code (tsfm_public/models/tinytimemixer/modeling_tinytimemixer.py, git 5991892):
// TinyTimeMixerForDecomposedPrediction = std scaler + robust_lowess_like trend split (host, Double), then a
// trend TinyTimeMixerForPrediction on the scaled context (512 steps, patch 64) and a residual one on the last
// 90 detrended steps (patch 9), each a multi-scale patcher + register tokens + gated MLP-mixer encoder, mixer
// decoder, linear head and multi-quantile head; quantiles combined by the maker's sqrt-of-squares width rule.
// Scope: one univariate series, horizon up to the checkpoint's 30 steps (one pass; the maker ships no rollout here).
import Foundation
import MLX
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.forecast"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["forecast"]

/// One forecaster's shape (TinyTimeMixerConfig after the decomposed model's trend/residual overrides).
struct TTMConfig {
    var contextLength: Int
    var patchLength: Int
    var dModel: Int
    var decoderDModel: Int
    var numLayers: Int
    var decoderNumLayers: Int
    var registerTokens: Int
    var predictionLength: Int
    var expansion: Int
    var normEps: Float
    var mqHidden: Int
    var mqDecoderDModel: Int
    var quantileCount: Int

    /// Patches kept by TinyTimeMixerPatchify (stride == length).
    var basePatches: Int { (max(contextLength, patchLength) - patchLength) / patchLength + 1 }

    /// Patch counts per scale, finest first (MultiScaleFromPatchedSequence on the patchified sequence).
    var scalePatchCounts: [Int] {
        let s = basePatches * patchLength
        var counts: [Int] = []
        var i = 0
        while true {
            let factor = 1 << i
            if s < factor || s / factor < patchLength { break }
            let n = (s / factor) / patchLength
            if n == 0 { break }
            counts.append(n)
            i += 1
        }
        return counts
    }

    /// Projector count of MultiScaleFromPatchedSequence (from context_length, not the patchified length).
    var maxScales: Int {
        var count = 0, length = contextLength
        while length >= patchLength { count += 1; length /= 2 }
        return count
    }

    /// num_patches as check_and_init_preprocessing computes it (multi-scale from context_length, plus registers).
    var numPatches: Int {
        var total = 0, i = 0
        while true {
            let down = contextLength / (1 << i)
            if down < patchLength { break }
            total += down / patchLength
            i += 1
        }
        return total + registerTokens
    }
}

/// x * softmax(Linear(x)) over the last axis (TinyTimeMixerGatedAttention, gate_mode softmax).
final class Gate: Module {
    @ModuleInfo(key: "attn_layer") var attnLayer: Linear
    init(_ size: Int) { _attnLayer.wrappedValue = Linear(size, size) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x * softmax(attnLayer(x).asType(.float32), axis: -1).asType(x.dtype)
    }
}

/// fc1 → exact GELU → fc2 (TinyTimeMixerMLP; dropout is off at inference).
final class MixerMLP: Module {
    @ModuleInfo(key: "fc1") var fc1: Linear
    @ModuleInfo(key: "fc2") var fc2: Linear
    init(_ size: Int, expansion: Int) {
        _fc1.wrappedValue = Linear(size, size * expansion)
        _fc2.wrappedValue = Linear(size * expansion, size)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(gelu(fc1(x))) }
}

/// TinyTimeMixerNormLayer with LayerNorm (norm_mlp = LayerNorm): the checkpoint nests it as norm.norm.
final class NormLayer: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    init(_ dims: Int, eps: Float) { _norm.wrappedValue = LayerNorm(dimensions: dims, eps: eps) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(x) }
}

/// PatchMixerBlock (patch axis) or FeatureMixerBlock (feature axis); x is [patches, d].
final class MixerBlock: Module {
    let acrossPatches: Bool
    @ModuleInfo(key: "norm") var norm: NormLayer
    @ModuleInfo(key: "mlp") var mlp: MixerMLP
    @ModuleInfo(key: "gating_block") var gatingBlock: Gate

    init(dModel: Int, patches: Int, acrossPatches: Bool, config: TTMConfig) {
        self.acrossPatches = acrossPatches
        let size = acrossPatches ? patches : dModel
        _norm.wrappedValue = NormLayer(dModel, eps: config.normEps)
        _mlp.wrappedValue = MixerMLP(size, expansion: config.expansion)
        _gatingBlock.wrappedValue = Gate(size)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = norm(x)
        if acrossPatches { h = h.transposed(1, 0) }
        h = gatingBlock(mlp(h))
        if acrossPatches { h = h.transposed(1, 0) }
        return h + x
    }
}

/// TinyTimeMixerLayer in common_channel mode: patch mixer, then feature mixer.
final class MixerLayer: Module {
    @ModuleInfo(key: "patch_mixer") var patchMixer: MixerBlock
    @ModuleInfo(key: "feature_mixer") var featureMixer: MixerBlock
    init(dModel: Int, patches: Int, config: TTMConfig) {
        _patchMixer.wrappedValue = MixerBlock(dModel: dModel, patches: patches, acrossPatches: true, config: config)
        _featureMixer.wrappedValue = MixerBlock(dModel: dModel, patches: patches, acrossPatches: false, config: config)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { featureMixer(patchMixer(x)) }
}

/// TinyTimeMixerBlock without adaptive patching.
final class MixerStack: Module {
    @ModuleInfo(key: "mixers") var mixers: [MixerLayer]
    init(layers: Int, dModel: Int, patches: Int, config: TTMConfig) {
        _mixers.wrappedValue = (0..<layers).map { _ in MixerLayer(dModel: dModel, patches: patches, config: config) }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for layer in mixers { h = layer(h) }
        return h
    }
}

final class MultiScalePatcher: Module {
    @ModuleInfo(key: "projectors") var projectors: [Linear]
    init(_ config: TTMConfig) {
        _projectors.wrappedValue = (0..<config.maxScales).map { _ in Linear(config.patchLength, config.dModel) }
    }
}

final class RegisterTokens: Module {
    @ParameterInfo(key: "patch_tokens") var patchTokens: MLXArray
    init(_ count: Int, _ dims: Int) { _patchTokens.wrappedValue = MLXArray.zeros([count, dims]) }
}

final class Encoder: Module {
    @ModuleInfo(key: "patcher") var patcher: MultiScalePatcher
    @ModuleInfo(key: "add_tokens") var addTokens: RegisterTokens
    @ModuleInfo(key: "base_norm") var baseNorm: LayerNorm
    @ModuleInfo(key: "mlp_mixer_encoder") var mixer: MixerStack
    init(_ config: TTMConfig) {
        _patcher.wrappedValue = MultiScalePatcher(config)
        _addTokens.wrappedValue = RegisterTokens(config.registerTokens, config.dModel)
        _baseNorm.wrappedValue = LayerNorm(dimensions: config.numPatches * config.dModel, eps: config.normEps)
        _mixer.wrappedValue = MixerStack(layers: config.numLayers, dModel: config.dModel, patches: config.numPatches, config: config)
    }
}

final class Backbone: Module {
    @ModuleInfo(key: "encoder") var encoder: Encoder
    init(_ config: TTMConfig) { _encoder.wrappedValue = Encoder(config) }
}

final class Decoder: Module {
    @ModuleInfo(key: "adapter") var adapter: Linear?
    @ModuleInfo(key: "decoder_block") var decoderBlock: MixerStack
    init(_ config: TTMConfig) {
        _adapter.wrappedValue = config.dModel != config.decoderDModel ? Linear(config.dModel, config.decoderDModel) : nil
        _decoderBlock.wrappedValue = MixerStack(
            layers: config.decoderNumLayers, dModel: config.decoderDModel, patches: config.numPatches, config: config)
    }
}

final class Head: Module {
    @ModuleInfo(key: "base_forecast_block") var baseForecastBlock: Linear
    init(_ config: TTMConfig) {
        _baseForecastBlock.wrappedValue = Linear(config.numPatches * config.decoderDModel, config.predictionLength)
    }
}

/// A raw weight and bias (the quantile head's Conv1d and InstanceNorm parameters, used with explicit math).
final class Raw: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    @ParameterInfo(key: "bias") var bias: MLXArray
    init(_ weightShape: [Int], _ biasCount: Int) {
        _weight.wrappedValue = MLXArray.zeros(weightShape)
        _bias.wrappedValue = MLXArray.zeros([biasCount])
    }
}

/// MultiQuantileHead with mq_use_decoder_pool, cond path flatten, cond mode concat, q50 type mean.
final class QuantileHead: Module {
    @ModuleInfo(key: "dw") var dw: Raw
    @ModuleInfo(key: "inorm") var inorm: Raw
    @ModuleInfo(key: "pw1") var pw1: Raw
    @ModuleInfo(key: "pw2") var pw2: Raw
    @ModuleInfo(key: "dec_tok_proj") var decTokProj: Linear
    @ModuleInfo(key: "dec_flat_to_hidden") var decFlatToHidden: Linear

    init(_ config: TTMConfig) {
        _dw.wrappedValue = Raw([1, 1, 3], 1)
        _inorm.wrappedValue = Raw([1], 1)
        _pw1.wrappedValue = Raw([config.mqHidden, 1, 1], config.mqHidden)
        _pw2.wrappedValue = Raw([config.quantileCount, 2 * config.mqHidden, 1], config.quantileCount)
        _decTokProj.wrappedValue = Linear(config.decoderDModel, config.mqDecoderDModel)
        _decFlatToHidden.wrappedValue = Linear(config.numPatches * config.mqDecoderDModel, config.mqHidden)
    }

    /// meanHat [T] float32 and the decoder output [P, D] → raw channels o [Q, T] in float32.
    func raw(_ meanHat: MLXArray, decoded: MLXArray) -> MLXArray {
        let t = meanHat.dim(0)
        let f32 = { (a: MLXArray) in a.asType(.float32) }
        // dw: Conv1d(1, 1, 3, padding 1)
        let w = f32(dw.weight).reshaped([3])
        let padded = concatenated([MLXArray.zeros([1]), meanHat, MLXArray.zeros([1])])
        var x = padded[0..<t] * w[0] + padded[1..<(t + 1)] * w[1] + padded[2..<(t + 2)] * w[2] + f32(dw.bias)[0]
        // InstanceNorm1d(1, affine): biased variance over time, eps 1e-5.
        let mu = x.mean()
        let variance = ((x - mu) * (x - mu)).mean()
        x = (x - mu) * rsqrt(variance + 1e-5) * f32(inorm.weight)[0] + f32(inorm.bias)[0]
        // pw1 (1 → H, kernel 1) then GELU: [H, T]
        let h = gelu(f32(pw1.weight).reshaped([-1, 1]) * x.reshaped([1, t]) + f32(pw1.bias).reshaped([-1, 1]))
        // Decoder conditioning: token projection, flatten, to hidden; broadcast over time.
        let cond = decFlatToHidden(decTokProj(decoded).reshaped([-1])).asType(.float32)
        let c = broadcast(cond.reshaped([-1, 1]), to: [cond.dim(0), t])
        let hc = concatenated([h, c], axis: 0)
        let w2 = f32(pw2.weight).reshaped([pw2.weight.dim(0), pw2.weight.dim(1)])
        return matmul(w2, hc) + f32(pw2.bias).reshaped([-1, 1])
    }
}

/// TinyTimeMixerForPrediction (scaling off inside the decomposed model; the outer scaler runs on the host).
final class Forecaster: Module {
    let config: TTMConfig
    @ModuleInfo(key: "backbone") var backbone: Backbone
    @ModuleInfo(key: "decoder") var decoder: Decoder
    @ModuleInfo(key: "head") var head: Head
    @ModuleInfo(key: "multi_quantile_head_block") var quantileHead: QuantileHead

    init(_ config: TTMConfig) {
        self.config = config
        _backbone.wrappedValue = Backbone(config)
        _decoder.wrappedValue = Decoder(config)
        _head.wrappedValue = Head(config)
        _quantileHead.wrappedValue = QuantileHead(config)
    }

    /// Context in scaled units → (mean forecast [P], raw quantile channels [Q, P]) in scaled units, Double.
    func callAsFunction(_ context: [Double]) -> (mean: [Double], raw: [[Double]]) {
        let c = config
        // Left-pad zeros or keep the last context_length steps (forward() in TinyTimeMixerForPrediction).
        var seq = [Double](repeating: 0, count: c.contextLength)
        let keep = min(context.count, c.contextLength)
        for i in 0..<keep { seq[c.contextLength - keep + i] = context[context.count - keep + i] }
        // Patchify keeps the last basePatches * patchLength steps.
        let s = c.basePatches * c.patchLength
        let trimmed = Array(seq[(c.contextLength - s)...])

        // Multi-scale: average-pool by 2^i (floor), keep the last whole patches, project; coarsest scale first.
        var scales: [MLXArray] = []
        for (i, count) in c.scalePatchCounts.enumerated() {
            let factor = 1 << i
            let pooled = (0..<(s / factor)).map { j in
                trimmed[(j * factor)..<((j + 1) * factor)].reduce(0, +) / Double(factor)
            }
            let tail = pooled.suffix(count * c.patchLength).map { Float($0) }
            let patches = MLXArray(tail, [count, c.patchLength]).asType(.float16)
            scales.append(backbone.encoder.patcher.projectors[i](patches))
        }
        let encoder = backbone.encoder
        var x = concatenated([encoder.addTokens.patchTokens] + scales.reversed(), axis: 0)  // [P, d]
        let p = x.dim(0), d = x.dim(1)
        x = encoder.baseNorm(x.reshaped([p * d])).reshaped([p, d])
        x = encoder.mixer(x)

        if let adapter = decoder.adapter { x = adapter(x) }
        let decoded = decoder.decoderBlock(x)  // [P, decoder d]
        let meanHat = head.baseForecastBlock(decoded.reshaped([-1])).asType(.float32)  // [prediction_length]
        let raw = quantileHead.raw(meanHat, decoded: decoded)
        eval(meanHat, raw)
        let mean = meanHat.asArray(Float.self).map(Double.init)
        let flat = raw.asArray(Float.self).map(Double.init)
        let t = mean.count
        let rows = (0..<raw.dim(0)).map { q in Array(flat[(q * t)..<((q + 1) * t)]) }
        return (mean, rows)
    }
}

final class TTMNetwork: Module {
    @ModuleInfo(key: "trend_forecaster") var trend: Forecaster
    @ModuleInfo(key: "residual_forecaster") var residual: Forecaster
    init(trend: TTMConfig, residual: TTMConfig) {
        _trend.wrappedValue = Forecaster(trend)
        _residual.wrappedValue = Forecaster(residual)
    }
}

private func softplus(_ x: Double) -> Double { x > 0 ? x + log1p(exp(-x)) : log1p(exp(x)) }

/// torch.median: the lower of the two middle values for an even count.
private func lowerMedian(_ x: [Double]) -> Double {
    let sorted = x.sorted()
    return sorted[(sorted.count - 1) / 2]
}

/// robust_lowess_like(x, frac 0.15, iters 2): Gaussian local average with reflect padding, DC guard,
/// and Tukey-bisquare reweighting with a MAD scale.
func robustLowessTrend(_ x: [Double], frac: Double = 0.15, iters: Int = 2, eps: Double = 1e-8) -> [Double] {
    let n = x.count
    var w = max(5, Int((frac * Double(n)).rounded(.toNearestOrEven)))
    if w % 2 == 0 { w += 1 }
    w = min(w, n % 2 == 1 ? n : n - 1)
    let r = (w - 1) / 2
    let sigma = 0.25 * Double(w)
    var kernel = (-r...r).map { exp(-0.5 * pow(Double($0) / sigma, 2)) }
    let total = kernel.reduce(0, +) + 1e-12
    kernel = kernel.map { $0 / total }
    // F.pad reflect: index -i → i, index n-1+i → n-1-i.
    func reflect(_ i: Int) -> Int { i < 0 ? -i : (i >= n ? 2 * (n - 1) - i : i) }
    func conv(_ y: [Double]) -> [Double] {
        (0..<n).map { t in (0..<w).reduce(0.0) { $0 + kernel[$1] * y[reflect(t - r + $1)] } }
    }
    func smooth(_ weights: [Double]?) -> [Double] {
        let wts = weights ?? [Double](repeating: 1, count: n)
        let num = conv(zip(x, wts).map { $0 * $1 })
        let den = conv(wts)
        var out = zip(num, den).map { $0 / ($1 + eps) }
        let shift = x.reduce(0, +) / Double(n) - out.reduce(0, +) / Double(n)
        out = out.map { $0 + shift }
        return out
    }
    var trend = smooth(nil)
    for _ in 0..<iters {
        let resid = zip(x, trend).map { $0 - $1 }
        let med = lowerMedian(resid)
        let centered = resid.map { $0 - med }
        let mad = lowerMedian(centered.map { abs($0) }) + eps
        let wts = centered.map { value -> Double in
            let u = value / (6 * mad)
            let c = min(max(u * u, 0), 1)
            return (1 - c) * (1 - c)
        }
        trend = smooth(wts)
    }
    return trend
}

final class Model {
    let net: TTMNetwork
    let trendConfig: TTMConfig
    let residualConfig: TTMConfig
    let quantileLevels: [Double]
    let minimumScale = 1e-5
    let mqEps: Double

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, *.safetensors, tokenizer files) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let json = try Files.json(weightsDir)
        func int(_ key: String) -> Int? { (json[key] as? NSNumber)?.intValue }
        func bool(_ key: String) -> Bool? { json[key] as? Bool }
        func string(_ key: String) -> String? { json[key] as? String }
        // Only the features this checkpoint uses are ported; refuse a config that needs others.
        let supported = bool("decompose") == true && bool("multi_scale") == true && (int("fft_length") ?? 0) == 0
            && (int("adaptive_patching_levels") ?? 0) == 0 && string("mode") == "common_channel"
            && string("decoder_mode") == "common_channel" && bool("decoder_raw_residual") != true
            && bool("use_positional_encoding") != true && bool("resolution_prefix_tuning") != true
            && bool("self_attn") != true && bool("gated_attn") == true && json["gate_mode"] == nil
            && bool("multi_quantile_head") == true && bool("mq_use_decoder_pool") == true
            && string("mq_cond_path") == "flatten" && string("mq_cond_mode") == "concat" && string("mq_q50_type") == "mean"
            && bool("combine_quantiles_via_variance") == true && string("scaling") == "std"
            && (int("num_input_channels") ?? 1) == 1 && string("norm_mlp") == "LayerNorm"
            && json["prediction_filter_length"] is NSNull && (int("trend_fft_length") ?? 0) == 0
        guard supported else { throw RunnerError.failed("This runner ports only the decomposed TTM R3 512-30-dec-90 layout.") }

        let levels = (json["quantile_levels"] as? [NSNumber])?.map { $0.doubleValue }.sorted() ?? [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        quantileLevels = levels
        mqEps = (json["mq_eps"] as? NSNumber)?.doubleValue ?? 1e-6
        let base = TTMConfig(
            contextLength: int("context_length") ?? 512, patchLength: int("patch_length") ?? 9,
            dModel: int("d_model") ?? 128, decoderDModel: int("decoder_d_model") ?? 64,
            numLayers: int("num_layers") ?? 10, decoderNumLayers: int("decoder_num_layers") ?? 4,
            registerTokens: int("register_tokens") ?? 2, predictionLength: int("prediction_length") ?? 30,
            expansion: int("expansion_factor") ?? 3, normEps: (json["norm_eps"] as? NSNumber)?.floatValue ?? 1e-5,
            mqHidden: int("mq_hidden") ?? 8, mqDecoderDModel: int("mq_decoder_d_model") ?? 8, quantileCount: levels.count)
        var trend = base
        if let v = int("trend_patch_length") { trend.patchLength = v }
        if let v = int("trend_d_model") { trend.dModel = v }
        if let v = int("trend_decoder_d_model") { trend.decoderDModel = v }
        if let v = int("trend_num_layers") { trend.numLayers = v }
        if let v = int("trend_decoder_num_layers") { trend.decoderNumLayers = v }
        if let v = int("trend_register_tokens") { trend.registerTokens = v }
        var residual = base
        if let v = int("residual_context_length") { residual.contextLength = v }
        trendConfig = trend
        residualConfig = residual
        guard levels.count % 2 == 1, abs(levels[levels.count / 2] - 0.5) < 1e-12 else {
            throw RunnerError.failed("quantile_levels must be odd in count with 0.5 in the middle.")
        }
        net = TTMNetwork(trend: trend, residual: residual)
        try Weights.apply(try Weights.load(weightsDir), to: net)
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let values = try params.numbers("values")
        let horizon = params.int("horizon", default: 0)
        let maxHorizon = trendConfig.predictionLength
        guard values.count <= 8192 else { throw RunnerError.invalid("values holds at most 8192 numbers.") }
        guard values.count >= 3 else { throw RunnerError.invalid("values needs at least 3 numbers for the trend split.") }
        guard values.allSatisfy({ $0.isFinite }) else { throw RunnerError.invalid("values must be finite numbers.") }
        guard horizon >= 1, horizon <= maxHorizon else {
            throw RunnerError.invalid("horizon must be from 1 to \(maxHorizon) steps for this model.")
        }
        let quantiles = forecast(values).prefix(horizon).map { $0 }
        let median = quantiles.map { $0[quantileLevels.count / 2] }
        return ["horizon": horizon, "quantile_levels": quantileLevels, "median": median, "quantiles": quantiles]
    }

    /// Per-forecaster quantiles [Q][T] in scaled units (MultiQuantileHead, q50 = the point forecast).
    private func quantiles(_ out: (mean: [Double], raw: [[Double]])) -> [[Double]] {
        let k = quantileLevels.count / 2
        let t = out.mean.count
        var rows = [[Double]](repeating: [Double](repeating: 0, count: t), count: 2 * k + 1)
        for step in 0..<t {
            var down = 0.0, up = 0.0
            rows[k][step] = out.mean[step]
            for j in 0..<k {
                down += softplus(out.raw[1 + j][step]) + mqEps
                up += softplus(out.raw[1 + k + j][step]) + mqEps
                rows[k - 1 - j][step] = out.mean[step] - down
                rows[k + 1 + j][step] = out.mean[step] + up
            }
        }
        return rows
    }

    /// Returns [step][quantile] in the data's units for the checkpoint's full prediction length.
    private func forecast(_ values: [Double]) -> [[Double]] {
        // TinyTimeMixerStdScaler over every observed step.
        let n = Double(values.count)
        let loc = values.reduce(0, +) / n
        let variance = values.reduce(0) { $0 + ($1 - loc) * ($1 - loc) } / n
        let scale = (variance + minimumScale).squareRoot()
        let scaled = values.map { ($0 - loc) / scale }

        let lowess = robustLowessTrend(scaled)
        let lRes = min(residualConfig.contextLength, scaled.count)
        let residualInput = (0..<lRes).map { i in
            scaled[scaled.count - lRes + i] - lowess[lowess.count - lRes + i]
        }
        let trendQ = quantiles(net.trend(scaled))
        let residQ = quantiles(net.residual(residualInput))

        // combine_quantiles_fast: medians add; widths combine as sign(sum) * sqrt(a² + b² + 1e-8); exact median.
        let k = quantileLevels.count / 2
        let t = trendQ[0].count
        return (0..<t).map { step in
            let median = trendQ[k][step] + residQ[k][step]
            return (0..<quantileLevels.count).map { q in
                var value = median
                if q != k {
                    let a = trendQ[q][step] - trendQ[k][step]
                    let b = residQ[q][step] - residQ[k][step]
                    let sum = a + b
                    let sign: Double = sum > 0 ? 1 : (sum < 0 ? -1 : 0)
                    value += sign * (a * a + b * b + 1e-8).squareRoot()
                }
                return value * scale + loc
            }
        }
    }
}
