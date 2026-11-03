// SPDX-License-Identifier: MIT
// Timestral 209M (Mistral AI, Modified MIT) ported to MLX from the maker's code (github.com/mistralai/timestral
// a534bdbd): layers.py (RMSNorm, ResidualBlock, FusedSwiGLU, SinCosRoPE with partial rotation, gated time and
// variate attention, TransformerBlock), model.py (PatchRepresentation, QuantileHead, the [Enc ; Dec] two-stream pass)
// and masks.py. The host side (scaling, encoding, block decoding, refine frame, clampsym) is Request.swift. The
// network runs on the float16 weights; norms reduce in float32 and the head's output returns in float32, as the
// maker's half path does.
import Foundation
import MLX
import MLXFast
import MLXNN

let modelKind = "local.forecast"
let modelOps = ["forecast"]

// MARK: - Layers (layers.py)

/// `x * rsqrt(mean(x²) + eps)` in float32, cast back, then times the learned scale.
final class TimestralRMSNorm: Module {
    @ParameterInfo(key: "scale") var scale: MLXArray
    let eps: Float
    init(_ dims: Int, eps: Double) {
        self.eps = Float(eps)
        _scale.wrappedValue = MLXArray.ones([dims])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let f = x.asType(.float32)
        return (f * rsqrt(mean(f * f, axis: -1, keepDims: true) + eps)).asType(x.dtype) * scale
    }
}

/// `output(silu(hidden(x))) + residual(x)`.
final class TimestralResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hidden: Linear
    @ModuleInfo(key: "output_layer") var output: Linear
    @ModuleInfo(key: "residual_layer") var residual: Linear
    init(_ input: Int, _ hiddenDims: Int, _ outputDims: Int, bias: Bool) {
        _hidden.wrappedValue = Linear(input, hiddenDims, bias: bias)
        _output.wrappedValue = Linear(hiddenDims, outputDims, bias: bias)
        _residual.wrappedValue = Linear(input, outputDims, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(silu(hidden(x))) + residual(x) }
}

/// SwiGLU with one fused projection: `w3(gate * silu(up))`, gate the first half.
final class FusedSwiGLU: Module {
    @ModuleInfo(key: "w12") var w12: Linear
    @ModuleInfo(key: "w3") var w3: Linear
    init(_ width: Int, _ hidden: Int) {
        _w12.wrappedValue = Linear(width, 2 * hidden, bias: false)
        _w3.wrappedValue = Linear(hidden, width, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = split(w12(x), parts: 2, axis: -1)
        return w3(parts[0] * silu(parts[1]))
    }
}

/// SinCosRoPE tables [S, headDim] for integer positions: rotate-half layout, only the first ropeDim / 2 frequencies
/// nonzero (the rest are the identity rotation).
struct RopeTables {
    let cos: MLXArray
    let sin: MLXArray
    init(positions: [Int], headDim: Int, ropeDim: Int, theta: Double, dtype: DType) {
        let half = headDim / 2
        let inverse: [Float] = (0..<half).map { i in
            i < ropeDim / 2 ? Float(1.0) / Float(pow(theta, Double(2 * i) / Double(ropeDim))) : 0
        }
        var c = [Float](repeating: 0, count: positions.count * headDim), s = c
        for (row, p) in positions.enumerated() {
            for i in 0..<half {
                let angle = Float(p) * inverse[i]
                c[row * headDim + i] = Foundation.cos(angle); c[row * headDim + i + half] = Foundation.cos(angle)
                s[row * headDim + i] = Foundation.sin(angle); s[row * headDim + i + half] = Foundation.sin(angle)
            }
        }
        cos = MLXArray(c, [positions.count, 1, headDim]).asType(dtype)
        sin = MLXArray(s, [positions.count, 1, headDim]).asType(dtype)
    }
    /// x: [batch, S, heads, headDim].
    func rotate(_ x: MLXArray) -> MLXArray {
        let half = x.dim(-1) / 2
        let rotated = concatenated([-x[.ellipsis, half...], x[.ellipsis, ..<half]], axis: -1)
        return x * cos + rotated * sin
    }
}

/// Multi-head attention over time (with RoPE) or over variates, gated by `sigmoid(_gate(x))`.
final class TimestralAttention: Module {
    let isTime: Bool, heads: Int, headDim: Int
    @ModuleInfo(key: "wQKV") var wQKV: Linear
    @ModuleInfo(key: "wO") var wO: Linear
    @ModuleInfo(key: "gate") var gate: Linear
    init(width: Int, heads: Int, isTime: Bool) {
        self.isTime = isTime; self.heads = heads; headDim = width / heads
        _wQKV.wrappedValue = Linear(width, 3 * width, bias: false)
        _wO.wrappedValue = Linear(width, width, bias: false)
        _gate.wrappedValue = Linear(width, width, bias: false)
    }
    /// x: [R, S, D]; time mask [R, 1, S, S] or variate mask [S, 1, R, R] (nil for one row).
    func callAsFunction(_ x: MLXArray, mask: MLXArray?, rope: RopeTables) -> MLXArray {
        let (rows, length, width) = (x.dim(0), x.dim(1), x.dim(2))
        let inputs = isTime ? x : x.transposed(1, 0, 2)
        let (batch, sequence) = (inputs.dim(0), inputs.dim(1))
        // The fused weight's layout is (qkv, head_dim, heads).
        let qkv = wQKV(inputs).reshaped(batch, sequence, 3, headDim, heads).transposed(2, 0, 1, 4, 3)
        var q = qkv[0], k = qkv[1]
        if isTime { q = rope.rotate(q); k = rope.rotate(k) }
        let attended = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: k.transposed(0, 2, 1, 3), values: qkv[2].transposed(0, 2, 1, 3),
            scale: 1 / Float(headDim).squareRoot(), mask: mask.map { .array($0) } ?? .none)
        var out = attended.transposed(0, 2, 1, 3).reshaped(batch, sequence, heads * headDim) * sigmoid(gate(inputs))
        if !isTime { out = out.reshaped(length, rows, width).transposed(1, 0, 2) }
        return wO(out)
    }
}

final class TimestralBlock: Module {
    @ModuleInfo(key: "attn_norms") var attnNorms: [TimestralRMSNorm]
    @ModuleInfo(key: "attns") var attns: [TimestralAttention]
    @ModuleInfo(key: "mlp_norm") var mlpNorm: TimestralRMSNorm
    @ModuleInfo(key: "mlp") var mlp: FusedSwiGLU
    init(kind: String, config: TimestralConfig) {
        let axes = TimestralConfig.kindAxes[kind]!
        _attnNorms.wrappedValue = axes.map { _ in TimestralRMSNorm(config.dModel, eps: config.normEps) }
        _attns.wrappedValue = axes.map { TimestralAttention(width: config.dModel, heads: config.heads, isTime: $0) }
        _mlpNorm.wrappedValue = TimestralRMSNorm(config.dModel, eps: config.normEps)
        _mlp.wrappedValue = FusedSwiGLU(config.dModel, config.mlpHidden)
    }
    func callAsFunction(_ input: MLXArray, timeMask: MLXArray, variateMask: MLXArray?, rope: RopeTables) -> MLXArray {
        var x = input
        for (norm, attention) in zip(attnNorms, attns) {
            x = x + attention(norm(x), mask: attention.isTime ? timeMask : variateMask, rope: rope)
        }
        return x + mlp(mlpNorm(x))
    }
}

final class TimestralTransformer: Module {
    @ModuleInfo(key: "layers") var layers: [TimestralBlock]
    @ModuleInfo(key: "out_norm") var outNorm: TimestralRMSNorm
    init(_ config: TimestralConfig) {
        _layers.wrappedValue = config.schedule.map { TimestralBlock(kind: $0, config: config) }
        _outNorm.wrappedValue = TimestralRMSNorm(config.dModel, eps: config.normEps)
    }
}

final class PatchRepresentation: Module {
    @ModuleInfo(key: "patch_proj") var patchProj: TimestralResidualBlock
    @ParameterInfo(key: "forecast_query") var forecastQuery: MLXArray
    init(_ config: TimestralConfig) {
        _patchProj.wrappedValue = TimestralResidualBlock(2 * config.patch, 4 * config.dModel, config.dModel, bias: true)
        _forecastQuery.wrappedValue = MLXArray.zeros([config.dModel])
    }
}

final class QuantileHead: Module {
    @ModuleInfo(key: "projection") var projection: TimestralResidualBlock
    init(_ config: TimestralConfig) {
        _projection.wrappedValue = TimestralResidualBlock(config.dModel, 4 * config.dModel, config.patch * config.knots.count, bias: false)
    }
}

final class TimestralNetwork: Module {
    @ModuleInfo(key: "representation") var representation: PatchRepresentation
    @ModuleInfo(key: "transformer") var transformer: TimestralTransformer
    @ModuleInfo(key: "head") var head: QuantileHead
    init(_ config: TimestralConfig) {
        _representation.wrappedValue = PatchRepresentation(config)
        _transformer.wrappedValue = TimestralTransformer(config)
        _head.wrappedValue = QuantileHead(config)
    }
}

// MARK: - Masks (masks.py), boolean, true = may attend

enum Masks {
    static func range(_ n: Int) -> MLXArray { MLXArray((0..<n).map(Int32.init)) }
    static func causal(_ n: Int) -> MLXArray { range(n).reshaped(n, 1) .>= range(n).reshaped(1, n) }
    static func eye(_ n: Int) -> MLXArray { range(n).reshaped(n, 1) .== range(n).reshaped(1, n) }

    /// `(R, L, L)` causal encoder mask over valid keys, diagonal open; `bidirectional` rows see every valid key.
    static func encoder(_ keyValid: MLXArray, bidirectional: MLXArray? = nil) -> MLXArray {
        let n = keyValid.dim(1)
        let keys = keyValid.expandedDimensions(axis: 1)
        let causalMask = (causal(n).expandedDimensions(axis: 0) .&& keys) .|| eye(n).expandedDimensions(axis: 0)
        guard let bidirectional else { return causalMask }
        let full = keys .|| eye(n).expandedDimensions(axis: 0)
        return MLX.where(bidirectional.reshaped(-1, 1, 1), full, causalMask)
    }

    /// No covariates: `[Enc(L) ; Dec(h)]`, (R, 1, L+h, L+h).
    static func forecastTime(keyValid: MLXArray, horizon h: Int) -> MLXArray {
        let (rows, l) = (keyValid.dim(0), keyValid.dim(1))
        let top = concatenated([encoder(keyValid), MLXArray.zeros([rows, l, h], dtype: .bool)], axis: 2)
        let decEnc = broadcast(keyValid.expandedDimensions(axis: 1), to: [rows, h, l])
        let decDec = broadcast(causal(h).expandedDimensions(axis: 0), to: [rows, h, h])
        return concatenated([top, concatenated([decEnc, decDec], axis: 2)], axis: 1).expandedDimensions(axis: 1)
    }

    /// The covariate window `[Enc(n) ; Dec(n)]`, (R, 1, 2n, 2n), every row's context ending at patch `l`.
    static func windowTime(keyValid: MLXArray, contextPatches l: Int, blockSize: Int, known: MLXArray) -> MLXArray {
        let (rows, n) = (keyValid.dim(0), keyValid.dim(1))
        let p = Array(0..<n)
        let activeValues = p.map { $0 >= l }
        let anchorValues = p.map { $0 >= l ? Int32(l + blockSize * (($0 - l) / blockSize)) : 0 }
        let active = MLXArray(activeValues), anchor = MLXArray(anchorValues)
        let j = range(n).reshaped(1, 1, n)
        let decEnc = active.reshaped(1, n, 1) .&& (j .< anchor.reshaped(1, n, 1)) .&& keyValid.expandedDimensions(axis: 1)
        let sameBlock = anchor.reshaped(n, 1) .== anchor.reshaped(1, n)
        let decDec = ((sameBlock .&& causal(n) .&& active.reshaped(n, 1) .&& active.reshaped(1, n)) .|| eye(n)).expandedDimensions(axis: 0)
        let top = concatenated([encoder(keyValid, bidirectional: known), MLXArray.zeros([rows, n, n], dtype: .bool)], axis: 2)
        let bottom = concatenated([broadcast(decEnc, to: [rows, n, n]), broadcast(decDec, to: [rows, n, n])], axis: 2)
        return concatenated([top, bottom], axis: 1).expandedDimensions(axis: 1)
    }

    /// `(Nenc + Ndec, 1, R, R)` variate mask for one series, or nil for one row. Covariate rows never read targets;
    /// known-future rows never read causal rows; the diagonal is open.
    static func variate(keyValid: MLXArray, decoderSlots: Int, target: [Bool]?, known: [Bool]?) -> MLXArray? {
        let rows = keyValid.dim(0)
        guard rows > 1 else { return nil }
        let slots = concatenated([keyValid, MLXArray.ones([rows, decoderSlots], dtype: .bool)], axis: 1).transposed(1, 0) // [S, R]
        var mask = broadcast(slots.reshaped(-1, 1, 1, rows), to: [slots.dim(0), 1, rows, rows])
        if target != nil || known != nil {
            var forbid = [Bool](repeating: false, count: rows * rows)
            for a in 0..<rows {
                for b in 0..<rows {
                    if let target, !target[a] && target[b] { forbid[a * rows + b] = true }
                    if let known, known[a] && !known[b] { forbid[a * rows + b] = true }
                }
            }
            mask = mask .&& .!MLXArray(forbid, [1, 1, rows, rows])
        }
        return mask .|| eye(rows).reshaped(1, 1, rows, rows)
    }
}

// MARK: - Model

final class Model {
    let config: TimestralConfig
    let net: TimestralNetwork
    let forecaster: Forecaster

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = try TimestralConfig(try Files.json(weightsDir))
        net = TimestralNetwork(config)
        // MLX's module tree skips keys that start with an underscore, so the maker's `_gate` loads as `gate`.
        try Weights.apply(try Weights.load(weightsDir), to: net, rename: { $0.replacingOccurrences(of: "._gate.", with: ".gate.") })
        forecaster = Forecaster(config: config)
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "forecast" else { throw RunnerError.invalid("unknown op \(op)") }
        let request = try ForecastRequest(params.raw, config: config)
        // All 21 trained levels come back; the median is always the model's own 0.5 level.
        let all = try forecaster.forecast(request) { input in try self.block(input, check: params.checkCancelled) }
        let columns = request.levels.map { level in config.knots.firstIndex { abs($0 - level) < 1e-9 }! }
        func series(_ steps: [[Double]]) -> [String: Any] {
            ["median": steps.map { $0[config.medianIndex] }, "quantiles": steps.map { row in columns.map { row[$0] } }]
        }
        var result = series(all[0])
        result["horizon"] = request.horizon
        result["quantile_levels"] = request.levels
        if all.count > 1 { result["co_target_forecasts"] = all.dropFirst().map(series) }
        return result
    }

    nonisolated(unsafe) static var captureCount = 0

    /// One decode block through the network: raw head outputs [rows][h·patch][Q] in float32.
    func block(_ input: BlockInput, check: () throws -> Void) throws -> [Float] {
        let ps = config.patch, rows = input.rows, n = input.encPatches, h = input.horizonPatches, l = input.contextPatches
        let decoderSlots = input.window ? n : h
        let keyValid = MLXArray(input.keyValid, [rows, n])
        let timeMask =
            input.window
            ? Masks.windowTime(keyValid: keyValid, contextPatches: l, blockSize: config.blockSize, known: MLXArray(input.knownRows))
            : Masks.forecastTime(keyValid: keyValid, horizon: h)
        let variateMask = Masks.variate(
            keyValid: keyValid, decoderSlots: decoderSlots, target: input.window ? input.targetRows : nil, known: input.window ? input.knownRows : nil)
        let encPositions = (0..<n).map { $0 - input.firstObserved }
        let positions = input.window ? encPositions + encPositions : (0..<(n + h)).map { $0 - input.firstObserved }
        let rope = RopeTables(positions: positions, headDim: config.headDim, ropeDim: config.ropeDim, theta: config.ropeTheta, dtype: .float16)

        let patches = MLXArray(input.patchInput, [rows, n, 2 * ps]).asType(.float16)
        let encoded = net.representation.patchProj(patches)
        let queries = broadcast(net.representation.forecastQuery.reshaped(1, 1, -1), to: [rows, decoderSlots, config.dModel])
        var x = concatenated([encoded, queries], axis: 1)
        let captureDir = ProcessInfo.processInfo.environment["ARSENAL_TIMESTRAL_CAPTURE_DIR"]
        var captured: [String: MLXArray] = [:]
        if captureDir != nil { captured["patch_input"] = patches.asType(.float32); captured["encoded"] = encoded.asType(.float32) }
        for (index, layer) in net.transformer.layers.enumerated() {
            x = layer(x, timeMask: timeMask, variateMask: variateMask, rope: rope)
            if captureDir != nil { captured["layer.\(index)"] = x.asType(.float32) }
            eval(x)
            try check()
        }
        x = net.transformer.outNorm(x)
        let decoder = input.window ? x[0..., (n + l)..<(2 * n), 0...] : x[0..., n..., 0...]
        let raw = net.head.projection(decoder).asType(.float32)
        eval(raw)
        if let captureDir {
            captured["out_norm"] = x.asType(.float32); captured["head_raw"] = raw
            Self.captureCount += 1
            try MLX.save(arrays: captured, url: URL(fileURLWithPath: captureDir).appendingPathComponent("block-\(Self.captureCount).safetensors"))
        }
        return raw.reshaped(rows * h * ps * config.knots.count).asArray(Float.self)
    }
}
