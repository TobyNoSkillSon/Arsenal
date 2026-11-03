// SPDX-License-Identifier: MIT
// Timestral (Mistral AI) host side, ported from the maker's code (github.com/mistralai/timestral a534bdbd):
// config.py, scaler.py, covariate_encoding.py, the quantile head's monotone_quantiles (model.py) and forecaster.py
// (TimestralForecaster.forecast with clampsym, row assembly, block-by-block decoding, the advancing refine frame and
// the quantile cap). Foundation only: the network is a closure, so this file compiles and runs without MLX.
// Precision follows the maker: series values are float32, causal statistics float64 rounded to float32, the
// normalized and denormalized quantiles float32, the refine-frame statistics float64.
import Foundation

struct TimestralConfig {
    let patch: Int
    let dModel: Int
    let heads: Int
    let mlpHidden: Int
    let schedule: [String]
    let knots: [Double]
    let blockSize: Int
    let contextLength: Int
    let ropeTheta: Double
    let ropePartial: Double
    let normEps: Double
    let minimumScale: Double
    let useAsinh: Bool
    let covariateEnabled: Bool
    let attendMissing: Bool
    let widthFactor: Double

    static let kindAxes: [String: [Bool]] = ["time": [true], "variate": [false], "time_variate": [true, false], "variate_time": [false, true]]
    var headDim: Int { dModel / heads }
    var ropeDim: Int { Int(Double(headDim) * ropePartial) }
    var medianIndex: Int { knots.firstIndex(of: 0.5)! }

    /// config.json, refusing the settings the maker's own inference code refuses or this port does not implement.
    init(_ json: [String: Any]) throws {
        func int(_ key: String) throws -> Int {
            guard let value = json[key] as? NSNumber else { throw RunnerError.failed("config.json lacks \(key)") }
            return value.intValue
        }
        func double(_ key: String, _ fallback: Double) -> Double { (json[key] as? NSNumber)?.doubleValue ?? fallback }
        func bool(_ key: String, _ fallback: Bool) -> Bool { (json[key] as? Bool) ?? fallback }
        patch = try int("patch_size")
        dModel = try int("d_model")
        heads = try int("num_heads")
        mlpHidden = try int("mlp_hidden_dim")
        blockSize = try int("block_size")
        contextLength = try int("context_length")
        schedule = (json["block_schedule"] as? [String]) ?? []
        knots = ((json["quantile_knots"] as? [NSNumber]) ?? []).map(\.doubleValue)
        ropeTheta = double("rope_theta", 10_000)
        ropePartial = double("rope_partial_factor", 0.5)
        normEps = double("norm_eps", 1e-6)
        minimumScale = double("minimum_scale", 1e-6)
        useAsinh = bool("use_asinh", true)
        covariateEnabled = bool("covariate_enabled", true)
        attendMissing = bool("attend_missing_patches", true)
        widthFactor = double("quantile_width_factor", 2)
        guard !schedule.isEmpty, schedule.allSatisfy({ Self.kindAxes[$0] != nil }), knots.contains(0.5),
            zip(knots, knots.dropFirst()).allSatisfy({ $0 < $1 }), zip(knots, knots.reversed()).allSatisfy({ abs($0 + $1 - 1) < 1e-9 }),
            dModel % heads == 0, ropeDim > 0, ropeDim % 2 == 0, patch > 0, blockSize > 0,
            !bool("stabilize_with_global", false), !bool("attn_bias", false), !bool("mlp_bias", false),
            (json["time_attn_gate"] as? String ?? "sdpa_sigmoid") == "sdpa_sigmoid",
            (json["variate_attn_gate"] as? String ?? "sdpa_sigmoid") == "sdpa_sigmoid"
        else { throw RunnerError.failed("unsupported Timestral configuration") }
    }

    /// Some time op is followed by at least one more time op and at least one variate op (config.py).
    var supportsKnownCovariates: Bool {
        guard covariateEnabled else { return false }
        let ops = schedule.flatMap { Self.kindAxes[$0]! }
        for (i, isTime) in ops.enumerated() where isTime {
            let rest = ops[(i + 1)...]
            if rest.contains(true) && rest.contains(false) { return true }
        }
        return false
    }
}

// MARK: - Request

struct ForecastRequest {
    static let maxContext = 32768
    static let maxHorizon = 4096
    static let maxRows = 32
    /// Numbers in the reply's forecasts (series x horizon x (levels + median)). The longest float as JSON is 24 bytes
    /// with its comma (-1.2345678442216013e-05, measured), plus up to 2 of brackets: 640,000 stay under the worker's
    /// 16 MiB - 64 KiB reply limit in the worst case.
    static let maxReplyNumbers = 640_000

    var values: [Double?]
    var horizon: Int
    var levels: [Double]
    var coTargets: [[Double?]]
    var pastCovariates: [[Double?]]
    var knownCovariates: [[Double?]]
    var categoricalPast: [[Double?]]
    var categoricalKnown: [[Double?]]
    var symmetricAveraging: Bool
    var clipNonnegative: Bool

    init(_ raw: [String: Any], config: TimestralConfig) throws {
        let allowed: Set<String> = [
            "values", "horizon", "quantile_levels", "co_targets", "past_covariates", "covariates", "categorical_past_covariates",
            "categorical_covariates", "symmetric_averaging", "clip_nonnegative"
        ]
        if let unknown = raw.keys.first(where: { !allowed.contains($0) && $0 != "out_dir" }) { throw RunnerError.invalid("unknown field \(unknown)") }
        values = try Self.series(raw["values"], "values")
        guard (1...Self.maxContext).contains(values.count), values.contains(where: { $0 != nil }) else {
            throw RunnerError.invalid("values must hold 1 to \(Self.maxContext) observations, at least one a number.")
        }
        guard let number = raw["horizon"] as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue == number.doubleValue.rounded(),
            (1...Double(Self.maxHorizon)).contains(number.doubleValue)
        else { throw RunnerError.invalid("horizon must be an integer from 1 to \(Self.maxHorizon).") }
        horizon = number.intValue
        if let requested = raw["quantile_levels"] {
            guard let list = requested as? [Any], !list.isEmpty else { throw RunnerError.invalid("quantile_levels must be a nonempty list.") }
            levels = try list.map { item in
                guard let level = (item as? NSNumber)?.doubleValue, CFGetTypeID(item as CFTypeRef) != CFBooleanGetTypeID(),
                    let trained = config.knots.first(where: { abs($0 - level) < 1e-9 })
                else {
                    throw RunnerError.invalid("quantile_levels must be among the 21 trained levels: " + config.knots.map { String($0) }.joined(separator: ", "))
                }
                return trained
            }
            // Any order is accepted; the reply lists the levels ascending.
            guard Set(levels).count == levels.count else { throw RunnerError.invalid("quantile_levels must be unique.") }
            levels.sort()
        } else {
            levels = config.knots
        }
        let context = values.count
        func rows(_ key: String, length: Int, what: String) throws -> [[Double?]] {
            guard let value = raw[key] else { return [] }
            guard let list = value as? [Any], list.count <= 31 else { throw RunnerError.invalid("\(key) must be a list of at most 31 series.") }
            return try list.enumerated().map { index, item in
                let row = try Self.series(item, "\(key)[\(index)]")
                guard row.count == length else { throw RunnerError.invalid("\(key)[\(index)] must have \(length) values (\(what)).") }
                return row
            }
        }
        coTargets = try rows("co_targets", length: context, what: "as many as values")
        pastCovariates = try rows("past_covariates", length: context, what: "as many as values")
        knownCovariates = try rows("covariates", length: context + horizon, what: "len(values) + horizon")
        categoricalPast = try rows("categorical_past_covariates", length: context, what: "as many as values")
        categoricalKnown = try rows("categorical_covariates", length: context + horizon, what: "len(values) + horizon")
        let total = 1 + coTargets.count + pastCovariates.count + knownCovariates.count + categoricalPast.count + categoricalKnown.count
        guard total <= Self.maxRows else { throw RunnerError.invalid("at most \(Self.maxRows) series in all (values, co_targets and covariates).") }
        let numbers = (1 + coTargets.count) * horizon * (levels.count + 1)
        guard numbers <= Self.maxReplyNumbers else {
            throw RunnerError.invalid(
                "The forecast would return \(numbers) numbers (\(1 + coTargets.count) series x \(horizon) steps x \(levels.count + 1) levels with the median), "
                    + "over the reply limit of \(Self.maxReplyNumbers) (16 MiB as JSON): ask for fewer co_targets or quantile_levels, or a shorter horizon.")
        }
        if !(knownCovariates.isEmpty && categoricalKnown.isEmpty) && !config.supportsKnownCovariates {
            throw RunnerError.invalid("this checkpoint cannot condition on known-future covariates")
        }
        symmetricAveraging = try Self.flag(raw["symmetric_averaging"], "symmetric_averaging", true)
        clipNonnegative = try Self.flag(raw["clip_nonnegative"], "clip_nonnegative", true)
    }

    static func series(_ value: Any?, _ name: String) throws -> [Double?] {
        guard let list = value as? [Any] else { throw RunnerError.invalid("\(name) must be a list of numbers or nulls.") }
        return try list.map { item in
            if item is NSNull { return nil }
            guard let number = item as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite else {
                throw RunnerError.invalid("\(name) must hold finite numbers or null.")
            }
            return number.doubleValue
        }
    }

    static func flag(_ value: Any?, _ name: String, _ fallback: Bool) throws -> Bool {
        guard let value else { return fallback }
        guard let number = value as? NSNumber, CFGetTypeID(number) == CFBooleanGetTypeID() else { throw RunnerError.invalid("\(name) must be true or false.") }
        return number.boolValue
    }
}

// MARK: - Scaling (scaler.py)

enum Scaling {
    /// Causal running mean and std per step (Welford form in float64), rounded to float32 like the maker.
    static func causal(_ x: [Float], _ observed: [Bool], minimumScale: Double) -> (loc: [Float], scale: [Float]) {
        var loc = [Float](repeating: 0, count: x.count), scale = loc
        var cumW = 0.0, cumX = 0.0, m2 = 0.0, previous = 0.0
        for t in x.indices {
            let w = observed[t] ? 1.0 : 0.0, v = Double(x[t])
            cumW += w
            cumX += w * v
            let den = Swift.max(cumW, 1)
            let mean = cumX / den
            m2 += (v - previous) * (v - mean) * w
            let variance = m2 / Swift.max(den - 1, 1)
            loc[t] = Float(mean)
            scale[t] = Float(Swift.max(variance.squareRoot(), minimumScale))
            previous = mean
        }
        return (loc, scale)
    }

    /// Per-patch statistics at each patch's last step, broadcast over the patch (causal_patch_loc_scale).
    static func patchCausal(_ x: [Float], _ observed: [Bool], patch: Int, minimumScale: Double) -> (loc: [Float], scale: [Float]) {
        let (loc, scale) = causal(x, observed, minimumScale: minimumScale)
        var pl = loc, ps = scale
        for start in stride(from: 0, to: x.count, by: patch) {
            let last = start + patch - 1
            for t in start..<(start + patch) { pl[t] = loc[last]; ps[t] = scale[last] }
        }
        return (pl, ps)
    }

    /// `asinh((x - loc) / scale)` on observed steps in float32; unobserved steps are 0.
    static func affine(_ x: [Float], loc: [Float], scale: [Float], observed: [Bool], asinh useAsinh: Bool) -> [Float] {
        x.indices.map { t in
            guard observed[t] else { return 0 }
            let z = (x[t] - loc[t]) / Swift.max(scale[t], Float.ulpOfOne)
            return useAsinh ? Foundation.asinh(z) : z
        }
    }

    /// Non-causal mean and std (no Bessel correction) for known-future covariate rows; 0/1 rows keep 0 and 1.
    /// Sums run in Double, so large finite values cannot overflow float32 on the way to the mean and std.
    static func instance(_ x: [Float], _ observed: [Bool], minimumScale: Double) -> (loc: Float, scale: Float, binary: Bool) {
        let n = Swift.max(Double(observed.filter { $0 }.count), 1)
        var sum = 0.0
        for t in x.indices where observed[t] { sum += Double(x[t]) }
        let loc = sum / n
        var squares = 0.0
        for t in x.indices where observed[t] { squares += (Double(x[t]) - loc) * (Double(x[t]) - loc) }
        let scale = Swift.max((squares / n).squareRoot(), minimumScale)
        let binary = observed.contains(true) && x.indices.allSatisfy { !observed[$0] || x[$0] == 0 || x[$0] == 1 }
        return binary ? (0, 1, true) : (Float(loc), Float(scale), false)
    }

    /// +/-inf to the nearest finite value along the quantile axis; a step with no finite value becomes 0.
    static func clampNonfinite(_ row: [Float]) -> [Float] {
        let finite = row.filter { $0.isFinite }
        return row.map { v in
            if v == .infinity { return finite.max() ?? 0 }
            if v == -.infinity { return finite.min() ?? 0 }
            return v
        }
    }

    /// Welford merge of `(weight, mean, m2)` states.
    static func combine(_ a: (Double, Double, Double), _ b: (Double, Double, Double)) -> (Double, Double, Double) {
        let w = a.0 + b.0
        guard w != 0 else { return (0, 0, 0) }
        let safe = Swift.max(w, Double.leastNormalMagnitude)
        let delta = b.1 - a.1
        return (w, a.1 + delta * b.0 / safe, a.2 + b.2 + delta * delta * a.0 * b.0 / safe)
    }

    /// The advancing refine frame (advancing_block_frames) of one row: per horizon patch, (loc, scale) in float64.
    static func advancingFrames(_ context: [Float], _ observed: [Bool], sinhMedian: [[Double]], patch: Int, minimumScale: Double)
        -> [(loc: Double, scale: Double)]
    {
        var state = (0.0, 0.0, 0.0)
        for start in stride(from: 0, to: context.count, by: patch) {
            var count = 0.0, total = 0.0
            for t in start..<(start + patch) where observed[t] { count += 1; total += Double(context[t]) }
            let mean = count == 0 ? 0 : total / Swift.max(count, 1)
            var m2 = 0.0
            for t in start..<(start + patch) where observed[t] { m2 += (Double(context[t]) - mean) * (Double(context[t]) - mean) }
            state = combine(state, count == 0 ? (0, 0, 0) : (count, mean, m2))
        }
        let offset = state.1
        let clip = Double(Float.greatestFiniteMagnitude).squareRoot() / 1e3
        var frames: [(loc: Double, scale: Double)] = []
        for median in sinhMedian {
            let variance = Swift.max(state.2 / Swift.max(state.0 - 1, 1), 0)
            let scale = Swift.max(variance.squareRoot(), minimumScale)
            frames.append((state.1, scale))
            let folded = median.map { Swift.min(Swift.max($0 * scale + state.1 - offset, -clip), clip) + offset }
            let mean = folded.reduce(0, +) / Double(folded.count)
            let m2 = folded.reduce(0) { $0 + ($1 - mean) * ($1 - mean) }
            state = combine(state, (Double(patch), mean, m2))
        }
        return frames
    }
}

// MARK: - Quantile head (model.py monotone_quantiles), float32

enum Quantiles {
    static func softplus(_ x: Float) -> Float { x > 20 ? x : Foundation.log1p(Foundation.exp(x)) }

    /// Non-crossing quantiles: the median is free, the others fan out by `width * dp * softplus(raw)`.
    static func monotone(_ raw: [Float], knots: [Double], widthFactor: Double) -> [Float] {
        let m = knots.firstIndex(of: 0.5)!
        var widths = [Float](repeating: 0, count: knots.count)
        for i in (m + 1)..<knots.count { widths[i] = Float(knots[i] - knots[i - 1]) }
        for i in 0..<m { widths[i] = Float(knots[i + 1] - knots[i]) }
        let increments = raw.indices.map { Float(widthFactor) * widths[$0] * softplus(raw[$0]) }
        var out = [Float](repeating: raw[m], count: raw.count)
        var up: Float = 0
        for i in (m + 1)..<raw.count { up += increments[i]; out[i] = raw[m] + up }
        var down: Float = 0
        for i in stride(from: m - 1, through: 0, by: -1) { down += increments[i]; out[i] = raw[m] - down }
        return out.map { Swift.min(Swift.max($0, -.greatestFiniteMagnitude), .greatestFiniteMagnitude) }
    }
}

// MARK: - Categorical covariates (covariate_encoding.py)

enum TargetEncoding {
    /// Per-series smoothed mean-target encoding (smooth 1); NaN codes are a category, unseen future codes the mean.
    static func encode(target: [Double], past: [Double], future: [Double]?, smooth: Double = 1) -> (past: [Float], future: [Float]?) {
        let observed = target.map { $0.isFinite }
        let values = target.indices.map { observed[$0] ? target[$0] : 0 }
        let count = Double(observed.filter { $0 }.count)
        let mean = count > 0 ? values.reduce(0, +) / count : 0
        let categories = Array(Set(past.filter { $0.isFinite })).sorted()
        let index = Dictionary(uniqueKeysWithValues: categories.enumerated().map { ($1, $0) })
        let nanSlot = categories.count
        func slot(_ code: Double, unseen: Int?) -> Int {
            if code.isNaN { return nanSlot }
            if let found = index[code] { return found }
            return unseen ?? 0
        }
        var sums = [Double](repeating: 0, count: nanSlot + 1), counts = sums
        for (t, code) in past.enumerated() {
            let s = slot(code, unseen: nil)
            if observed[t] { sums[s] += values[t]; counts[s] += 1 }
        }
        let lookup = (0...nanSlot).map { (smooth * mean + sums[$0]) / (smooth + counts[$0]) } + [mean]
        let encodedPast = past.map { Float(lookup[slot($0, unseen: nil)]) }
        return (encodedPast, future.map { $0.map { Float(lookup[slot($0, unseen: nanSlot + 1)]) } })
    }
}

// MARK: - Forecaster (forecaster.py)

/// One decode block for the network: packed rows, already normalized and patched.
struct BlockInput {
    let rows: Int
    /// Encoder patches: the context (no covariates) or context plus horizon (the covariate window).
    let encPatches: Int
    /// Context patches L; the horizon has `horizonPatches` patches.
    let contextPatches: Int
    let horizonPatches: Int
    let window: Bool
    /// [rows][encPatches][2 * patch]: normalized values then missing flags.
    let patchInput: [Float]
    /// [rows][encPatches] encoder key validity.
    let keyValid: [Bool]
    let targetRows: [Bool]
    let knownRows: [Bool]
    /// First patch any row observes (0 if none); positions count from it.
    let firstObserved: Int
}

/// Raw head outputs [rows][horizonPatches * patch][Q] in float32 for one block.
typealias Network = (BlockInput) throws -> [Float]

enum Role { case target, past, known }

struct Forecaster {
    let config: TimestralConfig
    let maxContext: Int
    let capK: Float = 1e4
    let extremeThreshold: Float = 1e30
    var perShot: Int { Swift.min(32, config.blockSize) }

    init(config: TimestralConfig) {
        self.config = config
        maxContext = config.contextLength
    }

    /// The whole request: [target + co-targets][horizon][all trained levels], float32 values as Double.
    func forecast(_ request: ForecastRequest, network: Network) throws -> [[[Double]]] {
        let targets = [request.values] + request.coTargets
        let inputs = targets.map {
            $0.map { value -> Float in
                let f = Float(value ?? 0); return f.isFinite ? f : 0
            }
        }
        // As the maker: float32 first, then a step is observed only when finite (a value beyond float32 is missing).
        let mask = targets.map { $0.map { $0.map { Float($0).isFinite } ?? false } }
        let context = request.values.count
        // Covariates as the targets: a value beyond float32 becomes infinite there, so it counts as missing (NaN).
        var past = Self.numeric(request.pastCovariates), known = Self.numeric(request.knownCovariates)
        if !(request.categoricalPast.isEmpty && request.categoricalKnown.isEmpty) {
            let target = encodingTarget(inputs, mask)
            past += encode(request.categoricalPast, target: target, context: context)
            known += encode(request.categoricalKnown, target: target, context: context)
        }
        var quantiles = try forecastRaw(inputs, mask, horizon: request.horizon, past: past, known: known, network: network)
        if request.symmetricAveraging {
            let mirrored = try forecastRaw(
                inputs.map { $0.map { -$0 } }, mask, horizon: request.horizon, past: past.map { $0.map { -$0 } }, known: known.map { $0.map { -$0 } },
                network: network)
            quantiles = quantiles.indices.map { r in
                quantiles[r].indices.map { s in
                    let q = quantiles[r][s], m = mirrored[r][s]
                    return q.indices.map { 0.5 * (q[$0] + -m[q.count - 1 - $0]) }
                }
            }
        }
        if request.clipNonnegative {
            for r in quantiles.indices {
                let seen = inputs[r].indices.filter { mask[r][$0] }
                if !seen.isEmpty && seen.allSatisfy({ inputs[r][$0] >= 0 }) { quantiles[r] = quantiles[r].map { $0.map { Swift.max($0, 0) } } }
            }
        }
        return quantiles.map { $0.map { row in row.map(Double.init) } }
    }

    /// Numeric covariate rows in float32; null and any value that is not finite in float32 are NaN (missing).
    static func numeric(_ rows: [[Double?]]) -> [[Float]] {
        rows.map { $0.map { value in value.map { Float($0) }.flatMap { $0.isFinite ? $0 : nil } ?? .nan } }
    }

    /// Forward-filled single target from its first observation, over the last maxContext steps; NaN elsewhere. Nil with co-targets.
    func encodingTarget(_ inputs: [[Float]], _ mask: [[Bool]]) -> [Double]? {
        guard inputs.count == 1 else { return nil }
        let filled = Self.forwardFill(inputs[0], mask[0]), t = inputs[0].count
        var started = false
        return (0..<t).map { i in
            started = started || mask[0][i]
            return started && i >= t - maxContext ? Double(filled[i]) : .nan
        }
    }

    func encode(_ rows: [[Double?]], target: [Double]?, context: Int) -> [[Float]] {
        rows.map { row in
            let codes = row.map { $0 ?? .nan }
            guard let target else { return codes.map { Float($0) } }
            let future = codes.count > context ? Array(codes[context...]) : nil
            let encoded = TargetEncoding.encode(target: target, past: Array(codes[..<context]), future: future)
            var out = (0..<context).map { t in target[t].isNaN && codes[t].isNaN ? Float.nan : encoded.past[t] }
            if let f = encoded.future { out += f }
            return out
        }
    }

    /// Unobserved steps take the last observed value; leading ones the first step's value (0 when missing).
    static func forwardFill(_ x: [Float], _ observed: [Bool]) -> [Float] {
        var index = 0
        return x.indices.map { t in
            if observed[t] { index = t }
            return x[index].isNaN ? 0 : x[index]
        }
    }

    /// One pass (`_forecast_raw`): [targets][horizon][Q] raw-space quantiles, capped.
    func forecastRaw(_ inputs: [[Float]], _ mask: [[Bool]], horizon: Int, past: [[Float]], known: [[Float]], network: Network) throws -> [[[Float]]] {
        let ps = config.patch, targets = inputs.count, context = inputs[0].count
        var rows = inputs.indices.map { Self.forwardFill(inputs[$0], mask[$0]) }
        var masks = mask
        var roles = [Role](repeating: .target, count: targets)
        for row in past {
            masks.append(row.map { !$0.isNaN }); rows.append(row.map { $0.isNaN ? 0 : $0 }); roles.append(.past)
        }
        var knownFuture: [[Float]] = [], knownFutureMask: [[Bool]] = []
        for row in known {
            let ctx = Array(row[..<context]), fut = Array(row[context...])
            masks.append(ctx.map { !$0.isNaN }); rows.append(ctx.map { $0.isNaN ? 0 : $0 }); roles.append(.known)
            knownFuture.append(fut.map { $0.isNaN ? 0 : $0 }); knownFutureMask.append(fut.map { !$0.isNaN })
        }
        if rows[0].count > maxContext {
            rows = rows.map { Array($0.suffix(maxContext)) }; masks = masks.map { Array($0.suffix(maxContext)) }
        }
        let pad = (ps - rows[0].count % ps) % ps
        rows = rows.map { [Float](repeating: 0, count: pad) + $0 }
        masks = masks.map { [Bool](repeating: false, count: pad) + $0 }
        let covariates = !(past.isEmpty && known.isEmpty)
        let horizonPadded = (horizon + ps - 1) / ps * ps
        // Known-future values per packed row: zeros and unobserved except on known rows.
        var kf = [[Float]](repeating: [Float](repeating: 0, count: horizonPadded), count: rows.count)
        var kfm = [[Bool]](repeating: [Bool](repeating: false, count: horizonPadded), count: rows.count)
        var k = 0
        for r in roles.indices where roles[r] == .known {
            for s in 0..<horizon { kf[r][s] = knownFuture[k][s]; kfm[r][s] = knownFutureMask[k][s] }
            k += 1
        }
        let decoded = try decode(rows, masks, horizon: horizon, roles: covariates ? roles : nil, knownFuture: kf, knownFutureMask: kfm, network: network)
        var result = Array(decoded.prefix(targets))
        for r in result.indices {
            let seen = rows[r].indices.filter { masks[r][$0] }
            let high = seen.map { rows[r][$0] }.max() ?? 0, low = seen.map { rows[r][$0] }.min() ?? 0
            let s = Scaling.patchCausal(rows[r], masks[r], patch: ps, minimumScale: config.minimumScale).scale.last! * capK
            result[r] = result[r].map { $0.map { Swift.min(Swift.max($0, low - s), high + s) } }
            if result[r].contains(where: { $0.contains { $0.isNaN } }) { throw RunnerError.failed("forecast produced NaN values; the input is degenerate") }
        }
        return result
    }

    /// Block-by-block decoding (`_block_ar_decode`), feeding medians (targets), real futures (known) or blanks (past).
    func decode(
        _ start: [[Float]], _ startMask: [[Bool]], horizon: Int, roles: [Role]?, knownFuture: [[Float]], knownFutureMask: [[Bool]], network: Network
    ) throws -> [[[Float]]] {
        let ps = config.patch
        var context = start, mask = startMask
        var blocks = [[[Float]]](repeating: [], count: start.count)
        var remaining = (horizon + ps - 1) / ps, offset = 0
        while remaining > 0 {
            let block = Swift.min(perShot, remaining), length = block * ps
            let kf = knownFuture.map { Array($0[offset..<(offset + length)]) }, kfm = knownFutureMask.map { Array($0[offset..<(offset + length)]) }
            var q = try predictBlock(context, mask, horizonPatches: block, roles: roles, knownFuture: kf, knownFutureMask: kfm, network: network)
            q = refine(q, context, mask, roles: roles)
            for r in q.indices { blocks[r] += q[r] }
            remaining -= block
            offset += length
            if remaining <= 0 { break }
            for r in context.indices {
                var feedback = q[r].map { row -> Float in
                    let m = row[config.medianIndex]
                    return m.isFinite && abs(m) < extremeThreshold ? m : 0
                }
                var feedbackMask = [Bool](repeating: true, count: length)
                if let roles {
                    if roles[r] == .known { feedback = kf[r]; feedbackMask = kfm[r] }
                    if roles[r] == .past { feedback = [Float](repeating: 0, count: length); feedbackMask = [Bool](repeating: false, count: length) }
                }
                context[r] += feedback
                mask[r] += feedbackMask
            }
            if context[0].count > maxContext {
                let keep = Swift.max(ps, maxContext / ps * ps)
                context = context.map { Array($0.suffix(keep)) }; mask = mask.map { Array($0.suffix(keep)) }
            }
        }
        return blocks.map { Array($0.prefix(horizon)) }
    }

    /// `predict_block`: normalize, patch, run the network, denormalize with the last context patch's frame.
    func predictBlock(
        _ inputs: [[Float]], _ observed: [[Bool]], horizonPatches h: Int, roles: [Role]?, knownFuture: [[Float]], knownFutureMask: [[Bool]], network: Network
    ) throws -> [[[Float]]] {
        let ps = config.patch, rows = inputs.count, contextPatches = inputs[0].count / ps
        let window = roles != nil && config.covariateEnabled && roles!.contains { $0 != .target }
        let full = window ? inputs.indices.map { inputs[$0] + knownFuture[$0] } : inputs
        let fullObserved = window ? observed.indices.map { observed[$0] + knownFutureMask[$0] } : observed
        let n = full[0].count / ps
        var locs: [[Float]] = [], scales: [[Float]] = [], scaled: [[Float]] = []
        for r in 0..<rows {
            var (loc, scale) = Scaling.patchCausal(full[r], fullObserved[r], patch: ps, minimumScale: config.minimumScale)
            var z = Scaling.affine(full[r], loc: loc, scale: scale, observed: fullObserved[r], asinh: config.useAsinh)
            if window, roles![r] == .known {
                let stats = Scaling.instance(full[r], fullObserved[r], minimumScale: config.minimumScale)
                loc = [Float](repeating: stats.loc, count: loc.count); scale = [Float](repeating: stats.scale, count: scale.count)
                z = Scaling.affine(full[r], loc: loc, scale: scale, observed: fullObserved[r], asinh: false)
                if config.useAsinh && !stats.binary { z = z.indices.map { fullObserved[r][$0] ? Foundation.asinh(z[$0]) : z[$0] } }
            }
            locs.append(loc); scales.append(scale); scaled.append(z)
        }
        var patchInput = [Float](repeating: 0, count: rows * n * 2 * ps)
        var keyValid = [Bool](repeating: false, count: rows * n)
        var firstObserved = Int.max
        for r in 0..<rows {
            var started = false
            for p in 0..<n {
                let base = (r * n + p) * 2 * ps
                var any = false
                for k in 0..<ps {
                    let t = p * ps + k
                    patchInput[base + k] = scaled[r][t]
                    patchInput[base + ps + k] = fullObserved[r][t] ? 0 : 1
                    any = any || fullObserved[r][t]
                }
                started = started || any
                keyValid[r * n + p] = any || (config.attendMissing && started)
                if any { firstObserved = Swift.min(firstObserved, p) }
            }
        }
        let input = BlockInput(
            rows: rows, encPatches: n, contextPatches: window ? n - h : contextPatches, horizonPatches: h, window: window, patchInput: patchInput,
            keyValid: keyValid, targetRows: roles.map { $0.map { $0 == .target } } ?? Array(repeating: true, count: rows),
            knownRows: roles.map { $0.map { $0 == .known } } ?? Array(repeating: false, count: rows),
            firstObserved: firstObserved == Int.max ? 0 : firstObserved)
        let raw = try network(input)
        let q = config.knots.count, ll = input.contextPatches
        return (0..<rows).map { r in
            let anchor = (ll - 1) * ps, loc = locs[r][anchor], scale = scales[r][anchor]
            return (0..<(h * ps)).map { s in
                let base = (r * h * ps + s) * q
                let normalized = Quantiles.monotone(Array(raw[base..<(base + q)]), knots: config.knots, widthFactor: config.widthFactor)
                return Scaling.clampNonfinite(normalized.map { (config.useAsinh ? Foundation.sinh($0) : $0) * scale + loc })
            }
        }
    }

    /// `_refine_block`: horizon patches after the first move to the advancing refine frame (target rows only).
    func refine(_ quantiles: [[[Float]]], _ context: [[Float]], _ mask: [[Bool]], roles: [Role]?) -> [[[Float]]] {
        let ps = config.patch, nPred = quantiles[0].count / ps
        guard nPred > 1 else { return quantiles }
        return quantiles.indices.map { r in
            if let roles, roles[r] != .target { return quantiles[r] }
            let (loc, scale) = Scaling.patchCausal(context[r], mask[r], patch: ps, minimumScale: config.minimumScale)
            let loc0 = loc.last!, scale0 = scale.last!
            let sinhMedian = (0..<nPred).map { j in
                (0..<ps).map { k in (Double(quantiles[r][j * ps + k][config.medianIndex]) - Double(loc0)) / Double(scale0) }
            }
            let frames = Scaling.advancingFrames(context[r], mask[r], sinhMedian: sinhMedian, patch: ps, minimumScale: config.minimumScale)
            return quantiles[r].indices.map { s in
                let j = s / ps
                let lj = j == 0 ? loc0 : Float(frames[j].loc), sj = j == 0 ? scale0 : Float(frames[j].scale)
                return Scaling.clampNonfinite(quantiles[r][s].map { ($0 - loc0) / scale0 * sj + lj })
            }
        }
    }
}
