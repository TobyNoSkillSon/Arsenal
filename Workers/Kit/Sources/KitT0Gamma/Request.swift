// SPDX-License-Identifier: MIT
// t0-gamma's host-side request handling, kept free of MLX so it compiles and is checked on its own
// (Tests/KitModels/t0-gamma/test_request.py): config, request validation against ops/forecast.input.json, the
// quantile-level maths of t0/levels.py and t0/quantile.py, and the adjustments of t0/forecaster.py.
import Foundation

/// Hyperparameters from config.json. Defaults are the maker's T0Config defaults (those reproduce t0-alpha), so a
/// key the file leaves out keeps the maker's meaning.
struct T0Config {
    var embedDim = 512
    var numLayers = 24
    var numHeads = 8
    var mlpHiddenDim = 2048
    var patchSize = 32
    var groupEveryN = 3
    var quantileLevels: [Double] = [0.1, 0.25, 0.5, 0.75, 0.9]
    var scalerUseArcsinh = true
    var scalerEps = 0.1
    /// "variance_offset": sqrt(variance + eps); "std_clamp": max(sqrt(variance), eps).
    var stdClamp = false
    /// "interleaved" (false) or "halves" (true): which rotary coordinates share an xPos scale.
    var xposHalves = false
    /// Whether cells without an observation enter the model as zero after scaling.
    var zeroNoValue = true
    /// DEFAULT_MAX_HORIZON in t0/forecaster.py: steps decoded in one forward pass.
    let maxHorizon = 1024

    init(_ json: [String: Any]) throws {
        embedDim = (json["embed_dim"] as? NSNumber)?.intValue ?? embedDim
        numLayers = (json["num_layers"] as? NSNumber)?.intValue ?? numLayers
        numHeads = (json["num_heads"] as? NSNumber)?.intValue ?? numHeads
        mlpHiddenDim = (json["mlp_hidden_dim"] as? NSNumber)?.intValue ?? mlpHiddenDim
        patchSize = (json["patch_size"] as? NSNumber)?.intValue ?? patchSize
        groupEveryN = (json["group_every_n"] as? NSNumber)?.intValue ?? groupEveryN
        if let levels = json["quantile_levels"] as? [NSNumber] { quantileLevels = levels.map { $0.doubleValue } }
        scalerUseArcsinh = json["scaler_use_arcsinh"] as? Bool ?? scalerUseArcsinh
        scalerEps = (json["scaler_eps"] as? NSNumber)?.doubleValue ?? scalerEps
        if let mode = json["scaler_eps_mode"] as? String {
            guard ["variance_offset", "std_clamp"].contains(mode) else { throw RunnerError.failed("unknown scaler_eps_mode \(mode)") }
            stdClamp = mode == "std_clamp"
        }
        if let layout = json["xpos_scale_layout"] as? String {
            guard ["interleaved", "halves"].contains(layout) else { throw RunnerError.failed("unknown xpos_scale_layout \(layout)") }
            xposHalves = layout == "halves"
        }
        zeroNoValue = json["scaler_zero_no_value"] as? Bool ?? zeroNoValue
        guard quantileLevels == quantileLevels.sorted(), Set(quantileLevels).count == quantileLevels.count, quantileLevels.count >= 2 else {
            throw RunnerError.failed("config.json quantile_levels must be ascending and unique")
        }
    }
}

// MARK: - Quantile levels (t0/levels.py, t0/quantile.py)

enum Levels {
    /// The request's levels, ascending, unique, in (0, 1) (validate_quantile_levels).
    static func validate(_ levels: [Double]) throws -> [Double] {
        guard !levels.isEmpty, levels.count <= maxRequested else { throw RunnerError.invalid("quantile_levels holds 1 to \(maxRequested) levels.") }
        guard levels.allSatisfy({ $0 > 0 && $0 < 1 }) else { throw RunnerError.invalid("each quantile level must be between 0 and 1, exclusive.") }
        // The tails read each level in float32: one that rounds to 0 or 1 there (1e-50, 0.99999999) has an infinite tail.
        guard levels.allSatisfy({ Float($0) > 0 && Float($0) < 1 }) else {
            throw RunnerError.invalid("each quantile level must stay inside (0, 1) in float32: from about 1e-45 to 0.9999999.")
        }
        guard levels == Array(Set(levels)).sorted() else { throw RunnerError.invalid("quantile_levels must be ascending without duplicates.") }
        return levels
    }

    static let maxRequested = 99

    private static func round6(_ x: Double) -> Double { (x * 1e6).rounded() / 1e6 }

    /// The trained levels the exponential tails pin through, or nil when no requested level lies outside them.
    static func tailKnots(trained: [Double], requested: [Double]) -> [Double]? {
        let ordered = trained.map(round6).sorted()
        let lo = ordered.first!, hi = ordered.last!
        guard requested.contains(where: { ($0 < lo || $0 > hi) && $0 > 0 && $0 < 1 }) else { return nil }
        return Array(Set([ordered[0], ordered[1], ordered[ordered.count - 2], ordered[ordered.count - 1]])).sorted()
    }

    /// The levels a pass must produce to serve a request (get_rollout_quantile_levels).
    static func rolloutLevels(trained: [Double], requested: [Double]) -> [Double] {
        guard let knots = tailKnots(trained: trained, requested: requested) else { return requested }
        let interior = requested.filter { $0 >= knots.first! && $0 <= knots.last! }
        return Array(Set(interior).union(knots)).sorted()
    }

    /// Linear interpolation of query levels among source columns, holding the edges flat (interpolation_plan,
    /// in float32 as the maker's plan is).
    static func interpolate(_ values: [Double], source: [Double], query: [Double]) -> [Double] {
        let levels = source.map { Float($0) }
        let padLeft = levels.first! > 0, padRight = levels.last! < 1
        let padded = (padLeft ? [Float(0)] : []) + levels + (padRight ? [Float(1)] : [])
        let shift = padLeft ? 1 : 0
        return query.map { level in
            let q = Float(level)
            let up = Swift.min(padded.firstIndex(where: { $0 > q }) ?? padded.count, padded.count - 1)
            let lo = up - 1
            let span = padded[up] - padded[lo]
            let lower = Swift.min(Swift.max(lo - shift, 0), levels.count - 1)
            let upper = Swift.min(Swift.max(up - shift, 0), levels.count - 1)
            let weight = span == 0 ? 0 : Double(Float((q - padded[lo]) / span))
            return values[lower] + weight * (values[upper] - values[lower])
        }
    }

    /// Quantiles at the query levels from a pass at `original` levels, with exponential tails beyond the trained
    /// range (extrapolate_quantiles and extend_quantile_tails).
    static func extrapolate(_ values: [Double], original: [Double], query: [Double], trained: [Double]) -> [Double] {
        guard let knots = tailKnots(trained: trained, requested: query) else { return values }
        let k = knots.map { Double(Float($0)) }
        let v = knots.map { knot in values[original.firstIndex(of: knot)!] }
        let leftSlope = Swift.max((v[1] - v[0]) / Foundation.log(k[1] / k[0]), 0)
        let n = k.count - 1
        let rightSlope = Swift.max((v[n] - v[n - 1]) / Foundation.log((1 - k[n - 1]) / (1 - k[n])), 0)
        return query.map { level in
            let q = Double(Float(level))
            if level < knots.first! { return v[0] + leftSlope * Foundation.log(q / k[0]) }
            if level > knots.last! { return v[n] + rightSlope * Foundation.log((1 - k[n]) / (1 - q)) }
            return values[original.firstIndex(of: level)!]
        }
    }
}

/// Numerically stable softplus, as F.softplus in the maker's quantile head.
func softplus(_ x: Double) -> Double { x > 0 ? x + log1p(exp(-x)) : log1p(exp(x)) }

// MARK: - Request

struct ForecastRequest {
    var targets: [[Double?]]
    var futures: [[Double?]]
    var horizon: Int
    var levels: [Double]
    var symmetricAveraging: Bool
    var clipNonnegative: Bool
    var snapIntegers: Bool

    static let maxContext = 8192
    /// ops/forecast.input.json: covariates and past_covariates each hold at most 31 series.
    static let maxCovariates = 31
    static let maxRows = 1 + 2 * maxCovariates
    /// Numbers in the reply's forecasts (series x horizon x (levels + median)). The longest float as JSON is 24 bytes
    /// with its comma (-1.2345678442216013e-05, measured), plus up to 2 of brackets: 640,000 stay under the worker's
    /// 16 MiB - 64 KiB reply limit in the worst case.
    static let maxReplyNumbers = 640_000

    static func isBool(_ number: NSNumber) -> Bool { CFGetTypeID(number) == CFBooleanGetTypeID() }

    /// A JSON boolean only; a number such as 1 is refused, as the schema's `"type": "boolean"` does.
    static func flag(_ raw: [String: Any], _ key: String) throws -> Bool {
        guard let value = raw[key] else { return false }
        guard let number = value as? NSNumber, isBool(number) else { throw RunnerError.invalid("\(key) must be true or false.") }
        return number.boolValue
    }

    static func series(_ raw: Any?, name: String) throws -> [Double?] {
        guard let items = raw as? [Any] else { throw RunnerError.invalid("\(name) must be an array of numbers (null for a missing value).") }
        return try items.map { item in
            if item is NSNull { return nil }
            guard let number = item as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), number.doubleValue.isFinite else {
                throw RunnerError.invalid("\(name) must hold finite numbers or null.")
            }
            // The network reads float32: a larger magnitude would overflow the scaling and come back as a forecast of 0.
            guard abs(number.doubleValue) <= Double(Float.greatestFiniteMagnitude) else {
                throw RunnerError.invalid("\(name) holds \(number.doubleValue), beyond the float32 range (about 3.4e38) the model reads.")
            }
            return number.doubleValue
        }
    }

    init(_ raw: [String: Any], config: T0Config) throws {
        let values = try Self.series(raw["values"], name: "values")
        guard !values.isEmpty, values.count <= Self.maxContext else { throw RunnerError.invalid("values holds 1 to \(Self.maxContext) numbers.") }
        guard values.contains(where: { $0 != nil }) else { throw RunnerError.invalid("values needs at least one number.") }
        guard let number = raw["horizon"] as? NSNumber, !Self.isBool(number), number.doubleValue == number.doubleValue.rounded(.towardZero),
            abs(number.doubleValue) <= 1e9
        else { throw RunnerError.invalid("horizon must be a whole number of steps.") }
        horizon = number.intValue
        guard horizon >= 1, horizon <= config.maxHorizon else { throw RunnerError.invalid("horizon must be from 1 to \(config.maxHorizon) steps.") }
        var targets = [values]
        if let past = raw["past_covariates"] {
            guard let rows = past as? [Any], rows.count <= Self.maxCovariates else {
                throw RunnerError.invalid("past_covariates must be an array of at most \(Self.maxCovariates) series.")
            }
            for (index, row) in rows.enumerated() {
                let series = try Self.series(row, name: "past_covariates[\(index)]")
                guard series.count == values.count else { throw RunnerError.invalid("past_covariates[\(index)] must have as many steps as values (\(values.count)).") }
                targets.append(series)
            }
        }
        var futures: [[Double?]] = []
        if let known = raw["covariates"] {
            guard let rows = known as? [Any], rows.count <= Self.maxCovariates else {
                throw RunnerError.invalid("covariates must be an array of at most \(Self.maxCovariates) series.")
            }
            for (index, row) in rows.enumerated() {
                let series = try Self.series(row, name: "covariates[\(index)]")
                guard series.count == values.count + horizon else {
                    throw RunnerError.invalid("covariates[\(index)] must span the past and the horizon: \(values.count + horizon) steps.")
                }
                futures.append(series)
            }
        }
        guard targets.count + futures.count <= Self.maxRows else {
            throw RunnerError.invalid("past_covariates and covariates together hold at most \(Self.maxRows - 1) series.")
        }
        self.targets = targets
        self.futures = futures
        if let requested = raw["quantile_levels"] {
            guard let list = requested as? [NSNumber], !list.contains(where: Self.isBool) else { throw RunnerError.invalid("quantile_levels must be an array of numbers.") }
            levels = try Levels.validate(list.map { $0.doubleValue })
        } else {
            levels = config.quantileLevels
        }
        let numbers = targets.count * horizon * (levels.count + 1)
        guard numbers <= Self.maxReplyNumbers else {
            throw RunnerError.invalid(
                "The forecast would return \(numbers) numbers (\(targets.count) series x \(horizon) steps x \(levels.count + 1) levels with the median), "
                    + "over the reply limit of \(Self.maxReplyNumbers) (16 MiB as JSON): ask for fewer past_covariates or quantile_levels, or a shorter horizon.")
        }
        symmetricAveraging = try Self.flag(raw, "symmetric_averaging")
        // Symmetric averaging also forecasts each level's mirror 1 - q, and a mirror that rounds to 0 or 1 in float32
        // (that of 1e-8 is 1) has an infinite tail too.
        if symmetricAveraging, !levels.allSatisfy({ Float(1 - $0) > 0 && Float(1 - $0) < 1 }) {
            throw RunnerError.invalid(
                "with symmetric_averaging, each quantile level's mirror 1 - q must also stay inside (0, 1) in float32: use levels from about 3e-8 to 0.9999999.")
        }
        clipNonnegative = try Self.flag(raw, "clip_nonnegative")
        snapIntegers = try Self.flag(raw, "snap_integers")
    }
}

enum Adjustments {
    /// clip_nonnegative and snap_integers as the maker applies them: the history tests read the series as the float32
    /// tensor it becomes (so 0.1 + 0.2 counts as whole only if float32 says so), and snapping rounds half to even, as
    /// np.round / torch.round do (2.5 becomes 2).
    static func adjust(_ steps: [[Double]], history: [Double], clip: Bool, snap: Bool) -> [[Double]] {
        let stored = history.map { Double(Float($0)) }
        var out = steps
        if clip && stored.min()! >= 0 { out = out.map { $0.map { Swift.max($0, 0) } } }
        if snap && stored.allSatisfy({ $0 == $0.rounded(.toNearestOrEven) }) { out = out.map { $0.map { $0.rounded(.toNearestOrEven) } } }
        return out
    }

    /// `levels` with 0.5 added when missing, ascending.
    static func withMedian(_ levels: [Double]) -> [Double] { levels.contains(0.5) ? levels : (levels + [0.5]).sorted() }

    /// `levels` plus each level's mirror 1 - q, ascending (_with_mirrors).
    static func withMirrors(_ levels: [Double]) -> [Double] {
        var closed = levels
        for level in levels where !closed.contains(where: { abs((1 - level) - $0) <= 1e-9 }) { closed.append(1 - level) }
        return closed.sorted()
    }
}
