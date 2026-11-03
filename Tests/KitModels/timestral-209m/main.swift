// SPDX-License-Identifier: MIT
// Compiled with models/timestral-209m/runners/swift/Sources/Runner/Request.swift alone (no MLX, no weights) by
// Timestral209MFolderTests.hostLogicMatchesTheMaker. Maker values come from the maker's own functions on the CPU
// (timestral a534bdbd, captured once). Prints "ok <checks>" or exits 1 on the first failure.
import Foundation

enum RunnerError: Error { case invalid(String), failed(String), cancelled }

var checks = 0
func check(_ condition: Bool, _ what: String) {
    checks += 1
    if !condition { FileHandle.standardError.write(Data("FAIL: \(what)\n".utf8)); exit(1) }
}
func close(_ a: [Float], _ b: [Double], _ tolerance: Double = 1e-5) -> Bool {
    a.count == b.count && zip(a, b).allSatisfy { abs(Double($0) - $1) <= tolerance * Swift.max(1, abs($1)) }
}
func json(_ text: String) -> [String: Any] { (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:] }

let configJSON = json(
    #"{"patch_size":32,"d_model":768,"num_heads":12,"mlp_hidden_dim":2048,"block_schedule":["time","time","time","time_variate","time","time","time","time_variate","time","time","time","time_variate","time","time","time","time_variate","time","time","time","time_variate","time","time","time","time_variate"],"quantile_knots":[0.01,0.05,0.1,0.15,0.2,0.25,0.3,0.35,0.4,0.45,0.5,0.55,0.6,0.65,0.7,0.75,0.8,0.85,0.9,0.95,0.99],"block_size":32,"context_length":32768,"rope_theta":10000.0,"rope_partial_factor":0.5,"norm_eps":1e-06,"attn_bias":false,"mlp_bias":false,"time_attn_gate":"sdpa_sigmoid","variate_attn_gate":"sdpa_sigmoid","minimum_scale":1e-06,"use_asinh":true,"stabilize_with_global":false,"covariate_enabled":true,"split_context_horizon_attn":true,"attend_missing_patches":true,"quantile_width_factor":3.0}"#
)
let config = try TimestralConfig(configJSON)

// 1. Configuration.
check(config.headDim == 64 && config.ropeDim == 32 && config.medianIndex == 10 && config.knots.count == 21, "derived sizes")
check(config.supportsKnownCovariates, "the 209M schedule routes known futures to the targets (maker: True)")
var stabilized = configJSON
stabilized["stabilize_with_global"] = true
check((try? TimestralConfig(stabilized)) == nil, "stabilize_with_global is refused, as the maker refuses it")

// 2. Causal statistics and the asinh transform (scaler.py), maker values.
let x: [Float] = [3, 5, 0, 4, 6, 7, 5, 1e6], observed = [true, true, false, true, true, true, true, true]
let stats = Scaling.causal(x, observed, minimumScale: 1e-6)
check(close(stats.loc, [3, 4, 4, 4, 4.5, 5, 5, 142861.421875]), "causal loc \(stats.loc)")
check(
    close(stats.scale, [9.999999974752427e-07, 1.4142135381698608, 1.4142135381698608, 1, 1.29099440574646, 1.5811388492584229, 1.4142135381698608, 377962.59375]),
    "causal scale \(stats.scale)")
let z = Scaling.affine(x, loc: stats.loc, scale: stats.scale, observed: observed, asinh: true)
check(close(z, [0, 0.6584789156913757, 0, 0, 0.9913485050201416, 1.05687415599823, 0, 1.5573581457138062]), "asinh transform \(z)")

// 3. Monotone quantiles (model.py), maker values for raw = linspace(-2, 2.5, 21).
let raw = (0..<21).map { -2 + 4.5 * Float($0) / 20 }
let mono = Quantiles.monotone(raw, knots: config.knots, widthFactor: config.widthFactor)
check(
    close(
        mono,
        [
            -0.2823566198348999, -0.2671252489089966, -0.2436409294605255, -0.21476948261260986, -0.17941346764564514, -0.13631317019462585,
            -0.08404648303985596, -0.021038174629211426, 0.05442044138908386, 0.14414125680923462, 0.2500000596046448, 0.3937884569168091,
            0.5592663288116455, 0.7481151223182678, 0.9618772268295288, 1.2019391059875488, 1.4695241451263428, 1.7656946182250977,
            2.0913591384887695, 2.447286605834961, 2.756753444671631
        ], 2e-5), "monotone quantiles \(mono)")
check(zip(mono, mono.dropFirst()).allSatisfy { $0 <= $1 }, "quantiles never cross")

// 4. Target encoding (covariate_encoding.py), maker values.
let encoded = TargetEncoding.encode(target: [10, 20, 12, 5, .nan, 30], past: [1, 2, 1, .nan, 2, 3], future: [1, 4, .nan])
check(close(encoded.past, [12.466666221618652, 17.700000762939453, 12.466666221618652, 10.199999809265137, 17.700000762939453, 22.700000762939453]), "encoded past \(encoded.past)")
check(close(encoded.future ?? [], [12.466666221618652, 15.399999618530273, 10.199999809265137]), "encoded future (unseen code → mean) \(encoded.future ?? [])")

// 5. The advancing refine frame (scaler.py advancing_block_frames), maker values, patch 4.
let frames = Scaling.advancingFrames(
    [1, 2, 4, 3, 5, 6, 0, 8], [true, true, true, true, true, true, false, true],
    sinhMedian: [[0.5, 1, 1.5, 2], [2, 2.5, 3, 3.5], [-1, 0, 1, 2]], patch: 4, minimumScale: 1e-6)
check(zip(frames.map(\.loc), [4.142857142857142, 5.238445951068724, 7.11132111392966]).allSatisfy { abs($0 - $1) < 1e-9 }, "frame locs \(frames.map(\.loc))")
check(zip(frames.map(\.scale), [2.410295378065479, 2.553920676628549, 3.946691910405822]).allSatisfy { abs($0 - $1) < 1e-9 }, "frame scales \(frames.map(\.scale))")

// 6. Known-future covariate statistics (instance_norm_stats), maker values.
let instance = Scaling.instance([1, 2, 0, 4], [true, true, false, true], minimumScale: 1e-6)
check(abs(instance.loc - 2.3333332538604736) < 1e-6 && abs(instance.scale - 1.247219204902649) < 1e-6 && !instance.binary, "instance stats")
let binary = Scaling.instance([0, 1, 1, 0], [true, true, true, true], minimumScale: 1e-6)
check(binary.loc == 0 && binary.scale == 1 && binary.binary, "a 0/1 row keeps loc 0, scale 1")
check(Forecaster.forwardFill([0, 2, 0, 0, 5], [false, true, false, false, true]) == [0, 2, 2, 2, 5], "forward fill, leading gap 0")
check(Scaling.clampNonfinite([-.infinity, 1, 2, .infinity]) == [1, 1, 2, 2] && Scaling.clampNonfinite([.infinity]) == [0], "inf clamps to the nearest finite")

// 7. Validation agrees with ops/forecast.input.json.
func refused(_ raw: [String: Any]) -> Bool {
    do { _ = try ForecastRequest(raw, config: config); return false } catch RunnerError.invalid { return true } catch { return false }
}
check(!refused(["values": [1.0, 2.0, 3.0], "horizon": 4]), "a plain request passes")
check(refused(json(#"{"values":[1,2,3],"horizon":2.5}"#)) && refused(json(#"{"values":[1,2,3],"horizon":true}"#)), "horizon must be an integer")
check(!refused(json(#"{"values":[1,2,3],"horizon":4.0}"#)), "4.0 is an integer in JSON Schema")
check(refused(["values": [1.0], "horizon": 4097]) && !refused(["values": [1.0], "horizon": 4096]), "horizon at most 4096")
check(refused(json(#"{"values":[null,null],"horizon":2}"#)), "at least one number")
check(refused(["values": [Double](repeating: 1, count: 32769), "horizon": 1]) && !refused(["values": [Double](repeating: 1, count: 32768), "horizon": 1]), "context at most 32768")
check(refused(json(#"{"values":[1,2,3],"horizon":2,"quantile_levels":[0.33]}"#)), "only trained levels")
check(!refused(json(#"{"values":[1,2,3],"horizon":2,"quantile_levels":[0.9,0.1]}"#)), "levels in any order pass (review fix)")
check((try? ForecastRequest(json(#"{"values":[1,2,3],"horizon":2,"quantile_levels":[0.9,0.1]}"#), config: config))?.levels == [0.1, 0.9], "levels come back ascending")
check(refused(json(#"{"values":[1,2,3],"horizon":2,"quantile_levels":[0.5,0.5]}"#)), "levels unique")
check(!refused(json(#"{"values":[1,2,3],"horizon":2,"quantile_levels":[0.1,0.9]}"#)), "a subset passes")
check(refused(json(#"{"values":[1,2,3],"horizon":2,"covariates":[[1,2,3]]}"#)), "known covariates span len(values) + horizon")
check(!refused(json(#"{"values":[1,2,3],"horizon":2,"covariates":[[1,2,3,4,null]],"past_covariates":[[1,null,3]]}"#)), "covariates with nulls pass")
check(refused(json(#"{"values":[1,2,3],"horizon":2,"symmetric_averaging":1}"#)), "flags are booleans")
check(refused(json(#"{"values":[1,2,3],"horizon":2,"snap_integers":true}"#)), "unknown fields are refused")
check(
    refused(["values": [1.0, 2.0], "horizon": 1, "co_targets": Array(repeating: [1.0, 2.0], count: 16), "past_covariates": Array(repeating: [1.0, 2.0], count: 16)]),
    "at most 32 series")

// 8. The forecaster around a fake network: counts, block layout and the frame arithmetic.
var calls: [BlockInput] = []
let flat: Network = { input in
    calls.append(input)
    return [Float](repeating: 0, count: input.rows * input.horizonPatches * 32 * 21)
}
let forecaster = Forecaster(config: config)
let constant = try ForecastRequest(["values": [Double](repeating: 5, count: 70), "horizon": 70], config: config)
let out = try forecaster.forecast(constant, network: flat)
check(out.count == 1 && out[0].count == 70 && out[0].allSatisfy { $0.count == 21 }, "counts: 1 series, 70 steps, 21 levels")
check(out[0].allSatisfy { row in row.allSatisfy { abs($0 - 5) < 1e-4 } }, "a constant series with a flat head forecasts the constant")
check(calls.count == 2 && calls.allSatisfy { !$0.window && $0.rows == 1 && $0.encPatches == 3 && $0.horizonPatches == 3 }, "two passes (clampsym), 3 context and 3 horizon patches")
check(calls[0].keyValid == [true, true, true] && calls[0].firstObserved == 0, "the left-padded first patch is observed")
calls = []
let long = try ForecastRequest(["values": (0..<64).map { Double($0) }, "horizon": 1100, "symmetric_averaging": false], config: config)
let longOut = try forecaster.forecast(long, network: flat)
check(
    longOut[0].count == 1100 && calls.map(\.horizonPatches) == [32, 3] && calls.map(\.encPatches) == [2, 34],
    "1100 steps decode as 32 + 3 patches, the context growing by one block")
calls = []
let covariates = try ForecastRequest(
    json(
        #"{"values":[1,2,3,4,5],"horizon":40,"covariates":[[0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0,1,0]],"past_covariates":[[1,null,3,4,5]],"symmetric_averaging":false}"#
    ),
    config: config)
_ = try forecaster.forecast(covariates, network: flat)
check(calls.count == 1 && calls[0].window && calls[0].rows == 3 && calls[0].encPatches == 3 && calls[0].contextPatches == 1, "the covariate window: 1 context + 2 horizon patches")
check(calls[0].targetRows == [true, false, false] && calls[0].knownRows == [false, false, true], "roles: target, past, known")
check(Array(calls[0].keyValid[6..<9]) == [true, true, true] && Array(calls[0].keyValid[0..<3]) == [true, true, true], "known-future patches are valid keys")
let multi = try forecaster.forecast(try ForecastRequest(json(#"{"values":[1,2,3],"co_targets":[[4,5,6]],"horizon":3}"#), config: config), network: flat)
check(multi.count == 2 && multi.allSatisfy { $0.count == 3 }, "co-targets come back as their own forecasts")

// Review fix: a value beyond float32 is missing, as the maker's isfinite mask after its float32 cast.
calls = []
_ = try forecaster.forecast(try ForecastRequest(["values": [1.0, 2.0, 1e39, 3.0], "horizon": 2, "symmetric_averaging": false], config: config), network: flat)
check(calls.count == 1 && calls[0].patchInput[28 + 32 + 2] == 1 && calls[0].patchInput[28 + 2].isFinite, "1e39 is a missing step, not infinity")

// 9. Release review fixes (2 Nov): a reply past the frame cap is refused before the model runs; covariates beyond
// float32 are missing, as targets are; the known-future statistics cannot overflow float32 on the way.
let longValues = [Double](repeating: 1, count: 32)
check(!refused(["values": longValues, "horizon": 4096, "co_targets": Array(repeating: longValues, count: 6)]), "7 series x 4096 x 22 pass (630,784 numbers)")
check(
    refused(["values": longValues, "horizon": 4096, "co_targets": Array(repeating: longValues, count: 7)]),
    "8 series x 4096 x 22 are refused (720,896 numbers, about 17 MB as JSON on small negative data)")
check(refused(["values": longValues, "horizon": 4096, "co_targets": Array(repeating: longValues, count: 8)]), "9 series x 4096 x 22 are refused")
check(refused(["values": longValues, "horizon": 4096, "co_targets": Array(repeating: longValues, count: 9)]), "10 series (16.8 MB measured) are refused")
check(
    !refused(["values": longValues, "horizon": 4096, "co_targets": Array(repeating: longValues, count: 31), "quantile_levels": [0.5]]),
    "32 series at one level pass (262,144 numbers)")
let covariateRows = Forecaster.numeric([[1, 1e39, nil, -1e39, 3e38]])
check(
    covariateRows[0][0] == 1 && covariateRows[0][1].isNaN && covariateRows[0][2].isNaN && covariateRows[0][3].isNaN && covariateRows[0][4] == 3e38,
    "covariates beyond float32 are missing (NaN), not infinite: \(covariateRows)")
let big = Scaling.instance([3e38, 3e38, -3e38, 1], [true, true, true, false], minimumScale: 1e-6)
check(big.loc.isFinite && big.scale.isFinite && abs(Double(big.loc) - 1e38) < 1e32, "large known-future values keep finite stats: \(big)")
calls = []
_ = try forecaster.forecast(
    try ForecastRequest(json(#"{"values":[1,2,3,4],"horizon":2,"covariates":[[1,2,1e39,4,5,6]],"symmetric_averaging":false}"#), config: config), network: flat)
check(calls.allSatisfy { $0.patchInput.allSatisfy(\.isFinite) }, "a covariate of 1e39 reaches the network as a missing step, not infinity")

print("ok \(checks)")
