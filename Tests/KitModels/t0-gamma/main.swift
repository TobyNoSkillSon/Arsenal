// SPDX-License-Identifier: MIT
// Compiled with models/t0-gamma/runners/swift/Sources/Runner/Request.swift alone (no MLX, no weights) by
// T0GammaFolderTests.requestLogicFollowsTheMakerAndTheSchema. Prints "ok <checks>" or exits 1 on the first failure.
import Foundation

enum RunnerError: Error { case invalid(String), failed(String), cancelled }

var checks = 0
func check(_ condition: Bool, _ what: String) {
    checks += 1
    if !condition { FileHandle.standardError.write(Data("FAIL: \(what)\n".utf8)); exit(1) }
}
func refused(_ raw: [String: Any], _ config: T0Config) -> Bool {
    do { _ = try ForecastRequest(raw, config: config); return false } catch RunnerError.invalid { return true } catch { return false }
}
func json(_ text: String) -> [String: Any] { (try? JSONSerialization.jsonObject(with: Data(text.utf8))) as? [String: Any] ?? [:] }

let config = try T0Config(["quantile_levels": [0.1, 0.25, 0.5, 0.75, 0.9]])

// 1. Snapping rounds half to even, as np.round / torch.round.
let snapped = Adjustments.adjust([[0.5, 1.5, 2.5, -2.5, 2.6]], history: [1, 2, 3], clip: false, snap: true)
check(snapped == [[0, 2, 2, -2, 3]], "snap rounds half to even: \(snapped)")

// 2. History tests read the float32 tensor the maker builds: 3.0000001 is 3 in float32, so it counts as whole.
check(Adjustments.adjust([[1.4]], history: [3.0000001], clip: false, snap: true) == [[1]], "float32 history counts 3.0000001 as whole")
check(Adjustments.adjust([[1.4]], history: [3.001], clip: false, snap: true) == [[1.4]], "3.001 is not whole in float32")
check(Adjustments.adjust([[-1]], history: [-1e-46], clip: true, snap: false) == [[0]], "-1e-46 is -0 in float32, so the history is non-negative")
check(Adjustments.adjust([[-1]], history: [-1e-3], clip: true, snap: false) == [[-1]], "a negative history is not clipped")

// 3. The median always comes from the 0.5 level of the same pass.
check(Adjustments.withMedian([0.1, 0.9]) == [0.1, 0.5, 0.9], "0.5 joins the pass levels")
check(Adjustments.withMedian([0.25, 0.5]) == [0.25, 0.5], "0.5 is not duplicated")
check(Adjustments.withMedian([0.6, 0.7]) == [0.5, 0.6, 0.7], "0.5 sorts into place")

// 4. Validation agrees with ops/forecast.input.json.
let base: [String: Any] = ["values": [1.0, 2.0, 3.0], "horizon": 4]
check(!refused(base, config), "a plain request passes")
check(refused(json(#"{"values":[1,2,3],"horizon":2.5}"#), config), "a fractional horizon is refused (schema: integer)")
check(!refused(json(#"{"values":[1,2,3],"horizon":4.0}"#), config), "4.0 is an integer in JSON Schema")
check(refused(json(#"{"values":[1,2,3],"horizon":true}"#), config), "a boolean horizon is refused")
check(refused(json(#"{"values":[1,2,3],"horizon":"4"}"#), config), "a string horizon is refused")
check(refused(json(#"{"values":[1,2,3],"horizon":4,"quantile_levels":[true]}"#), config), "a boolean level is refused")
check(refused(json(#"{"values":[1,2,3],"horizon":4,"snap_integers":1}"#), config), "a number flag is refused (schema: boolean)")
check(!refused(json(#"{"values":[1,2,3],"horizon":4,"snap_integers":true}"#), config), "a boolean flag passes")
let levels99 = (1...99).map { Double($0) / 100 }
check(!refused(base.merging(["quantile_levels": levels99]) { $1 }, config), "99 levels pass")
check(refused(base.merging(["quantile_levels": (1...100).map { Double($0) / 101 }]) { $1 }, config), "100 levels are refused (schema maxItems 99)")
let past = Array(repeating: [1.0, 2.0, 3.0], count: 31), future = Array(repeating: Array(repeating: 1.0, count: 7), count: 31)
check(!refused(base.merging(["past_covariates": past, "covariates": future]) { $1 }, config), "31 past and 31 known covariates pass (schema maxima)")
check(refused(base.merging(["past_covariates": past + [[1.0, 2.0, 3.0]]]) { $1 }, config), "32 past covariates are refused (schema maxItems 31)")
check(refused(base.merging(["covariates": future + [future[0]]]) { $1 }, config), "32 known covariates are refused (schema maxItems 31)")

// 5. Release review fixes (2 Nov): levels that round to 0 or 1 in float32 are refused (their tails are infinite),
// magnitudes past float32 are refused rather than forecast as 0, and a reply past the frame cap is refused up front.
check(refused(base.merging(["quantile_levels": [0.99999999]]) { $1 }, config), "0.99999999 is 1 in float32 and is refused")
check(refused(base.merging(["quantile_levels": [1e-50]]) { $1 }, config), "1e-50 is 0 in float32 and is refused")
check(!refused(base.merging(["quantile_levels": [1e-7, 0.9999999]]) { $1 }, config), "1e-7 and 0.9999999 pass")
let trained5 = [0.1, 0.25, 0.5, 0.75, 0.9], extreme = [1e-7, 0.9999999]
let rollout5 = Levels.rolloutLevels(trained: trained5, requested: extreme)
let tails = Levels.extrapolate(Array(repeating: 1, count: rollout5.count), original: rollout5, query: extreme, trained: trained5)
check(tails.allSatisfy(\.isFinite), "the tails of a flat series at the extreme accepted levels are finite: \(tails)")
check(refused(json(#"{"values":[1e200,2e200,3e200],"horizon":4}"#), config), "values beyond float32 are refused, not forecast as 0")
check(refused(json(#"{"values":[1,2,3],"horizon":4,"covariates":[[1,2,3,1e39,1,1,1]]}"#), config), "a covariate beyond float32 is refused")
check(!refused(json(#"{"values":[3e38,-3e38,1],"horizon":4}"#), config), "values inside float32 pass")
let wide = (1...99).map { Double($0) / 100 }
let eight = Array(repeating: Array(repeating: 1.0, count: 32), count: 7)
let longBase: [String: Any] = ["values": Array(repeating: 1.0, count: 32), "horizon": 1024, "quantile_levels": wide]
check(!refused(longBase.merging(["past_covariates": Array(eight.prefix(5))]) { $1 }, config), "6 series x 1024 steps x 100 levels pass (614,400 numbers)")
check(refused(longBase.merging(["past_covariates": Array(eight.prefix(6))]) { $1 }, config), "7 series x 1024 x 100 are refused (716,800 numbers, up to 18 MB as JSON)")
check(refused(longBase.merging(["past_covariates": eight]) { $1 }, config), "8 series x 1024 x 100 are refused (819,200 numbers)")
check(refused(longBase.merging(["past_covariates": eight + [eight[0]]]) { $1 }, config), "9 series (17.9 MB measured) are refused")

// 6. Release review E (2 Nov): with symmetric averaging the mirror 1 - q is forecast too, so a level whose mirror
// rounds to 1 in float32 (1e-8) is refused before the model runs, and the tails of the extremes that pass are finite.
let symmetric = base.merging(["symmetric_averaging": true]) { $1 }
check(refused(symmetric.merging(["quantile_levels": [1e-8, 0.5]]) { $1 }, config), "1e-8 is refused under symmetric averaging (its mirror is 1 in float32)")
check(!refused(base.merging(["quantile_levels": [1e-8, 0.5]]) { $1 }, config), "1e-8 still passes without symmetric averaging")
check(!refused(symmetric.merging(["quantile_levels": [1e-7, 0.9999999]]) { $1 }, config), "1e-7 and 0.9999999 pass under symmetric averaging")
let mirrored = Adjustments.withMirrors([1e-7, 0.5])
let mirroredRollout = Levels.rolloutLevels(trained: trained5, requested: mirrored)
let mirroredTails = Levels.extrapolate(
    mirroredRollout.indices.map(Double.init), original: mirroredRollout, query: mirrored, trained: trained5)
check(mirroredTails.allSatisfy(\.isFinite), "the mirrored tails at 1e-7 are finite: \(mirroredTails)")

print("ok \(checks)")
