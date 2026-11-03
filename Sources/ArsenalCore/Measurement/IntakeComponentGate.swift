// SPDX-License-Identifier: MIT
import Foundation

/// Host-side statistics only. This reader never executes a model or changes captured arrays.
struct IntakeTensor {
    let shape: [Int]
    let values: [Double]
    let dtype: String
    static func read(_ url: URL) throws -> [String: IntakeTensor] {
        let data = try Data(contentsOf: url, options: .mappedIfSafe)
        guard data.count >= 8 else { throw ManifestError.invalid("Truncated component archive") }
        let size = data.withUnsafeBytes { $0.loadUnaligned(as: UInt64.self).littleEndian }
        guard size <= 64 << 20, size <= data.count - 8,
            let header = try JSONSerialization.jsonObject(with: data.subdata(in: 8..<(8 + Int(size)))) as? [String: Any]
        else { throw ManifestError.invalid("Malformed component archive") }
        var result: [String: IntakeTensor] = [:]
        for (key, raw) in header where key != "__metadata__" {
            guard let entry = raw as? [String: Any], let shape = entry["shape"] as? [Int], shape.allSatisfy({ $0 > 0 }), let type = entry["dtype"] as? String,
                let range = entry["data_offsets"] as? [Int], range.count == 2, range[0] >= 0, range[1] >= range[0],
                range[1] <= data.count - 8 - Int(size), ["BF16", "F16", "F32"].contains(type)
            else { throw ManifestError.invalid("Invalid component tensor: \(key)") }
            let width = type == "F32" ? 4 : 2
            guard shape.reduce(1, *) * width == range[1] - range[0] else { throw ManifestError.invalid("Tensor count mismatch") }
            let start = 8 + Int(size) + range[0]
            let values: [Double] = data.withUnsafeBytes { bytes in
                stride(from: start, to: start + range[1] - range[0], by: width).map { offset in
                    if type == "F32" { return Double(Float(bitPattern: bytes.loadUnaligned(fromByteOffset: offset, as: UInt32.self).littleEndian)) }
                    let bits = bytes.loadUnaligned(fromByteOffset: offset, as: UInt16.self).littleEndian
                    return type == "F16" ? Double(Float16(bitPattern: bits)) : Double(Float(bitPattern: UInt32(bits) << 16))
                }
            }
            result[key] = IntakeTensor(shape: shape, values: values, dtype: type)
        }
        return result
    }
}

struct IntakeComparison: Codable {
    var relativeL2: Double
    var cosine: Double
    var finite: Bool
    var maxAbs: Double
    static func compare(reference: [Double], candidate: [Double]) throws -> Self {
        guard !reference.isEmpty, reference.count == candidate.count, reference.allSatisfy(\.isFinite) else {
            throw ManifestError.invalid("Component count or reference finiteness mismatch")
        }
        guard candidate.allSatisfy(\.isFinite) else { return .init(relativeL2: 1e300, cosine: -1, finite: false, maxAbs: 1e300) }
        var r2 = 0.0, c2 = 0.0, d2 = 0.0, dot = 0.0, maximum = 0.0
        for (r, c) in zip(reference, candidate) { r2 += r * r; c2 += c * c; d2 += (r - c) * (r - c); dot += r * c; maximum = max(maximum, abs(c)) }
        guard r2 > 0 else { throw ManifestError.invalid("Zero reference cannot calibrate relative component gate") }
        return .init(relativeL2: sqrt(d2 / r2), cosine: c2 > 0 ? min(1, max(-1, dot / sqrt(r2 * c2))) : 0, finite: true, maxAbs: maximum)
    }
}

/// Named, pre-frozen component policies; source rounding and reproducibility are distinct evidence.
/// Bounds are immutable before any candidate is evaluated. Internal precision parity is distinct from vendor qualification.
enum IntakeComponentGate {
    /// Historical receipts predate additive hashes and scale diagnostics; their original
    /// numerical results, bounds, component keys and verdicts must remain identical.
    static func validateSavedEvaluation(_ saved: JSONObject, current: JSONObject) throws {
        guard current["diagnostic_only"] != true else { throw ManifestError.invalid("Diagnostic-only capture cannot replay format qualification") }
        var replay = current
        for key in ["reference_outputs_sha256", "candidate_outputs_sha256", "fixture_inputs_sha256", "diagnostic_only"] where saved[key] == nil {
            replay.removeValue(forKey: key)
        }
        guard let oldComponents = saved["components"]?.object, !oldComponents.isEmpty,
            var components = current["components"]?.object, Set(oldComponents.keys) == Set(components.keys)
        else { throw ManifestError.invalid("Saved component set changed") }
        for (key, old) in oldComponents {
            guard let old = old.object, var component = components[key]?.object else { throw ManifestError.invalid("Invalid saved component") }
            for field in ["reference_rms", "candidate_rms", "reference_max_abs"] where old[field] == nil { component.removeValue(forKey: field) }
            components[key] = .object(component)
        }
        replay["components"] = .object(components)
        guard replay == saved else { throw ManifestError.invalid("Saved format qualification changed") }
    }

    static let fp32SourcePolicyID = "fp32-source-bf16-nearest-truncation-fp16-ulp-v1"
    static let fp32SourceBasis =
        "FP32-source qualification: GPU BF16 nearest versus BF16 truncation source-weight rounding floor; "
        + "relative L2 <= 2 * floor + one FP16 ULP, matching cosine; separate exact BF16 A/A reproducibility; "
        + "finite, no new clamps, candidate FP16 range 65504. Frozen before FP16 execution."

    static func freezeFP32Source(reference: URL, roundingCapture: URL, repeatCapture: URL, to file: URL, expectedKeys: Set<String>) throws {
        try freeze(reference: reference, repeatCapture: repeatCapture, to: file, half: "bf16", expectedKeys: expectedKeys, sourceRoundingCapture: roundingCapture)
    }

    static let quantizedRelativeL2Max = 0.05
    static let quantizedCosineMin = 0.999
    static let quantizedBasis =
        "Quantized-tier quality policy: relative L2 <= 0.05, cosine >= 0.999, all finite, no new clamp events; "
        + "activations within the selected 16-bit format range. A 16-bit rounding floor rejects intentional quantization error by construction."
    static func freeze(reference: URL, repeatCapture: URL, to file: URL, half: String, quantized: Bool = false, expectedKeys: Set<String>? = nil, sourceRoundingCapture: URL? = nil)
        throws
    {
        guard ["fp16", "bf16"].contains(half) else { throw ManifestError.invalid("Expected selected 16-bit format") }
        let sourceFormat = sourceRoundingCapture != nil
        guard !sourceFormat || (half == "bf16" && !quantized) else { throw ManifestError.invalid("FP32-source policy needs BF16 calibration and an FP16 candidate") }
        let candidateHalf = sourceFormat ? "fp16" : half
        let formatMax = candidateHalf == "bf16" ? Double(Float(bitPattern: 0x7f7f0000)) : 65504.0
        guard !FileManager.default.fileExists(atPath: file.path) else { throw ManifestError.invalid("Bounds already frozen") }
        let ref = try IntakeTensor.read(reference.appendingPathComponent("outputs.safetensors"))
        let again = try IntakeTensor.read(repeatCapture.appendingPathComponent("outputs.safetensors"))
        let calibration = try sourceRoundingCapture.map { try IntakeTensor.read($0.appendingPathComponent("outputs.safetensors")) } ?? again
        guard Set(ref.keys) == Set(calibration.keys), Set(ref.keys) == Set(again.keys), expectedKeys.map({ Set(ref.keys) == $0 }) ?? (ref.count == 5) else {
            throw ManifestError.invalid("Expected matching nonempty component keys")
        }
        let epsilon = candidateHalf == "bf16" ? Foundation.pow(2.0, -7.0) : Foundation.pow(2.0, -10.0)
        var components: JSONObject = [:]
        for key in ref.keys.sorted() {
            let a = ref[key]!, b = calibration[key]!, repeated = again[key]!
            guard a.shape == b.shape, a.shape == repeated.shape else { throw ManifestError.invalid("Reference repeat shape mismatch") }
            let reproducibility = try IntakeComparison.compare(reference: a.values, candidate: repeated.values)
            if sourceFormat {
                guard a.dtype == "BF16", b.dtype == "BF16", repeated.dtype == "BF16", reproducibility.finite, reproducibility.relativeL2 == 0 else {
                    throw ManifestError.invalid("Source-format calibration needs BF16 captures and exact A/A reproducibility")
                }
            }
            let floor = try IntakeComparison.compare(reference: a.values, candidate: b.values)
            guard floor.finite else { throw ManifestError.invalid("Nonfinite reference repeat") }
            // One ULP at each reference value, using its own exponent, then combined in L2.
            let energy = a.values.reduce(0) { $0 + $1 * $1 }
            let margin2 = a.values.reduce(0) { sum, x in
                let ulp =
                    x == 0
                    ? Foundation.pow(2.0, candidateHalf == "bf16" ? -133.0 : -24.0)
                    : max(Foundation.pow(2.0, candidateHalf == "bf16" ? -133.0 : -24.0), Foundation.pow(2.0, floorLog2(abs(x))) * epsilon)
                return sum + ulp * ulp
            }
            let margin = sqrt(margin2 / energy)
            components[key] = [
                "shape": .array(a.shape.map { .number(Double($0)) }), "count": .number(Double(a.values.count)),
                "repeat_relative_l2": .number(reproducibility.relativeL2), "repeat_cosine": .number(reproducibility.cosine),
                "floor_relative_l2": .number(floor.relativeL2), "floor_cosine": .number(floor.cosine),
                "one_ulp_relative_l2": .number(margin), "relative_l2_max": .number(quantized ? quantizedRelativeL2Max : 2 * floor.relativeL2 + margin),
                "cosine_min": .number(quantized ? quantizedCosineMin : max(-1, 1 - 2 * (1 - floor.cosine) - margin * margin)),
                "max_abs": .number(a.values.map(abs).max()! * (1 + epsilon) + abs(floor.maxAbs - a.values.map(abs).max()!))
            ]
        }
        let refTrace = try traces(reference), repeatTrace = try traces(sourceRoundingCapture ?? repeatCapture)
        guard Set(refTrace.keys) == Set(repeatTrace.keys) else { throw ManifestError.invalid("Activation case count mismatch") }
        var activations: JSONObject = [:]
        for (key, row) in refTrace {
            guard row["finite"]?.bool == true, repeatTrace[key]?["finite"]?.bool == true, let maxAbs = row["max_abs"]?.number,
                let repeatMax = repeatTrace[key]?["max_abs"]?.number, row["shape"] == repeatTrace[key]?["shape"]
            else { throw ManifestError.invalid("Invalid reference activation: \(key)") }
            activations[key] = [
                "shape": row["shape"]!, "count": row["count"]!, "max_abs": .number(quantized || sourceFormat ? formatMax : maxAbs * (1 + epsilon) + 2 * abs(maxAbs - repeatMax)),
                "clamp_count_max": .number(sourceFormat ? (row["clamp_count"]?.number ?? 0) : max(row["clamp_count"]?.number ?? 0, repeatTrace[key]?["clamp_count"]?.number ?? 0))
            ]
        }
        try WireJSON.encode([
            "schema": 1, "policy_kind": .string(sourceFormat ? fp32SourcePolicyID : quantized ? "quantized-tier" : "16-bit-format"),
            "candidate_half": .string(candidateHalf),
            "floor_calibration": .string(sourceFormat ? "BF16 nearest versus truncation of original source weights" : "A/A reproducibility"),
            "reference_outputs_sha256": .string(try FileDigest.sha256(reference.appendingPathComponent("outputs.safetensors"))),
            "calibration_outputs_sha256": .string(try FileDigest.sha256((sourceRoundingCapture ?? repeatCapture).appendingPathComponent("outputs.safetensors"))),
            "repeat_outputs_sha256": .string(try FileDigest.sha256(repeatCapture.appendingPathComponent("outputs.safetensors"))),
            "basis": .string(
                sourceFormat
                    ? fp32SourceBasis
                    : quantized ? quantizedBasis : "Reference A/A repeat floor ×2 plus one reference-format ULP; frozen before candidates; no stochastic final-output threshold"),
            "reference_half": .string(half), "components": .object(components), "activations": .object(activations)
        ]).write(to: file, options: .withoutOverwriting)
    }
    /// Selection may use a predeclared safety margin; qualification keeps the shared tier policy.
    static func freezeSelectionSafety(from bounds: URL, to file: URL, relativeL2: Double, cosine: Double) throws {
        guard relativeL2 > 0, relativeL2 <= quantizedRelativeL2Max, cosine >= quantizedCosineMin, cosine <= 1 else {
            throw ManifestError.invalid("Selection safety bounds must be at least as strict as qualification")
        }
        var policy = try WireJSON.decode(Data(contentsOf: bounds))
        guard policy["policy_kind"]?.string == "quantized-tier", let components = policy["components"]?.object else {
            throw ManifestError.invalid("Selection safety needs frozen shared tier bounds")
        }
        policy["policy_kind"] = "selection-safety-margin-v1"
        policy["basis"] = "Predeclared selection-only safety margin; final selection and held-out qualification retain the shared .05/.999 policy."
        policy["qualification_bounds_sha256"] = .string(try FileDigest.sha256(bounds))
        policy["components"] = .object(
            components.mapValues { value in
                var row = value.object!; row["relative_l2_max"] = .number(relativeL2); row["cosine_min"] = .number(cosine); return .object(row)
            })
        try WireJSON.encode(policy).write(to: file, options: .withoutOverwriting)
    }
    private static func floorLog2(_ x: Double) -> Double { Foundation.floor(Foundation.log2(x)) }
    static func traces(_ capture: URL) throws -> [String: JSONObject] {
        let raw = try WireJSON.decode(Data(contentsOf: capture.appendingPathComponent("activations.json")))
        return try raw.mapValues {
            guard let value = $0.object else { throw ManifestError.invalid("Invalid activation capture") }; return value
        }
    }
    static func evaluate(reference: URL, candidate: URL, bounds file: URL, diagnosticKeys: Set<String>? = nil) throws -> JSONObject {
        let policy = try WireJSON.decode(Data(contentsOf: file)), bounds = try requireObject(policy["components"])
        let ref = try IntakeTensor.read(reference.appendingPathComponent("outputs.safetensors")),
            cand = try IntakeTensor.read(candidate.appendingPathComponent("outputs.safetensors"))
        guard Set(ref.keys) == Set(cand.keys), Set(ref.keys) == Set(bounds.keys) else { throw ManifestError.invalid("Component keys/count differ") }
        let trace = try traces(candidate), traceBounds = try requireObject(policy["activations"])
        guard Set(trace.keys) == Set(traceBounds.keys) else { throw ManifestError.invalid("Activation keys/count differ") }
        let sourceFormat = policy["policy_kind"]?.string == fp32SourcePolicyID
        if sourceFormat {
            guard policy["reference_outputs_sha256"]?.string == (try FileDigest.sha256(reference.appendingPathComponent("outputs.safetensors"))) else {
                throw ManifestError.invalid("Source-format reference differs from frozen calibration")
            }
        }
        if let diagnosticKeys {
            guard policy["policy_kind"]?.string == "selection-safety-margin-v1", !diagnosticKeys.isEmpty, diagnosticKeys.isSubset(of: Set(ref.keys)) else {
                throw ManifestError.invalid("Partial evaluation is only a declared selection diagnostic")
            }
        }
        var components: JSONObject = [:], allPass = true
        for key in (diagnosticKeys ?? Set(ref.keys)).sorted() {
            let a = ref[key]!, b = cand[key]!, bound = bounds[key]!
            guard a.shape == b.shape else { throw ManifestError.invalid("Component shape mismatch: \(key)") }
            let metric = try IntakeComparison.compare(reference: a.values, candidate: b.values)
            let names = trace.keys.filter { $0.hasPrefix(key + "/") }
            guard !names.isEmpty else { throw ManifestError.invalid("Missing activation coverage") }
            var activationPass = true, finite = metric.finite, clamps = 0.0, maximum = 0.0, ratio = 0.0
            for name in names {
                let row = trace[name]!, limit = traceBounds[name]!
                guard row["shape"] == limit["shape"], row["count"] == limit["count"] else { throw ManifestError.invalid("Activation shape/count mismatch: \(name)") }
                let isFinite = row["finite"]?.bool == true
                finite = finite && isFinite
                let value = row["max_abs"]?.number ?? 1e300, maxBound = limit["max_abs"]?.number ?? 0
                maximum = max(maximum, value); ratio = max(ratio, maxBound > 0 ? value / maxBound : (value == 0 ? 0 : 1e300))
                clamps += row["clamp_count"]?.number ?? 0
                activationPass = activationPass && isFinite && value <= maxBound && (row["clamp_count"]?.number ?? 0) <= (limit["clamp_count_max"]?.number ?? 0)
            }
            if sourceFormat { activationPass = activationPass && metric.maxAbs <= 65504 && b.dtype == "F16" }
            let passed = finite && activationPass && metric.relativeL2 <= bound["relative_l2_max"]!.number! && metric.cosine >= bound["cosine_min"]!.number!
            allPass = allPass && passed
            let normalBF16 = Foundation.pow(2.0, -126.0), normalFP16 = Foundation.pow(2.0, -14.0)
            let underflowZero = sourceFormat ? zip(a.values, b.values).filter { abs($0.0) >= normalBF16 && $0.1 == 0 }.count : 0
            let underflowSubnormal = sourceFormat ? zip(a.values, b.values).filter { abs($0.0) >= normalBF16 && abs($0.1) > 0 && abs($0.1) < normalFP16 }.count : 0
            components[key] = [
                "fp16_zero_where_bf16_normal": sourceFormat ? .number(Double(underflowZero)) : .null,
                "fp16_subnormal_where_bf16_normal": sourceFormat ? .number(Double(underflowSubnormal)) : .null,
                "relative_l2": .number(metric.relativeL2), "cosine": .number(metric.cosine), "finite": .bool(finite),
                "reference_rms": .number(sqrt(a.values.reduce(0) { $0 + $1 * $1 } / Double(a.values.count))),
                "candidate_rms": .number(metric.finite ? sqrt(b.values.reduce(0) { $0 + $1 * $1 } / Double(b.values.count)) : 1e300),
                "reference_max_abs": .number(a.values.map(abs).max()!),
                "activation_max": .number(maximum), "worst_activation_bound_ratio": .number(ratio), "clamp_count": .number(clamps),
                "activation_passed": .bool(activationPass), "bounds": bound, "passed": .bool(passed)
            ]
        }
        var fixtureIdentity: JSONValue = .null
        let identityFile = candidate.appendingPathComponent("fixture-identity.json")
        if FileManager.default.fileExists(atPath: identityFile.path) {
            let identity = try WireJSON.decode(Data(contentsOf: identityFile))
            let referenceIdentity = try WireJSON.decode(Data(contentsOf: reference.appendingPathComponent("fixture-identity.json")))
            guard identity["fixture_inputs_sha256"] == referenceIdentity["fixture_inputs_sha256"],
                identity["component_count"]?.number == Double(cand.count)
            else { throw ManifestError.invalid("Capture fixture identity/count mismatch") }
            fixtureIdentity = identity["fixture_inputs_sha256"] ?? .null
        }
        return [
            "diagnostic_only": .bool(diagnosticKeys != nil),
            "fixture_inputs_sha256": fixtureIdentity,
            "reference_outputs_sha256": .string(try FileDigest.sha256(reference.appendingPathComponent("outputs.safetensors"))),
            "candidate_outputs_sha256": .string(try FileDigest.sha256(candidate.appendingPathComponent("outputs.safetensors"))),
            "policy_kind": policy["policy_kind"] ?? .null, "components": .object(components), "component_gate_passed": .bool(allPass),
            "bounds_sha256": .string(try FileDigest.sha256(file))
        ]
    }
    private static func requireObject(_ value: JSONValue?) throws -> JSONObject {
        guard let o = value?.object else { throw ManifestError.invalid("Malformed component bounds") }; return o
    }
}

struct IntakeTimingSummary {
    let samples: [Double]
    var median: Double { let s = samples.sorted(); return s.count % 2 == 0 ? (s[s.count / 2 - 1] + s[s.count / 2]) / 2 : s[s.count / 2] }
    var json: JSONValue {
        [
            "samples_s": .array(samples.map(JSONValue.number)), "median_s": .number(median), "min_s": .number(samples.min()!), "max_s": .number(samples.max()!),
            "spread_s": .number(samples.max()! - samples.min()!)
        ]
    }
}
