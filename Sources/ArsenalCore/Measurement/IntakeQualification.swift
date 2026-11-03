// SPDX-License-Identifier: MIT
import Foundation

/// Shared selection/qualification boundary. Freeze before search; call completion only after
/// reloading the final stored recipe. A failed held-out attempt is terminal for this plan:
/// changing a recipe needs selection-only evidence and a new untouched held-out plan.
public enum IntakeQualification {
    public static let policyID = "distinct-heldout-stored-recipe-v1"
    public static func freeze(selection: URL, heldout: URL, selectionBounds: URL, heldoutBounds: URL, to file: URL) throws -> JSONObject {
        func identity(_ directory: URL, _ bounds: URL) throws -> JSONValue {
            let policy = try WireJSON.decode(Data(contentsOf: bounds))
            guard let keys = policy["components"]?.object, !keys.isEmpty else { throw ManifestError.invalid("Empty qualification bounds") }
            return [
                "inputs_sha256": .string(try FileDigest.sha256(directory.appendingPathComponent("inputs.safetensors"))),
                "plan_sha256": .string(try FileDigest.sha256(directory.appendingPathComponent("fixture-plan.json"))),
                "bounds_sha256": .string(try FileDigest.sha256(bounds)),
                "component_keys": .array(keys.keys.sorted().map(JSONValue.string)), "component_count": .number(Double(keys.count)),
                "bounds_file": .string(bounds.lastPathComponent), "bounds_policy": policy["policy_kind"] ?? .null, "bounds": .object(policy)
            ]
        }
        let a = try identity(selection, selectionBounds), b = try identity(heldout, heldoutBounds)
        guard a["inputs_sha256"] != b["inputs_sha256"], a["plan_sha256"] != b["plan_sha256"] else {
            throw ManifestError.invalid("Qualification requires distinct selection and held-out inputs")
        }
        let plan: JSONObject = ["policy": .string(policyID), "selection": a, "heldout": b]
        try WireJSON.encode(plan).write(to: file, options: .withoutOverwriting)
        return plan
    }
    /// A declared recipe with no adaptive selection needs the same frozen evidence checks,
    /// but has no searched-recipe held-out claim.
    public static func freeze(selection: URL, selectionBounds: URL, to file: URL) throws -> JSONObject {
        try freezeRecorded(inputsSHA256: FileDigest.sha256(selection.appendingPathComponent("inputs.safetensors")), selectionBounds: selectionBounds, to: file)
    }
    /// Rebind retained measurements using their captured inventory identity and the original,
    /// already-frozen bounds. This does not claim a new capture or a new pre-run freeze.
    public static func freezeRecorded(inputsSHA256: String, selectionBounds: URL, to file: URL) throws -> JSONObject {
        let bounds = try WireJSON.decode(Data(contentsOf: selectionBounds))
        guard inputsSHA256.count == 64, inputsSHA256.allSatisfy({ "0123456789abcdef".contains($0) }),
            let keys = bounds["components"]?.object, !keys.isEmpty
        else { throw ManifestError.invalid("Empty qualification bounds or input identity") }
        let plan: JSONObject = [
            "policy": .string(policyID), "searched": false,
            "selection": [
                "inputs_sha256": .string(inputsSHA256), "bounds_sha256": .string(try FileDigest.sha256(selectionBounds)),
                "bounds_file": .string(selectionBounds.lastPathComponent), "bounds": .object(bounds),
                "component_keys": .array(keys.keys.sorted().map(JSONValue.string)), "component_count": .number(Double(keys.count))
            ]
        ]
        try WireJSON.encode(plan).write(to: file, options: .withoutOverwriting)
        return plan
    }
    /// Validate numerical verdicts even when every redundant pass flag was forged together.
    public static func validateGate(_ gate: JSONObject, bounds: JSONObject? = nil) throws -> Bool {
        guard let components = gate["components"]?.object, !components.isEmpty,
            let aggregate = gate["component_gate_passed"]?.bool, gate["diagnostic_only"] != true
        else { throw ManifestError.invalid("Missing complete component gate") }
        let limits = bounds?["components"]?.object
        if let limits, Set(limits.keys) != Set(components.keys) { throw ManifestError.invalid("Frozen component set differs") }
        var passed = true
        for (key, value) in components {
            guard let entry = value.object, let bound = entry["bounds"]?.object,
                limits == nil || limits?[key] == .object(bound),
                let relative = entry["relative_l2"]?.number, relative.isFinite, relative >= 0,
                let cosine = entry["cosine"]?.number, cosine.isFinite, (-1...1).contains(cosine),
                let maximum = bound["relative_l2_max"]?.number, maximum.isFinite, maximum >= 0,
                let minimum = bound["cosine_min"]?.number, minimum.isFinite, (-1...1).contains(minimum),
                let finite = entry["finite"]?.bool, let activation = entry["activation_passed"]?.bool,
                let result = entry["passed"]?.bool
            else { throw ManifestError.invalid("Invalid component metrics or frozen bounds: \(key)") }
            if activation, let ratio = entry["worst_activation_bound_ratio"]?.number, !ratio.isFinite || ratio > 1 || ratio < 0 {
                throw ManifestError.invalid("Activation verdict contradicts its frozen bound: " + key)
            }
            let expected = finite && activation && relative <= maximum && cosine >= minimum
            guard result == expected else { throw ManifestError.invalid("Component verdict contradicts numerical evidence: \(key)") }
            passed = passed && expected
        }
        if gate["max_probability_delta"] != nil || gate["chosen_answers_unchanged"] != nil {
            guard let delta = gate["max_probability_delta"]?.number, delta.isFinite, delta >= 0,
                gate["probability_delta_max"] == 0.017, let unchanged = gate["chosen_answers_unchanged"]?.bool
            else { throw ManifestError.invalid("Invalid decision answer qualification") }
            passed = passed && unchanged && delta <= 0.017
        }
        if let format = gate["format_qualification"]?.object {
            let accepted = try validateGate(format)
            passed = passed && (gate["selected_half"] == "bf16" ? !accepted : accepted)
        }
        guard aggregate == passed else { throw ManifestError.invalid("Aggregate verdict contradicts component or task evidence") }
        return passed
    }
    /// All completion writers use the same strict row check before publishing anything.
    public static func validatedRow(_ row: JSONObject, model: ModelManifest) throws -> JSONObject {
        guard let q = row["qualification"]?.object, let plan = q["fixtures"]?.object,
            let selection = q["selection_gate"]?.object, let searched = q["searched"]?.bool,
            q["passed"] == row["qualified"], q["selection_gate"] == row["selection_gate"],
            (q["heldout_gate"] ?? q["selection_gate"]) == row["gate"],
            q["ordinary_load"]?["ok"] == true, q["ordinary_load"]?["requested_load"]?["dtype"] == row["precision"],
            try complete(plan: plan, selection: selection, heldout: q["heldout_gate"]?.object, searched: searched, ordinaryLoad: q["ordinary_load"]?.object) == q
        else { throw ManifestError.invalid("Completion requires exact shared qualification and ordinary Load") }
        for gate in [q["selection_gate"], q["heldout_gate"]].compactMap({ $0?.object }) {
            if model.kind == "decisions" {
                guard gate["chosen_answers_unchanged"]?.bool != nil, gate["max_probability_delta"]?.number != nil,
                    gate["probability_delta_max"] == 0.017
                else { throw ManifestError.invalid("Decision qualification requires retained answer checks") }
            }
            if plan["searched"] == false, plan["heldout"] == nil, row["precision"]?.string == model.basePrecision, model.weights.dtype == "fp32" {
                guard gate["format_qualification"]?.object != nil else { throw ManifestError.invalid("Source cast requires its frozen format qualification") }
            }
        }
        return q
    }
    public static func validateCompletion(rows: [JSONValue], model: ModelManifest, evidence: URL) throws {
        for value in rows {
            guard let row = value.object, let dtype = row["precision"]?.string else { throw ManifestError.invalid("Incomplete completion row") }
            let q = try validatedRow(row, model: model), plan = q["fixtures"]!.object!
            let fixed = plan["searched"] == false && plan["heldout"] == nil
            let name = fixed ? "qualification-plan-" + (["int8", "int4"].contains(dtype) ? "quantized" : dtype) + ".json" : "qualification-plan.json"
            guard try WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent(name))) == plan else { throw ManifestError.invalid("Completion differs from frozen plan") }
            for arm in fixed ? ["selection"] : ["selection", "heldout"] {
                let identity = plan[arm]!.object!, gate = q[arm + "_gate"]!.object!
                let file = identity["bounds_file"]?.string ?? (arm == "heldout" ? "heldout-quantized-bounds.json" : "quantized-bounds.json")
                guard SafePath.isRelative(file), !file.contains("/"),
                    try FileDigest.sha256(evidence.appendingPathComponent(file)) == identity["bounds_sha256"]?.string
                else { throw ManifestError.invalid("Completion bounds digest differs") }
                _ = try validateGate(gate, bounds: WireJSON.decode(Data(contentsOf: evidence.appendingPathComponent(file))))
                if let format = gate["format_qualification"]?.object {
                    let formatFile = evidence.appendingPathComponent("format-qualification/format-gate.json")
                    let formatBounds = evidence.appendingPathComponent("format-qualification/bounds.json")
                    guard try WireJSON.decode(Data(contentsOf: formatFile)) == format,
                        try FileDigest.sha256(formatFile) == gate["format_qualification_sha256"]?.string,
                        try FileDigest.sha256(formatBounds) == format["bounds_sha256"]?.string
                    else { throw ManifestError.invalid("Nested format qualification differs from retained frozen evidence") }
                    _ = try validateGate(format, bounds: WireJSON.decode(Data(contentsOf: formatBounds)))
                }
            }
        }
    }
    public static func complete(plan: JSONObject, selection: JSONObject, heldout: JSONObject?, searched: Bool, ordinaryLoad: JSONObject?) throws -> JSONObject {
        if !searched, plan["searched"] == false, plan["heldout"] == nil {
            guard plan["policy"]?.string == policyID, heldout == nil, let ordinaryLoad,
                ordinaryLoad["ok"] == true, ordinaryLoad["requested_load"]?.object != nil,
                let components = selection["components"]?.object, !components.isEmpty,
                selection["diagnostic_only"]?.bool != true,
                selection["bounds_sha256"] == plan["selection"]?["bounds_sha256"],
                selection["fixture_inputs_sha256"]?.string != nil, selection["fixture_inputs_sha256"] == plan["selection"]?["inputs_sha256"],
                isDigest(selection["reference_outputs_sha256"]), isDigest(selection["candidate_outputs_sha256"]),
                Set(components.keys) == Set(plan["selection"]?["component_keys"]?.array?.compactMap(\.string) ?? []),
                Double(components.count) == plan["selection"]?["component_count"]?.number,
                components.values.allSatisfy({ $0["passed"]?.bool != nil }),
                let passed = selection["component_gate_passed"]?.bool,
                passed == (try validateGate(selection, bounds: plan["selection"]?["bounds"]?.object))
            else { throw ManifestError.invalid("Fixed recipe differs from frozen complete evidence or ordinary Load") }
            return [
                "policy": .string(policyID), "searched": false, "fixtures": .object(plan),
                "selection_gate": .object(selection), "ordinary_load": .object(ordinaryLoad), "passed": .bool(passed)
            ]
        }
        guard plan["policy"]?.string == policyID, let heldout, let ordinaryLoad,
            ordinaryLoad["ok"] == true, ordinaryLoad["requested_load"]?.object != nil,
            plan["selection"]?["inputs_sha256"] != plan["heldout"]?["inputs_sha256"],
            plan["selection"]?["plan_sha256"] != plan["heldout"]?["plan_sha256"]
        else { throw ManifestError.invalid("Cannot bind a recipe without distinct held-out evidence through ordinary Load") }
        for (name, gate) in [("selection", selection), ("heldout", heldout)] {
            guard let components = gate["components"]?.object, !components.isEmpty, gate["diagnostic_only"]?.bool != true,
                gate["bounds_sha256"] == plan[name]?["bounds_sha256"],
                gate["fixture_inputs_sha256"]?.string != nil, gate["fixture_inputs_sha256"] == plan[name]?["inputs_sha256"],
                isDigest(gate["reference_outputs_sha256"]), isDigest(gate["candidate_outputs_sha256"]),
                Set(components.keys) == Set(plan[name]?["component_keys"]?.array?.compactMap(\.string) ?? []),
                Double(components.count) == plan[name]?["component_count"]?.number,
                components.values.allSatisfy({ $0["passed"]?.bool != nil }), gate["component_gate_passed"]?.bool != nil,
                gate["component_gate_passed"]?.bool == (try validateGate(gate, bounds: plan[name]?["bounds"]?.object))
            else { throw ManifestError.invalid("Qualification evidence differs from frozen complete \(name) set") }
        }
        return [
            "policy": .string(policyID), "searched": .bool(searched), "fixtures": .object(plan),
            "selection_gate": .object(selection), "heldout_gate": .object(heldout), "ordinary_load": .object(ordinaryLoad),
            "passed": .bool(selection["component_gate_passed"]?.bool == true && heldout["component_gate_passed"]?.bool == true)
        ]
    }
    private static func isDigest(_ value: JSONValue?) -> Bool {
        guard let text = value?.string else { return false }
        return text.count == 64 && text.allSatisfy { "0123456789abcdef".contains($0) }
    }

}
