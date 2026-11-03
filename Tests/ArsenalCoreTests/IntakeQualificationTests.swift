// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct IntakeQualificationTests {
    @Test func searchedRecipeRequiresDistinctCompleteHeldoutEvidence() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let selection = root.appendingPathComponent("selection"), heldout = root.appendingPathComponent("heldout")
        let a = root.appendingPathComponent("a.json"), b = root.appendingPathComponent("b.json")
        let bound: JSONObject = ["components": ["a": ["relative_l2_max": 0.05, "cosine_min": 0.999]]]
        for (directory, text, bounds) in [(selection, "selection", a), (heldout, "different text, tokens, prefix and conditioning", b)] {
            try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
            try Data(text.utf8).write(to: directory.appendingPathComponent("inputs.safetensors"))
            try WireJSON.encode(["case": .string(text)]).write(to: directory.appendingPathComponent("fixture-plan.json"))
            try WireJSON.encode(bound).write(to: bounds)
        }
        let plan = try IntakeQualification.freeze(selection: selection, heldout: heldout, selectionBounds: a, heldoutBounds: b, to: root.appendingPathComponent("plan.json"))
        let gate: JSONObject = [
            "component_gate_passed": true,
            "components": [
                "a": ["passed": true, "finite": true, "activation_passed": true, "relative_l2": 0, "cosine": 1, "bounds": ["relative_l2_max": 0.05, "cosine_min": 0.999]]
            ], "bounds_sha256": .string(try FileDigest.sha256(a)),
            "fixture_inputs_sha256": plan["selection"]!["inputs_sha256"]!,
            "reference_outputs_sha256": .string(String(repeating: "a", count: 64)), "candidate_outputs_sha256": .string(String(repeating: "b", count: 64))
        ]
        var heldoutGate = gate; heldoutGate["fixture_inputs_sha256"] = plan["heldout"]?["inputs_sha256"]
        let load: JSONObject = ["ok": true, "requested_load": ["dtype": "int8", "model_dir": "/lab/final-stored-recipe"]]
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: plan, selection: gate, heldout: nil, searched: true, ordinaryLoad: load) }
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: plan, selection: gate, heldout: gate, searched: true, ordinaryLoad: nil) }
        var bad = plan; bad["heldout"] = plan["selection"]
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: bad, selection: gate, heldout: gate, searched: true, ordinaryLoad: load) }
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: plan, selection: gate, heldout: gate, searched: true, ordinaryLoad: load) }
        let complete = try IntakeQualification.complete(plan: plan, selection: gate, heldout: heldoutGate, searched: true, ordinaryLoad: load)
        #expect(complete["passed"] == true)
        var diagnostic = heldoutGate; diagnostic["diagnostic_only"] = true
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: plan, selection: gate, heldout: diagnostic, searched: true, ordinaryLoad: load) }
        var missing = gate; missing["components"] = [:]
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: plan, selection: gate, heldout: missing, searched: true, ordinaryLoad: load) }
        var failed = heldoutGate

        failed["components"] = [
            "a": ["passed": false, "finite": false, "activation_passed": false, "relative_l2": 1.05, "cosine": 0.9, "bounds": ["relative_l2_max": 0.05, "cosine_min": 0.999]]
        ]; failed["component_gate_passed"] = false
        #expect(try IntakeQualification.complete(plan: plan, selection: gate, heldout: failed, searched: true, ordinaryLoad: load)["passed"] == false)
        #expect(throws: (any Error).self) {
            try IntakeQualification.freeze(selection: selection, heldout: selection, selectionBounds: a, heldoutBounds: a, to: root.appendingPathComponent("invalid.json"))
        }
    }
    @Test func fixedRecipeUsesFrozenEvidenceButCannotClaimSearchQualification() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Data("fixed inputs".utf8).write(to: root.appendingPathComponent("inputs.safetensors"))
        let bounds = root.appendingPathComponent("bounds.json")
        try WireJSON.encode(["components": ["a": ["relative_l2_max": 0.05, "cosine_min": 0.999]]]).write(to: bounds)
        let plan = try IntakeQualification.freeze(selection: root, selectionBounds: bounds, to: root.appendingPathComponent("plan.json"))
        let gate: JSONObject = [
            "component_gate_passed": true,
            "components": [
                "a": ["passed": true, "finite": true, "activation_passed": true, "relative_l2": 0, "cosine": 1, "bounds": ["relative_l2_max": 0.05, "cosine_min": 0.999]]
            ],
            "bounds_sha256": .string(try FileDigest.sha256(bounds)), "fixture_inputs_sha256": plan["selection"]!["inputs_sha256"]!,
            "reference_outputs_sha256": .string(String(repeating: "a", count: 64)), "candidate_outputs_sha256": .string(String(repeating: "b", count: 64))
        ]
        let load: JSONObject = ["ok": true, "requested_load": ["dtype": "int8", "model_dir": "/lab/stored"]]
        #expect(try IntakeQualification.complete(plan: plan, selection: gate, heldout: nil, searched: false, ordinaryLoad: load)["passed"] == true)
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: plan, selection: gate, heldout: nil, searched: true, ordinaryLoad: load) }
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: plan, selection: gate, heldout: nil, searched: false, ordinaryLoad: nil) }
        var changed = gate; changed["fixture_inputs_sha256"] = "changed"
        #expect(throws: (any Error).self) { try IntakeQualification.complete(plan: plan, selection: changed, heldout: nil, searched: false, ordinaryLoad: load) }
    }

    @Test func reconciliationPreservesUnmeasuredCellsAndRejectsMissingMeasuredProvenance() throws {
        let missing: JSONValue = ["state": "notMeasured"]
        #expect(try ModelIntakeCommand.rebindProvenance(missing, oldRun: "old", run: "new", path: "results/new.json") == missing)
        #expect(throws: (any Error).self) {
            try ModelIntakeCommand.rebindProvenance(["state": "quickCheck", "value": 1], oldRun: "old", run: "new", path: "results/new.json")
        }
    }

    @Test func everyCurrentCatalogBindingQualifiesEveryTierThroughTheSharedBoundary() throws {
        let folders = try FileManager.default.contentsOfDirectory(at: intakeFixtureRepository.appendingPathComponent("models"), includingPropertiesForKeys: nil)
        var models = 0, tiers = 0
        for folder in folders {
            let files = (try? FileManager.default.contentsOfDirectory(at: folder.appendingPathComponent("gate/evidence"), includingPropertiesForKeys: nil)) ?? []
            guard files.contains(where: { $0.lastPathComponent.hasPrefix("quality-binding-20") }) else { continue }
            let binding = try latestIntakeBinding(folder), run = try #require(binding["run"]?.string)
            let path = folder.appendingPathComponent("results/intake/" + run + ".json")
            let captured = try WireJSON.decode(Data(contentsOf: path)), rows = try #require(captured["rows"]?.array)
            #expect(binding["schema"] == 3 && binding["qualification_policy"]?.string == IntakeQualification.policyID)
            let model = try ModelManifest.load(folder), outcomes = try #require(binding["outcomes"]?.array)
            // The binding covers every declared tier; a tier withdrawn since keeps its recorded outcome (removedQualityTiers).
            let bound = Set(outcomes.compactMap { $0["dtype"]?.string })
            #expect(bound.isSuperset(of: model.precisions))
            for tier in bound.subtracting(model.precisions) {
                #expect(removedQualityTiers.contains { $0.model == model.id && $0.tier == tier }, "\(model.id) \(tier)")
            }
            try IntakeQualification.validateCompletion(rows: rows, model: model, evidence: folder.appendingPathComponent("gate/evidence/intake/" + run))
            for outcome in outcomes {
                let row = try #require(rows.first { $0["precision"] == outcome["dtype"] })
                #expect(row["qualification"] == outcome["qualification"] && row["qualified"] == outcome["passed"])
                #expect(outcome["capture_result_sha256"]?.string == (try FileDigest.sha256(path)))
                tiers += 1
            }
            models += 1
        }
        #expect(models == 16 && tiers == 48)
    }

}
