// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore

struct IntakeComponentGateTests {
    @Test func savedFormatReplayAllowsOnlyAdditiveDiagnostics() throws {
        let saved: JSONObject = ["bounds_sha256": "frozen", "component_gate_passed": true, "components": ["decoder": ["relative_l2": 0.01, "passed": true]]]
        var current = saved
        current["reference_outputs_sha256"] = "reference"; current["candidate_outputs_sha256"] = "candidate"
        current["fixture_inputs_sha256"] = .null; current["diagnostic_only"] = false
        current["components"] = ["decoder": ["relative_l2": 0.01, "passed": true, "reference_rms": 1, "candidate_rms": 1, "reference_max_abs": 2]]
        try IntakeComponentGate.validateSavedEvaluation(saved, current: current)
        var changed = current; changed["component_gate_passed"] = false
        #expect(throws: (any Error).self) { try IntakeComponentGate.validateSavedEvaluation(saved, current: changed) }
        changed = current; changed["components"] = ["decoder": ["relative_l2": 0.02, "passed": true]]
        #expect(throws: (any Error).self) { try IntakeComponentGate.validateSavedEvaluation(saved, current: changed) }
        changed = current; changed["diagnostic_only"] = true
        #expect(throws: (any Error).self) { try IntakeComponentGate.validateSavedEvaluation(saved, current: changed) }
        changed = current; changed["unknown"] = true
        #expect(throws: (any Error).self) { try IntakeComponentGate.validateSavedEvaluation(saved, current: changed) }
    }
    @Test func signReversalCannotPassCosineOrRelativeError() throws {
        let metric = try IntakeComparison.compare(reference: [1, 2, -3], candidate: [-1, -2, 3])
        #expect(metric.relativeL2 == 2 && abs(metric.cosine + 1) < 1e-14)
        #expect(throws: (any Error).self) { try IntakeComparison.compare(reference: [1], candidate: [1, 2]) }
        #expect(try IntakeComparison.compare(reference: [1], candidate: [.infinity]).finite == false)
    }
    @Test func captureReaderAllowsMLXNullMetadataAndRejectsTruncation() throws {
        let file = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: file) }
        let header = Data(#"{"__metadata__":null,"x":{"dtype":"F16","shape":[1],"data_offsets":[0,2]}}"#.utf8)
        var n = UInt64(header.count).littleEndian
        var bytes = withUnsafeBytes(of: &n) { Data($0) }; bytes.append(header); bytes.append(contentsOf: [0, 60])
        try bytes.write(to: file)
        #expect(try IntakeTensor.read(file)["x"]?.values == [1])
        bytes.removeLast(); try bytes.write(to: file)
        #expect(throws: (any Error).self) { try IntakeTensor.read(file) }
    }
    @Test func voiceCaptureMustMatchTheSelectedHalfStream() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try capture(root)
        try ModelIntakeCommand.requireVoiceHalfCapture(root, half: "fp16")
        #expect(throws: (any Error).self) { try ModelIntakeCommand.requireVoiceHalfCapture(root, half: "bf16") }
        let file = root.appendingPathComponent("outputs.safetensors")
        let header = Data(#"{"decoder":{"dtype":"F32","shape":[1],"data_offsets":[0,4]}}"#.utf8)
        var length = UInt64(header.count).littleEndian
        var bytes = withUnsafeBytes(of: &length) { Data($0) }
        bytes.append(header); bytes.append(contentsOf: [0, 0, 128, 63])
        try bytes.write(to: file)
        #expect(throws: (any Error).self) { try ModelIntakeCommand.requireVoiceHalfCapture(root, half: "bf16") }
    }
    @Test func boundsFreezeBeforeCandidatesAndGateEveryActivation() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let reference = root.appendingPathComponent("reference"), candidate = root.appendingPathComponent("candidate"), bounds = root.appendingPathComponent("bounds.json")
        try capture(reference); try capture(candidate)
        try IntakeComponentGate.freeze(reference: reference, repeatCapture: reference, to: bounds, half: "fp16")
        let hash = try FileDigest.sha256(bounds)
        #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: bounds)["component_gate_passed"]?.bool == true)
        #expect(throws: (any Error).self) { try IntakeComponentGate.freeze(reference: reference, repeatCapture: candidate, to: bounds, half: "fp16") }
        try capture(candidate, maximum: 10)
        #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: bounds)["component_gate_passed"]?.bool == false)
        #expect(try FileDigest.sha256(bounds) == hash)
        try capture(candidate, omit: "denoiser-1")
        #expect(throws: (any Error).self) { try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: bounds) }
    }
    @Test func quantizedQualityAllowsSmallErrorButRejectsRangeClampsAndLargeError() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let reference = root.appendingPathComponent("reference"), candidate = root.appendingPathComponent("candidate")
        let format = root.appendingPathComponent("format.json"), quant = root.appendingPathComponent("quant.json")
        try capture(reference); try capture(candidate, maximum: 3, scale: 1.02)
        try IntakeComponentGate.freeze(reference: reference, repeatCapture: reference, to: format, half: "fp16")
        try IntakeComponentGate.freeze(reference: reference, repeatCapture: reference, to: quant, half: "fp16", quantized: true)
        #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: format)["component_gate_passed"]?.bool == false)
        #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: quant)["component_gate_passed"]?.bool == true)
        for (maximum, scale, clamps) in [(65505.0, Float16(1), 0), (2.0, Float16(1.1), 0), (2.0, Float16(1), 1)] {
            try capture(candidate, maximum: maximum, scale: scale, clamps: clamps)
            #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: quant)["component_gate_passed"]?.bool == false)
        }
    }
    @Test func partialEvaluationIsMarkedAndCannotUseQualificationBounds() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let reference = root.appendingPathComponent("reference"), quality = root.appendingPathComponent("quality.json"), search = root.appendingPathComponent("search.json")
        try capture(reference)
        try IntakeComponentGate.freeze(reference: reference, repeatCapture: reference, to: quality, half: "fp16", quantized: true)
        try IntakeComponentGate.freezeSelectionSafety(from: quality, to: search, relativeL2: 0.05, cosine: 0.999)
        let gate = try IntakeComponentGate.evaluate(reference: reference, candidate: reference, bounds: search, diagnosticKeys: ["decoder"])
        #expect(gate["diagnostic_only"] == true && gate["component_gate_passed"] == true)
        #expect(throws: (any Error).self) { try IntakeComponentGate.evaluate(reference: reference, candidate: reference, bounds: quality, diagnosticKeys: ["decoder"]) }
        #expect(throws: (any Error).self) { try IntakeComponentGate.evaluate(reference: reference, candidate: reference, bounds: search, diagnosticKeys: ["missing"]) }
        #expect(throws: (any Error).self) { try IntakeComponentGate.evaluate(reference: reference, candidate: reference, bounds: search, diagnosticKeys: []) }
    }
    @Test func selectionSafetyNeverWeakensOrMutatesQualification() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let reference = root.appendingPathComponent("reference"), candidate = root.appendingPathComponent("candidate")
        let quality = root.appendingPathComponent("quality.json"), search = root.appendingPathComponent("search.json")
        try capture(reference); try capture(candidate, scale: 1.03)
        try IntakeComponentGate.freeze(reference: reference, repeatCapture: reference, to: quality, half: "fp16", quantized: true)
        let frozen = try FileDigest.sha256(quality)
        try IntakeComponentGate.freezeSelectionSafety(from: quality, to: search, relativeL2: 0.025, cosine: 0.99975)
        #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: quality)["component_gate_passed"] == true)
        #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: search)["component_gate_passed"] == false)
        #expect(try FileDigest.sha256(quality) == frozen)
        #expect(throws: (any Error).self) {
            try IntakeComponentGate.freezeSelectionSafety(from: quality, to: root.appendingPathComponent("weak.json"), relativeL2: 0.06, cosine: 0.999)
        }
    }
    @Test func sourceFormatChecksRangeWithoutRequiringBF16ActivationExtrema() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let keys = ["decoder", "seconds", "padding"]
        try capture(root.appendingPathComponent("bf16-reference"), keys: keys, bf16: true)
        try capture(root.appendingPathComponent("bf16-rounding"), scale: 1.01, keys: keys, bf16: true)
        try capture(root.appendingPathComponent("bf16-repeat"), keys: keys, bf16: true)
        try ModelIntakeCommand.soundFormatCommand(["--freeze-sound-format", "--format-evidence", root.path])
        let candidate = root.appendingPathComponent("fp16-candidate")
        // The FP16 format contract accepts a changed intermediate maximum that remains in range.
        try capture(candidate, maximum: 2.01, keys: keys)
        let bounds = root.appendingPathComponent("bounds.json"), reference = root.appendingPathComponent("bf16-reference")
        #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: bounds)["component_gate_passed"] == true)
        let frozen = try WireJSON.decode(Data(contentsOf: bounds))
        #expect(frozen["policy_kind"]?.string == IntakeComponentGate.fp32SourcePolicyID)
        #expect(frozen["candidate_half"] == "fp16")
        let component = try #require(frozen["components"]?["decoder"]?.object)
        #expect(component["repeat_relative_l2"]?.number == 0)
        #expect(component["floor_relative_l2"]?.number ?? 0 > 0)
        #expect(abs((component["one_ulp_relative_l2"]?.number ?? 0) - sqrt((1.0 / 1024 / 1024 + 0.25 / 1024 / 1024 + 4.0 / 1024 / 1024) / 5.25)) < 1e-14)
        #expect(throws: (any Error).self) { try ModelIntakeCommand.soundFormatCommand(["--freeze-sound-format", "--format-evidence", root.path]) }
        for (maximum, clamps) in [(65505.0, 0), (2.0, 1)] {
            try capture(candidate, maximum: maximum, clamps: clamps, keys: keys)
            #expect(try IntakeComponentGate.evaluate(reference: reference, candidate: candidate, bounds: bounds)["component_gate_passed"] == false)
        }
    }
    @Test func sourceFormatReportsUnderflowAndRejectsAARoundingSubstitution() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a"), floor = root.appendingPathComponent("floor"), b = root.appendingPathComponent("b")
        func tiny(_ dir: URL, bf16: Bool) throws {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            let values: [Float] = [1, 1e-9, 1e-5]
            var payload = Data()
            for x in values {
                var bits = (bf16 ? UInt16(x.bitPattern >> 16) : Float16(x).bitPattern).littleEndian
                withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
            }
            let header = try JSONSerialization.data(withJSONObject: ["x": ["dtype": bf16 ? "BF16" : "F16", "shape": [3], "data_offsets": [0, 6]]])
            var length = UInt64(header.count).littleEndian
            var data = withUnsafeBytes(of: &length) { Data($0) }; data.append(header); data.append(payload)
            try data.write(to: dir.appendingPathComponent("outputs.safetensors"))
            try WireJSON.encode(["x/output": ["shape": [3], "count": 3, "finite": true, "max_abs": 1, "clamp_count": 0]])
                .write(to: dir.appendingPathComponent("activations.json"))
        }
        try tiny(a, bf16: true); try tiny(floor, bf16: true); try tiny(b, bf16: false)
        let bounds = root.appendingPathComponent("bounds.json")
        try IntakeComponentGate.freezeFP32Source(reference: a, roundingCapture: floor, repeatCapture: a, to: bounds, expectedKeys: ["x"])
        let gate = try IntakeComponentGate.evaluate(reference: a, candidate: b, bounds: bounds)
        #expect(gate["components"]?["x"]?["fp16_zero_where_bf16_normal"] == 1)
        #expect(gate["components"]?["x"]?["fp16_subnormal_where_bf16_normal"] == 1)
        #expect(throws: (any Error).self) {
            try IntakeComponentGate.freezeFP32Source(reference: a, roundingCapture: floor, repeatCapture: b, to: root.appendingPathComponent("invalid.json"), expectedKeys: ["x"])
        }
    }

    @Test(arguments: ["decisions", "sound", "images", "voices"]) func figuresRejectAnyCaptureWorker(kind: String) async throws {
        let captureWorker = WorkerProcess(), servingWorker = WorkerProcess()
        do {
            try await captureWorker.start(stubLaunch(kind))
            _ = try? await captureWorker.request(op: "measurement_components")
            #expect(await captureWorker.componentCaptureRequests == 1)
            _ = try? await captureWorker.request(op: "measurement_voice_patch")
            #expect(await captureWorker.componentCaptureRequests == 2)
            await #expect(throws: (any Error).self) { try await ModelIntakeCommand.servingPeak(captureWorker) }
            let (fresh, _) = try await ModelIntakeCommand.freshServingWorker(capture: captureWorker) {
                try await servingWorker.start(stubLaunch(kind))
                return (servingWorker, try await servingWorker.request(op: "load", params: ["weights_dir": "/tmp/none"]))
            }
            #expect(await captureWorker.exited)
            #expect(fresh === servingWorker)
            #expect(try await ModelIntakeCommand.servingPeak(servingWorker) > 0)
            #expect(await ModelIntakeCommand.servingWorkerReceipt(servingWorker)["measurement_components_requests"] == 0)
            await captureWorker.unload(); await servingWorker.unload()
        } catch { await captureWorker.unload(); await servingWorker.unload(); throw error }
    }

    private func capture(
        _ dir: URL, maximum: Double = 2, omit: String? = nil, scale: Float16 = 1, clamps: Int = 0, keys: [String] = ["prompt", "denoiser-0", "denoiser-1", "denoiser-3", "decoder"],
        bf16: Bool = false
    ) throws {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        var header: [String: Any] = ["__metadata__": NSNull()], payload = Data(), traces: JSONObject = [:]
        for key in keys where key != omit {
            let start = payload.count
            for x in [Float16(-1), 0.5, 2] {
                var bits = (bf16 ? UInt16(Float(x * scale).bitPattern >> 16) : (x * scale).bitPattern).littleEndian; withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
            }
            header[key] = ["dtype": bf16 ? "BF16" : "F16", "shape": [3], "data_offsets": [start, payload.count]]
            traces[key + "/output"] = ["shape": [3], "count": 3, "finite": true, "max_abs": .number(maximum), "clamp_count": .number(Double(clamps))]
        }
        let json = try JSONSerialization.data(withJSONObject: header)
        var n = UInt64(json.count).littleEndian
        var data = withUnsafeBytes(of: &n) { Data($0) }; data.append(json); data.append(payload)
        try data.write(to: dir.appendingPathComponent("outputs.safetensors"))
        try WireJSON.encode(traces).write(to: dir.appendingPathComponent("activations.json"))
    }
    @Test func warmSummarySeparatesMedianAndSpread() {
        let summary = IntakeTimingSummary(samples: [4, 2, 9, 3])
        #expect(summary.median == 3.5)
        #expect(summary.json["spread_s"]?.number == 7)
    }
    @Test func stagingRecoveryPreservesLiveLeaseAndReclaimsAbandonedLease() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let live = try ConversionStaging(root.appendingPathComponent(".cast-" + UUID().uuidString))
        let dead = root.appendingPathComponent(".cast-" + UUID().uuidString)
        try FileManager.default.createDirectory(at: dead, withIntermediateDirectories: true)
        try Data().write(to: dead.appendingPathComponent(".owner-lock"))
        let foreign = root.appendingPathComponent(".cast-user-data")
        try FileManager.default.createDirectory(at: foreign, withIntermediateDirectories: true)
        try Data().write(to: foreign.appendingPathComponent(".owner-lock"))
        try ConversionStaging.recover(in: root)
        #expect(FileManager.default.fileExists(atPath: live.root.path))
        #expect(!FileManager.default.fileExists(atPath: dead.path))
        #expect(FileManager.default.fileExists(atPath: foreign.path))
        #expect(throws: (any Error).self) { try ConversionStaging.preflight(in: root, sourceBytes: Int64.max) }
        withExtendedLifetime(live) {}
    }
}

extension IntakeComponentGateTests {
    @Test func d1ProbabilityGateChecksFloorAndExactCases() throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let a = root.appendingPathComponent("a"), b = root.appendingPathComponent("b")
        try probabilityCapture(a, offset: 0)
        try probabilityCapture(b, offset: 0.016)
        let component: JSONObject = ["component_gate_passed": true]
        #expect(try ModelIntakeCommand.decisionGate(component, reference: a, candidate: b)["component_gate_passed"]?.bool == true)
        try probabilityCapture(b, offset: 0.018)
        #expect(try ModelIntakeCommand.decisionGate(component, reference: a, candidate: b)["component_gate_passed"]?.bool == false)
        try probabilityCapture(b, offset: 0, missing: true)
        #expect(throws: (any Error).self) { try ModelIntakeCommand.decisionGate(component, reference: a, candidate: b) }
        #expect(ModelIntakeCommand.d1ComponentKeys.count == 93)
    }
    private func probabilityCapture(_ directory: URL, offset: Float, missing: Bool = false) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let names = ["case-0/question-0", "case-1/question-0", "case-1/question-1"]
        var header: [String: Any] = [:], payload = Data()
        for name in names where !missing || name != names.last {
            let start = payload.count
            for value in [Float(0.5) + offset, Float(0.5) - offset] {
                var bits = value.bitPattern.littleEndian
                withUnsafeBytes(of: &bits) { payload.append(contentsOf: $0) }
            }
            header[name] = ["dtype": "F32", "shape": [2], "data_offsets": [start, payload.count]]
        }
        let encoded = try JSONSerialization.data(withJSONObject: header, options: [.sortedKeys])
        var length = UInt64(encoded.count).littleEndian
        var bytes = withUnsafeBytes(of: &length) { Data($0) }; bytes.append(encoded); bytes.append(payload)
        try bytes.write(to: directory.appendingPathComponent("probabilities.safetensors"))
    }
}
