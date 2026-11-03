// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalKernels
import VerdictSmallMGEMM
@testable import VerdictEngine

/// CPU-only: the small-M gate credits a pass only when the folder tile kernel ran (no MLXArray, no dispatch), and the
/// load-time gates key their verdicts by the weights they ran on.
@Suite(.serialized) struct KernelGateTests {
    func tile(_ id: String) throws -> FolderMLXKernel {
        let spec = KernelSpec(
            id: id, mode: .fast, replaces: "linear.down", entry: "smallm_tile", files: ["/k/t.metal"],
            inputs: ["x", "w", "bias", "res"], outputs: ["out"])
        let file = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("arsenal-decisions-gate-\(UUID().uuidString)/kernels.json")
        return try FolderMLXKernel(try LoadedKernel(spec: spec, header: "", body: "out[0] = x[0];"), verdicts: KernelVerdicts(file: file))
    }

    @Test(.heavy) func smallMPassNeedsTheFolderKernelToRun() throws {
        defer { SmallMGEMM.tile = nil }
        SmallMGEMM.tile = nil
        #expect(SmallMGate.engaged(nil) == "no folder tile kernel")
        // A failing check keeps its own reason.
        #expect(SmallMGate.engaged("library self-test: m16") == "library self-test: m16")
        // Loaded but never dispatched: every call fell back to stock MLX, so the "with kernel" side measured stock.
        let idle = try tile("idle")
        SmallMGEMM.tile = idle
        #expect(SmallMGate.engaged(nil, since: 0) == "the check never dispatched the folder kernel")
        // A preflight failure (here: an argument mismatch) refuses the pass with that reason.
        _ = idle.preflight(inputs: [], template: [], outputDTypes: [], threadGroup: (1, 1, 1))
        #expect(SmallMGate.engaged(nil)?.hasPrefix("folder kernel preflight: Kernel idle called with 0 inputs") == true)
    }

    @Test func weightsIdentityNamesRevisionAndCheckpoint() throws {
        let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("arsenal-decisions-weights-\(UUID().uuidString)")
        let a = root.appendingPathComponent("snapshots/laya-english/abc123/fp16"), b = root.appendingPathComponent("snapshots/laya-english/def456/fp16")
        for dir in [a, b] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
            try Data(count: 16).write(to: dir.appendingPathComponent("model.safetensors"))
        }
        let ia = LayaModel.weightsIdentity(a)
        #expect(ia.hasPrefix("abc123/fp16:16:"))
        #expect(ia != LayaModel.weightsIdentity(b))
        try Data(count: 32).write(to: a.appendingPathComponent("model.safetensors"))
        #expect(LayaModel.weightsIdentity(a) != ia)
    }
}
