// SPDX-License-Identifier: MIT
import Foundation
import Darwin
import Testing
import MLX
import MLXNN
import IntegrationKit
@testable import LanguageWorker

/// Mirrors the pinned Qwen3-VL tower's module paths: vision_tower.blocks.N.mlp.act plus a merger GELU that must stay exact.
private final class StandInMLP: Module {
    @ModuleInfo(key: "act") var act: GELU
    override init() { _act.wrappedValue = GELU(approximation: .fast); super.init() }
}
private final class StandInBlock: Module {
    @ModuleInfo(key: "mlp") var mlp: StandInMLP
    override init() { _mlp.wrappedValue = StandInMLP(); super.init() }
}
private final class StandInMerger: Module {
    @ModuleInfo(key: "act") var act: GELU
    override init() { _act.wrappedValue = GELU(); super.init() }
}
private final class StandInTower: Module {
    @ModuleInfo(key: "blocks") var blocks: [StandInBlock]
    @ModuleInfo(key: "merger") var merger: StandInMerger
    init(depth: Int) { _blocks.wrappedValue = (0..<depth).map { _ in StandInBlock() }; _merger.wrappedValue = StandInMerger(); super.init() }
}
private final class StandInGraph: Module {
    @ModuleInfo(key: "vision_tower") var tower: StandInTower
    init(depth: Int) { _tower.wrappedValue = StandInTower(depth: depth); super.init() }
}

struct VisionActivationTests {
    init() { MLXReady.ensure() }
    /// Transformers' gelu_pytorch_tanh: 0.5·x·(1 + tanh(√(2/π)·(x + 0.044715·x³))).
    private func tanhGELU(_ x: Double) -> Double { 0.5 * x * (1 + Darwin.tanh((2 / Double.pi).squareRoot() * (x + 0.044715 * x * x * x))) }

    @Test(.heavy) func declaredTanhActivationMatchesFormula() throws {
        NativeRuntime.prepare()
        let inputs: [Float] = [-4, -2.5, -1, -0.5, -0.1, 0, 0.1, 0.5, 1, 2.5, 4]
        let gelu = GELU(approximation: try VisionActivation.approximation("gelu_pytorch_tanh"))
        let actual = gelu(MLXArray(inputs)).asArray(Float.self)
        for (x, y) in zip(inputs, actual) { #expect(abs(Double(y) - tanhGELU(Double(x))) < 1e-5, "x=\(x)") }
        // The library default x·σ(1.702x) is measurably different (0.0046 at x=1), which is why the swap matters.
        let fast = GELU(approximation: .fast)(MLXArray([Float(1)])).item(Float.self)
        #expect(abs(Double(fast) - tanhGELU(1)) > 1e-3)
        #expect(throws: (any Error).self) { try VisionActivation.approximation("relu") }
    }

    @Test func replacesEveryBlockMLPOnlyForQwenVision() throws {
        let config = try JSONSerialization.data(withJSONObject: ["vision_config": ["hidden_act": "gelu_pytorch_tanh", "depth": 3]])
        let graph = StandInGraph(depth: 3)
        #expect(try VisionActivation.apply(graph, config: config, architecture: "gemma4") == 0)
        #expect(graph.tower.blocks.allSatisfy { $0.mlp.act.approximation == .fast })
        #expect(try VisionActivation.apply(graph, config: config, architecture: "qwen3_5") == 3)
        #expect(graph.tower.blocks.allSatisfy { $0.mlp.act.approximation == .precise })
        #expect(graph.tower.merger.act.approximation == GELU.Approximation.none)
        // A depth that does not match the reached blocks is a mapping failure, not a silent partial swap.
        let wrong = try JSONSerialization.data(withJSONObject: ["vision_config": ["hidden_act": "gelu_pytorch_tanh", "depth": 4]])
        #expect(throws: (any Error).self) { try VisionActivation.apply(StandInGraph(depth: 3), config: wrong, architecture: "qwen3_vl") }
    }
}
