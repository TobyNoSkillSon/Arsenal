// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import ArsenalWire

/// The pinned MLXVLM Qwen3-VL vision tower (`Qwen3VLVision.MLP`, Qwen3VL.swift:576, which Qwen3.5 reuses) builds every
/// block MLP with `GELU(approximation: .fast)`, i.e. x·σ(1.702x). Qwen3-VL and Qwen3.5 checkpoints declare the vision
/// activation in `vision_config.hidden_act` (`gelu_pytorch_tanh` in every pinned language folder). After the graph is
/// built, the declared activation replaces the hard-coded one; the merger's exact GELU is left alone.
enum VisionActivation {
    /// Transformers ACT2FN names for the GELU family, mapped to MLXNN's GELU variants.
    static func approximation(_ name: String) throws -> GELU.Approximation {
        switch name {
        case "gelu": return .none
        // gelu_new and gelu_fast are the same tanh formula written differently.
        case "gelu_pytorch_tanh", "gelu_new", "gelu_fast": return .precise
        case "quick_gelu": return .fast
        default: throw WorkerFailure(code: "unsupported_model", message: "Unsupported vision activation: \(name).")
        }
    }

    /// Replaces each `vision_tower.blocks.N.mlp.act` for Qwen3-VL/Qwen3.5 graphs; returns the number of replaced MLPs.
    @discardableResult
    static func apply(_ model: Module, config: Data, architecture: String) throws -> Int {
        guard ["qwen3_5", "qwen3_vl"].contains(architecture) else { return 0 }
        let root = try JSONSerialization.jsonObject(with: config) as? [String: Any] ?? [:]
        let vision = root["vision_config"] as? [String: Any] ?? [:]
        // Transformers' Qwen3VLVisionConfig and Qwen3_5VisionConfig default hidden_act is gelu_pytorch_tanh.
        let approximation = try approximation(vision["hidden_act"] as? String ?? "gelu_pytorch_tanh")
        var replaced = 0
        for (path, module) in model.namedModules() where isBlockMLP(path) {
            try module.update(modules: ModuleChildren(values: ["act": .value(GELU(approximation: approximation))]), verify: .none)
            replaced += 1
        }
        let depth = vision["depth"] as? Int ?? replaced
        guard replaced > 0, replaced == depth else {
            throw WorkerFailure(code: "unsupported_model", message: "Vision activation reached \(replaced) of \(depth) blocks.")
        }
        return replaced
    }

    static func isBlockMLP(_ path: String) -> Bool {
        let parts = path.split(separator: ".")
        return parts.count == 4 && parts[0] == "vision_tower" && parts[1] == "blocks" && Int(parts[2]) != nil && parts[3] == "mlp"
    }
}
