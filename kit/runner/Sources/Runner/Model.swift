// SPDX-License-Identifier: MIT
// The only file you write. Port the maker's PyTorch code (its modeling_*.py or model.py) to MLX here.
// Kit.swift has the helpers: Params, Weights.load/apply, Images, Audio, Text.tokenizer, list().
import Foundation
import MLX
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.example"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["predict"]

final class Model {
    let net: Example

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, *.safetensors, tokenizer files) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        let config = try Files.json(weightsDir)
        net = Example(inputs: config["input_size"] as? Int ?? 4, outputs: config["num_labels"] as? Int ?? 2)
        try Weights.apply(try Weights.load(weightsDir), to: net)
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let values = try params.numbers("values")
        let logits = net(MLXArray(values.map(Float.init)).asType(.float16).expandedDimensions(axis: 0))
        let probabilities = softmax(logits.asType(.float32), axis: -1)[0]
        return ["probabilities": list(probabilities)]
    }
}

/// Replace with the real network. Property names must match the checkpoint's key names
/// (use @ModuleInfo(key: "...") when they differ), so Weights.apply loads it with no renaming.
final class Example: Module, UnaryLayer {
    @ModuleInfo(key: "classifier") var classifier: Linear
    init(inputs: Int, outputs: Int) { _classifier.wrappedValue = Linear(inputs, outputs) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { classifier(x) }
}
