// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ArsenalWire

final class SmartTurnAdapter: AudioAdapter {
    override var architecture: String { "smart_turn" }
    override var operations: Set<String> { ["end_of_turn"] }
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        var configuration = try config(context, as: SmartTurnConfig.self)
        configuration.dtype = "float16"
        let model = SmartTurnModel(configuration)
        try prepare(
            model, context, cancellation: cancellation,
            map: { name, tensor in
                guard let item = SmartTurnModel.sanitize([name: tensor]).first else { return nil }
                return (item.key, item.value)
            })
    }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "threshold"])
        guard operation == "end_of_turn", let model = graph as? SmartTurnModel else { throw audioFailure("Smart Turn is not loaded.") }
        let threshold = input["threshold"]?.number ?? 0.5
        guard threshold.isFinite, (0...1).contains(threshold) else { throw audioFailure("Threshold must be between zero and one.") }
        try cancellation.check()
        let audio = try audioInput(input, rate: model.config.processorConfig.samplingRate)
        let features = try model.prepareInputFeatures(MLXArray(audio.samples), sampleRate: audio.sampleRate)
        try cancellation.check()
        let output = model(features)
        eval(output)
        let probability = output.item(Float.self)
        guard probability.isFinite else { throw audioFailure("Turn prediction is non-finite.") }
        try cancellation.check()
        return ["probability": .number(Double(probability)), "prediction": .bool(Double(probability) > threshold)]
    }
}
