// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import RIFENative
import MLX
import MLXNN
import ArsenalWire

final class RIFEAdapter: NativeModel {
    let architecture = "rife"
    let operations: Set<String> = ["interpolate_video"]
    private var model: RIFEModel?
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] { model.map { Dictionary(uniqueKeysWithValues: $0.flownet.parameters().flattened()) } ?? [:] }
    // RIFE consists entirely of spatial kernels and vectors, with no eligible Linear/Embedding matrix.
    static func eligible(_ name: String, _ shape: [Int]) -> Bool { false }
    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let model = RIFEModel()
        if context.randomWeights {
            try NativeParameters.initializeRandom(model.flownet)
        } else {
            _ = try NativePrecisionTier(context.precision).load(
                model.flownet, archives: [context.weights.appendingPathComponent("model.safetensors")], map: { ($0, $1) }, eligible: Self.eligible)
        }
        model.flownet.train(false); self.model = model
    }
    private func interpolate(_ first: MLXArray, _ second: MLXArray, t: Double, cancellation: NativeCancellation) throws -> MLXArray {
        try cancellation.check()
        guard let model, first.shape == second.shape, t.isFinite, (0...1).contains(t) else {
            throw MediaBridge.error("Frames must have equal sizes and t must be between zero and one.")
        }
        if t == 0 { return first }; if t == 1 { return second }
        let output = model.inference(img0: first, img1: second, timestep: Float(t)); eval(output)
        try cancellation.check(); return output
    }
    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "interpolate_video", model != nil else { throw MediaBridge.error("RIFE is not loaded or operation is unsupported.") }
        return try InterpolationDriver.run(input, cancellation: cancellation, emit: emit) { a, b in
            { t in try self.interpolate(a, b, t: t, cancellation: cancellation) }
        }
    }
    func unload() { model = nil }
}
