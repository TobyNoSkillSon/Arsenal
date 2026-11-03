// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
import MLXLMCommon
import ChatEngine
import ChatContract
import ArsenalCore

private final class TinyHalfLM: Module, LanguageModel {
    var calls: [Int] = []
    let embedding = MLXArray([Float16(2), 0, 0, 2, 1, 0, 0, 1], [4, 2])
    func newCache(parameters: GenerateParameters?) -> [KVCache] { [ArraysCache(size: 1)] }
    func prepare(_ input: LMInput, cache: [KVCache], windowSize: Int?) -> PrepareResult { .tokens(input.text) }
    func callAsFunction(_ input: MLXArray, cache: [KVCache]?) -> MLXArray {
        calls.append(input.size)
        let hidden = embedding[input]
        (cache?.first as? ArraysCache)?[0] = hidden.sum(axis: 1)
        return matmul(hidden, embedding.T)
    }
}

@Suite(.serialized) struct TeacherForcedProbeTests {
    init() {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] { GPU.metallib = URL(fileURLWithPath: path) }
        Device.setDefault(device: .gpu)
    }
    @Test(.heavy) func halfLogitsAreExtractedWithoutAGPUFloat32Cast() throws {
        let bits = MLXArray([UInt16(0x3f80), 0xc000, 0x4080]).view(dtype: .bfloat16)
        #expect(bits.dtype == .bfloat16)
        #expect(try ChatMeasurementProbe.hostLogits(bits) == [1, -2, 4])
        #expect(bits.dtype == .bfloat16)
        let half = MLXArray([Float16(1), -2, 4])
        #expect(try ChatMeasurementProbe.hostLogits(half) == [1, -2, 4])
    }
    @Test(.heavy) func timingWorkloadSelectsSeededDefaultCategoricalSampler() throws {
        let input = try MeasurementWorkload.request(kind: .chat, model: "fixture", batched: false, count: 1, directory: URL(fileURLWithPath: "/unused"))
        let object = try #require(JSONSerialization.jsonObject(with: JSONEncoder().encode(input.params)) as? [String: Any])
        let request = try ChatRequest(object)
        #expect(request.temperature == 0.6 && request.topP == 1 && request.seed == 0)
        let parameters = GenerateParameters(temperature: request.temperature, topP: request.topP, seed: request.seed)
        let first = parameters.sampler(), second = parameters.sampler()
        #expect(first is CategoricalSampler)
        #expect(!(first is ArgMaxSampler))
        let logits = MLXArray([Float16(1), 2, 3, 4], [1, 4])
        for _ in 0..<4 {
            let lhs = first.sample(logits: logits), rhs = second.sample(logits: logits)
            eval(lhs, rhs)
            #expect(lhs.asArray(Int32.self) == rhs.asArray(Int32.self))
        }
    }
    @Test(.heavy) func logitsReadLastPrefixThenEachFixedTokenWithExactCounts() throws {
        let model = TinyHalfLM()
        let measured = try ChatMeasurementProbe.capture(model: model, prefix: [0, 1], continuation: [2, 3])
        #expect(measured.count == 2 && measured.allSatisfy { $0.count == 4 })
        let expected = matmul(model.embedding[MLXArray([1, 2])], model.embedding.T); eval(expected)
        #expect(expected.dtype == .float16)
        let host = expected.asArray(Float16.self).map(Double.init)
        #expect(measured.flatMap { $0 } == host)
        #expect(throws: (any Error).self) { try ChatMeasurementProbe.capture(model: model, prefix: [], continuation: [1]) }
    }
}
