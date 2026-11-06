// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import MLXNN
@testable import StableAudio3
import SoundEngine
import VireoCore

@Test(.heavy) func sameSStoredHalfAndAffineMetadataAreNeverWidened() throws {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
    for dtype: DType in [.float16, .bfloat16] {
        var config = SAMESConfig()
        config.latentDim = 4; config.dim = 8; config.numHeads = 2; config.headDim = 4
        config.ropeDims = 2; config.numBlocks = 2; config.ffInner = 16; config.outChannels = 4
        let decoder = SAMESDecoder(config)
        // Deterministic small learned tensors; includes shifted and unshifted attention, convolution and residuals.
        let weights = Dictionary(
            uniqueKeysWithValues: decoder.parameters().flattened().map { key, value in
                (key, MLXArray.full(value.shape, values: MLXArray(Float(0.01)), dtype: dtype))
            })
        try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
        let mapped = sa3SAMESDecoderWeights(
            [
                "mapping.weight": MLXArray.ones([4, 8, 3], dtype: dtype),
                "project_in.scales": MLXArray.ones([8, 1], dtype: dtype),
                "project_in.weight": MLXArray.zeros([8, 1], dtype: .uint32)
            ], config: config)
        #expect(mapped["mapping.weight"]?.shape == [4, 3, 8])
        #expect(mapped["mapping.weight"]?.dtype == dtype && mapped["project_in.scales"]?.dtype == dtype)
        #expect(mapped["project_in.weight"]?.dtype == .uint32)
        var traces: [DType] = []
        decoder.activationTrace = { _, value in traces.append(value.dtype) }
        let output = sa3DecodeLatents(decoder, MLXArray.ones([1, 4, 14], dtype: .float32))
        eval(output)
        #expect(output.dtype == dtype && output.shape == [1, 4, 224])
        #expect(!traces.isEmpty && traces.allSatisfy { $0 == dtype })
        #expect(output.asArray(Float.self).allSatisfy { $0.isFinite })
        #expect(decoder.parameters().flattened().allSatisfy { $0.1.dtype == dtype })
    }
}

@Test(.heavy) func sharedDecoderDispatchPreservesMediumFixedTensorOutputs() throws {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
    var config = SAMELConfig()
    config.latentDim = 4; config.dim = 8; config.numHeads = 2; config.headDim = 4
    config.ropeDims = 2; config.numBlocks = 1; config.ffInner = 16; config.outChannels = 4; config.sinStartBlock = 0
    let decoder = SAMELDecoder(config)
    let weights = Dictionary(
        uniqueKeysWithValues: decoder.parameters().flattened().map { key, value in
            (key, MLXArray.full(value.shape, values: MLXArray(Float(0.01)), dtype: .float16))
        })
    try decoder.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    let input = MLXArray.ones([1, 4, 2], dtype: .float32)
    let original = decoder(input), captured = decoder.decodeWindow(input), served = sa3DecodeLatents(decoder, input)
    eval(original, captured, served)
    #expect(original.shape == [1, 4, 32] && original.dtype == .float16)
    #expect(original.asArray(Float.self) == captured.asArray(Float.self))
    #expect(original.asArray(Float.self) == served.asArray(Float.self))
}

@Test(.heavy) func sameSRejectsUnconvertedFP32BeforeConstructingAModel() throws {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
    let file = FileManager.default.temporaryDirectory.appendingPathComponent("same-s-legacy-" + UUID().uuidString + ".safetensors")
    defer { try? FileManager.default.removeItem(at: file) }
    try save(arrays: ["running_std": MLXArray([Float(1)])], url: file)
    #expect(throws: (any Error).self) { try sa3LoadSAMESDecoder(file) }
}

@Test(.heavy, arguments: ["stable-audio-3-small-sfx", "stable-audio-3-medium"])
func soundQuantizationRebuildsReadonlySourceCoverage(id: String) throws {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
    let root = FileManager.default.temporaryDirectory.appendingPathComponent("same-s-conversion-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source"), output = root.appendingPathComponent("output")
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    let file = source.appendingPathComponent("fixture.safetensors"), coverage = file.appendingPathExtension("coverage.json")
    try save(
        arrays: [
            "project_in.weight": MLXArray.ones([256, 256], dtype: .float16),
            "cond.seconds_total_weight": MLXArray.ones([768, 256], dtype: .float16)
        ], url: file)
    let original = Data("{\"quantized_tensors\":0}".utf8)
    try original.write(to: coverage)
    try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: coverage.path)
    let model = ModelSource(
        id: id, engine: "stable-audio-3", directory: source, precision: "fp16", options: ["variant": id == "stable-audio-3-small-sfx" ? "sm-sfx" : "medium", "intake": "1"])
    try SoundEngines.convert(model, dtype: "int8", into: output)
    let receipt = try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent(coverage.lastPathComponent))) as? [String: Any]
    #expect(receipt?["quantized_tensors"] as? Int == (id == "stable-audio-3-small-sfx" ? 2 : 1))
    let stored = try loadArrays(url: output.appendingPathComponent("fixture.safetensors"))
    if id == "stable-audio-3-small-sfx" {
        #expect(stored["cond.seconds_total.weight"]?.dtype == .uint32)
        #expect(stored["cond.seconds_total.scales"]?.dtype == .float16 && stored["cond.seconds_total_weight"] == nil)
    } else {
        #expect(stored["cond.seconds_total_weight"]?.dtype == .float16 && stored["cond.seconds_total.weight"] == nil)
    }
    #expect(try Data(contentsOf: coverage) == original)
}

@Test(.heavy) func smallDurationProjectionPacksWhileMediumKeepsItsOriginalFixedTensorOutput() throws {
    if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
    let weight = MLXArray((0..<(768 * 256)).map { Float($0 % 17 + 1) / 1000 }).reshaped([768, 256]).asType(.float16)
    let bias = MLXArray.zeros([768], dtype: .float16)
    let medium = SecondsTotalEmbedder(weight: weight, bias: bias)
    let features = expoFourierFeaturesSeconds(MLXArray([Float(5) / 384]), dim: 256)
    let original = (MLX.matmul(features, weight.transposed()) + bias).expandedDimensions(axis: 1)
    let mediumOutput = medium(5)
    eval(original, mediumOutput)
    #expect(original.asArray(Float.self) == mediumOutput.asArray(Float.self))
    #expect(mediumOutput.dtype == .float32)
    let (packed, scales, offsets) = MLX.quantized(weight, groupSize: 64, bits: 8)
    let small = SecondsTotalEmbedder(weight: weight, bias: bias, halfOperands: true)
    let quant = SecondsTotalEmbedder(weight: packed, bias: bias, scales: scales, affineBiases: offsets, quantBits: 8, halfOperands: true)
    let a = small(5), b = quant(5); eval(a, b)
    #expect(a.dtype == .float16 && b.dtype == .float16 && quant.weight.dtype == .uint32)
    #expect(a.shape == [1, 1, 768] && b.shape == a.shape)
    #expect(b.asArray(Float.self).allSatisfy { $0.isFinite })
}
