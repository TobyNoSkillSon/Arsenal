import Foundation
import IntegrationKit
import MLX
import MLXNN
import ModelPrecision
import ArsenalWire
import Testing
@testable import AudioWorker

@Suite(.serialized) struct PrecisionTests {
    @Test(.heavy) func t5ClampsHalfActivationsLikeHuggingFace() throws {
        guard let path = (ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"]) else {
            throw audioFailure("The family GPU test needs the pinned no-fast-math metallib.")
        }
        GPU.metallib = URL(fileURLWithPath: path)
        // 70,000 overflows float16 to infinity: values are then held at 65,504 - 1,000, which float16 stores as 64,512.
        let overflowed = t5HalfClamp(MLXArray([Float(70000), -70000, 1]).asType(.float16)).asType(.float32).asArray(Float.self)
        #expect(overflowed == [64512, -64512, 1])
        let finite = t5HalfClamp(MLXArray([Float(60000), -2]).asType(.float16)).asType(.float32).asArray(Float.self)
        #expect(finite == [60000, -2])
        #expect(t5HalfClamp(MLXArray([Float(1e6)])).asArray(Float.self) == [1e6], "float32 is left as it is")
    }
    @Test(.heavy) func recurrentEquationsAndStoredTiers() throws {
        guard let path = (ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"]) else {
            throw audioFailure("The family GPU test needs the pinned no-fast-math metallib.")
        }
        GPU.metallib = URL(fileURLWithPath: path)
        let source = LSTM(inputSize: 64, hiddenSize: 64)
        try NativeParameters.initializeRandom(source)
        let raw = Dictionary(uniqueKeysWithValues: source.parameters().flattened())
        let native = AudioLSTM(inputSize: 64, hiddenSize: 64)
        let mapped = raw.map { name, value -> (String, MLXArray) in
            let key = name == "Wx" ? "x_proj.weight" : name == "Wh" ? "h_proj.weight" : "x_proj.bias"
            return (key, value)
        }
        try native.update(parameters: ModuleParameters.unflattened(mapped), verify: .all)
        let input = MLXArray(Array(0..<128).map { Float($0 % 13 - 6) / 20 }).reshaped(1, 2, 64).asType(.float16)
        let baseline = source(input), candidate = native(input)
        #expect((baseline.0 - candidate.0).abs().max().item(Float.self) < 0.002)
        #expect((baseline.1 - candidate.1).abs().max().item(Float.self) < 0.002)
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        let half = root.appendingPathComponent("half.safetensors")
        try save(arrays: Dictionary(uniqueKeysWithValues: mapped), url: half)
        for bits in [8, 4] {
            let archive = try TensorArchive(half), packed = root.appendingPathComponent("int\(bits).safetensors")
            let converted = try StreamedPrecision.convert(keys: archive.entries.keys.sorted(), read: archive.array, to: packed, half: .float16, bits: bits, eligible: audioEligible)
            #expect(converted.quantizedTensors == 2)
            let model = AudioLSTM(inputSize: 64, hiddenSize: 64)
            let coverage = try NativePrecisionTier("int\(bits)").load(model, archives: [packed], map: { ($0, $1) }, eligible: audioEligible)
            #expect(coverage.quantizedTensors == 2)
            let values = Dictionary(uniqueKeysWithValues: model.parameters().flattened())
            #expect(values["x_proj.weight"]?.dtype == .uint32)
            #expect(values["h_proj.weight"]?.dtype == .uint32)
            #expect(values["x_proj.bias"]?.dtype == .float16)
            #expect(values["x_proj.bias"]?.asArray(Float.self) == raw["bias"]?.asArray(Float.self))
            let result = model(input)
            #expect(MLX.all(MLX.isFinite(result.0)).item(Bool.self))
            #expect(result.0.shape == [1, 2, 64])
        }
    }
    @Test func checkpointAliases() {
        #expect(audioLSTMKey("audio_codec.decoder.lstm.layers.0.Wx") == "audio_codec.decoder.lstm.layers.0.x_proj.weight")
        #expect(audioLSTMKey("model_0.forward_lstms.1.bias") == "model_0.forward_lstms.1.x_proj.bias")
        #expect(audioLSTMKey("encoder.layers.0.bias") == "encoder.layers.0.bias")
    }
}
