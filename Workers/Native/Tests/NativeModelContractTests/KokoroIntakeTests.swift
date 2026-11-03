// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import Testing
import ModelPrecision
@testable import MLXAudioTTS
@testable import VireoEngine

@Suite(.serialized) struct KokoroIntakeTests {
    @Test(.heavy) func auxiliaryNormAndAffineProjectionNeverRetainFloat32Weights() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.gpu) {
            let raw = MLXArray((0..<65536).map { Float($0 % 29 - 14) / 16 }, [512, 128])
            for half in ["fp16", "bf16"] {
                let dtype: DType = half == "fp16" ? .float16 : .bfloat16
                let norm = BARTLayerNorm(dimensions: 128, weight: MLXArray.ones([128], dtype: dtype), bias: MLXArray.zeros([128], dtype: dtype))
                #expect(norm.parameters().flattened().allSatisfy { $0.1.dtype == dtype })
                let view = BARTWeightView(["fc.weight": raw], precision: BARTPrecision(half: half, bits: 8))
                let packed = view.linear("fc")
                #expect(packed is QuantizedLinear)
                #expect(packed.parameters().flattened().allSatisfy { !$0.1.dtype.isFloatingPoint || $0.1.dtype == dtype })
                let selector = "matrix@" + BARTWeightView.file + ":fc.weight"
                #expect(view.receipt["quantized_parameters"] as? Int == 65536)
                let retained = BARTWeightView(["fc.weight": raw], precision: BARTPrecision(half: half, bits: 8, keeping: [selector]))
                #expect(!(retained.linear("fc") is QuantizedLinear))
                #expect(retained.receipt["fallback_parameters"] as? Int == 65536)
                let dense = Linear(weight: raw.asType(dtype), bias: nil)
                let array = FeedForward(linear1: packed, linear2: Linear(weight: MLXArray.ones([128, 512], dtype: dtype), bias: nil))
                let hashes = [selector: try #require((view.receipt["matrix_scales_sha256"] as? [String: String])?[selector])]
                let donor = FeedForward(linear1: dense, linear2: Linear(weight: MLXArray.ones([128, 512], dtype: dtype), bias: nil))
                let pairs = try VoiceMatrixDiagnosis.pairs(roots: [("g2p", array)], denseRoots: ["g2p": donor], hashes: hashes)
                #expect(pairs.count == 1)
                try StoredVoiceWeights.replace(pairs[0].root, path: pairs[0].path, with: pairs[0].half)
                #expect(!(array.linear1 is QuantizedLinear))
            }
        }
    }
    @Test(.heavy) func recurrentPaddingAlignmentAndHarmonicProjectionKeepTheLearnedHalf() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.gpu) {
            for dtype in [DType.float16, .bfloat16] {
                let recurrent = BiLSTM(inputSize: 8, hiddenSize: 4)
                recurrent.update(parameters: ModuleParameters.unflattened(recurrent.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
                for fused in [false, true] {
                    recurrent.fused = fused
                    let output = recurrent(MLXArray.ones([1, 3, 8], dtype: .float32)).0
                    #expect(output.dtype == dtype && output.shape == [1, 3, 8])
                }
                let conv = WeightNormedConv(inChannels: 4, outChannels: 4, kernelSize: 3)
                conv.update(parameters: ModuleParameters.unflattened(conv.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
                #expect(conv(MLXArray.ones([1, 5, 4], dtype: dtype)).dtype == dtype)
                let encoder = KokoroTextEncoder(channels: 8, kernelSize: 3, depth: 1, nSymbols: 32)
                encoder.update(parameters: ModuleParameters.unflattened(encoder.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
                let mask = MLXArray.zeros([1, 3], type: Bool.self), lengths = MLXArray([Int32(3)])
                let output = encoder(MLXArray([Int32(1), 2, 3]).reshaped(1, 3), inputLengths: lengths, mask: mask)
                #expect(output.dtype == dtype)
                let duration = KokoroDurationEncoder(styDim: 4, dModel: 8, nLayers: 1, dropout: 0)
                duration.update(parameters: ModuleParameters.unflattened(duration.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
                #expect(duration(MLXArray.ones([1, 8, 3], dtype: dtype), style: MLXArray.ones([1, 4], dtype: dtype), textLengths: lengths, mask: mask).dtype == dtype)
                let source = SourceModule(samplingRate: 24000, upsampleScale: 2)
                source.update(parameters: ModuleParameters.unflattened(source.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }))
                #expect(source(MLXArray.ones([1, 16, 1], dtype: dtype) * 120).0.dtype == dtype)
            }
        }
    }
    @Test(.heavy) func kokoroDerivedConvolutionLayoutIsNotSanitizedAgain() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try Device.withDefaultDevice(.gpu) {
            let source = root.appendingPathComponent("source"), target = root.appendingPathComponent("int8")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            try Data(#"{"arsenal_voice_intake":{"half":"fp16","bits":16}}"#.utf8).write(to: source.appendingPathComponent("config.json"))
            let writer = try TensorWriter(source.appendingPathComponent("kokoro-v1_0.safetensors"))
            try writer.append("text_encoder.cnn.0.0.weight_v", MLXArray.ones([16, 5, 16], dtype: .float16)); try writer.finish()
            _ = try WholeVoiceConversion.convert(
                source: source, files: ["config.json", "kokoro-v1_0.safetensors"], model: "kokoro-82m", dtype: "int8", half: "fp16", matrixPolicy: true, into: target)
            #expect(try TensorArchive(target.appendingPathComponent("kokoro-v1_0.safetensors")).entries["text_encoder.cnn.0.0.weight_v"]?.shape == [16, 5, 16])
        }
    }
}
