// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import Testing
import ModelPrecision
@testable import MLXAudioTTS
@testable import VireoEngine
@testable import VoicesWorker

@Suite(.serialized) struct VoiceIntakeAdapterTests {
    @Test func conversionDispatchResolvesEveryWholeVoiceRecipe() {
        for model in WholeVoiceConversion.models {
            for prefix in ["cast:", "intake-v1:", "mlx-affine-g64:"] { #expect(Runner.recipeModel(prefix + model) == model) }
            #expect(Runner.recipeModel("cast:" + model + ":bf16-truncation") == model)
            #expect(Runner.recipeModel("intake-v3:" + model + ":all") == model)
        }
    }
    @Test func mixedRecipesAreCanonicalAndInt4CannotRetainEligibleComponents() throws {
        let recipe = WholeVoiceConversion.recipe(model: "qwen3-tts-base", keeping: ["talker", "predictor"])
        #expect(recipe == "intake-v2:qwen3-tts-base:predictor,talker")
        #expect(Runner.recipeModel(recipe) == "qwen3-tts-base")
        #expect(try WholeVoiceConversion.keeping(recipe: recipe, model: "qwen3-tts-base", dtype: "int8") == ["talker", "predictor"])
        for bad in [recipe, "intake-v2:qwen3-tts-base:unmapped", "intake-v2:qwen3-tts-base:talker,talker"] {
            #expect(throws: (any Error).self) { try WholeVoiceConversion.keeping(recipe: bad, model: "qwen3-tts-base", dtype: "int4") }
        }
        #expect(WholeVoiceConversion.owner(file: "model.safetensors", key: "talker.code_predictor.model.layers.0.weight") == "predictor")
        #expect(WholeVoiceConversion.owner(file: "model.safetensors", key: "talker.model.text_embedding.weight") == "text-embedding")
        #expect(WholeVoiceConversion.owner(file: "model.safetensors", key: "talker.code_predictor.model.codec_embedding.0.weight") == "predictor-embedding")
        #expect(WholeVoiceConversion.owner(file: "model.safetensors", key: "mimi.decoder_transformer.transformer.layers.0.weight") == "mimi-transformer-upsampling")
    }
    @Test(.heavy) func affinePackingAndPointwiseLayoutsMatchIndependentFixedTensorReference() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.gpu) {
            func fixed(_ i: Int) -> Float {
                var v = UInt64(i) &+ 0x9e3779b97f4a7c15
                v = (v ^ (v >> 30)) &* 0xbf58476d1ce4e5b9
                v = (v ^ (v >> 27)) &* 0x94d049bb133111eb
                return Float((v ^ (v >> 31)) & 65535) / 32768 - 1
            }
            for dtype in [DType.bfloat16, .float16] {
                let raw = MLXArray((0..<65536).map { fixed($0) }, [256, 256]).asType(dtype)
                let (weight, scales, biases) = MLX.quantized(raw, groupSize: 64, bits: 8, mode: .affine)
                let audit = try WholeVoiceConversion.packingAudit(dense: raw, packed: weight, scales: scales, biases: biases!)
                #expect(audit["manual_unpack_max_abs"] == 0)
                #expect((audit["weight_relative_l2"] ?? 1) < 0.03)
                let x = MLXArray((0..<768).map { fixed($0 + 100000) }, [1, 3, 256]).asType(dtype)
                let conv = PointwiseFixture()
                conv.conv.weight._updateInternal(raw.expandedDimensions(axis: 1))
                let reference = conv.conv(x)
                let arrays = ["conv.weight": weight, "conv.scales": scales, "conv.biases": biases!]
                #expect(try StoredVoiceWeights.prepare(conv, weights: arrays) == 1)
                try conv.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
                let packed = conv.conv(x)
                let decoded = dequantized(weight, scales: scales, biases: biases!, groupSize: 64, bits: 8)
                let explicit = matmul(x, decoded.T)
                let a = reference.asArray(Float.self), b = packed.asArray(Float.self), d = explicit.asArray(Float.self)
                let energy = zip(a, a).reduce(0.0) { $0 + Double($1.0) * Double($1.1) }
                let rel = sqrt(zip(a, b).reduce(0.0) { $0 + pow(Double($1.0 - $1.1), 2) } / energy)
                let kernelRel = sqrt(zip(b, d).reduce(0.0) { $0 + pow(Double($1.0 - $1.1), 2) } / energy)
                print("VOICE_PACK_AUDIT \(dtype) weight=\(audit["weight_relative_l2"]!) output=\(rel) qmm_vs_dequant=\(kernelRel)")
                #expect(rel < 0.03 && kernelRel < 0.015)
            }
        }
    }
    @Test(.heavy) func pocketDecoderBoundaryIsTheRealNextInputAndPreservesServedPCM() throws {
        _ = NativePrecisionTestRuntime.ready
        let config =
            #"{"sample_rate":8,"channels":1,"frame_rate":4,"seanet":{"dimension":64,"channels":1,"n_filters":4,"n_residual_layers":1,"ratios":[2],"kernel_size":3,"residual_kernel_size":3,"last_kernel_size":3,"dilation_base":2,"pad_mode":"constant","compress":2},"transformer":{"d_model":64,"input_dimension":64,"output_dimensions":[64],"num_heads":4,"num_layers":1,"layer_scale":0.01,"context":8,"dim_feedforward":128,"max_period":10000},"quantizer":{"dimension":64,"output_dimension":64}}"#
        let parsed = try JSONDecoder().decode(PocketTTSMimiConfig.self, from: Data(config.utf8))
        try Device.withDefaultDevice(.gpu) {
            let model = MimiAdapter.fromConfig(parsed)
            try model.update(parameters: ModuleParameters.unflattened(model.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }), verify: .all)
            let latent = MLXArray.ones([1, 64, 2], dtype: .bfloat16)
            let served = model.decodeFromLatent(latent)
            eval(served)
            model.resetState()
            let nextInput = model.decoderInput(latent)
            let diagnostic = model.decoder(nextInput)
            #expect(nextInput.shape == [1, 64, 2] && nextInput.dtype == .bfloat16)
            #expect(diagnostic.shape == served.shape && diagnostic.asArray(Float.self) == served.asArray(Float.self))
        }
    }
    @Test(.heavy) func immutableInputsAndIndependentComponentReplay() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try Device.withDefaultDevice(.cpu) {
            let fixture = root.appendingPathComponent("fixture")
            let reference = try VoiceIntakeCapture(fixture: fixture, create: true)
            let mutable = MLXArray([Float(1), 2, 3, 4], [1, 4]).asType(.bfloat16)
            let x = try reference.input("hidden") { mutable }
            mutable[0, 0] = MLXArray(Float(999)).asType(.bfloat16)
            eval(mutable)
            #expect(x.asArray(Float.self) == [1, 2, 3, 4])
            reference.component("projection") { x * 2 }
            #expect(try reference.finish(fixture: fixture, output: root.appendingPathComponent("ref")) == 1)
            let bytes = try Data(contentsOf: fixture.appendingPathComponent("inputs.safetensors"))
            let candidate = try VoiceIntakeCapture(fixture: fixture, create: false)
            var called = false
            let saved = try candidate.input("hidden") {
                called = true; return MLXArray([Float(999)])
            }
            #expect(!called && saved.shape == [1, 4])
            #expect(saved.asArray(Float.self) == [1, 2, 3, 4])
            #expect(throws: (any Error).self) { try candidate.input("missing") { saved } }
            #expect(try Data(contentsOf: fixture.appendingPathComponent("inputs.safetensors")) == bytes)
            #expect(throws: (any Error).self) { try reference.finish(fixture: fixture, output: root.appendingPathComponent("again")) }
        }
    }
    @Test(.heavy) func selectiveDiagnosisExecutesOnlyRequestedComponentsAndKeepsItsReferenceImmutable() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let fixture = root.appendingPathComponent("fixture"), ref = root.appendingPathComponent("reference")
        let reference = try VoiceIntakeCapture(fixture: fixture, create: true)
        _ = try reference.input("input") { MLXArray([Float(1), 2]).asType(.bfloat16) }
        reference.component("a") { MLXArray([Float(3), 4]).asType(.bfloat16) }
        reference.component("b") { MLXArray([Float(5), 6]).asType(.bfloat16) }
        _ = try reference.finish(fixture: fixture, output: ref)
        let before = try Data(contentsOf: ref.appendingPathComponent("outputs.safetensors"))
        let trial = try VoiceIntakeCapture(fixture: fixture, create: false, only: ["b"], reference: ref)
        var called = false
        let unchanged = trial.component("a") {
            called = true; return MLXArray([Float(999)])
        }
        let changed = trial.component("b") { MLXArray([Float(7), 8]).asType(.bfloat16) }
        #expect(!called && unchanged.asArray(Float.self) == [3, 4] && changed.asArray(Float.self) == [7, 8])
        #expect(try trial.finish(fixture: fixture, output: root.appendingPathComponent("diagnostic")) == 2)
        #expect(try Data(contentsOf: ref.appendingPathComponent("outputs.safetensors")) == before)
        #expect(Set(trial.traces.keys) == Set(reference.traces.keys))
    }
    @Test(.heavy) func inactiveCaptureNeverEvaluatesClampReadback() throws {
        _ = NativePrecisionTestRuntime.ready
        let previous = VoiceIntakeCapture.active
        VoiceIntakeCapture.active = nil
        defer { VoiceIntakeCapture.active = previous }
        var reductions = 0
        func readback() -> Int { reductions += 1; return 7 }
        VoiceIntakeCapture.observe("pre-clip", MLXArray(Float(2)), clamps: readback())
        #expect(reductions == 0)
        let capture = try VoiceIntakeCapture(fixture: FileManager.default.temporaryDirectory, create: true)
        capture.prefix = "decoder"
        VoiceIntakeCapture.active = capture
        VoiceIntakeCapture.observe("pre-clip", MLXArray(Float(2)), clamps: readback())
        #expect(reductions == 1)
        #expect((capture.traces["decoder/pre-clip"] as? [String: Any])?["clamp_count"] as? Int == 7)
    }
    @Test(.heavy) func stockAndCachedDepthTracesUseTheSameLastTokenShape() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        let capture = try VoiceIntakeCapture(fixture: root, create: true)
        capture.prefix = "depth"
        VoiceIntakeCapture.active = capture
        defer { VoiceIntakeCapture.active = nil }
        Device.withDefaultDevice(.cpu) {
            let prefill = MLXArray([Float(1), 2, 3, 4, 5, 6], [1, 2, 3]).asType(.bfloat16)
            VoiceIntakeCapture.observeLastToken("layer-0", prefill)
            let stock = capture.traces["depth/layer-0"] as? [String: Any]
            VoiceIntakeCapture.observeLastToken("layer-0", prefill[0..., -1, 0...])
            let cached = capture.traces["depth/layer-0"] as? [String: Any]
            #expect(stock?["shape"] as? [Int] == [1, 3] && cached?["shape"] as? [Int] == [1, 3])
            #expect(stock?["max_abs"] as? Double == cached?["max_abs"] as? Double)
        }
    }

    @Test(.heavy) func flowHeadAndTimeEmbeddingsKeepSelectedHalfArithmetic() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu) {
            for dtype in [DType.bfloat16, .float16] {
                let flow = SimpleMLPAdaLN(inChannels: 64, modelChannels: 64, outChannels: 64, condChannels: 64, numResBlocks: 1, numTimeConds: 2)
                let weights = flow.parameters().flattened().map { ($0.0, $0.1.asType(dtype)) }
                try flow.update(parameters: ModuleParameters.unflattened(weights), verify: .none)
                let y = flow(MLXArray.ones([1, 64], dtype: dtype), MLXArray.zeros([1, 1], dtype: dtype), MLXArray.ones([1, 1], dtype: dtype), MLXArray.ones([1, 64], dtype: dtype))
                #expect(y.dtype == dtype && y.shape == [1, 64])
            }
        }
    }
    @Test(.heavy) func quantizedPocketAllowsEmptyTextAndLatentPrefixes() async throws {
        _ = NativePrecisionTestRuntime.ready
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: root) }
        try Data(#"{"model":{"unk_id":0,"vocab":[["<unk>",0],["a",0]]}}"#.utf8).write(to: root.appendingPathComponent("tokenizer.json"))
        let conditioner = try await LUTConditioner(nBins: 3, modelFolder: root, dim: 64, outputDim: 64)
        try Device.withDefaultDevice(.cpu) {
            let flow = FlowLMModel(
                conditioner: conditioner,
                flowNet: SimpleMLPAdaLN(inChannels: 64, modelChannels: 64, outChannels: 64, condChannels: 64, numResBlocks: 1, numTimeConds: 2),
                transformer: PocketStreamingTransformer(dModel: 64, numHeads: 4, numLayers: 1, dimFeedforward: 128, maxPeriod: 10000, layerScale: nil), dim: 64, ldim: 64)
            var weights = Dictionary(uniqueKeysWithValues: flow.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) })
            for key in ["input_linear.weight", "conditioner.embed.weight"] {
                let raw = try #require(weights[key])
                let (weight, scales, biases) = MLX.quantized(raw, groupSize: 64, bits: 8, mode: .affine)
                weights[key] = weight
                let prefix = String(key.dropLast(7))
                weights[prefix + ".scales"] = scales; weights[prefix + ".biases"] = biases
            }
            #expect(try StoredVoiceWeights.prepare(flow, weights: weights) == 2)
            try flow.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            let empty = conditioner(TokenizedText(tokens: MLXArray.zeros([1, 0], type: Int32.self)))
            #expect(empty.shape == [1, 0, 64] && empty.dtype == .bfloat16)
            let result = flow(
                sequence: MLXArray.zeros([1, 0, 64], dtype: .bfloat16),
                textEmbeddings: MLXArray.ones([1, 1, 64], dtype: .bfloat16), cache: flow.makeCache(),
                lsdDecodeSteps: 1, temperature: 0.7, noiseClamp: nil, eosThreshold: -4)
            #expect(result.0.shape == [1, 64] && result.0.dtype == .bfloat16)
        }
    }

    @Test(.heavy) func sparsePackInMixedModuleArrayPreservesTheUnpackedFirstElement() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu) {
            let block = PocketResBlock(channels: 256)
            let source = try #require(block.adaLN_modulation[1] as? Linear)
            let (weight, scales, biases) = MLX.quantized(source.weight.asType(.bfloat16), groupSize: 64, bits: 8, mode: .affine)
            let weights = ["adaLN_modulation.1.weight": weight, "adaLN_modulation.1.scales": scales, "adaLN_modulation.1.biases": biases!]
            #expect(try StoredVoiceWeights.prepare(block, weights: weights) == 1)
            #expect(block.adaLN_modulation[0] is SiLU)
            #expect(block.adaLN_modulation[1] is QuantizedLinear)
        }
    }
    @Test(.heavy) func batchedHeadsAreContiguousPackedAndIndexedByTheirRealStoredKeys() throws {
        _ = NativePrecisionTestRuntime.ready
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try Device.withDefaultDevice(.cpu) {
            let source = root.appendingPathComponent("source"), output = root.appendingPathComponent("int8")
            try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
            try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
            let writer = try TensorWriter(source.appendingPathComponent("model.safetensors"))
            let values: [Float] = (0..<131584).map { index in Float(index % 31 - 15) / 16 }
            let raw = MLXArray(values, [2, 256, 257]).asType(.bfloat16)
            try writer.append("depth_decoder.codebooks_head.weight", raw); try writer.finish()
            try Data(#"{"weight_map":{"depth_decoder.codebooks_head.weight":"model.safetensors"}}"#.utf8).write(to: source.appendingPathComponent("model.safetensors.index.json"))
            try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: source.appendingPathComponent("model.safetensors.index.json").path)
            let files = ["config.json", "model.safetensors", "model.safetensors.index.json"]
            let report = try WholeVoiceConversion.convert(source: source, files: files, model: "breeze-tts-2", dtype: "int8", half: "bf16", into: output)
            #expect(report.quantized.count == 2)
            let archive = try TensorArchive(output.appendingPathComponent("model.safetensors"))
            let index = try JSONSerialization.jsonObject(with: Data(contentsOf: output.appendingPathComponent("model.safetensors.index.json"))) as? [String: Any]
            let mapped = try #require(index?["weight_map"] as? [String: String])
            #expect(Set(mapped.keys) == Set(archive.entries.keys) && mapped.count == 6)
            #expect(archive.entries["depth_decoder.codebooks_head.heads.1.weight"]?.shape == [257, 64])
        }
    }

    final class PointwiseFixture: Module {
        @ModuleInfo var conv: Conv1d
        override init() { _conv.wrappedValue = Conv1d(inputChannels: 256, outputChannels: 256, kernelSize: 1, bias: false) }
    }
    @Test(.heavy) func voiceCodecQuantizerStartsItsResidualSumInTheStoredHalf() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu) {
            for dtype in [DType.bfloat16, .float16] {
                let quantizer = ResidualVectorQuantization(numQuantizers: 2, dim: 4, codebookSize: 8)
                for layer in quantizer.layers {
                    layer.codebook.embedding_sum._updateInternal(MLXArray.ones([8, 4], dtype: dtype))
                    layer.codebook.cluster_usage._updateInternal(MLXArray.ones([8], dtype: dtype))
                    layer.codebook.updateInPlace()
                }
                let out = quantizer.decode(MLXArray.zeros([2, 1, 3], type: Int32.self))
                #expect(out.dtype == dtype && out.shape == [1, 4, 3])
                #expect(out.asArray(Float.self).allSatisfy { $0 == 2 })
            }
        }
    }

    @Test(.heavy) func speakerPointwiseConvolutionUsesStoredPacksAtEveryTimePosition() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu) {
            #expect(WholeVoiceConversion.skipReason(key: "speaker_encoder.mfa.conv.weight", shape: [256, 1, 256]) == nil)
            #expect(WholeVoiceConversion.skipReason(key: "speaker_encoder.blocks.0.conv.weight", shape: [512, 5, 128]) != nil)
            #expect(WholeVoiceConversion.skipReason(key: "mimi.conv.weight", shape: [256, 1, 256]) != nil)
            let model = PointwiseFixture()
            let dense = MLXArray.ones([256, 256], dtype: .bfloat16)
            let (weight, scales, biases) = MLX.quantized(dense, groupSize: 64, bits: 8, mode: .affine)
            let arrays = ["conv.weight": weight, "conv.scales": scales, "conv.biases": biases!]
            #expect(try StoredVoiceWeights.prepare(model, weights: arrays) == 1)
            try model.update(parameters: ModuleParameters.unflattened(arrays), verify: .all)
            #expect(model.conv is StoredVoicePointwise)
            let output = model.conv(MLXArray.ones([1, 3, 256], dtype: .bfloat16))
            #expect(output.shape == [1, 3, 256] && output.dtype == .bfloat16)
            #expect(output.asArray(Float.self).allSatisfy { $0 == 256 })
        }
    }

    @Test(.heavy) func breezePackedTextTablePreservesItsSeparateEndOfInstructionEmbedding() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu) {
            let layer = BreezeTTSTextEmbedding(vocabSize: 256, dimensions: 256, eoiTokenIndex: 5)
            let (weight, scales, biases) = MLX.quantized(MLXArray.ones([256, 256], dtype: .bfloat16), groupSize: 64, bits: 8, mode: .affine)
            let weights = [
                "table.weight": weight, "table.scales": scales, "table.biases": biases!, "eoi_embedding": MLXArray.full([256], values: MLXArray(Float(2)).asType(.bfloat16))
            ]
            #expect(try StoredVoiceWeights.prepare(layer, weights: weights) == 1)
            try layer.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
            let result = layer(MLXArray([Int32(1), 5]))
            #expect(result[0].asArray(Float.self).allSatisfy { $0 == 16 })
            #expect(result[1].asArray(Float.self).allSatisfy { $0 == 2 })
        }
    }

    @Test(.heavy) func breezeBatchedOutputHeadsMapToPackableLinears() throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu) {
            let weights = BreezeTTSModel.sanitize(weights: ["depth_decoder.codebooks_head.weight": MLXArray.ones([3, 64, 65], dtype: .bfloat16)])
            #expect(weights.count == 3)
            #expect(weights["depth_decoder.codebooks_head.heads.2.weight"]?.shape == [65, 64])
            #expect(WholeVoiceConversion.skipReason(key: "speaker_proj_weight", shape: [1024, 512]) == nil)
            #expect(WholeVoiceConversion.skipReason(key: "depth_decoder.codebooks_head.heads.14.weight", shape: [2051, 1024]) == nil)
        }
    }
}
