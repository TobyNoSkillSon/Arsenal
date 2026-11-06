// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
@testable import D1Engine

enum NativePrecisionTestRuntime {
    // MLX's global path setter is not thread-safe. Swift initializes this static exactly once.
    static let ready: Bool = {
        if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) }
        return true
    }()
}

@Test func concurrentPrecisionFixtureSetupIsSafe() async {
    let initialized = await withTaskGroup(of: Bool.self, returning: [Bool].self) { group in
        for _ in 0..<32 { group.addTask { NativePrecisionTestRuntime.ready } }
        var values: [Bool] = []
        for await value in group { values.append(value) }
        return values
    }
    #expect(initialized.count == 32 && initialized.allSatisfy { $0 })
}

struct D1PrecisionTests {
    private func cpu(_ body: () throws -> Void) throws {
        _ = NativePrecisionTestRuntime.ready
        try Device.withDefaultDevice(.cpu, body)
    }
    @Test(.heavy) func readoutReplaysExactHiddenInputRegardlessOfPredecessorAndRejectsMissingOrWrongShape() throws {
        try cpu {
            let key = "case-0/readout/input"
            let reference = MLXArray([Float(1), 2, 3, 4], [2, 2]).asType(.bfloat16)
            let capture = D1ComponentSession(inputs: [:], capture: true)
            _ = try capture.readout(key, hidden: reference) { $0 }
            #expect(Set(capture.inputs.keys) == [key] && capture.inputs[key]?.shape == [2, 2])
            let replay = D1ComponentSession(inputs: capture.inputs, capture: false)
            let changed = MLXArray.full([2, 2], values: MLXArray(Float(99))).asType(.bfloat16)
            let rows = MLX.eye(2).asType(.bfloat16)
            let actual = try replay.readout(key, hidden: changed) { D1Model.readout($0, rows: rows) }
            #expect(arrayEqual(actual, reference.asType(.float32)).item(Bool.self))
            #expect(replay.used == [key] && replay.validShapes)
            #expect(throws: (any Error).self) {
                try replay.readout(key, hidden: changed.reshaped(1, 4)) { _ in
                    Issue.record("readout ran before shape check"); return changed
                }
            }
            #expect(throws: (any Error).self) {
                try replay.readout(key, hidden: changed.asType(.float16)) { _ in
                    Issue.record("readout ran before dtype check"); return changed
                }
            }
            let missing = D1ComponentSession(inputs: [:], capture: false)
            #expect(throws: (any Error).self) {
                try missing.readout(key, hidden: changed) { _ in
                    Issue.record("readout ran without saved input"); return changed
                }
            }
        }
    }
    @Test(.heavy, arguments: [4, 8]) func packedMatmulEmbeddingAndFusedSlicesUseStoredGroups(bits: Int) throws {
        try cpu {
            let source = MLXArray((0..<(8 * 64)).map { Float($0 % 23 - 11) / 10 }, [8, 64]).asType(.bfloat16)
            let (w, s, b) = quantized(source, groupSize: 64, bits: bits)
            let matrix = try D1Matrix(weight: w, scales: s, biases: b)
            #expect(matrix.bits == bits && matrix.shape == [8, 64] && matrix.dtype == .bfloat16)
            let input = MLXArray.ones([2, 3, 64], dtype: .bfloat16)
            let expected = quantizedMM(input, w, scales: s, biases: b, transpose: true, groupSize: 64, bits: bits)
            #expect(max(abs(matrix(input) - expected)).item(Float.self) == 0)
            let fused = try D1Matrix.concatenate([matrix.rows(0..<3), matrix.rows(3..<8)])
            #expect(max(abs(fused(input) - expected)).item(Float.self) == 0)
            let ids = MLXArray([Int32(7), 0, 4])
            let dense = dequantized(w, scales: s, biases: b, groupSize: 64, bits: bits)
            #expect(max(abs(matrix[ids] - dense[ids])).item(Float.self) == 0)
            #expect(matrix.arrays.map(\.nbytes).reduce(0, +) < source.nbytes)
            let partial = matrix.rows(1..<5)(input)
            #expect(max(abs(partial - expected[.ellipsis, 1..<5])).item(Float.self) == 0)
        }
    }
    @Test(.heavy) func denseArithmeticIsUnchangedAndBadPacksAreRejected() throws {
        try cpu {
            let w = MLXArray.ones([4, 64], dtype: .bfloat16)
            let matrix = try D1Matrix(weight: w)
            let x = MLXArray.ones([2, 64], dtype: .bfloat16)
            #expect(max(abs(matrix(x) - matmul(x, w.T))).item(Float.self) == 0)
            let fused = try D1Matrix.concatenate([matrix, matrix])
            #expect(fused.shape == [8, 64] && fused.bits == 16)
            #expect(throws: (any Error).self) { try D1Matrix(weight: MLXArray.zeros([4, 8], dtype: .uint32)) }
            let (packed, scales, biases) = quantized(w, groupSize: 64, bits: 2)
            #expect(throws: (any Error).self) { try D1Matrix(weight: packed, scales: scales, biases: biases) }
        }
    }
}

@testable import VoicesWorker
@testable import SoundWorker

@Test func allVoiceAndSoundFamiliesAdmitInt4BeforeWeightsAreRead() throws {
    for spec in SpeechCatalog.all { try SpeechCatalog.checkPrecision("int4", spec: spec) }
    for model in ["re-use", "stable-audio-3-medium", "stable-audio-3-small-sfx"] { try checkDtype("int4", model: model) }
    #expect(throws: (any Error).self) { try checkDtype("fp32", model: "stable-audio-3-small-sfx") }
}

import ModelPrecision

@Test(.heavy, arguments: [4, 8]) func d1StreamingConversionWritesStandalonePackWithoutChangingSource(bits: Int) throws {
    _ = NativePrecisionTestRuntime.ready
    let root = URL(fileURLWithPath: NSTemporaryDirectory()).appendingPathComponent("arsenal-d1-pack-" + UUID().uuidString)
    defer { try? FileManager.default.removeItem(at: root) }
    let source = root.appendingPathComponent("source"), destination = root.appendingPathComponent("int4")
    try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
    try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
    // The production content-addressed source is sealed read-only; a copied config must be replaced atomically.
    try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: source.appendingPathComponent("config.json").path)
    try Data("tokenizer fixture".utf8).write(to: source.appendingPathComponent("tokenizer.json"))
    try Device.withDefaultDevice(.cpu) {
        let file = source.appendingPathComponent("model.safetensors")
        let writer = try TensorWriter(file)
        try writer.append("model.language_model.embed_tokens.weight", MLXArray.ones([256, 256], dtype: .bfloat16))
        try writer.append("model.vision_tower.position_embedding.weight", MLXArray.ones([256, 256], dtype: .bfloat16))
        try writer.finish()
        let before = try Data(contentsOf: file)
        try D1Precision.convert(source: source, destination: destination, bits: bits)
        #expect(try Data(contentsOf: source.appendingPathComponent("config.json")) == Data("{}".utf8))
        #expect(try Data(contentsOf: file) == before)
        let archive = try TensorArchive(destination.appendingPathComponent("model.safetensors"))
        #expect(
            Set(archive.entries.keys)
                == Set([
                    "model.language_model.embed_tokens.weight", "model.language_model.embed_tokens.scales", "model.language_model.embed_tokens.biases",
                    "model.vision_tower.position_embedding.weight"
                ]))
        let store = try SafetensorsFile(url: destination.appendingPathComponent("model.safetensors"))
        let embedding = try store.matrix("model.language_model.embed_tokens.weight")
        #expect(embedding.bits == bits && embedding.shape == [256, 256])
        #expect(embedding[MLXArray([Int32(5)])].asArray(Float.self).allSatisfy { $0 == 1 })
        #expect(try Data(contentsOf: destination.appendingPathComponent("tokenizer.json")) == Data("tokenizer fixture".utf8))
        let config = try JSONSerialization.jsonObject(with: Data(contentsOf: destination.appendingPathComponent("config.json"))) as? [String: Any]
        #expect((config?["quantization"] as? [String: Any])?["bits"] as? Int == bits)
    }
}

@testable import VireoEngine
@testable import VerdictEngine
import SoundEngine
import VireoCore
import MLXNN

@Suite(.serialized) struct Int4RoundTwoTests {
    func setup() {
        _ = NativePrecisionTestRuntime.ready
    }
    @Test(.heavy) func d1HalfTapsKeepReferenceRounding() throws {
        setup()
        try Device.withDefaultDevice(.cpu) {
            let taps = MLXArray((0..<192).map { Float($0 % 7 - 3) / 8 }, [3, 64]).asType(.bfloat16)
            let y = MLXArray((0..<576).map { Float($0 % 11 - 5) / 8 }, [3, 192]).asType(.bfloat16)
            let p1 = MLXArray([Int32(0), 1, 2]), p2 = MLXArray([Int32(0), 0, 1])
            #expect(taps.dtype == .bfloat16)
            #expect(max(abs(convOps(y, p1, p2, taps) - convOps(y, p1, p2, taps.asType(.float32)))).item(Float.self) == 0)
        }
    }
    @Test(.heavy) func d1ReadoutPreservesUnroundedNearTieProbabilities() throws {
        setup()
        try Device.withDefaultDevice(.cpu) {
            // All operands are BF16-exact. Row 0 exposes output rounding; row 1 cancellation.
            let h = MLXArray([Float(1), 1, 0, 1, -1, -1], [2, 3]).asType(.bfloat16)
            let rows = MLXArray([Float(32), 0.12451171875, 32, 32, 0.126953125, 32], [2, 3]).asType(.bfloat16)
            let actual = D1Model.readout(h, rows: rows)
            let reference: [[Double]] = [[32.12451171875, 32.126953125], [-0.12451171875, -0.126953125]]
            let groups = [[0], [1]]
            #expect(actual.dtype == .float32)
            #expect(actual.shape == [2, 2] && reference.count == 2 && groups == [[0], [1]])
            let logits = actual.asArray(Float.self).map(Double.init)
            let rounded = actual.asType(.bfloat16).asArray(Float.self).map(Double.init)
            #expect(logits.count == 4 && rounded.count == 4)
            let floor = 0.017 // Existing d1 gate's maximum absolute probability rounding floor.
            var roundedDrift = 0.0
            for index in reference.indices {
                #expect(reference[index].count == 2)
                let expected = Readout.probabilities(groups) { reference[index][$0] }
                let probabilities = Readout.probabilities(groups) { logits[index * 2 + $0] }
                let lossy = Readout.probabilities(groups) { rounded[index * 2 + $0] }
                #expect(probabilities.count == expected.count && expected.count == 2)
                for option in groups.indices {
                    #expect(logits[index * 2 + option] == reference[index][option])
                    #expect(abs(probabilities[option] - expected[option]) < 1e-12)
                    #expect(abs(probabilities[option] - expected[option]) < floor)
                    roundedDrift = max(roundedDrift, abs(lossy[option] - expected[option]))
                }
                #expect((probabilities[0] > probabilities[1]) == (index == 1))
            }
            #expect(roundedDrift > 0.06 && roundedDrift > floor)
        }
    }
    @Test(.heavy) func smallSfxGetDispatchPacksEveryEligibleComponent() throws {
        setup()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), output = root.appendingPathComponent("int4")
        try FileManager.default.createDirectory(at: source.appendingPathComponent("MLX"), withIntermediateDirectories: true)
        try Device.withDefaultDevice(.cpu) {
            for name in ["dit_sm-sfx_f16", "t5gemma_f16", "same_s_decoder_f32"] {
                let writer = try TensorWriter(source.appendingPathComponent("MLX/" + name + ".safetensors"))
                try writer.append("projection.weight", MLXArray.ones([256, 256], dtype: .float16))
                try writer.append("norm.weight", MLXArray.ones([256], dtype: .float16))
                try writer.finish()
            }
            let input = ModelSource(
                id: "stable-audio-3-small-sfx", engine: "stable-audio-3", directory: source,
                precision: "fp16", options: ["variant": "sm-sfx"])
            try SoundEngines.convert(input, dtype: "int4", into: output)
            for name in ["dit_sm-sfx_f16", "t5gemma_f16", "same_s_decoder_f32"] {
                let archive = try TensorArchive(output.appendingPathComponent("MLX/" + name + ".safetensors"))
                #expect(Set(archive.entries.keys) == Set(["projection.weight", "projection.scales", "projection.biases", "norm.weight"]))
                #expect(archive.entries["projection.weight"]?.dtype == .uint32)
                #expect(archive.entries["norm.weight"]?.dtype == .float16)
            }
        }
    }
    @Test(.heavy) func sharedVoiceConversionReleasesEachDenseTensorBeforeReadingNext() throws {
        setup()
        let output = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".safetensors")
        defer { try? FileManager.default.removeItem(at: output) }
        try Device.withDefaultDevice(.cpu) {
            weak var previous: MLXArray?
            var reads = 0
            let result = try VireoEngine.QuantizedWeights.streamTensors(
                keys: ["a.weight", "b.weight", "c.weight"],
                read: { _ in
                    #expect(previous == nil)
                    let value = MLXArray.ones([64, 64], dtype: .float16)
                    previous = value; reads += 1
                    return value
                }, output: output, bits: 4, group: 64, selects: { _, _ in true })
            #expect(previous == nil && reads == 3 && result.selected.count == 3 && result.packed == 3 * 64 * 64)
            let archive = try TensorArchive(output)
            #expect(archive.entries.count == 9)
            for name in ["a", "b", "c"] { #expect(archive.entries[name + ".weight"]?.dtype == .uint32) }
        }
    }
    /// Every Linear and the token embedding are packed at 8/4 bits, one dense tensor at a time (none retained at Load);
    /// at 16 bits the embedding stays a dense half table. Lookups read the packed rows.
    @Test(.heavy, arguments: [16, 8, 4]) func layaLoadsAndReleasesOneLinearAtATimeAndPacksTheEmbedding(bits: Int) throws {
        setup()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        let config = try LayaEncoderConfiguration(data: Data(#"{"hidden_size":64,"num_hidden_layers":1,"num_attention_heads":1,"intermediate_size":64,"vocab_size":64}"#.utf8))
        try Device.withDefaultDevice(.cpu) {
            let writer = try TensorWriter(root.appendingPathComponent("model.safetensors"))
            let linears = [
                "scorer.1", "scorer.3", "act_head.0", "act_head.2", "encoder.layers.0.attn.Wqkv",
                "encoder.layers.0.attn.Wo", "encoder.layers.0.mlp.Wi", "encoder.layers.0.mlp.Wo"
            ]
            for name in linears { try writer.append(name + ".weight", MLXArray.ones([64, 64], dtype: .float16)) }
            // Row r alternates r / 8 and (r + 1) / 8: each group's endpoints, so packed lookups return them closely.
            let parity = (MLXArray(0..<64) % 2).asType(.float32).reshaped(1, 64)
            let table = (MLXArray(0..<64).asType(.float32).reshaped(64, 1) + parity) / 8
            try writer.append("encoder.embeddings.tok_embeddings.weight", table.asType(.float16))
            for name in ["encoder.embeddings.norm", "encoder.final_norm", "type_emb", "scorer.0", "encoder.layers.0.mlp_norm"] {
                try writer.append(name + ".weight", MLXArray.ones([64], dtype: .float16))
            }
            try writer.finish()
            weak var previous: MLXArray?
            var denseReads = 0
            let network = try LayaNetwork(
                snapshot: root, config: config, headLayers: 0, bits: bits,
                read: { archive, key in
                    #expect(bits == 16 || previous == nil) // 16-bit Linears keep their dense weights by design
                    let value = try archive.array(key)
                    if value.ndim == 2 { previous = value; denseReads += 1 }
                    return value
                })
            // 16 bits reads the dense embedding; 8/4 bits read it in row blocks straight from the file (never through `read`).
            #expect(denseReads == (bits == 16 ? 9 : 8) && (bits == 16 || previous == nil))
            if bits == 16 {
                #expect(network.embedding == nil)
            } else {
                let packed = try #require(network.embedding)
                #expect(packed.bits == bits && packed.groupSize == 64 && packed.weight.dtype == .uint32)
                #expect(packed.weight.shape == [64, 64 * bits / 32] && packed.scales.shape == [64, 1])
                let rows = packed(MLXArray([Int32(3), Int32(40)]))
                #expect(rows.shape == [2, 64] && rows.dtype == .float16)
                let expected = (MLXArray([Float(3), Float(40)]).reshaped(2, 1) + parity) / 8
                #expect(allClose(rows.asType(.float32), expected, rtol: 0, atol: 0.02).item(Bool.self))
                // Row blocks pack exactly as the whole table (groups run along a row).
                let archive = try TensorArchive(root.appendingPathComponent("model.safetensors"))
                let blocked = try LayaNetwork.packedEmbedding(snapshot: root, archive: archive, key: "encoder.embeddings.tok_embeddings.weight", bits: bits, blockRows: 5)
                let whole = MLX.quantized(table.asType(.float16), groupSize: 64, bits: bits)
                #expect(arrayEqual(blocked.weight, whole.0).item(Bool.self) && arrayEqual(blocked.scales, whole.1).item(Bool.self))
                #expect(arrayEqual(try #require(blocked.biases), try #require(whole.2)).item(Bool.self))
                // Nine packed tables: far below the dense half bytes.
                #expect(network.residentBytes < 9 * 64 * 64 * 2 * (bits + 2) / 16)
            }
        }
    }
}

import StableAudio3

extension Int4RoundTwoTests {
    @Test(.heavy) func smallSfxDecoderInstallsPackedMatricesWithLegacyNumerics() throws {
        setup()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
        try Device.withDefaultDevice(.cpu) {
            var config = SAMESConfig()
            config.dim = 64; config.latentDim = 64; config.numHeads = 1; config.headDim = 64
            config.numBlocks = 1; config.ffInner = 64; config.outChannels = 4
            let skeleton = SAMESDecoder(config)
            let source = root.appendingPathComponent("source.safetensors"), output = root.appendingPathComponent("int4.safetensors")
            let tensors = Dictionary(uniqueKeysWithValues: skeleton.parameters().flattened().map { ($0.0, MLXArray.ones($0.1.shape, dtype: .float16)) })
            try MLX.save(arrays: tensors, url: source)
            let archive = try TensorArchive(source)
            let coverage = try StreamedPrecision.convert(
                keys: archive.entries.keys.sorted(), read: archive.array,
                to: output, half: .float16, bits: 4, eligible: { _, _ in true })
            let (model, report) = try sa3LoadSAMESDecoder(output, config: config)
            #expect(report.ok && coverage.quantizedTensors > 0)
            #expect(model.projectIn is QuantizedLinear)
            #expect(model.projectIn.weight.dtype == .uint32)
            // The Get snapshot stores FP16 scales (models/stable-audio-3-small-sfx/log/004); FP32 scales would break the FP16-only policy.
            #expect((model.projectIn as? QuantizedLinear)?.scales.dtype == .float16)
        }
    }

    @Test(.heavy) func voiceCheckpointDispatchPacksAllFixtureMatrices() throws {
        setup()
        let root = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source"), output = root.appendingPathComponent("int4")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try FileManager.default.createDirectory(at: output, withIntermediateDirectories: true)
        try Data("{}".utf8).write(to: source.appendingPathComponent("config.json"))
        // The production content-addressed source is sealed read-only; a copied config must be replaced atomically.
        try FileManager.default.setAttributes([.posixPermissions: 0o444], ofItemAtPath: source.appendingPathComponent("config.json").path)
        try Device.withDefaultDevice(.cpu) {
            let writer = try TensorWriter(source.appendingPathComponent("model.safetensors"))
            for name in ["a.weight", "b.weight"] { try writer.append(name, MLXArray.ones([64, 64], dtype: .float16)) }
            try writer.finish()
            let rule = VireoEngine.QuantRule(id: "fixture", version: 1, include: [".*"], exclude: [], note: "fixture")
            var report = VireoEngine.QuantReport(rule: rule, bits: 4, groupSize: 64)
            try VireoEngine.QuantizedWeights.convertCheckpoint(source: source, output: output, bits: 4, report: &report, sanitize: { $0 }, expectedLayers: 2)
            #expect(report.quantized.sorted() == ["a", "b"])
            #expect(report.quantizedParameters == 8192 && report.parameters == 8192)
            #expect(try TensorArchive(output.appendingPathComponent("model.safetensors")).entries.count == 6)
        }
    }
}
