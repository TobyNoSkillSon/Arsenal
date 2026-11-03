// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ModelPrecision
import ArsenalWire
import Testing
@testable import AudioWorker

/// SAM Audio Large from Meta's original `checkpoint.pt` (`cast:audio-sam-v2`). The fixture holds the official
/// checkpoint's real tensor records (read from its `data.pkl`) and the layout the port loads (the safetensors headers of
/// the FP16 re-upload it was ported against). No model weights and no inference.
@Suite struct SAMOfficialCheckpointTests {
    typealias Layout = [String: (dtype: String, shape: [Int])]

    init() {
        if let metallib = ProcessInfo.processInfo.environment["ARSENAL_MLX_METALLIB"] ?? ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] {
            GPU.metallib = URL(fileURLWithPath: metallib)
        }
    }

    static func fixture() throws -> (official: Layout, port: Layout) {
        let url = try #require(Bundle.module.url(forResource: "sam-audio-large-layout", withExtension: "json", subdirectory: "Fixtures"))
        let object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        func layout(_ key: String) throws -> Layout {
            let entries = try #require(object[key] as? [String: [Any]])
            return try entries.mapValues { entry in
                (dtype: try #require(entry.first as? String), shape: try entry.dropFirst().map { try #require($0 as? Int) })
            }
        }
        return (try layout("official"), try layout("port"))
    }

    /// Counts, then names, then shapes and dtypes: the conversion plan of the real official checkpoint is exactly the
    /// tensor set the port loads, and the port's own large configuration declares that same set.
    @Test func officialLayoutPlansExactlyThePortLayout() throws {
        let (official, port) = try Self.fixture()
        #expect(official.count == 1_345)
        #expect(port.count == 735)
        #expect(Set(official.values.map(\.dtype)) == ["F32"])
        #expect(Set(port.values.map(\.dtype)) == ["F16"])
        let shapes = official.mapValues(\.shape)
        let plan = try SAMConversion.Official.plan(shapes)
        #expect(plan.count == port.count)
        #expect(Set(plan.keys) == Set(port.keys))
        let mismatched = plan.filter { SAMConversion.Official.shape($0.value, in: shapes) != port[$0.key]?.shape }.keys.sorted()
        #expect(mismatched == [])
        let portModules = Dictionary(uniqueKeysWithValues: port.map { (audioLSTMKey($0.key), $0.value.shape) })
        #expect(portModules.count == port.count)
        try SAMConversion.Official.require(plan, shapes: shapes, equals: portModules)

        // What is dropped, summed and transposed.
        let used = Set(plan.values.flatMap(\.names))
        let unused = Set(official.keys).subtracting(used)
        #expect(unused.count == 606)
        #expect(unused.filter { $0.hasPrefix("vision_encoder.") }.count == 601)
        #expect(unused.filter { $0.hasPrefix("align_masked_video.") }.count == 5)
        #expect(plan.values.filter { $0.names.count == 2 }.count == 4)
        #expect(plan.values.filter { $0.permutation == [0, 2, 1] }.count == 142)
        #expect(plan.values.filter { $0.permutation == [1, 2, 0] }.count == 12)
        #expect(plan["transformer.x_embedder.block.block1.project.weight"]?.permutation == [0, 2, 1])
        #expect(plan["audio_codec.decoder.blocks.0.block_1.weight_g"]?.permutation == [1, 2, 0])
        #expect(plan["audio_codec.encoder.blocks.0.res1.conv1.weight_g"]?.permutation == nil)

        // The port built from the release's config.json declares exactly the fixture's port layout.
        let url = try #require(Bundle.module.url(forResource: "sam-audio-large-layout", withExtension: "json", subdirectory: "Fixtures"))
        let object = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let config = try JSONDecoder().decode(SAMAudioConfig.self, from: JSONSerialization.data(withJSONObject: try #require(object["config"])))
        Device.withDefaultDevice(.cpu) {
            let expected = Dictionary(uniqueKeysWithValues: SAMAudio(config: config).parameters().flattened().map { ($0.0, $0.1.shape) })
            #expect(expected.count == port.count)
            let differing = Set(expected.keys).union(portModules.keys).filter { expected[$0] != portModules[$0] }.sorted()
            #expect(differing == [])
        }
    }

    @Test func refusesDuplicateAndUnpairedTensors() throws {
        let lstm = "audio_codec.decoder.wm_model.encoder_block.post.0.lstm"
        #expect(throws: (any Error).self) { try SAMConversion.Official.plan([lstm + ".bias_ih_l0": [8]]) }
        #expect(throws: (any Error).self) { try SAMConversion.Official.plan([lstm + ".bias_ih_l0": [8], lstm + ".bias_hh_l0": [4]]) }
        // Both names become audio_codec.quantizer_in_proj.bias.
        #expect(throws: (any Error).self) {
            try SAMConversion.Official.plan(["audio_codec.quantizer.in_proj.bias": [4], "audio_codec.quantizer_in_proj.bias": [4]])
        }
        #expect(SAMConversion.Official.name(lstm + ".weight_ih_l1") == "audio_codec.decoder.wm_model.encoder_block.post_0.lstm.layers.1.Wx")
        #expect(SAMConversion.Official.name("audio_codec.encoder.block.2.block.1.block.3.weight_v") == "audio_codec.encoder.blocks.1.res2.conv2.weight_v")
        #expect(SAMConversion.Official.name("audio_codec.decoder.model.3.block.6.block.1.bias") == "audio_codec.decoder.blocks.2.block_6.conv1.bias")
        #expect(SAMConversion.Official.name("audio_codec.decoder.wm_model.encoder_block.pre.1.weight_g") == "audio_codec.decoder.conv_out.weight_g")
    }

    /// Get's base conversion on a small synthetic torch.save archive in the official layout (four encoder and decoder
    /// blocks like the large model, narrow widths, one transformer layer): it drops the vision and video tensors, sums
    /// the LSTM biases, transposes every convolution, writes FP16 only, copies the config, prepares the T5 encoder and
    /// the result loads in the port.
    @Test func convertsASyntheticOfficialCheckpoint() throws {
        try Device.withDefaultDevice(.cpu) {
            let (official, _) = try Self.fixture()
            let folder = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
                .deletingLastPathComponent().appendingPathComponent("models/sam-audio-large")
            var config = try #require(try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/no-weights.json")))["config"]?.object)
            var codec = try #require(config["audio_codec"]?.object)
            // Four blocks each way like the large model; a decoder width of 96 keeps every watermark unit (width / 3) nonempty.
            codec["encoder_rates"] = [2, 2, 2, 2]; codec["decoder_rates"] = [2, 2, 2, 2]; codec["decoder_dim"] = 96
            config["audio_codec"] = .object(codec)
            let configData = try WireJSON.encode(config)
            let port = SAMAudio(config: try JSONDecoder().decode(SAMAudioConfig.self, from: configData))
            let portShapes = Dictionary(uniqueKeysWithValues: port.parameters().flattened().map { ($0.0, $0.1.shape) })
            #expect(portShapes.values.allSatisfy { !$0.contains(0) })

            // Stored and official names for the small model's parameters, found by mapping the real official names forward.
            var inverse: [String: (stored: String, official: [String])] = [:]
            for key in official.keys.sorted() {
                let stored: String
                if let (base, _, layer) = SAMConversion.Official.lstmBias(key) {
                    stored = SAMConversion.Official.name(base + ".combined_bias_l" + layer)
                } else {
                    stored = SAMConversion.Official.name(key)
                }
                inverse[audioLSTMKey(stored), default: (stored, [])].official.append(key)
            }
            var tensors: [(name: String, shape: [Int], values: [Float])] = []
            var expected: [String: [Float]] = [:]
            var storedShapes: [String: [Int]] = [:]
            for (index, module) in portShapes.keys.sorted().enumerated() {
                let entry = try #require(inverse[module], "no official name for \(module)")
                let key = entry.stored, names = entry.official
                let shape = portShapes[module]!
                storedShapes[key] = shape
                let permutation = SAMConversion.Official.permutation(key, rank: shape.count)
                var sourceShape = shape
                if let permutation { for (axis, from) in permutation.enumerated() { sourceShape[from] = shape[axis] } }
                let count = shape.reduce(1, *)
                var sum = [Float](repeating: 0, count: count)
                for (part, name) in names.enumerated() {
                    // Exactly representable in FP16, so the expected output is exact.
                    let values = (0..<count).map { Float(($0 * 7 + index * 13 + part * 5) % 61 - 30) / 16 }
                    tensors.append((name, sourceShape, values))
                    for i in 0..<count { sum[i] += values[i] }
                }
                expected[key] = permutation.map { Self.permute(sum, shape: sourceShape, axes: $0) } ?? sum
            }
            #expect(tensors.filter { $0.name.contains(".lstm.bias_") }.count == 8)
            tensors.append(("vision_encoder.model.proj", [4, 4], (0..<16).map(Float.init)))
            tensors.append(("align_masked_video.gate", [1], [70_000])) // past FP16's range: proves it is never cast

            let root = FileManager.default.temporaryDirectory.appendingPathComponent("sam-official-" + UUID().uuidString)
            defer { try? FileManager.default.removeItem(at: root) }
            let source = root.appendingPathComponent("source"), asset = root.appendingPathComponent("t5"), output = root.appendingPathComponent("fp16")
            for directory in [source, asset] { try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true) }
            try Data(Self.checkpoint(tensors)).write(to: source.appendingPathComponent("checkpoint.pt"))
            try configData.write(to: source.appendingPathComponent("config.json"))
            try Self.writeTextAsset(asset)

            let request: JSONObject = [
                "src_dir": .string(source.path), "dst_dir": .string(output.path), "dtype": "fp16", "group_size": 64,
                "recipe": .string(SAMConversion.official.recipe), "assets": [SAMConversion.textAsset: .string(asset.path)]
            ]
            #expect(SAMConversion.official.recipe == "cast:audio-sam-v2")
            let receipt = try SAMConversion.official.run(request, cancellation: NativeCancellation())
            let paths = Set(receipt["files"]?.array?.compactMap { $0["path"]?.string } ?? [])
            #expect(
                paths == [
                    "config.json", "model.safetensors", SAMConversion.marker, "text_encoder/config.json", "text_encoder/model.safetensors",
                    "text_encoder/tokenizer.json"
                ])
            #expect(try Data(contentsOf: output.appendingPathComponent("config.json")) == configData)

            let archive = try TensorArchive(output.appendingPathComponent("model.safetensors"))
            #expect(archive.entries.count == storedShapes.count)
            #expect(Set(archive.entries.keys) == Set(storedShapes.keys))
            #expect(archive.entries.values.allSatisfy { $0.dtype == .float16 })
            for key in storedShapes.keys.sorted() {
                let value = try archive.array(key)
                #expect(value.shape == storedShapes[key], "\(key)")
                #expect(value.asType(.float32).asArray(Float.self) == expected[key], "\(key)")
            }
            let text = try TensorArchive(output.appendingPathComponent("text_encoder/model.safetensors"))
            #expect(text.entries.values.allSatisfy { $0.dtype == .float16 })

            // The snapshot loads in the port, every parameter half.
            let adapter = SAMAdapter()
            try adapter.load(
                NativeLoadContext([
                    "model_dir": .string(folder.path), "weights_dir": .string(output.path), "dtype": "fp16", "config": .object(config)
                ]), cancellation: NativeCancellation())
            try NativeParameters.requireHalf(adapter.parameters)
            adapter.unload()

            // A source with anything besides checkpoint.pt and config.json is refused.
            try Data().write(to: source.appendingPathComponent("extra.bin"))
            let again: JSONObject = request.merging(["dst_dir": .string(root.appendingPathComponent("again").path)]) { _, new in new }
            #expect(throws: (any Error).self) { try SAMConversion.official.run(again, cancellation: NativeCancellation()) }
        }
    }

    /// Host-side transpose of a row-major rank-3 array: output axis i is source axis `axes[i]`.
    static func permute(_ values: [Float], shape: [Int], axes: [Int]) -> [Float] {
        let strides = [shape[1] * shape[2], shape[2], 1]
        let out = axes.map { shape[$0] }
        var result: [Float] = []
        result.reserveCapacity(values.count)
        for a in 0..<out[0] {
            for b in 0..<out[1] {
                for c in 0..<out[2] {
                    result.append(values[a * strides[axes[0]] + b * strides[axes[1]] + c * strides[axes[2]]])
                }
            }
        }
        return result
    }

    /// A one-layer T5 encoder asset in google-t5's naming.
    static func writeTextAsset(_ directory: URL) throws {
        let text = T5Encoder(config: T5ModelConfig(vocabSize: 512, dModel: 64, dKV: 16, dFF: 128, numLayers: 1, numHeads: 4))
        try WireJSON.encode(["vocab_size": 512, "d_model": 64, "d_kv": 16, "d_ff": 128, "num_layers": 1, "num_heads": 4] as JSONObject)
            .write(to: directory.appendingPathComponent("config.json"))
        let unigram: JSONObject = [
            "version": "1.0", "added_tokens": [], "pre_tokenizer": ["type": "Whitespace"],
            "model": ["type": "Unigram", "unk_id": 2, "vocab": [["<pad>", 0], ["</s>", 0], ["<unk>", 0], ["voice", -1]]]
        ]
        try WireJSON.encode(unigram).write(to: directory.appendingPathComponent("tokenizer.json"))
        var weights: [String: MLXArray] = [:]
        for (name, value) in text.parameters().flattened() {
            weights[name.replacingOccurrences(of: ".self_attention.", with: ".layer.0.").replacingOccurrences(of: ".ff.", with: ".layer.1.")] = value
        }
        try save(arrays: weights, url: directory.appendingPathComponent("model.safetensors"))
    }

    /// A stored-ZIP `torch.save` archive of FP32 tensors, one storage each, as `torch.save(state_dict)` writes it.
    static func checkpoint(_ tensors: [(name: String, shape: [Int], values: [Float])]) -> [UInt8] {
        func unicode(_ text: String) -> [UInt8] {
            let bytes = Array(text.utf8)
            return [0x58] + withUnsafeBytes(of: UInt32(bytes.count).littleEndian, Array.init) + bytes
        }
        func int(_ value: Int) -> [UInt8] { [0x4A] + withUnsafeBytes(of: Int32(value).littleEndian, Array.init) }
        func tuple(_ values: [Int]) -> [UInt8] { [0x28] + values.flatMap(int) + [0x74] }
        let orderedDict: [UInt8] = [0x63] + Array("collections\nOrderedDict\n".utf8) + [0x29, 0x52]
        var pickle: [UInt8] = [0x80, 0x02] + orderedDict + [0x28]
        var entries: [(String, [UInt8])] = []
        for (index, tensor) in tensors.enumerated() {
            let strides = tensor.shape.indices.map { tensor.shape[($0 + 1)...].reduce(1, *) }
            let rebuild: [UInt8] = [0x63] + Array("torch._utils\n_rebuild_tensor_v2\n".utf8)
            let storage: [UInt8] = [0x63] + Array("torch\nFloatStorage\n".utf8)
            pickle += unicode(tensor.name)
            pickle += rebuild
            pickle += [0x28, 0x28]
            pickle += unicode("storage")
            pickle += storage
            pickle += unicode(String(index))
            pickle += unicode("cpu")
            pickle += int(tensor.values.count)
            pickle += [0x74, 0x51]
            pickle += int(0)
            pickle += tuple(tensor.shape)
            pickle += tuple(strides)
            pickle += [0x89]
            pickle += orderedDict
            pickle += [0x74, 0x52]
            entries.append(("archive/data/\(index)", tensor.values.flatMap { withUnsafeBytes(of: $0.bitPattern.littleEndian, Array.init) }))
        }
        pickle += [0x75, 0x2E]
        return zip([("archive/data.pkl", pickle)] + entries)
    }

    static func zip(_ entries: [(String, [UInt8])]) -> [UInt8] {
        func le16(_ v: Int) -> [UInt8] { [UInt8(v & 0xFF), UInt8(v >> 8 & 0xFF)] }
        func le32(_ v: Int) -> [UInt8] { le16(v & 0xFFFF) + le16(v >> 16 & 0xFFFF) }
        var out: [UInt8] = [], central: [UInt8] = []
        for (name, data) in entries {
            let offset = out.count, n = Array(name.utf8)
            out += le32(0x0403_4b50) + le16(20) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0) + le32(data.count) + le32(data.count)
            out += le16(n.count) + le16(0) + n + data
            central += le32(0x0201_4b50) + le16(20) + le16(20) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0) + le32(data.count)
            central += le32(data.count) + le16(n.count) + le16(0) + le16(0) + le16(0) + le16(0) + le32(0) + le32(offset) + n
        }
        let start = out.count
        out += central
        out += le32(0x0605_4b50) + le16(0) + le16(0) + le16(entries.count) + le16(entries.count) + le32(central.count) + le32(start) + le16(0)
        return out
    }
}
