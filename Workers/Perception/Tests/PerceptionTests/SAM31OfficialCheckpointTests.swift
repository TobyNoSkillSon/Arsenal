// SPDX-License-Identifier: MIT
import Foundation
import Testing
import MLX
import IntegrationKit
import ModelPrecision
import ArsenalWire
@testable import SAM31

/// Get converts Meta's facebook/sam3.1 checkpoint (`sam3.1_multiplex.pt`, FP32, Meta's key names and PyTorch layouts)
/// into the tensor set the port loads (`cast:sam31-fp16-v3`). No weights: the real layout is a names-and-shapes fixture,
/// and the converter runs end to end on a tiny synthetic torch.save archive, on the CPU stream.
@Suite(.serialized) struct SAM31OfficialCheckpointTests {
    init() { if let path = ProcessInfo.processInfo.environment["ARSENAL_TEST_METALLIB"] { setTestMetallib(URL(fileURLWithPath: path)) } }

    struct Layout {
        let source: [String: OfficialCheckpoint.Entry]
        let expected: [String: [Int]]
    }
    static func layout() throws -> Layout {
        let url = try #require(Bundle.module.url(forResource: "sam31-official-layout", withExtension: "json", subdirectory: "Fixtures"))
        let root = try #require(try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any])
        let dtypes: [String: DType] = ["float32": .float32, "complex64": .complex64]
        var source: [String: OfficialCheckpoint.Entry] = [:], expected: [String: [Int]] = [:]
        for item in try #require(root["source"] as? [[Any]]) {
            let name = try #require(item[0] as? String), shape = try #require(item[1] as? [Int]), dtype = try #require(dtypes[item[2] as? String ?? ""])
            source[name] = OfficialCheckpoint.Entry(shape: shape, dtype: dtype)
        }
        for item in try #require(root["expected"] as? [[Any]]) { expected[try #require(item[0] as? String)] = try #require(item[1] as? [Int]) }
        return Layout(source: source, expected: expected)
    }

    /// The official checkpoint's 1,623 tensors map onto exactly the 1,960 tensors the port loads: counts, then names,
    /// then shapes; every kept source tensor is FP32 (cast to FP16 at Get); only RoPE buffers and the unused CLIP
    /// projection are left out.
    @Test func officialLayoutMapsOntoExactlyThePortsTensors() throws {
        let layout = try Self.layout()
        #expect(layout.source.count == 1623 && layout.expected.count == 1960)
        let planned = try OfficialCheckpoint.plan(layout.source, dropping: NativeSAM31.unusedTensors)
        #expect(planned.count == layout.expected.count)
        #expect(Set(planned.keys).subtracting(layout.expected.keys).sorted() == [])
        #expect(Set(layout.expected.keys).subtracting(planned.keys).sorted() == [])
        #expect(planned.filter { layout.expected[$0.key] != $0.value.shape }.keys.sorted() == [])

        var used = Set<String>()
        for item in planned.values {
            switch item.source {
            case .tensor(let key), .third(let key, _): used.insert(key)
            case .stacked(let keys): used.formUnion(keys)
            }
        }
        #expect(used.allSatisfy { layout.source[$0]?.dtype == .float32 })
        let left = Set(layout.source.keys).subtracting(used)
        #expect(left.count == 33)
        #expect(left.filter { !$0.hasSuffix(".attn.freqs_cis") } == ["detector.backbone.language_backbone.encoder.text_projection"])
        #expect(left.allSatisfy { $0.hasSuffix(".attn.freqs_cis") == (layout.source[$0]?.dtype == .complex64) })

        // Layout spot checks: a fused ViT q/k/v, a Conv2d, a ConvTranspose2d, the class-token strip.
        let vit = "detector_model.vision_encoder.backbone."
        #expect(
            planned[vit + "layers.0.attention.k_proj.weight"]
                == .init(source: .third("detector.backbone.vision_backbone.trunk.blocks.0.attn.qkv.weight", 1), layout: .keep, shape: [1024, 1024]))
        #expect(planned[vit + "embeddings.patch_embeddings.projection.weight"]?.layout == .conv)
        #expect(planned["tracker_model.sam_mask_decoder.upscale_conv1.weight"]?.layout == .convTranspose)
        #expect(planned["detector_model.vision_encoder.neck.convs.0.scale_layers.0.weight"]?.layout == .convTranspose)
        #expect(planned[vit + "embeddings.position_embeddings"]?.layout == .dropFirstToken)
        #expect(planned["tracker_model.memory_temporal_positional_encoding"]?.layout == .keep)
    }

    /// What the port loads: the converted names pass the load-time map unchanged, and the full module (built from the
    /// converted form of Meta's config shape) has exactly these parameters.
    @Test func convertedLayoutIsTheFullModulesParameterSet() throws {
        let layout = try Self.layout()
        let planned = try OfficialCheckpoint.plan(layout.source, dropping: NativeSAM31.unusedTensors)
        for (name, item) in planned {
            let mapped = try #require(try NativeSAM31.map(name, MLXArray.zeros(item.shape)))
            #expect(mapped.0 == name && mapped.1.shape == item.shape)
        }
        let config = try JSONDecoder().decode(ModelConfig.self, from: OfficialCheckpoint.config(Self.metaConfigShape))
        let parameters = Dictionary(uniqueKeysWithValues: SAM31Root(config).parameters().flattened().map { ($0.0, $0.1.shape) })
        #expect(parameters.count == layout.expected.count)
        #expect(parameters == layout.expected)
    }

    /// The fields of Meta's config.json that differ from the form the port reads (the rest decode to the defaults).
    static let metaConfigShape = Data(
        #"{"model_type":"sam3_video","detector_config":{"vision_config":{"scale_factors":[4.0,2.0,1.0,0.5],"fpn_hidden_size":256}},"tracker_config":{"vision_config":{"scale_factors":[4.0,2.0,1.0,0.5]}},"low_res_mask_size":288}"#
            .utf8)

    @Test func configDropsTheMissingHalfScaleLevelAndIsIdempotent() throws {
        let once = try OfficialCheckpoint.config(Self.metaConfigShape)
        let config = try JSONDecoder().decode(ModelConfig.self, from: once)
        #expect(config.modelType == "sam3.1_video")
        #expect(config.detectorConfig.visionConfig.scaleFactors == [4, 2, 1] && config.trackerConfig.visionConfig.scaleFactors == [4, 2, 1])
        let root = try #require(try JSONSerialization.jsonObject(with: once) as? [String: Any])
        #expect((root["detector_config"] as? [String: Any])?["vision_config"].flatMap { ($0 as? [String: Any])?["fpn_hidden_size"] as? Int } == 256)
        #expect(try OfficialCheckpoint.config(once) == once)
    }

    // MARK: Synthetic checkpoint

    struct Tensor {
        let name: String
        let shape: [Int]
        var values: [Float]
        var complex = false
        init(_ name: String, _ shape: [Int], start: Float = 0) {
            self.name = name; self.shape = shape
            values = (0..<shape.reduce(1, *)).map { start + Float($0) }
        }
    }

    /// A `torch.save` protocol-2 pickle: OrderedDict(name → _rebuild_tensor_v2(storage, 0, shape, stride, False, OrderedDict())).
    static func pickle(_ tensors: [Tensor]) -> [UInt8] {
        func unicode(_ s: String) -> [UInt8] { let b = Array(s.utf8); return [0x58] + withUnsafeBytes(of: UInt32(b.count).littleEndian, Array.init) + b }
        func int(_ v: Int) -> [UInt8] { [0x4A] + withUnsafeBytes(of: Int32(v).littleEndian, Array.init) }
        func global(_ module: String, _ name: String) -> [UInt8] { [0x63] + Array("\(module)\n\(name)\n".utf8) }
        var p: [UInt8] = [0x80, 0x02] + global("collections", "OrderedDict") + [0x29, 0x52]
        for (index, tensor) in tensors.enumerated() {
            let strides = tensor.shape.indices.map { tensor.shape[($0 + 1)...].reduce(1, *) }
            p += unicode(tensor.name) + global("torch._utils", "_rebuild_tensor_v2") + [0x28]
            p += [0x28] + unicode("storage") + global("torch", tensor.complex ? "ComplexFloatStorage" : "FloatStorage")
            p += unicode(String(index)) + unicode("cpu") + int(tensor.shape.reduce(1, *)) + [0x74, 0x51]
            let shape: [UInt8] = tensor.shape.flatMap(int), stride: [UInt8] = strides.flatMap(int)
            p += int(0)
            p += [0x28] + shape + [0x74]
            p += [0x28] + stride + [0x74, 0x89]
            p += global("collections", "OrderedDict") + [0x29, 0x52] + [0x74, 0x52, 0x73]
        }
        return p + [0x2E]
    }

    static func archive(_ tensors: [Tensor]) -> [UInt8] {
        var entries: [(String, [UInt8])] = [("archive/data.pkl", pickle(tensors))]
        for (index, tensor) in tensors.enumerated() {
            let values = tensor.complex ? tensor.values.flatMap { [$0, 0] } : tensor.values
            entries.append(("archive/data/\(index)", values.flatMap { withUnsafeBytes(of: $0.bitPattern.littleEndian, Array.init) }))
        }
        return TorchZip.zip(entries)
    }

    /// Row-major element at `index` of `shape`.
    static func at(_ values: [Float], _ shape: [Int], _ index: [Int]) -> Float {
        var offset = 0
        for (i, dim) in shape.enumerated() { offset = offset * dim + index[i] }
        return values[offset]
    }
    /// `values` (shape `shape`) permuted by `axes`, in row-major order of the result.
    static func permuted(_ values: [Float], _ shape: [Int], _ axes: [Int]) -> [Float] {
        let out = axes.map { shape[$0] }
        var result: [Float] = []
        func walk(_ prefix: [Int]) {
            if prefix.count == out.count {
                var index = [Int](repeating: 0, count: shape.count)
                for (k, axis) in axes.enumerated() { index[axis] = prefix[k] }
                result.append(at(values, shape, index)); return
            }
            for i in 0..<out[prefix.count] { walk(prefix + [i]) }
        }
        walk([])
        return result
    }

    /// The whole Get conversion on a tiny checkpoint in Meta's names: renames, q/k/v splits, Conv2d and ConvTranspose2d
    /// transposes, the class-token strip, stacked point embeddings, the shared Gaussian copy, dropped complex RoPE buffers,
    /// and the unused CLIP projection whose -9.58e18 values would fail FP16 (22 Oct regression). Output is FP16 only.
    @Test func convertsASyntheticOfficialCheckpointEndToEnd() throws {
        let block = "detector.backbone.vision_backbone.trunk.blocks.0.attn."
        var projection = Tensor("detector.backbone.language_backbone.encoder.text_projection", [4, 2])
        projection.values[2] = -9.58471553820708e18
        var rope = Tensor(block + "freqs_cis", [3, 2]); rope.complex = true
        let qkv = Tensor(block + "qkv.weight", [6, 2]), qkvBias = Tensor(block + "qkv.bias", [6], start: 100)
        let out = Tensor(block + "proj.weight", [2, 2], start: 200)
        let patch = Tensor("detector.backbone.vision_backbone.trunk.patch_embed.proj.weight", [2, 3, 2, 2], start: 300)
        let deconv = Tensor("detector.backbone.vision_backbone.convs.0.dconv_2x2_0.weight", [3, 2, 2, 2], start: 400)
        let upscale = Tensor("tracker.model.sam_mask_decoder.output_upscaling.0.weight", [2, 3, 2, 2], start: 500)
        let position = Tensor("detector.backbone.vision_backbone.trunk.pos_embed", [1, 577, 2], start: 600)
        let tpos = Tensor("tracker.model.maskmem_tpos_enc", [2, 1, 1, 3], start: 1800)
        let points = (0..<4).map { Tensor("tracker.model.interactive_sam_prompt_encoder.point_embeddings.\($0).weight", [1, 2], start: 1900 + Float(2 * $0)) }
        let gaussian = Tensor("tracker.model.interactive_sam_prompt_encoder.pe_layer.positional_encoding_gaussian_matrix", [2, 2], start: 1950)
        let final = Tensor("tracker.model.maskmem_backbone.mask_downsampler.encoder.12.weight", [2, 2, 1, 1], start: 1960)
        let all = [projection, rope, qkv, qkvBias, out, patch, deconv, upscale, position, tpos, gaussian, final] + points

        let root = URL(fileURLWithPath: "/private/tmp/sam31-official-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let source = root.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try Data(Self.archive(all)).write(to: source.appendingPathComponent(OfficialCheckpoint.checkpoint))
        try Self.metaConfigShape.write(to: source.appendingPathComponent("config.json"))
        func request(_ src: URL, _ output: String) -> JSONObject {
            ["recipe": .string(NativeSAM31.recipe), "src_dir": .string(src.path), "dst_dir": .string(root.appendingPathComponent(output).path), "dtype": "fp16", "group_size": 64]
        }
        let receipt = try NativeSAM31.conversion.run(request(source, "output"), cancellation: NativeCancellation())
        #expect(receipt["files"]?.array?.compactMap { $0.object?["path"]?.string }.sorted() == ["config.json", "model.safetensors"])
        let converted = try TensorArchive(root.appendingPathComponent("output/model.safetensors"))

        let vit = "detector_model.vision_encoder.backbone.", prompt = "tracker_model.interactive_sam_prompt_encoder."
        let expected: [String: ([Int], [Float])] = [
            vit + "layers.0.attention.q_proj.weight": ([2, 2], Array(qkv.values[0..<4])),
            vit + "layers.0.attention.k_proj.weight": ([2, 2], Array(qkv.values[4..<8])),
            vit + "layers.0.attention.v_proj.weight": ([2, 2], Array(qkv.values[8..<12])),
            vit + "layers.0.attention.q_proj.bias": ([2], Array(qkvBias.values[0..<2])),
            vit + "layers.0.attention.k_proj.bias": ([2], Array(qkvBias.values[2..<4])),
            vit + "layers.0.attention.v_proj.bias": ([2], Array(qkvBias.values[4..<6])),
            vit + "layers.0.attention.o_proj.weight": ([2, 2], out.values),
            vit + "embeddings.patch_embeddings.projection.weight": ([2, 2, 2, 3], Self.permuted(patch.values, patch.shape, [0, 2, 3, 1])),
            "detector_model.vision_encoder.neck.convs.0.scale_layers.0.weight": ([2, 2, 2, 3], Self.permuted(deconv.values, deconv.shape, [1, 2, 3, 0])),
            "tracker_model.sam_mask_decoder.upscale_conv1.weight": ([3, 2, 2, 2], Self.permuted(upscale.values, upscale.shape, [1, 2, 3, 0])),
            vit + "embeddings.position_embeddings": ([1, 576, 2], Array(position.values[2...])),
            "tracker_model.memory_temporal_positional_encoding": ([2, 1, 1, 3], tpos.values),
            prompt + "point_embed.weight": ([4, 2], points.flatMap(\.values)),
            prompt + "shared_embedding.positional_embedding": ([2, 2], gaussian.values),
            "tracker_model.shared_image_embedding.positional_embedding": ([2, 2], gaussian.values),
            "tracker_model.memory_encoder.mask_downsampler.final_conv.weight": ([2, 1, 1, 2], Self.permuted(final.values, final.shape, [0, 2, 3, 1]))
        ]
        #expect(converted.entries.count == expected.count)
        #expect(converted.entries.keys.sorted() == expected.keys.sorted())
        for (name, (shape, values)) in expected.sorted(by: { $0.key < $1.key }) {
            let tensor = try converted.array(name)
            #expect(tensor.dtype == .float16, "\(name)")
            #expect(tensor.shape == shape, "\(name)")
            #expect(tensor.asType(.float32, stream: .cpu).asArray(Float.self) == values, "\(name)")
        }
        let config = try JSONDecoder().decode(ModelConfig.self, from: Data(contentsOf: root.appendingPathComponent("output/config.json")))
        #expect(config.detectorConfig.visionConfig.scaleFactors == [4, 2, 1] && config.modelType == "sam3.1_video")

        // Get's per-file cast hands the recipe config.json alone; it yields no tensors.
        let configOnly = root.appendingPathComponent("config-only")
        try FileManager.default.createDirectory(at: configOnly, withIntermediateDirectories: true)
        try Self.metaConfigShape.write(to: configOnly.appendingPathComponent("config.json"))
        let configReceipt = try NativeSAM31.conversion.run(request(configOnly, "config-output"), cancellation: NativeCancellation())
        #expect(configReceipt["files"]?.array?.compactMap { $0.object?["path"]?.string } == ["config.json"])

        // The v3 recipe converts Meta's checkpoint only; the former community safetensors source is refused.
        let community = root.appendingPathComponent("community")
        try FileManager.default.createDirectory(at: community, withIntermediateDirectories: true)
        let writer = try TensorWriter(community.appendingPathComponent("model.safetensors"))
        try writer.append("detector_model.text_projection.weight", MLXArray([Float](repeating: 0.5, count: 4), [2, 2]))
        try writer.finish()
        #expect(throws: PrecisionError.self) { try NativeSAM31.conversion.run(request(community, "community-output"), cancellation: NativeCancellation()) }
    }
}

/// Stored (uncompressed) ZIP writer for synthetic torch.save archives.
enum TorchZip {
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
