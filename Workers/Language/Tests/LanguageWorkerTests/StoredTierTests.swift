// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire
import Testing
@testable import LanguageWorker

/// Get derives int8/int4 from the checkpoint's own tensor names; Load maps those names to the graph and drops what the
/// graph does not use. Qwen3.5 checkpoints carry an `mtp.*` head (and Chandra a tied `lm_head`) that Load drops: when
/// Get packed them, their scales/biases had no weight left to attach to and Load refused the snapshot ("Orphan affine
/// companions"). Each catalog model here derives a shrunk random checkpoint in its pinned checkpoint's layout through
/// the registered converter for its folder's recipe, then loads every tier through the real adapter.
@Suite(.serialized)
struct StoredTierTests {
    init() { MLXReady.ensure() }

    /// The pinned checkpoint layouts, read from their safetensors headers (20 Oct): tensor naming, and the matrices Load
    /// drops that the checkpoint carries.
    enum Naming { case hfQwen35, hfQwen25VL, mlx }
    struct Layout: Sendable, CustomStringConvertible {
        let id: String, naming: Naming, mtp: Bool, tiedHead: Bool
        var description: String { id }
    }
    static let layouts: [Layout] = [
        Layout(id: "chandra-ocr-2", naming: .hfQwen35, mtp: true, tiedHead: true),
        Layout(id: "index-translate-2b", naming: .hfQwen35, mtp: true, tiedHead: false),
        Layout(id: "index-translate-9b", naming: .hfQwen35, mtp: true, tiedHead: false),
        Layout(id: "nuextract3", naming: .hfQwen35, mtp: true, tiedHead: false),
        Layout(id: "ui-venus-2-9b", naming: .hfQwen35, mtp: false, tiedHead: false),
        Layout(id: "ui-venus-1.5-2b", naming: .mlx, mtp: false, tiedHead: false),
        Layout(id: "teleocr", naming: .hfQwen25VL, mtp: false, tiedHead: false)
    ]

    static let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent()

    static func checkpointName(_ name: String, _ naming: Naming) -> String {
        switch naming {
        case .mlx: return name
        case .hfQwen35:
            if name.hasPrefix("language_model.model.") { return "model.language_model." + name.dropFirst("language_model.model.".count) }
            if name.hasPrefix("vision_tower.") { return "model.visual." + name.dropFirst("vision_tower.".count) }
            if name.hasPrefix("language_model.lm_head.") { return String(name.dropFirst("language_model.".count)) }
            return name
        case .hfQwen25VL:
            if name.hasPrefix("language_model.model.") { return "model." + name.dropFirst("language_model.model.".count) }
            if name.hasPrefix("vision_tower.") { return "visual." + name.dropFirst("vision_tower.".count) }
            if name.hasPrefix("language_model.lm_head.") { return String(name.dropFirst("language_model.".count)) }
            return name
        }
    }

    /// A BPE tokenizer with the Qwen vision and chat tokens, enough for the adapter's real-weight Load.
    static func writeTokenizer(_ directory: URL) throws {
        let special = ["<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>", "<|endoftext|>"]
        var vocab: [String: JSONValue] = [:]
        for (index, token) in (["a", "b", "c", "Ġ"]).enumerated() { vocab[token] = .number(Double(index)) }
        let added: [JSONValue] = special.enumerated().map { index, token in
            vocab[token] = .number(Double(4 + index))
            return ["id": .number(Double(4 + index)), "content": .string(token), "single_word": false, "lstrip": false, "rstrip": false, "normalized": false, "special": true]
        }
        let tokenizer: JSONObject = [
            "version": "1.0", "added_tokens": .array(added), "normalizer": .null, "post_processor": .null,
            "pre_tokenizer": ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true],
            "decoder": ["type": "ByteLevel", "add_prefix_space": false, "trim_offsets": false, "use_regex": true],
            "model": ["type": "BPE", "vocab": .object(vocab), "merges": [], "unk_token": .null]
        ]
        try WireJSON.encode(tokenizer).write(to: directory.appendingPathComponent("tokenizer.json"))
        try WireJSON.encode(["tokenizer_class": "Qwen2Tokenizer", "eos_token": "<|im_end|>", "pad_token": "<|endoftext|>"])
            .write(to: directory.appendingPathComponent("tokenizer_config.json"))
    }

    static func tomlString(_ toml: String, _ key: String) throws -> String {
        let line = try #require(toml.split(separator: "\n").first { $0.hasPrefix(key + " = \"") })
        return String(line.dropFirst(key.count + 4).dropLast())
    }

    @Test(.heavy, arguments: layouts)
    func derivedTiersLoadThroughTheAdapter(_ layout: Layout) throws {
        setenv("ARSENAL_NATIVE_LAB", "1", 1)
        let folder = Self.root.appendingPathComponent("models/" + layout.id)
        let toml = try String(contentsOf: folder.appendingPathComponent("model.toml"), encoding: .utf8)
        let architecture = try Self.tomlString(toml, "architecture")
        let recipe = try Self.tomlString(toml, "int8")
        // A declared int4 recipe matches int8's. A tier removed after its quality check is no longer declared; its packing
        // is still exercised below, from the same recipe.
        if toml.split(separator: "\n").contains(where: { $0.hasPrefix("int4 = \"") }) { #expect(try Self.tomlString(toml, "int4") == recipe) }
        let fixture = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("gate/no-weights.json")))
        let config = try #require(fixture["config"]?.object)
        let spec = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("runners/native/language.json")))

        let temporary = FileManager.default.temporaryDirectory.appendingPathComponent("language-tier-\(layout.id)-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: temporary) }
        let source = temporary.appendingPathComponent("source")
        try FileManager.default.createDirectory(at: source, withIntermediateDirectories: true)
        try WireJSON.encode(config).write(to: source.appendingPathComponent("config.json"))
        try WireJSON.encode(try #require(spec["processor_config"]?.object)).write(to: source.appendingPathComponent("preprocessor_config.json"))
        try Self.writeTokenizer(source)

        // The random graph through the adapter, then its tensors in the pinned checkpoint's layout.
        let random = LanguageNative()
        try random.load(
            NativeLoadContext([
                "weights_dir": .string(source.path), "model_dir": .string(folder.path), "dtype": "fp16",
                "config": ["random_weights": true, "inventory_only": true, "architecture": .string(architecture)]
            ]), cancellation: NativeCancellation())
        let graph = random.parameters
        random.unload()
        var tensors: [String: MLXArray] = [:]
        for (name, value) in graph { tensors[Self.checkpointName(name, layout.naming)] = value }
        // Matrices the pinned checkpoint carries that Load drops: Qwen3.5's MTP block and Chandra's tied lm_head.
        var dropped: [String] = []
        if layout.mtp {
            func shape(_ suffix: String) throws -> [Int] { try #require(graph.first { $0.key.hasSuffix(suffix) }?.value.shape) }
            let hidden = try shape("embed_tokens.weight")[1]
            var extra: [String: [Int]] = ["mtp.fc.weight": [hidden, 2 * hidden]]
            for projection in ["self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj"] {
                extra["mtp.layers.0.\(projection).weight"] = try shape(".\(projection).weight")
            }
            for norm in ["mtp.norm.weight", "mtp.pre_fc_norm_embedding.weight", "mtp.pre_fc_norm_hidden.weight", "mtp.layers.0.input_layernorm.weight"] {
                extra[norm] = [hidden]
            }
            for (key, shape) in extra { tensors[key] = MLXRandom.uniform(Float(-0.1)..<Float(0.1), shape).asType(.float16) }
            dropped += extra.keys
        }
        if layout.tiedHead {
            #expect(tensors["lm_head.weight"] == nil)
            tensors["lm_head.weight"] = try #require(tensors.first { $0.key.hasSuffix("embed_tokens.weight") }?.value)
            dropped.append("lm_head.weight")
        }
        try MLX.save(arrays: tensors, url: source.appendingPathComponent("model.safetensors"))

        let converter = try #require(LanguageWorker.registrations.compactMap(\.conversion).first { $0.recipe == recipe })
        func convert(_ conversion: NativeConversion, _ precision: String, into name: String) throws -> URL {
            let destination = temporary.appendingPathComponent(name)
            _ = try conversion.run(
                [
                    "src_dir": .string(source.path), "dst_dir": .string(destination.path), "model_dir": .string(folder.path),
                    "dtype": .string(precision), "group_size": 64, "recipe": .string(conversion.recipe)
                ], cancellation: NativeCancellation())
            return destination
        }
        func load(_ weights: URL, _ precision: String) throws -> (quantized: Int, bits: Set<Int>) {
            let native = LanguageNative()
            defer { native.unload() }
            try native.load(
                NativeLoadContext([
                    "weights_dir": .string(weights.path), "model_dir": .string(folder.path), "dtype": .string(precision),
                    "config": ["architecture": .string(architecture)]
                ]), cancellation: NativeCancellation())
            let model = try #require(native.model)
            var bits = Set<Int>(), quantized = 0
            for (_, module) in model.leafModules().flattened() {
                if let layer = module as? QuantizedLinear { quantized += 1; bits.insert(layer.bits) }
                if let layer = module as? QuantizedEmbedding { quantized += 1; bits.insert(layer.bits) }
            }
            return (quantized, bits)
        }
        func keys(_ directory: URL) throws -> [String: MLXArray] { try MLX.loadArrays(url: directory.appendingPathComponent("model.safetensors")) }

        for precision in ["int8", "int4"] {
            let derived = try convert(converter, precision, into: precision)
            let stored = try keys(derived)
            let packs = stored.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(7)) }
            #expect(!packs.isEmpty)
            // What Load drops is stored half and whole: no pack, so no companion outlives its weight.
            for key in dropped {
                #expect(stored[key]?.dtype == .float16, "\(layout.id) \(precision): \(key)")
                #expect(stored[String(key.dropLast(7)) + ".scales"] == nil, "\(layout.id) \(precision): \(key)")
            }
            let loaded = try load(derived, precision)
            #expect(loaded.quantized == packs.count, "\(layout.id) \(precision)")
            #expect(loaded.bits == [precision == "int8" ? 8 : 4], "\(layout.id) \(precision)")

            // The former policy (every eligible checkpoint matrix) fails exactly where the checkpoint carries a dropped
            // matrix; elsewhere it wrote the same snapshot, which is why those folders keep the v2 recipe.
            let former = try convert(NativeConversion(recipe: recipe, eligible: LanguageNative.eligible), precision, into: precision + "-former")
            if dropped.isEmpty {
                #expect(recipe == "cast:language-half-g64-v2")
                let same = try Data(contentsOf: former.appendingPathComponent("model.safetensors")) == Data(contentsOf: derived.appendingPathComponent("model.safetensors"))
                #expect(same, "\(layout.id) \(precision)")
            } else {
                #expect(recipe == "cast:language-half-g64-v3")
                let failure = #expect(throws: (any Error).self, "\(layout.id) \(precision)") { _ = try load(former, precision) }
                #expect(failure?.localizedDescription == "Orphan affine companions in checkpoint", "\(layout.id) \(precision)")
            }
        }
    }
}
