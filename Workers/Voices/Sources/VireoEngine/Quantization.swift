import Foundation
import MLX
import MLXNN
import ModelPrecision
import VireoCore
import MLXAudioTTS

// Tier derivation, converted once at Get and loaded like any checkpoint (tiers: TABLE-INTERFACE.md Contract 1).
//
// - 16 is the published 16-bit checkpoint, except for FP32-only models (Kokoro, RE-USE): `castTo16` casts them to
//   FP16 once (`<support>/quantized/<id>/16/`) and the FP32 source is deleted.
// - 8 and 4 are affine g64 from the 16-bit tier, never from a quantized source. Two kinds of rule:
//   * module rules (Pocket, Stable Audio 3, RE-USE): the selected Linear layers of one weights file, checked against the
//     model's module tree; the loader reads the rest from the 16-bit tier;
//   * checkpoint rules (Qwen3, Kokoro, Breeze): a whole mlx-community-layout checkpoint (quantized files + a config
//     with `quantization`, every other file linked to the 16-bit tier), since those loaders quantize whatever layer
//     carries `.scales`. The layer sets are the ones mlx-community quantized (lab/notes/COMPONENTS.md).
//
// Rule: only the large transformer/DiT Linear layers named by a model's `QuantRule` are quantized (affine,
// group 64, 8 or 4 bits); a selected layer whose input width is not a multiple of 64 stays at base precision.
// Audio codecs/decoders/VAEs, embeddings, output heads, conditioning MLPs, norms and convolutions are never
// quantized. `Engines.convertQuantized` writes the converted weights; the helper adds `manifest.json`
// (source revision + blob hashes, rule id/version, output hashes). Loading a converted file never quantizes:
// `QuantizedWeights.prepare` builds each QuantizedLinear directly from the stored tensors, so no random
// placeholder is ever quantized (Vella's leak note) and loading costs what loading the base costs.
//
// Earlier trials this replaces: quantize-at-load over every Linear outside the codec clipped Pocket's audio (it
// quantized the flow head and the EOS head too); the DiT variant touched only some layers. Rules below are
// explicit per model and unit-tested against the real module trees (Tests/EngineTests).

/// Which Linear layers of one model are quantized. Patterns are regular expressions over module paths
/// (`flow_lm.transformer.layers.3.linear1`), matched against the whole path.
public struct QuantRule: Sendable, Equatable {
    public let id: String
    /// Bump when include/exclude change: converted files made with an older version count as missing.
    public let version: Int
    public let include: [String]
    /// Documented exclusions; a path matching both lists is excluded.
    public let exclude: [String]
    /// One line for people (catalog `note`, manifest, docs).
    public let note: String

    public init(id: String, version: Int, include: [String], exclude: [String], note: String) {
        self.id = id; self.version = version; self.include = include; self.exclude = exclude; self.note = note
    }

    public func includes(_ path: String) -> Bool { include.contains { Self.whole($0, path) } }
    public func excludes(_ path: String) -> Bool { exclude.contains { Self.whole($0, path) } }
    public func selects(_ path: String) -> Bool { includes(path) && !excludes(path) }

    static func whole(_ pattern: String, _ path: String) -> Bool {
        guard let regex = try? NSRegularExpression(pattern: "^(?:" + pattern + ")$") else { return false }
        return regex.firstMatch(in: path, range: NSRange(path.startIndex..., in: path)) != nil
    }

    public var manifest: [String: Any] { ["id": id, "version": version, "include": include, "exclude": exclude, "note": note] }
}

public enum QuantRules {
    /// Group size for every Vireo quantization (MLX affine).
    public static let groupSize = 64
    public static let bitsOffered = [8, 4]

    /// Pocket TTS: the FlowLM backbone transformer only.
    public static let pocket = QuantRule(
        id: "pocket-tts", version: 1,
        include: [#"flow_lm\.transformer\.layers\.\d+\.(self_attn\.in_proj|self_attn\.out_proj|linear1|linear2)"#],
        exclude: [#"mimi(\..*)?"#,                          // Mimi codec: SEANet encoder/decoder and its transformers
                  #"flow_lm\.flow_net(\..*)?"#,              // flow-matching head that writes the latents
                  #"flow_lm\.conditioner(\..*)?"#,           // text embedding table (+ projection)
                  #"flow_lm\.input_linear"#,              // latent → model input projection
                  #"flow_lm\.out_eos"#],                  // end-of-speech head
        note: "8/4-bit: the 6-layer FlowLM transformer (attention in/out and feed-forward, 24 Linear layers, 75 M of 118 M parameters); Mimi codec, flow head, text embedding, input projection and EOS head stay BF16.")

    // MARK: Checkpoint rules (tensor names of the 16-bit checkpoint, after the loader's own key mapping)

    /// Qwen3-TTS (VoiceDesign, Base): the talker's Linear layers and codec head, as mlx-community's 8/4-bit repos
    /// (250 layers); embeddings, the speaker encoder, the speech tokenizer and norms stay BF16.
    public static let qwen3Talker = QuantRule(
        id: "qwen3-tts-talker", version: 1,
        include: [#"talker\.(model|code_predictor\.model)\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"#,
                  #"talker\.code_predictor\.lm_head\.\d+"#,
                  #"talker\.code_predictor\.small_to_mtp_projection"#,
                  #"talker\.codec_head"#,
                  #"talker\.text_projection\.linear_fc(1|2)"#],
        exclude: [],
        note: "8/4-bit: the talker's Linear layers and codec head (250 layers); embeddings, speaker encoder, speech tokenizer and norms stay BF16.")

    /// Kokoro 82M: mlx-community's 87 Linear and embedding layers (ALBERT, projections, prosody predictor, AdaIN
    /// projections), over the FP16 tier; convolutions, LSTMs, norms and the vocoder stay FP16.
    public static let kokoro = QuantRule(
        id: "kokoro", version: 1,
        include: [#"bert\.embeddings\.(word|position|token_type)_embeddings"#,
                  #"bert\.encoder\.albert_layer_groups\.\d+\.albert_layers\.\d+\.(attention\.(query|key|value|dense)|ffn|ffn_output)"#,
                  #"bert\.encoder\.embedding_hidden_mapping_in"#,
                  #"bert\.pooler"#,
                  #"bert_encoder"#,
                  #"text_encoder\.embedding"#,
                  #"predictor\.text_encoder\.lstms\.\d+\.fc"#,
                  #"predictor\.duration_proj\.linear_layer"#,
                  #"predictor\.(F0|N)\.\d+\.norm(1|2)\.fc"#,
                  #"decoder\.(encode|decode\.\d+)\.norm(1|2)\.fc"#,
                  #"decoder\.generator\.(resblocks|noise_res)\.\d+\.adain(1|2)\.\d+\.fc"#],
        exclude: [],
        note: "8-bit: 87 Linear and embedding layers (ALBERT text encoder, projections, prosody predictor, AdaIN projections); convolutions, LSTMs, norms and the vocoder stay FP16.")

    /// Breeze TTS 2: mlx-community's 468 quantized layers (Linear layers and token embeddings), here affine g64
    /// instead of MXFP8; the audio tokenizer and norms stay BF16.
    public static let breeze = QuantRule(
        id: "breeze-tts-2", version: 1,
        include: [#"(text_encoder|backbone_model|depth_decoder\.model)\.layers\.\d+\.(self_attn\.(q|k|v|o)_proj|mlp\.(gate|up|down)_proj)"#,
                  #"backbone_model\.embed_tokens\.embed_audio_tokens"#,
                  #"depth_decoder\.model\.(embed_tokens|inputs_embeds_projector)"#,
                  #"embed_text_tokens"#,
                  #"lm_head"#,
                  #"text_encoder_proj"#],
        exclude: [],
        note: "8-bit: 468 Linear layers and token embeddings (text encoder, backbone, depth decoder, heads); audio tokenizer and norms stay BF16.")

    /// Layers a checkpoint rule must select (the mlx-community sets): a different count means another checkpoint.
    public static let checkpointLayers: [String: Int] = [qwen3Talker.id: 250, kokoro.id: 87, breeze.id: 468]
    public static func isCheckpointRule(_ rule: QuantRule) -> Bool { checkpointLayers[rule.id] != nil }

    /// The rule for a catalog model's 8/4 tiers, nil when it has none.
    public static func rule(engine: String, id: String) -> QuantRule? {
        switch engine {
        case "mlx-audio":
            if id == "pocket-tts" { return pocket }
            if id.hasPrefix("qwen3-tts") { return qwen3Talker }
            if id == "kokoro-82m" { return kokoro }
            if id == "breeze-tts-2" { return breeze }
            return nil
        default: return nil
        }
    }
}

/// What a conversion did; the helper stores it in the manifest.
public struct QuantReport: Sendable, Equatable {
    public struct File: Sendable, Equatable { public let source: String; public let output: String }
    public var rule: QuantRule
    public var bits: Int
    public var groupSize: Int
    /// "affine" (8/4) or "cast" (16 from an FP32 source).
    public var mode = "affine"
    public var files: [File] = []
    /// Checkpoint rules: output path → absolute file in the 16-bit tier it links to (tokenizers, codecs, voices).
    public var links: [String: String] = [:]
    /// Module paths quantized, sorted.
    public var quantized: [String] = []
    /// Linear layers left at base precision → reason ("excluded by rule", "not selected by rule", "input width 257 not a multiple of 64").
    public var kept: [String: String] = [:]
    public var linears = 0
    public var quantizedParameters = 0
    public var parameters = 0
    public var fallbackComponents: [String] = []
    public var fallbackMatrices: [String] = []
    public var matrixParameters: [String: Int] = [:]
    public var matrixComponents: [String: String] = [:]
    public var matrixScalesSHA256: [String: String] = [:]
    public var fallbackParameters = 0
    public var componentParameters: [String: Int] = [:]
    public var packingAudit: [String: [String: Double]] = [:]

    public var manifest: [String: Any] {
        var reasons: [String: Int] = [:]
        for reason in kept.values { reasons[reason, default: 0] += 1 }
        return ["rule": rule.manifest, "bits": bits, "group_size": groupSize, "mode": mode,
                "files": files.map { ["source": $0.source, "output": $0.output] }, "links": links,
                "linear_layers": linears, "quantized_layers": quantized.count, "quantized": quantized,
                "kept_layers": kept.count, "kept_reasons": reasons, "kept": kept,
                "quantized_parameters": quantizedParameters, "parameters": parameters,
                "fallback_components": fallbackComponents, "fallback_matrices": fallbackMatrices,
                "matrix_parameters": matrixParameters, "matrix_components": matrixComponents, "matrix_scales_sha256": matrixScalesSHA256, "fallback_parameters": fallbackParameters,
                "component_eligible_parameters": componentParameters, "packing_audit": packingAudit]
    }
}

public enum QuantizedWeights {
    /// A layer the rule selects: module path and weight shape [out, in].
    public struct Selection: Equatable { public let path: String; public let shape: [Int] }

    /// Which Linear layers of `skeleton` the rule quantizes. `kept` gives the reason for every other Linear.
    /// The skeleton is a freshly constructed model; its lazily initialised parameters are never evaluated.
    public static func plan(_ skeleton: Module, rule: QuantRule, group: Int = QuantRules.groupSize)
        -> (selected: [Selection], kept: [String: String], linears: Int) {
        var selected: [Selection] = [], kept: [String: String] = [:], linears = 0
        for (path, layer) in skeleton.leafModules().flattened() {
            guard let linear = layer as? Linear, !(layer is QuantizedLinear) else { continue }
            linears += 1
            let shape = linear.weight.shape
            if rule.excludes(path) { kept[path] = "excluded by rule"; continue }
            if !rule.includes(path) { kept[path] = "not selected by rule"; continue }
            guard shape.count == 2, shape[1] % group == 0 else {
                kept[path] = "input width \(shape.last ?? 0) not a multiple of \(group)"; continue
            }
            selected.append(Selection(path: path, shape: shape))
        }
        return (selected.sorted { $0.path < $1.path }, kept, linears)
    }

    /// Quantizes the selected layers of one checkpoint file's tensors. `modulePath` maps a file key to the
    /// module key it loads into (identity for most models); output keeps the file's own key names, with
    /// `<prefix>.scales` / `<prefix>.biases` beside each packed `<prefix>.weight`. Runs on the CPU stream.
    public static func quantize(_ tensors: [String: MLXArray], selected: [Selection], bits: Int, group: Int = QuantRules.groupSize,
                                modulePath: (String) -> String = { $0 }) throws -> (arrays: [String: MLXArray], parameters: Int) {
        guard QuantRules.bitsOffered.contains(bits) else { throw EngineError("Vireo quantizes to 8 or 4 bits only, not \(bits)") }
        var fileKey: [String: String] = [:]
        for key in tensors.keys { fileKey[modulePath(key)] = key }
        var out = tensors
        var parameters = 0
        for selection in selected {
            guard let key = fileKey[selection.path + ".weight"], let weight = out[key] else {
                throw EngineError("The rule selects \(selection.path), but the checkpoint has no weight for it")
            }
            guard weight.shape == selection.shape else {
                throw EngineError("\(selection.path): checkpoint weight \(weight.shape) ≠ model \(selection.shape)")
            }
            guard weight.dtype == .float16 || weight.dtype == .bfloat16 || weight.dtype == .float32 else {
                throw EngineError("\(selection.path) is already \(weight.dtype); only float weights are quantized")
            }
            let prefix = String(key.dropLast(".weight".count))
            let (wq, scales, biases) = MLX.quantized(weight, groupSize: group, bits: bits, mode: .affine, stream: .cpu)
            guard let biases else { throw EngineError("affine quantization returned no biases") }
            eval(wq, scales, biases)
            out[key] = wq
            out[prefix + ".scales"] = scales
            out[prefix + ".biases"] = biases
            parameters += weight.size
        }
        return (out, parameters)
    }

    /// Before `update(parameters:)` with a converted checkpoint: replaces every Linear whose `<path>.scales` is in
    /// `weights` (module keys) with a QuantizedLinear built directly from the stored tensors (bits inferred from
    /// the packed shape). Returns the number of layers replaced; throws on anything inconsistent.
    @discardableResult
    public static func prepare(_ module: Module, weights: [String: MLXArray], group: Int = QuantRules.groupSize) throws -> Int {
        let scaled = Set(weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(".scales".count)) })
        guard !scaled.isEmpty else { return 0 }
        var updates: [(String, Module)] = []
        for (path, layer) in module.leafModules().flattened() where scaled.contains(path) {
            guard let linear = layer as? Linear, !(layer is QuantizedLinear) else {
                throw EngineError("\(path) has quantization scales but is not a Linear layer")
            }
            guard let weight = weights[path + ".weight"], let scales = weights[path + ".scales"], let biases = weights[path + ".biases"],
                  weight.ndim == 2, scales.ndim == 2 else {
                throw EngineError("\(path): incomplete quantized tensors (weight, scales, biases)")
            }
            let inputs = linear.weight.dim(1), outputs = linear.weight.dim(0)
            guard scales.dim(1) * group == inputs, weight.dim(0) == outputs, scales.dim(0) == outputs,
                  (weight.dim(1) * 32) % inputs == 0 else {
                throw EngineError("\(path): quantized shapes \(weight.shape)/\(scales.shape) do not fit [\(outputs), \(inputs)] at group \(group)")
            }
            let bits = weight.dim(1) * 32 / inputs
            guard QuantRules.bitsOffered.contains(bits) else { throw EngineError("\(path): \(bits)-bit weights are not supported") }
            updates.append((path, QuantizedLinear(weight: weight, bias: linear.bias, scales: scales, biases: biases, groupSize: group, bits: bits)))
        }
        let found = Set(updates.map(\.0))
        if let missing = scaled.subtracting(found).sorted().first {
            throw EngineError("Converted weights quantize \(missing), which this model does not have as a Linear layer")
        }
        do { try module.update(modules: ModuleChildren.unflattened(updates), verify: .none) }
        catch { throw EngineError("Quantized layers cannot be installed: \(error)") }
        return updates.count
    }

    /// Bits of a converted checkpoint (all quantized layers share them), nil if nothing is quantized.
    public static func bits(of weights: [String: MLXArray], group: Int = QuantRules.groupSize) -> Int? {
        guard let key = weights.keys.filter({ $0.hasSuffix(".scales") }).sorted().first,
              let scales = weights[key], let weight = weights[String(key.dropLast(".scales".count)) + ".weight"] else { return nil }
        return weight.dim(-1) * 32 / (scales.dim(-1) * group)
    }

    /// Writes safetensors via a temporary name in the same directory, then renames.
    public static func save(_ arrays: [String: MLXArray], to url: URL, metadata: [String: String]) throws {
        let tmp = url.deletingLastPathComponent().appendingPathComponent(".\(url.lastPathComponent).\(UUID().uuidString).tmp.safetensors")
        do {
            try MLX.save(arrays: arrays, metadata: metadata, url: tmp, stream: .cpu)
            if FileManager.default.fileExists(atPath: url.path) { try FileManager.default.removeItem(at: url) }
            try FileManager.default.moveItem(at: tmp, to: url)
        } catch {
            try? FileManager.default.removeItem(at: tmp)
            throw error
        }
    }

    /// Indexed source reads and immediate output writes; no dense source dictionary survives a tensor.
    /// Sanitizers operate on one tensor; Breeze's tied embedding is handled explicitly by the caller.
    static func streamTensors(keys: [String], read: (String) throws -> MLXArray, output: URL,
                              bits: Int, group: Int, sanitize: ([String: MLXArray]) -> [String: MLXArray] = { $0 },
                              selects: (String, [Int]) -> Bool) throws -> (selected: [Selection], parameters: Int, packed: Int) {
        guard QuantRules.bitsOffered.contains(bits), group == 64 else { throw EngineError("Use affine int8/int4, group 64") }
        guard !keys.contains(where: { $0.hasSuffix(".scales") || $0.hasSuffix(".biases") }) else {
            throw EngineError("Source is already packed")
        }
        let writer = try TensorWriter(output)
        var selected: [Selection] = [], parameters = 0, packed = 0
        for key in keys.sorted() {
            try autoreleasepool {
                let source = try read(key)
                for (name, tensor) in sanitize([key: source]).sorted(by: { $0.key < $1.key }) {
                    let value = tensor.dtype == .float32 ? tensor.asType(.float16, stream: .cpu) : tensor
                    parameters += value.size
                    if name.hasSuffix(".weight"), value.ndim == 2, value.dim(1) % group == 0, selects(name, value.shape) {
                        guard value.dtype == .float16 || value.dtype == .bfloat16 else { throw EngineError("Only half weights can be packed") }
                        let prefix = String(name.dropLast(7))
                        let (w, scale, bias) = MLX.quantized(value, groupSize: group, bits: bits, mode: .affine, stream: .cpu)
                        guard let bias else { throw EngineError("Missing affine biases") }
                        eval(w, scale, bias)
                        try writer.append(name, w); try writer.append(prefix + ".scales", scale); try writer.append(prefix + ".biases", bias)
                        packed += value.size; selected.append(.init(path: prefix, shape: value.shape))
                    } else {
                        eval(value); try writer.append(name, value)
                    }
                }
            }
            Memory.clearCache()
        }
        try writer.finish()
        return (selected, parameters, packed)
    }

    static func convertFile(source: URL, output: URL, skeleton: Module, rule: QuantRule, bits: Int, report: inout QuantReport,
                            modulePath: (String) -> String = { $0 },
                            check: (_ saved: [String: MLXArray], _ expectedLayers: Int) throws -> Void) throws {
        let archive = try TensorArchive(source)
        let plan = plan(skeleton, rule: rule, group: report.groupSize)
        guard !plan.selected.isEmpty else { throw EngineError("Rule \(rule.id) selects no layer of \(source.lastPathComponent)") }
        let byPath = Dictionary(uniqueKeysWithValues: plan.selected.map { ($0.path + ".weight", $0.shape) })
        let converted = try streamTensors(keys: archive.entries.keys.sorted(), read: archive.array, output: output,
                                          bits: bits, group: report.groupSize) { key, shape in byPath[modulePath(key)] == shape }
        guard converted.selected.count == plan.selected.count else { throw EngineError("Checkpoint does not cover every selected layer") }
        let saved = try TensorArchive(output).arrays()
        try check(saved, plan.selected.count)
        report.files.append(.init(source: source.lastPathComponent, output: output.lastPathComponent))
        report.quantized += plan.selected.map(\.path)
        report.kept.merge(plan.kept) { a, _ in a }
        report.linears += plan.linears
        report.quantizedParameters += converted.packed
        report.parameters += converted.parameters
    }

    /// The loader's own key mapping a checkpoint rule's files go through before quantizing: Kokoro's MLX layout (its
    /// loader transposes convolutions only while no `.scales` are present), Breeze's tied embeddings; identity for Qwen3.
    public static func sanitizer(_ rule: QuantRule) -> ([String: MLXArray]) -> [String: MLXArray] {
        switch rule.id {
        case QuantRules.kokoro.id: return KokoroModel.sanitizeWeights
        case QuantRules.breeze.id: return BreezeTTSModel.sanitize
        default: return { $0 }
        }
    }

    /// Checkpoint rule (Qwen3, Kokoro, Breeze): every top-level `*.safetensors` of the 16-bit tier `source`, mapped with the
    /// loader's own `sanitize` (Kokoro's MLX layout, Breeze's tied embeddings), the rule's layers quantized (affine,
    /// group 64) and saved under the same name; `config.json` gains `quantization`; every other file (tokenizers,
    /// codecs, voices) is linked to the 16-bit tier; the stale `model.safetensors.index.json` is dropped. The loaders then
    /// quantize exactly the layers that carry `.scales`. Refuses a quantized source.
    static func convertCheckpoint(source: URL, output: URL, bits: Int, report: inout QuantReport,
                                  sanitize: ([String: MLXArray]) -> [String: MLXArray],
                                  expectedLayers: Int? = nil) throws {
        let fm = FileManager.default
        let rule = report.rule, group = report.groupSize
        let configURL = source.appendingPathComponent("config.json")
        guard var config = (try? JSONSerialization.jsonObject(with: Data(contentsOf: configURL))) as? [String: Any] else {
            throw EngineError("\(source.lastPathComponent): no readable config.json")
        }
        guard config["quantization"] == nil, config["quantization_config"] == nil else {
            throw EngineError("The source is already quantized; tiers derive only from the 16-bit checkpoint")
        }
        let files = try fm.contentsOfDirectory(at: source, includingPropertiesForKeys: nil)
            .filter { $0.pathExtension == "safetensors" }.sorted { $0.lastPathComponent < $1.lastPathComponent }
        guard !files.isEmpty else { throw EngineError("\(source.path) holds no weights") }
        for file in files {
            let archive = try TensorArchive(file)
            let tied = rule.id == QuantRules.breeze.id ? "depth_decoder.model.embed_tokens.weight" : ""
            let hasTied = files.contains { (try? TensorArchive($0).entries[tied]) != nil }
            let keys = archive.entries.keys.filter { !(hasTied && $0 == "backbone_model.embed_tokens.embed_audio_tokens.weight") }.sorted()
            let converted = try streamTensors(keys: keys, read: archive.array,
                                              output: output.appendingPathComponent(file.lastPathComponent), bits: bits, group: group,
                                              sanitize: sanitize) { key, shape in
                let path = String(key.dropLast(7))
                if rule.selects(path) { return true }
                report.kept[path] = "not selected by rule"
                return false
            }
            let selected = converted.selected
            // Verify descriptors only: no second whole-checkpoint load during conversion.
            let saved = try TensorArchive(output.appendingPathComponent(file.lastPathComponent))
            for selection in selected {
                guard saved.entries[selection.path + ".scales"] != nil, saved.entries[selection.path + ".biases"] != nil,
                      saved.entries[selection.path + ".weight"]?.dtype == .uint32 else { throw EngineError("Incomplete packed group") }
            }
            report.files.append(.init(source: file.lastPathComponent, output: file.lastPathComponent))
            report.quantized += selected.map(\.path)
            report.linears += selected.count
            report.quantizedParameters += converted.packed
            report.parameters += converted.parameters
        }
        if let expected = expectedLayers, report.quantized.count != expected {
            throw EngineError("Rule \(rule.id) selected \(report.quantized.count) layers, expected \(expected): not the checkpoint it was written for")
        }
        let quantization: [String: Any] = ["group_size": group, "bits": bits, "mode": "affine"]
        config["quantization"] = quantization
        config["quantization_config"] = quantization
        let data = try JSONSerialization.data(withJSONObject: config, options: [.prettyPrinted, .sortedKeys])
        try data.write(to: output.appendingPathComponent("config.json"))
        report.files.append(.init(source: "config.json", output: "config.json"))
        // Everything else the loader reads (tokenizer, codec, voices): links into the 16-bit tier.
        // A converted tier 16 (Kokoro) has its own manifest.json, which is not the loader's and never linked.
        let written = Set(files.map(\.lastPathComponent) + ["config.json", "model.safetensors.index.json", "manifest.json", "conversion.json"])
        guard let walker = fm.enumerator(atPath: source.path) else { throw EngineError("Cannot list \(source.path)") }
        for case let relative as String in walker {
            if written.contains(relative) || (relative as NSString).lastPathComponent.hasPrefix(".") { continue }
            let target = source.appendingPathComponent(relative).resolvingSymlinksInPath()
            var isDirectory: ObjCBool = false
            guard fm.fileExists(atPath: target.path, isDirectory: &isDirectory), !isDirectory.boolValue else { continue }
            let link = output.appendingPathComponent(relative)
            try fm.createDirectory(at: link.deletingLastPathComponent(), withIntermediateDirectories: true)
            try fm.createSymbolicLink(at: link, withDestinationURL: target)
            report.links[relative] = target.path
        }
    }

    /// Structural check used after saving: quantize a fresh skeleton from the file's scales, then a strict update.
    static func strictCheck(_ fresh: Module, _ weights: [String: MLXArray], expected: Int) throws {
        let n = try prepare(fresh, weights: weights)
        guard n == expected else { throw EngineError("Converted file installs \(n) quantized layers, expected \(expected)") }
        do { try fresh.update(parameters: ModuleParameters.unflattened(weights), verify: .all) }
        catch { throw EngineError("Converted weights do not map onto the model: \(error)") }
    }
}

extension Engines {
    /// Vireo's own quantization rule for a catalog model (nil: pinned repos or none).
    public static func quantRule(engine: String, id: String) -> QuantRule? { QuantRules.rule(engine: engine, id: id) }

    /// Converts `source` (the 16-bit tier's directory) to `bits` into `directory` (created by the caller, empty). CPU
    /// only: safe while the GPU generates. Returns what was quantized; the caller writes the manifest.
    public static func convertQuantized(_ source: ModelSource, bits: Int, into directory: URL) async throws -> QuantReport {
        guard let rule = QuantRules.rule(engine: source.engine, id: source.id) else {
            throw EngineError("\(source.id) has no Vireo quantization rule")
        }
        return try await Device.withDefaultDevice(.cpu) {
            var report = QuantReport(rule: rule, bits: bits, groupSize: QuantRules.groupSize)
            switch rule.id {
            case QuantRules.qwen3Talker.id, QuantRules.breeze.id, QuantRules.kokoro.id:
                try QuantizedWeights.convertCheckpoint(source: source.directory, output: directory, bits: bits, report: &report,
                                                       sanitize: QuantizedWeights.sanitizer(rule), expectedLayers: QuantRules.checkpointLayers[rule.id])
            case QuantRules.pocket.id: try await MLXSpeech.convertPocket(source, bits: bits, into: directory, report: &report)
            default: throw EngineError("No converter for rule \(rule.id)")
            }
            report.quantized.sort()
            return report
        }
    }

    /// Tier 16 of an FP32-only model: every FP32 tensor of `files` (catalog paths in `source.directory`) cast to FP16 into
    /// `directory`; other files (config, Kokoro's voice files) copied unchanged, so the tier stands alone once the FP32
    /// source is deleted. CPU only. Refuses a source that is not FP32 (never casts a quantized or 16-bit checkpoint).
    public static func castTo16(_ source: ModelSource, files: [String], into directory: URL) async throws -> QuantReport {
        let rule = QuantRule(id: "\(source.id)-fp16", version: 1, include: [".*"], exclude: [],
                             note: "16-bit: every FP32 tensor of the model file cast to FP16 once; other files unchanged.")
        return try await Device.withDefaultDevice(.cpu) {
            var report = QuantReport(rule: rule, bits: 16, groupSize: 0)
            report.mode = "cast"
            for path in files.sorted() {
                let input = source.directory.appendingPathComponent(path)
                let output = directory.appendingPathComponent(path)
                try FileManager.default.createDirectory(at: output.deletingLastPathComponent(), withIntermediateDirectories: true)
                let isVoice = path.hasPrefix("voices/")
                if path.hasSuffix(".safetensors") && !isVoice {
                    let tensors = try MLX.loadArrays(url: input, stream: .cpu)
                    if let bad = tensors.first(where: { $0.value.dtype != .float32 }) {
                        throw EngineError("\(path): \(bad.key) is \(bad.value.dtype); only an FP32 checkpoint is cast to FP16")
                    }
                    var cast: [String: MLXArray] = [:]
                    for (key, value) in tensors { cast[key] = value.asType(.float16, stream: .cpu) }
                    eval(Array(cast.values))
                    try QuantizedWeights.save(cast, to: output, metadata: ["format": "mlx", "vireo_tier": "16", "vireo_cast": "fp32-to-fp16",
                                                                           "vireo_source": (path as NSString).lastPathComponent])
                    report.parameters += tensors.values.reduce(0) { $0 + $1.size }
                    report.quantizedParameters += tensors.values.reduce(0) { $0 + $1.size }
                } else {
                    try FileManager.default.copyItem(at: input.resolvingSymlinksInPath(), to: output)
                }
                report.files.append(.init(source: path, output: path))
            }
            return report
        }
    }

    /// `ModelSource.options` key the helper sets for a converted precision: the directory holding its files.
    public static let quantizedDirectoryOption = "quantized_dir"
    static func quantizedDirectory(_ source: ModelSource) -> URL? {
        source.options[quantizedDirectoryOption].map { URL(fileURLWithPath: $0, isDirectory: true) }
    }
}
