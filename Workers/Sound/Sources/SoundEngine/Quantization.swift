// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXNN
import VireoCore

// Legacy tier derivation, converted once at Get and loaded like any checkpoint.
// The all-component intake-v1 recipe is implemented separately by StableAudio3Engine.convertIntake;
// it includes eligible encoder/decoder matrices and half-stored decoder weights, with its own quality evidence.
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

    /// Stable Audio 3 Medium: the DiT blocks' attention and feed-forward projections. Applies to the DiT file only;
    /// the T5Gemma text encoder (F16) and the SAME-L decoder (FP32) are not touched.
    public static let stableAudio3 = QuantRule(
        id: "stable-audio-3-dit", version: 1,
        include: [#"transformer\.layers\.\d+\.(self_attn\.(to_qkv|to_out)|cross_attn\.(to_q|to_kv|to_out)|ff\.ff\.(glu\.proj|out))"#],
        exclude: [#"transformer\.layers\.\d+\.to_local_embed(\..*)?"#,                 // per-block local conditioning MLP (257-wide input)
                  #"transformer\.(project_in|project_out)"#,                         // latent in/out projections
                  #"transformer\.global_cond_embedder(\..*)?"#,                          // scale/shift/gate modulation
                  #"(to_cond_embed|to_global_embed|to_timestep_embed)(\..*)?"#],        // conditioning embedders
        note: "8/4-bit: the 24 DiT blocks' attention and feed-forward projections (168 Linear layers, 1.36 B of 1.45 B DiT parameters); SAME-L decoder (FP32), T5Gemma text encoder, conditioning embedders and input/output projections stay at base precision.")

    /// Stable Audio 3 Small SFX: the same layer selection over the small DiT (20 blocks, plain attention). Applies to the
    /// DiT file only; the T5Gemma text encoder (F16) and the SAME-S decoder (FP32) are not touched.
    public static let stableAudio3Small = QuantRule(
        id: "stable-audio-3-small-dit", version: 1, include: stableAudio3.include, exclude: stableAudio3.exclude,
        note: "8/4-bit: the 20 DiT blocks' attention and feed-forward projections (140 Linear layers, 419 M of 459 M DiT parameters); SAME-S decoder (FP32), T5Gemma text encoder, conditioning embedders and input/output projections stay at base precision.")

    /// RE-USE: the Mamba input/output projections and each bidirectional block's output projection.
    public static let reUse = QuantRule(
        id: "re-use", version: 1,
        include: [#"tfMamba\.\d+\.(timeMamba|freqMamba)\.((forward|backward)\.(inProj|outProj)|outputProj)"#],
        exclude: [#".*\.xProj"#,                                  // SSM Δ/B/C projection; the Metal scan needs its FP32 [36, 256] layout
                  #".*\.dtProj"#,                                 // 4-wide input
                  #"(denseEncoder|maskDecoder|phaseDecoder)(\..*)?"#], // convolutional encoder and magnitude/phase decoders
        note: "8/4-bit: the Mamba input/output and block output projections (300 Linear layers, 6.4 M of 9.6 M parameters); SSM projections, convolutions, encoder, decoders and norms stay FP32.")


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

    public var manifest: [String: Any] {
        var reasons: [String: Int] = [:]
        for reason in kept.values { reasons[reason, default: 0] += 1 }
        return ["rule": rule.manifest, "bits": bits, "group_size": groupSize, "mode": mode,
                "files": files.map { ["source": $0.source, "output": $0.output] }, "links": links,
                "linear_layers": linears, "quantized_layers": quantized.count, "quantized": quantized,
                "kept_layers": kept.count, "kept_reasons": reasons, "kept": kept,
                "quantized_parameters": quantizedParameters, "parameters": parameters]
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

    /// Converts one checkpoint file: read → plan on `skeleton` → quantize → save → verify the saved file maps
    /// strictly onto a quantized copy of the skeleton (`check`). Everything on the CPU.
    static func convertFile(source: URL, output: URL, skeleton: Module, rule: QuantRule, bits: Int, report: inout QuantReport,
                            read: (URL) throws -> [String: MLXArray] = { try MLX.loadArrays(url: $0, stream: .cpu) },
                            modulePath: (String) -> String = { $0 },
                            check: (_ saved: [String: MLXArray], _ expectedLayers: Int) throws -> Void) throws {
        let tensors = try read(source)
        let plan = plan(skeleton, rule: rule, group: report.groupSize)
        guard !plan.selected.isEmpty else { throw EngineError("Rule \(rule.id) selects no layer of \(source.lastPathComponent)") }
        let (arrays, quantizedParameters) = try quantize(tensors, selected: plan.selected, bits: bits, group: report.groupSize, modulePath: modulePath)
        let metadata = ["format": "mlx", "vireo_quant_rule": "\(rule.id)/\(rule.version)", "vireo_quant_bits": String(bits),
                        "vireo_quant_group_size": String(report.groupSize), "vireo_quant_source": source.lastPathComponent]
        try save(arrays, to: output, metadata: metadata)
        let saved = try MLX.loadArrays(url: output, stream: .cpu)
        try check(saved, plan.selected.count)
        report.files.append(.init(source: source.lastPathComponent, output: output.lastPathComponent))
        report.quantized += plan.selected.map(\.path)
        report.kept.merge(plan.kept) { a, _ in a }
        report.linears += plan.linears
        report.quantizedParameters += quantizedParameters
        report.parameters += tensors.values.reduce(0) { $0 + $1.size }
    }

    /// Structural check used after saving: quantize a fresh skeleton from the file's scales, then a strict update.
    static func strictCheck(_ fresh: Module, _ weights: [String: MLXArray], expected: Int) throws {
        let n = try prepare(fresh, weights: weights)
        guard n == expected else { throw EngineError("Converted file installs \(n) quantized layers, expected \(expected)") }
        do { try fresh.update(parameters: ModuleParameters.unflattened(weights), verify: .all) }
        catch { throw EngineError("Converted weights do not map onto the model: \(error)") }
    }
}
