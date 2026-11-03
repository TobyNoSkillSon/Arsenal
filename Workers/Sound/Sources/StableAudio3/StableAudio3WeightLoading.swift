import Foundation
import MLX
import MLXNN
import ModelPrecision

// Adapted from soniqo/speech-swift `StableAudio3WeightLoading.swift` (Apache-2.0,
// commit 2af9c6c). Vireo changes: reads Stability's official `.npz` files as well
// as the aufklarer safetensors bundles; F16 DiT loads without quantization; every
// update is strict (no missing or unexpected tensors), with an explicit report.

public enum StableAudio3Error: Error, LocalizedError {
    case missingFile(String)
    case missingTensor(String)
    case weightLoadFailed(String)
    case keyMismatch(component: String, report: SA3KeyReport)
    case tokenizerLoadFailed(String)
    case precision(String)

    public var errorDescription: String? {
        switch self {
        case .missingFile(let f): return "Stable Audio 3: missing file \(f)"
        case .missingTensor(let k): return "Stable Audio 3: missing tensor \(k)"
        case .weightLoadFailed(let m): return "Stable Audio 3: weight load failed: \(m)"
        case .keyMismatch(let c, let r): return "Stable Audio 3: \(c) keys do not match: \(r.summary)"
        case .tokenizerLoadFailed(let m): return "Stable Audio 3: tokenizer load failed: \(m)"
        case .precision(let m): return "Stable Audio 3: \(m)"
        }
    }
}

/// Result of matching checkpoint tensors against a module's parameters.
public struct SA3KeyReport: Sendable, Equatable {
    public var component: String
    public var matched: Int
    public var missing: [String]          // module parameters with no tensor
    public var unexpected: [String]       // tensors with no parameter
    public var shapeMismatches: [String]  // "key: checkpoint [..] vs model [..]"
    public var ignored: [String]          // known extra tensors, dropped on purpose
    public var dtypes: [String: Int]      // checkpoint dtype histogram

    public var ok: Bool { missing.isEmpty && unexpected.isEmpty && shapeMismatches.isEmpty }
    public var summary: String {
        var s = "\(component): \(matched) tensors matched"
        if !missing.isEmpty { s += ", \(missing.count) missing (\(missing.prefix(5).joined(separator: ", ")))" }
        if !unexpected.isEmpty { s += ", \(unexpected.count) unexpected (\(unexpected.prefix(5).joined(separator: ", ")))" }
        if !shapeMismatches.isEmpty { s += ", \(shapeMismatches.count) shape mismatches (\(shapeMismatches.prefix(3).joined(separator: "; ")))" }
        if !ignored.isEmpty { s += ", ignored \(ignored.joined(separator: ", "))" }
        s += ", dtypes " + dtypes.sorted { $0.key < $1.key }.map { "\($0.key)×\($0.value)" }.joined(separator: " ")
        return s
    }
}

// MARK: - Reading

/// Load a component from either an `.npz` file (official Stability MLX export) or
/// a directory holding `model.safetensors` (aufklarer bundle layout).
public func sa3ReadTensors(_ url: URL) throws -> [String: MLXArray] {
    var isDir: ObjCBool = false
    guard FileManager.default.fileExists(atPath: url.path, isDirectory: &isDir) else {
        throw StableAudio3Error.missingFile(url.path)
    }
    do {
        if isDir.boolValue {
            let st = url.appendingPathComponent("model.safetensors")
            guard FileManager.default.fileExists(atPath: st.path) else { throw StableAudio3Error.missingFile(st.path) }
            return try MLX.loadArrays(url: st)
        }
        switch url.pathExtension {
        case "npz": return try NPZArchive(url: url).arrays()
        case "safetensors": return try TensorArchive(url).arrays()
        default: throw StableAudio3Error.weightLoadFailed("unknown weight format \(url.lastPathComponent)")
        }
    } catch let e as StableAudio3Error {
        throw e
    } catch {
        throw StableAudio3Error.weightLoadFailed("\(url.lastPathComponent): \(error)")
    }
}

// MARK: - Matching

/// Compare checkpoint tensors with the module's parameters (after any quantization).
public func sa3KeyReport(component: String, module: Module, weights: [String: MLXArray],
                         ignored: [String] = []) -> SA3KeyReport {
    let params = Dictionary(uniqueKeysWithValues: module.parameters().flattened())
    var missing: [String] = [], unexpected: [String] = [], mism: [String] = []
    var matched = 0
    for (k, p) in params {
        guard let w = weights[k] else { missing.append(k); continue }
        if w.shape != p.shape { mism.append("\(k): checkpoint \(w.shape) vs model \(p.shape)") } else { matched += 1 }
    }
    for k in weights.keys where params[k] == nil { unexpected.append(k) }
    var dtypes: [String: Int] = [:]
    for w in weights.values { dtypes["\(w.dtype)", default: 0] += 1 }
    return SA3KeyReport(component: component, matched: matched, missing: missing.sorted(),
                        unexpected: unexpected.sorted(), shapeMismatches: mism.sorted(),
                        ignored: ignored.sorted(), dtypes: dtypes)
}

/// Strict update: refuse unless every parameter has a tensor of the right shape and
/// no tensor is left over.
func sa3ApplyStrict(component: String, module: Module, weights: [String: MLXArray],
                    ignored: [String] = []) throws -> SA3KeyReport {
    try StreamedPrecision.prepare(module, weights: weights)
    let report = sa3KeyReport(component: component, module: module, weights: weights, ignored: ignored)
    guard report.ok else { throw StableAudio3Error.keyMismatch(component: component, report: report) }
    try module.update(parameters: ModuleParameters.unflattened(weights), verify: .all)
    return report
}

// MARK: - DiT (medium)

/// Rewrite list indices (`X.0.` / `X.2.`, where index 1 is the SiLU) into named
/// children: two-linear MLPs → `inProj`/`outProj`, GeGLU feed-forward → `glu`/`out`.
public func sa3RewriteDiTMediumKeys(_ weights: [String: MLXArray]) -> [String: MLXArray] {
    var out: [String: MLXArray] = [:]
    for (k, v) in weights { out[sa3DiTModuleKey(k)] = v }
    return out
}

/// One official DiT checkpoint key → the module key it loads into (idempotent). Vireo: split out of
/// `sa3RewriteDiTMediumKeys` so the quantization converter can map single keys.
public func sa3DiTModuleKey(_ k: String) -> String {
    let owners = ["to_cond_embed", "to_global_embed", "to_timestep_embed", "transformer.global_cond_embedder"]
    func rewrite(_ k: String) -> String {
        for o in owners {
            if k.hasPrefix(o + ".0.") { return o + ".inProj." + k.dropFirst(o.count + 3) }
            if k.hasPrefix(o + ".2.") { return o + ".outProj." + k.dropFirst(o.count + 3) }
        }
        let prefix = "transformer.layers."
        guard k.hasPrefix(prefix) else { return k }
        let tail = k.dropFirst(prefix.count)
        guard let dot = tail.firstIndex(of: ".") else { return k }
        let base = prefix + tail[..<dot] + "."
        let body = String(tail[tail.index(after: dot)...])
        for (o, a, b) in [("to_local_embed.seq", "inProj", "outProj"), ("ff.ff", "glu", "out")] {
            if body.hasPrefix(o + ".0.") { return base + o + "." + a + "." + body.dropFirst(o.count + 3) }
            if body.hasPrefix(o + ".2.") { return base + o + "." + b + "." + body.dropFirst(o.count + 3) }
        }
        return k
    }
    return rewrite(k)
}

/// Tensors in the official DiT export that the MLX model does not use.
/// `transformer.rotary_pos_emb.inv_freq` is recomputed by `MLXFast.RoPE`
/// (base 10000, 32 rotary dims); `SA3KeyCheck` verifies the stored values agree.
public let sa3DiTIgnoredKeys: Set<String> = ["transformer.rotary_pos_emb.inv_freq"]

public struct SA3DiTLoad {
    public let model: DiTMedium
    public let padding: MLXArray                   // [768], stored precision (legacy fp32; intake fp16)
    public let secondsEmbedder: SecondsTotalEmbedder
    public let report: SA3KeyReport
    public let weightDType: DType                  // storage dtype of the big linears
    public let quantBits: Int?                     // nil = unquantized 16-bit
}

/// Load the medium DiT. Official F16 `.npz` → plain `Linear`s at F16. An aufklarer
/// int8/int4 bundle (`.scales` tensors present) → `quantize(model:)` first.
/// `precision` may force the storage type of a 16-bit checkpoint: "f16" (as
/// published), "bf16" (converted at load) or "fp32".
public func sa3LoadDiTMedium(_ url: URL, config: DiTMediumConfig = .medium,
                             precision: String? = nil, component: String = "dit_medium") throws -> SA3DiTLoad {
    var raw = try sa3ReadTensors(url)
    guard let padding = raw.removeValue(forKey: "cond.padding_embedding") else {
        throw StableAudio3Error.missingTensor("cond.padding_embedding")
    }
    let small = component == "dit_sm-sfx"
    let secondsKey = small && raw["cond.seconds_total.weight"] != nil ? "cond.seconds_total.weight" : "cond.seconds_total_weight"
    guard let secsW = raw.removeValue(forKey: secondsKey) else {
        throw StableAudio3Error.missingTensor("cond.seconds_total_weight")
    }
    guard let secsB = raw.removeValue(forKey: "cond.seconds_total_bias") else {
        throw StableAudio3Error.missingTensor("cond.seconds_total_bias")
    }
    let secondsScales = small ? raw.removeValue(forKey: "cond.seconds_total.scales") : nil
    let secondsAffineBiases = small ? raw.removeValue(forKey: "cond.seconds_total.biases") : nil
    var secondsBits: Int?
    if let scales = secondsScales {
        guard let offsets = secondsAffineBiases, scales.dtype == .float16, offsets.dtype == scales.dtype,
            scales.shape == [768, 4], offsets.shape == scales.shape, secsW.dtype == .uint32, secsW.ndim == 2, secsW.dim(0) == 768 else {
            throw StableAudio3Error.precision("Invalid packed Small SFX seconds conditioner")
        }
        secondsBits = secsW.dim(1) * 32 / 256
        guard secondsBits == 4 || secondsBits == 8 else { throw StableAudio3Error.precision("Invalid seconds conditioner bits") }
    } else if small && (secsW.dtype == .uint32 || secondsAffineBiases != nil) {
        throw StableAudio3Error.precision("Incomplete packed Small SFX seconds conditioner")
    }
    var ignored: [String] = []
    for k in sa3DiTIgnoredKeys where raw.removeValue(forKey: k) != nil { ignored.append(k) }

    var weights = sa3RewriteDiTMediumKeys(raw)
    let model = DiTMedium(config)

    // Quantized bundle? Infer bits from the packed weight / scales shapes (group size 64).
    let scaled = Set(weights.keys.filter { $0.hasSuffix(".scales") }.map { String($0.dropLast(".scales".count)) })
    var bits: Int? = nil
    if let any = scaled.sorted().first, let w = weights[any + ".weight"], let s = weights[any + ".scales"] {
        let groupSize = 64
        bits = w.dim(-1) * 32 / (s.dim(-1) * groupSize)
        let b = bits!
        // Vireo: build each QuantizedLinear directly from the stored tensors (RuntimeCaches rule 3: quantizing the
        // random placeholders, as quantize(model:) does, leaks per load), and throw instead of trapping.
        var updates: [(String, Module)] = []
        for (path, layer) in model.leafModules().flattened() where scaled.contains(path) {
            guard let linear = layer as? Linear, !(layer is QuantizedLinear),
                  let qw = weights[path + ".weight"], let qs = weights[path + ".scales"] else {
                throw StableAudio3Error.weightLoadFailed("quantized tensors for \(path) do not match a Linear layer")
            }
            updates.append((path, QuantizedLinear(weight: qw, bias: linear.bias, scales: qs, biases: weights[path + ".biases"],
                                                  groupSize: groupSize, bits: b)))
        }
        guard updates.count == scaled.count else {
            throw StableAudio3Error.weightLoadFailed("\(scaled.count - updates.count) quantized tensor groups match no Linear layer")
        }
        try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
    } else if let p = precision {
        let target: DType
        switch p.lowercased() {
        case "f16", "fp16", "float16": target = .float16
        case "bf16", "bfloat16": target = .bfloat16
        case "fp32", "f32", "float32": target = .float32
        default: throw StableAudio3Error.precision("unsupported DiT precision \(p); use f16, bf16 or fp32 (8-bit needs a quantized bundle)")
        }
        weights = weights.mapValues { $0.dtype.sa3IsFloat ? $0.asType(target) : $0 }
    }
    let report = try sa3ApplyStrict(component: component, module: model, weights: weights, ignored: ignored)
    eval(model.parameters())
    let storage = weights["transformer.layers.0.self_attn.to_qkv.weight"]?.dtype ?? .float16
    return SA3DiTLoad(model: model, padding: padding,
                      secondsEmbedder: SecondsTotalEmbedder(weight: secsW, bias: secsB, scales: secondsScales,
                          affineBiases: secondsAffineBiases, quantBits: secondsBits, halfOperands: small),
                      report: report, weightDType: bits == nil ? storage : .uint32, quantBits: bits)
}

// MARK: - SAME-L decoder (legacy FP32; intake half/packed weights with safe residual accumulation)

public func sa3LoadSAMELDecoder(_ url: URL) throws -> (SAMELDecoder, SA3KeyReport) {
    var raw = try sa3ReadTensors(url)
    if let w = raw["mapping.weight"], w.ndim == 3, w.dim(2) == 1 {
        raw["mapping.weight"] = w.reshaped([w.dim(0), w.dim(1)])
    }
    // Legacy source keeps its original reference numerics. Intake snapshots store half/packed matrices.
    // Differential subtraction and residuals still accumulate safely; no dense FP32 matrix copy is retained.
    let model = SAMELDecoder()
    try sa3PrepareHalfLinears(model, weights: raw)
    let report = try sa3ApplyStrict(component: "same_l_decoder", module: model, weights: raw)
    eval(model.parameters())
    return (model, report)
}

// MARK: - SAME-S decoder (half/packed intake weights)

/// The mapping Conv1d is stored in PyTorch layout [out, in, k] (weight norm already fused); MLX wants [out, k, in].
public func sa3SAMESDecoderWeights(_ raw: [String: MLXArray], config: SAMESConfig = .small) -> [String: MLXArray] {
    var out = raw
    if let w = raw["mapping.weight"], w.ndim == 3, w.dim(1) == config.dim, w.dim(2) == 3 {
        out["mapping.weight"] = w.transposed(0, 2, 1)
    }
    // Preserve half weights and affine metadata. Widening these at Load defeats the Get recipe.
    return out
}

public func sa3LoadSAMESDecoder(_ url: URL, config: SAMESConfig = .small) throws -> (SAMESDecoder, SA3KeyReport) {
    var raw = try sa3ReadTensors(url)
    guard raw.values.allSatisfy({ [.float16, .bfloat16, .uint32].contains($0.dtype) }) else {
        throw StableAudio3Error.precision("SAME-S needs a half/packed Get snapshot; Get the updated model before Load")
    }
    let model = SAMESDecoder(config)
    try StreamedPrecision.prepare(model, weights: raw)
    raw = sa3SAMESDecoderWeights(raw, config: config)
    // QuantizedLinear already installed; strict application must not replace the stored affine metadata.
    let report = sa3KeyReport(component: "same_s_decoder", module: model, weights: raw)
    guard report.ok else { throw StableAudio3Error.keyMismatch(component: "same_s_decoder", report: report) }
    try model.update(parameters: ModuleParameters.unflattened(raw), verify: .all)
    eval(model.parameters())
    return (model, report)
}

// MARK: - T5Gemma (F16) + bundled SentencePiece model

public func sa3LoadT5Gemma(_ url: URL) throws -> (T5GemmaText, SA3KeyReport) {
    var raw = try sa3ReadTensors(url)
    guard let tokBlob = raw.removeValue(forKey: "TOKENIZER_MODEL") else {
        throw StableAudio3Error.missingTensor("TOKENIZER_MODEL")
    }
    var ignored = ["TOKENIZER_MODEL"]
    if let meta = raw.removeValue(forKey: "META") {
        ignored.append("META")
        try sa3CheckT5GemmaMeta(Data(meta.asType(.uint8).asArray(UInt8.self)))
    }
    var invFreq: [Float]? = nil
    if let f = raw.removeValue(forKey: "rope_inv_freq") {
        ignored.append("rope_inv_freq (used for RoPE)")
        invFreq = f.asType(.float32).asArray(Float.self)
    }
    let tokenizer: UnigramTokenizer
    do {
        let spModel = try SentencePieceModel(data: Data(tokBlob.asType(.uint8).asArray(UInt8.self)))
        tokenizer = UnigramTokenizer(model: spModel)
    } catch {
        throw StableAudio3Error.tokenizerLoadFailed("\(error)")
    }
    let model = T5GemmaEncoderModel()
    model.ropeInvFreq = invFreq
    let report = try sa3ApplyStrict(component: "t5gemma", module: model, weights: raw, ignored: ignored)
    eval(model.parameters())
    return (T5GemmaText(encoder: model, tokenizer: tokenizer), report)
}

/// The META blob is the Hugging Face config JSON; refuse a checkpoint whose shape
/// differs from the hard-coded `T5GemmaDims`.
func sa3CheckT5GemmaMeta(_ data: Data) throws {
    guard let obj = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else { return }
    let expect: [(String, Int)] = [
        ("hidden_size", T5GemmaDims.hiddenSize), ("num_hidden_layers", T5GemmaDims.numLayers),
        ("num_attention_heads", T5GemmaDims.numAttentionHeads), ("num_key_value_heads", T5GemmaDims.numKeyValueHeads),
        ("head_dim", T5GemmaDims.headDim), ("intermediate_size", T5GemmaDims.intermediateSize),
        ("vocab_size", T5GemmaDims.vocabSize),
    ]
    for (k, v) in expect {
        if let got = obj[k] as? Int, got != v {
            throw StableAudio3Error.weightLoadFailed("t5gemma META \(k)=\(got), port expects \(v)")
        }
    }
}

extension DType {
    var sa3IsFloat: Bool { self == .float16 || self == .bfloat16 || self == .float32 || self == .float64 }
}

/// Half-stored decoder GEMMs use the decoder stream dtype without promoting dense weights.
private final class SA3HalfLinear: Linear {
    override func callAsFunction(_ x: MLXArray) -> MLXArray {
        super.callAsFunction(x.asType(weight.dtype)).asType(x.dtype)
    }
}
private func sa3PrepareHalfLinears(_ model: Module, weights: [String: MLXArray]) throws {
    var updates: [(String, Module)] = []
    for (path, layer) in model.leafModules().flattened() {
        if let linear = layer as? Linear, let w = weights[path + ".weight"], w.dtype == .float16 || w.dtype == .bfloat16 {
            updates.append((path, SA3HalfLinear(weight: w, bias: linear.bias)))
        }
    }
    try model.update(modules: ModuleChildren.unflattened(updates), verify: .none)
}
