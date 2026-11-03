// Inference port of laya-mlx 0.2.0 model.py (Apache-2.0; Convai Innovations / mizorewww).
import Foundation
import MLX
import MLXNN
import ModelPrecision
import ArsenalKernels

enum LayaError: Error, LocalizedError {
    case invalid(String)
    var errorDescription: String? { if case .invalid(let message) = self { return message }; return nil }
}

struct LayaEncoderConfiguration: Decodable {
    let hidden_size: Int
    let num_hidden_layers: Int
    let num_attention_heads: Int
    let intermediate_size: Int
    let vocab_size: Int
    var model_type: String = "modernbert"
    var norm_eps: Float = 1e-5
    var local_attention: Int = 128
    var global_attn_every_n_layers: Int = 3
    var global_rope_theta: Float = 160000
    var local_rope_theta: Float = 10000
    var layer_types: [String]?
    /// The positions the checkpoint was configured for (Von 1.1: 2048); nil when the config omits it.
    var max_position_embeddings: Int?
    var rope_parameters: [String: RopeParameters]?
    struct RopeParameters: Decodable { let rope_theta: Float; var rope_type: String? }

    init(data: Data) throws {
        let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] ?? [:]
        func integer(_ key: String) throws -> Int {
            guard let result = object[key] as? Int, result > 0 else { throw LayaError.invalid("Missing encoder field \(key)") }
            return result
        }
        hidden_size = try integer("hidden_size")
        num_hidden_layers = try integer("num_hidden_layers")
        num_attention_heads = try integer("num_attention_heads")
        intermediate_size = try integer("intermediate_size")
        vocab_size = try integer("vocab_size")
        model_type = object["model_type"] as? String ?? "modernbert"
        norm_eps = (object["norm_eps"] as? NSNumber)?.floatValue ?? 1e-5
        local_attention = object["local_attention"] as? Int ?? 128
        global_attn_every_n_layers = object["global_attn_every_n_layers"] as? Int ?? 3
        global_rope_theta = (object["global_rope_theta"] as? NSNumber)?.floatValue ?? 160000
        local_rope_theta = (object["local_rope_theta"] as? NSNumber)?.floatValue ?? 10000
        layer_types = object["layer_types"] as? [String]
        max_position_embeddings = (object["max_position_embeddings"] as? Int).flatMap { $0 > 0 ? $0 : nil }
        if let parameters = object["rope_parameters"] {
            rope_parameters = try JSONDecoder().decode([String: RopeParameters].self, from: JSONSerialization.data(withJSONObject: parameters))
        }
        guard model_type == "modernbert", object["hidden_activation"] as? String ?? "gelu" == "gelu",
              hidden_size % num_attention_heads == 0, (hidden_size / num_attention_heads) % 2 == 0,
              global_attn_every_n_layers > 0 else { throw LayaError.invalid("Unsupported ModernBERT configuration") }
        if let types = layer_types, types.count != num_hidden_layers || types.contains(where: { !["full_attention", "sliding_attention"].contains($0) }) {
            throw LayaError.invalid("Invalid ModernBERT layer_types")
        }
        if rope_parameters?.values.contains(where: { ($0.rope_type ?? "default") != "default" }) == true {
            throw LayaError.invalid("Only default (unscaled) ModernBERT RoPE is supported")
        }
    }

    func kind(_ layer: Int) -> String { layer_types?[layer] ?? (layer % global_attn_every_n_layers == 0 ? "full_attention" : "sliding_attention") }
    func ropeBase(_ kind: String) -> Float { rope_parameters?[kind]?.rope_theta ?? (kind == "full_attention" ? global_rope_theta : local_rope_theta) }
}

/// Frozen, inference-only weights. Serial use is owned by the helper's model service.
/// Uses the exact MLX Linear / QuantizedLinear primitives used by the Python model.
final class LayaNetwork {
    /// Tolerance-curve candidates, not defaults: VERDICT_LAYA_DTYPE=bf16 runs weights/activations in bfloat16;
    /// VERDICT_LAYA_QUANT_SCOPE=mlp|attn quantizes (at 8/4 bits) only the MLP or only the attention Linears.
    static let activationType: DType = ProcessInfo.processInfo.environment["VERDICT_LAYA_DTYPE"] == "bf16" ? .bfloat16 : .float16
    static let quantScope = ProcessInfo.processInfo.environment["VERDICT_LAYA_QUANT_SCOPE"] ?? "all"
    static func quantizes(_ name: String) -> Bool {
        let mlp = name.contains(".mlp.") || name.hasSuffix(".linear1") || name.hasSuffix(".linear2")
        let attention = name.contains(".attn.") || name.contains(".self_attn.")
        switch quantScope { case "mlp": return mlp; case "attn": return attention; default: return true }
    }
    let config: LayaEncoderConfiguration
    let headLayers: Int
    private var arrays: [String: MLXArray] = [:]
    private var linears: [String: Linear] = [:]
    static let embeddingKey = "encoder.embeddings.tok_embeddings.weight"
    /// The packed token embedding at 8/4 bits; nil at 16 bits (the dense table stays in `arrays`).
    private(set) var embedding: QuantizedEmbedding?
    private(set) var residentBytes = 0
    /// Active attention path for the sliding-window layers, reported in /status.
    var kernelPath: String {
        guard stockForced && windowedValidated else { return validatedKernelPath }
        return standardSelected ? "stock (Standard selected)" : "stock (windowed attention switched off after an inference failure)"
    }
    /// The model's recipe is Standard: the stock path is the choice, not a fallback.
    var standardSelected = false
    private var validatedKernelPath = "stock"
    /// Windowed attention passed its load-time self-test.
    private(set) var windowedValidated = false
    private var stockForced = false
    /// True while sliding-window layers take the windowed path.
    var windowedActive: Bool { windowedValidated && !stockForced }
    /// Stock attention and matmuls (true) or back to the validated optimized components (false).
    func useStockAttention(_ on: Bool) {
        stockForced = on; smallM.stockForced = on; graph.windowed = windowedActive; graph.smallM = smallMActive; graph.fuseGeGLU = fusedGeGLU && !on
        graph.exactDown = on ? nil : exactDown
    }
    /// The folder's exact split-K kernel for `linear.down` once its Exact gate passed (FolderKernelRoute).
    private var exactDown: FolderMLXKernel?
    var exactDownActive: Bool { graph.exactDown != nil }
    func activateExactDown(_ kernel: FolderMLXKernel) { exactDown = kernel; graph.exactDown = stockForced ? nil : kernel }
    /// The kernel on (candidate) or off for the Exact gate's two runs.
    func withExactDown<T>(_ kernel: FolderMLXKernel?, _ body: () throws -> T) rethrows -> T {
        let saved = graph.exactDown; graph.exactDown = kernel; defer { graph.exactDown = saved }
        return try body()
    }
    private var graph: LayaGraph!
    /// Small-row Linears on the family small-M GEMM (SmallMRoute), gated per model and precision.
    private let smallM = SmallMComponent()
    var smallMActive: Bool { smallM.active }
    var smallMPath: String { smallM.path }
    /// Non-finite output with the kernel on: serve without it until reload.
    func disableSmallM() { smallM.disable(); graph.smallM = false }
    /// Kernel on/off for the load-time comparison (the gate's model stage).
    func withSmallM<T>(_ on: Bool, _ body: () throws -> T) rethrows -> T {
        let saved = graph.smallM; graph.smallM = on; defer { graph.smallM = saved }
        return try body()
    }
    func qualifySmallM(model: String, bits: Int, allowed: Bool, modelCheck: () -> (pass: Bool, detail: String)) {
        let half = LayaNetwork.activationType == .float16 || LayaNetwork.activationType == .bfloat16
        smallM.qualify(model: model, bits: bits, defaultOn: true, allowed: allowed, ineligible: !half || !(bits == 0 || bits == 16) ? "\(bits)-bit weights" : nil, modelCheck: modelCheck)
        graph.smallM = smallMActive
    }

    static func parameterKey(_ original: String) -> String {
        var key = original.replacingOccurrences(of: ".in_proj_weight", with: ".in_proj.weight")
            .replacingOccurrences(of: ".in_proj_bias", with: ".in_proj.bias")
        for prefix in ["scorer", "act_head"] where key.hasPrefix(prefix + ".") && !key.hasPrefix(prefix + ".layers.") {
            key = prefix + ".layers." + key.dropFirst(prefix.count + 1)
        }
        return key
    }
    static func packedEmbedding(snapshot: URL, archive: TensorArchive, key: String, bits: Int, blockRows: Int = 8192) throws -> QuantizedEmbedding {
        guard let entry = archive.entries[key], entry.shape.count == 2, entry.shape[0] > 0, entry.shape[1] % 64 == 0 else {
            throw LayaError.invalid("Missing or invalid token embedding")
        }
        let handle = try FileHandle(forReadingFrom: snapshot.appendingPathComponent("model.safetensors"))
        defer { try? handle.close() }
        let rows = entry.shape[0], rowBytes = entry.bytes / rows
        var weights: [MLXArray] = [], scales: [MLXArray] = [], biases: [MLXArray] = []
        for start in stride(from: 0, to: rows, by: blockRows) {
            try autoreleasepool {
                let count = min(blockRows, rows - start)
                try handle.seek(toOffset: entry.offset + UInt64(start * rowBytes))
                guard let data = try handle.read(upToCount: count * rowBytes), data.count == count * rowBytes else {
                    throw LayaError.invalid("Truncated token embedding")
                }
                let block = MLXArray(data, [count, entry.shape[1]], dtype: entry.dtype).asType(activationType)
                let (w, s, b) = MLX.quantized(block, groupSize: 64, bits: bits)
                guard let b else { throw LayaError.invalid("Affine quantization returned no biases") }
                eval(w, s, b); weights.append(w); scales.append(s); biases.append(b)
            }
        }
        let packed = QuantizedEmbedding(
            weight: concatenated(weights, axis: 0), scales: concatenated(scales, axis: 0), biases: concatenated(biases, axis: 0), groupSize: 64, bits: bits)
        eval(packed.parameters())
        return packed
    }

    init(snapshot: URL, config: LayaEncoderConfiguration, headLayers: Int, bits: Int, read: ((TensorArchive, String) throws -> MLXArray)? = nil) throws {
        self.config = config; self.headLayers = headLayers
        let archive = try TensorArchive(snapshot.appendingPathComponent("model.safetensors"))
        var sourceKeys: [String: String] = [:]
        for original in archive.entries.keys {
            let key = Self.parameterKey(original)
            guard sourceKeys.updateValue(original, forKey: key) == nil else { throw LayaError.invalid("Duplicate checkpoint parameter \(key)") }
        }
        func take(_ key: String) throws -> MLXArray? {
            guard let original = sourceKeys.removeValue(forKey: key) else { return nil }
            return try (read?(archive, original) ?? archive.array(original)).asType(Self.activationType)
        }
        // Linear layers and (Arsenal) the token embedding are quantized at 8/4 bits; LayerNorm and small tensors stay half.
        var linearNames = ["scorer.layers.1", "scorer.layers.3", "act_head.layers.0", "act_head.layers.2"]
        for i in 0..<config.num_hidden_layers {
            linearNames += ["attn.Wqkv", "attn.Wo", "mlp.Wi", "mlp.Wo"].map { "encoder.layers.\(i)." + $0 }
        }
        for i in 0..<headLayers {
            linearNames += ["self_attn.in_proj", "self_attn.out_proj", "linear1", "linear2"].map { "head.layers.\(i)." + $0 }
        }
        for name in linearNames {
            try autoreleasepool {
            guard let weight = try take(name + ".weight"), weight.ndim == 2 else {
                throw LayaError.invalid("Missing or invalid Linear weight \(name)")
            }
            let bias = try take(name + ".bias")
            if (bits == 4 || bits == 8) && weight.dim(-1) % 64 == 0 && LayaNetwork.quantizes(name) {
                linears[name] = QuantizedLinear(weight: weight, bias: bias, groupSize: 64, bits: bits)
            } else { linears[name] = Linear(weight: weight, bias: bias) }
            eval(linears[name]!.parameters())
            }
            Memory.clearCache()
        }
        // Arsenal: the token embedding is a large part (Multilingual: 197M of 322M parameters), so int8/int4 pack it too
        // and look rows up from the packed table. It is read and packed in row blocks straight from the file, so the
        // dense FP16 table never exists in memory (affine groups run along a row, so blocks pack exactly as the whole).
        if bits == 4 || bits == 8 {
            guard let original = sourceKeys.removeValue(forKey: Self.embeddingKey) else { throw LayaError.invalid("Missing token embedding") }
            embedding = try Self.packedEmbedding(snapshot: snapshot, archive: archive, key: original, bits: bits)
            Memory.clearCache()
        }
        for key in sourceKeys.keys.sorted() {
            arrays[key] = try autoreleasepool {
                let value = try take(key)!
                eval(value)
                return value
            }
        }
        var needed = ["encoder.embeddings.tok_embeddings.weight", "encoder.embeddings.norm.weight", "encoder.final_norm.weight", "type_emb.weight", "scorer.layers.0.weight"]
        for i in 0..<config.num_hidden_layers {
            needed.append("encoder.layers.\(i).mlp_norm.weight")
            if i > 0 { needed.append("encoder.layers.\(i).attn_norm.weight") }
        }
        for i in 0..<headLayers { needed += ["head.layers.\(i).norm1.weight", "head.layers.\(i).norm2.weight"] }
        for key in needed where arrays[key] == nil && !(key == Self.embeddingKey && embedding != nil) {
            throw LayaError.invalid("Missing checkpoint parameter \(key)")
        }
        var all = Array(arrays.values)
        for linear in linears.values { all += linear.parameters().flattened().map(\.1) }
        if let embedding { all += embedding.parameters().flattened().map(\.1) }
        eval(all)
        residentBytes = all.reduce(0) { $0 + $1.nbytes }
        // No MLX compile (same finding as VonNetwork): MLX's compile cache is thread_local and keyed by the
        // closure's address, while requests run on arbitrary GCD threads. Traced graphs, which hold every
        // weight, then outlive /unload and precision switches (~0.8 GB leaked per English reload), and
        // per-shape traces grow the footprint. The plain graph gives identical outputs at the same speed.
        var windowed = false
        if LayaWindowedAttention.enabled {
            let heads = config.num_attention_heads
            if let diff = LayaWindowedAttention.selfTest(half: config.local_attention / 2, heads: heads, dims: config.hidden_size / heads) {
                windowed = true
                validatedKernelPath = String(format: "windowed-attention (L>=%d, self-test max diff %.1e)", LayaWindowedAttention.minimumLength, diff)
            } else { validatedKernelPath = "stock (windowed-attention self-test failed)" }
            Memory.clearCache()   // self-test buffers; not weights
        }
        windowedValidated = windowed
        graph = LayaGraph(config: config, headLayers: headLayers, arrays: arrays, linears: linears, embedding: embedding, windowed: windowed)
    }

    /// Queue the forward on the GPU without blocking; the caller reads the result later.
    /// `unmasked`: every row fills the padded length (no padding), so full-attention layers need no mask.
    func launch(_ inputs: [MLXArray], unmasked: Bool = false) -> MLXArray {
        let result = graph.forward(inputs, unmasked: unmasked)
        asyncEval(result)
        return result
    }

    func forward(_ inputs: [MLXArray], unmasked: Bool = false) -> MLXArray {
        let result = graph.forward(inputs, unmasked: unmasked)
        eval(result)
        return result
    }

    /// Arsenal intake only: observe (and, on replay, replace) each component's input and output.
    func withIntake<T>(_ session: LayaIntakeSession, prefix: String, _ body: () throws -> T) rethrows -> T {
        graph.intake = session; graph.intakePrefix = prefix
        defer { graph.intake = nil; graph.intakePrefix = "" }
        return try body()
    }
}

/// GELU(a) * b in one kernel: MLXNN's gelu is itself a shapeless compiled closure (same formula); fusing the
/// product saves one pass over [M, I]. Captures no arrays, so the thread-local compile cache holds no weights.
private let geglu: @Sendable (MLXArray, MLXArray) -> MLXArray = compile(shapeless: true) { a, b in a * (1 + erf(a / sqrt(2))) / 2 * b }
private let addmmResidual = ProcessInfo.processInfo.environment["VERDICT_LAYA_ADDMM"] == "1"
private let fusedGeGLU = ProcessInfo.processInfo.environment["VERDICT_LAYA_GEGLU"] != "0"

private final class LayaGraph {
    let config: LayaEncoderConfiguration
    let headLayers: Int
    let arrays: [String: MLXArray]
    let linears: [String: Linear]
    var windowed: Bool
    /// Small-row Linears on SmallMGEMM (gated by LayaNetwork.qualifySmallM).
    var smallM = false
    /// K > N Linears on the folder's exact split-K kernel (bit-identical to MLX; gated by LayaModel's Exact check).
    var exactDown: FolderMLXKernel?
    /// GELU(a)*b fused (bit-identical to the two-kernel form); off on the stock path.
    var fuseGeGLU = fusedGeGLU
    /// Arsenal intake observer; nil when serving.
    var intake: LayaIntakeSession?
    var intakePrefix = ""
    private func observeInput(_ key: String, _ x: MLXArray) -> MLXArray { intake.map { $0.input(intakePrefix + key, x) } ?? x }
    private func observeOutput(_ key: String, _ x: MLXArray) { intake?.output(intakePrefix + key, x) }
    let embedding: QuantizedEmbedding?
    init(config: LayaEncoderConfiguration, headLayers: Int, arrays: [String: MLXArray], linears: [String: Linear], embedding: QuantizedEmbedding?, windowed: Bool) {
        self.embedding = embedding
        self.config = config; self.headLayers = headLayers; self.arrays = arrays; self.linears = linears; self.windowed = windowed
    }
    func norm(_ x: MLXArray, _ name: String, eps: Float = 1e-5) -> MLXArray {
        MLXFast.layerNorm(x, weight: arrays[name + ".weight"], bias: arrays[name + ".bias"], eps: eps)
    }
    func linear(_ x: MLXArray, _ name: String) -> MLXArray {
        let layer = linears[name]!
        if let k = exactDown, let y = FolderKernelRoute.splitK(x, layer, residual: nil, kernel: k) { return y }
        if smallM, let y = SmallMRoute.linear(x, layer) { return y }
        return layer(x)
    }
    /// residual + x W^T. With VERDICT_LAYA_ADDMM=1 and a plain bias-free Linear, one addmm kernel (the residual
    /// is added in the fp32 epilogue before rounding: not bit-identical); otherwise matmul then add.
    func linear(_ x: MLXArray, _ name: String, residual: MLXArray) -> MLXArray {
        let layer = linears[name]!
        if let k = exactDown, let y = FolderKernelRoute.splitK(x, layer, residual: residual, kernel: k) { return y }
        if smallM, let y = SmallMRoute.linear(x, layer, residual: residual) { return y }
        if addmmResidual, type(of: layer) == Linear.self, layer.bias == nil { return addMM(residual, x, layer.weight.T) }
        return residual + layer(x)
    }
    /// `mask` nil: no mask (unpadded chunk). `window`: the block mask (LayaWindowedAttention.mask) when this
    /// sliding layer takes the windowed path.
    func attention(_ x: MLXArray, name: String, heads: Int, mask: MLXArray?, ropeBase: Float? = nil, window: MLXArray? = nil, residual: MLXArray) -> MLXArray {
        let b = x.dim(0), length = x.dim(1), dims = x.dim(2), d = dims / heads
        let qkv = linear(x, name + (ropeBase == nil ? ".in_proj" : ".Wqkv")).reshaped(b, length, 3, heads, d)
        var q = qkv[0..., 0..., 0, 0..., 0...].transposed(0, 2, 1, 3)
        var k = qkv[0..., 0..., 1, 0..., 0...].transposed(0, 2, 1, 3)
        let v = qkv[0..., 0..., 2, 0..., 0...].transposed(0, 2, 1, 3)
        if let base = ropeBase {
            q = MLXFast.RoPE(q, dimensions: d, traditional: false, base: base, scale: 1, offset: 0)
            k = MLXFast.RoPE(k, dimensions: d, traditional: false, base: base, scale: 1, offset: 0)
        }
        if let blockMask = window {
            let attended = LayaWindowedAttention.attend(q: q, k: k, v: v, mask: blockMask, half: config.local_attention / 2, scale: pow(Float(d), -0.5))
            return linear(attended, name + ".Wo", residual: residual)
        }
        let attended = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: pow(Float(d), -0.5), mask: mask.map { .array($0) } ?? .none)
        return linear(attended.transposed(0, 2, 1, 3).reshaped(b, length, dims), name + (ropeBase == nil ? ".out_proj" : ".Wo"), residual: residual)
    }
    func forward(_ inputs: [MLXArray], unmasked: Bool) -> MLXArray {
        let ids = inputs[0], valid = inputs[1], markers = inputs[2], markerMask = inputs[3], qtype = inputs[4]
        let b = ids.dim(0), length = ids.dim(1)
        let fullMask = valid.reshaped(b, 1, 1, length)
        let full: MLXArray? = unmasked ? nil : fullMask
        let windowed = self.windowed && length >= LayaWindowedAttention.minimumLength
        var local: MLXArray? = nil, blockMask: MLXArray? = nil
        if windowed {
            blockMask = LayaWindowedAttention.mask(valid: valid, heads: config.num_attention_heads, half: config.local_attention / 2)
        } else {
            let positions = MLXArray(0..<length)
            let localDistances = abs(positions.expandedDimensions(axis: 1) - positions.expandedDimensions(axis: 0)) .<= (config.local_attention / 2)
            local = logicalAnd(logicalOr(localDistances.reshaped(1, 1, length, length), logicalNot(valid.reshaped(b, 1, length, 1))), fullMask)
        }
        var x = embedding.map { $0(ids) } ?? arrays[LayaNetwork.embeddingKey]![ids]
        x = norm(x, "encoder.embeddings.norm", eps: config.norm_eps)
        observeOutput("embedding", x)
        for i in 0..<config.num_hidden_layers {
            x = observeInput("layer-\(i)", x)
            let prefix = "encoder.layers.\(i)", kind = config.kind(i)
            let normalized = i == 0 ? x : norm(x, prefix + ".attn_norm", eps: config.norm_eps)
            let global = kind == "full_attention"
            x = attention(normalized, name: prefix + ".attn", heads: config.num_attention_heads, mask: global ? full : local,
                          ropeBase: config.ropeBase(kind), window: global ? nil : blockMask, residual: x)
            let mlp = linear(norm(x, prefix + ".mlp_norm", eps: config.norm_eps), prefix + ".mlp.Wi")
            let halves = split(mlp, parts: 2, axis: -1)
            x = linear(fuseGeGLU ? geglu(halves[0], halves[1]) : gelu(halves[0]) * halves[1], prefix + ".mlp.Wo", residual: x)
            observeOutput("layer-\(i)", x)
        }
        x = norm(x, "encoder.final_norm", eps: config.norm_eps)
        x = x + arrays["type_emb.weight"]![qtype].expandedDimensions(axis: 1)
        for i in 0..<headLayers {
            x = observeInput("head-\(i)", x)
            let prefix = "head.layers.\(i)"
            x = attention(norm(x, prefix + ".norm1"), name: prefix + ".self_attn", heads: max(1, config.hidden_size / 64), mask: full, residual: x)
            // residual + layer(y): the same expression as x + linear(...) on the stock path.
            x = linear(relu(linear(norm(x, prefix + ".norm2"), prefix + ".linear1")), prefix + ".linear2", residual: x)
            observeOutput("head-\(i)", x)
        }
        x = observeInput("scorer", x)
        let gathered = x[MLXArray(0..<b).expandedDimensions(axis: 1), maximum(markers, 0)]
        let logits = linear(gelu(linear(norm(gathered, "scorer.layers.0"), "scorer.layers.1")), "scorer.layers.3").squeezed(axis: -1).asType(.float32)
        // act_head is loaded/quantized to preserve residency, but its outputs are discarded by the reference worker.
        let result = MLX.where(markerMask, logits, MLXArray(Float(-1e4)))
        observeOutput("scorer", result)
        return result
    }
}
