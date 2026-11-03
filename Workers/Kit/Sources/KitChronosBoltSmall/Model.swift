// SPDX-License-Identifier: MIT
// Chronos-Bolt Small (Amazon, Apache-2.0) ported to MLX from the maker's chronos-forecasting code
// (src/chronos/chronos_bolt.py, git 1224faf: ChronosBoltModelForForecasting and ChronosBoltPipeline.predict)
// and transformers' T5Stack (models/t5/modeling_t5.py). Instance scaling runs on the host in Double; the
// patch embedding, T5 encoder, one-step T5 decoder and quantile head run on the float16 weights.
// Horizons past the checkpoint's 64 steps use the maker's rollout heuristic (each quantile appended to the
// context, 81 forecasts reduced to 9 by torch.quantile's linear interpolation).
import Foundation
import MLX
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.forecast"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["forecast"]

struct BoltConfig {
    var dModel = 512
    var dFF = 2048
    var heads = 8
    var dKV = 64
    var encoderLayers = 6
    var decoderLayers = 6
    var buckets = 32
    var maxDistance = 128
    var eps: Float = 1e-6
    var contextLength = 2048
    var patchSize = 16
    var predictionLength = 64
    var quantiles: [Double] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    var useRegToken = true
    var regTokenId = 1
    var decoderStartTokenId = 0

    init(_ json: [String: Any]) throws {
        func int(_ key: String) -> Int? { (json[key] as? NSNumber)?.intValue }
        dModel = int("d_model") ?? dModel
        dFF = int("d_ff") ?? dFF
        heads = int("num_heads") ?? heads
        dKV = int("d_kv") ?? dKV
        encoderLayers = int("num_layers") ?? encoderLayers
        decoderLayers = int("num_decoder_layers") ?? encoderLayers
        buckets = int("relative_attention_num_buckets") ?? buckets
        maxDistance = int("relative_attention_max_distance") ?? maxDistance
        eps = (json["layer_norm_epsilon"] as? NSNumber)?.floatValue ?? eps
        decoderStartTokenId = int("decoder_start_token_id") ?? decoderStartTokenId
        guard (json["dense_act_fn"] as? String ?? "relu") == "relu", json["is_gated_act"] as? Bool != true else {
            throw RunnerError.failed("Only the ReLU T5 feed-forward of Chronos-Bolt is ported.")
        }
        guard let chronos = json["chronos_config"] as? [String: Any] else { throw RunnerError.failed("config.json has no chronos_config.") }
        contextLength = (chronos["context_length"] as? NSNumber)?.intValue ?? contextLength
        patchSize = (chronos["input_patch_size"] as? NSNumber)?.intValue ?? patchSize
        guard (chronos["input_patch_stride"] as? NSNumber)?.intValue ?? patchSize == patchSize else {
            throw RunnerError.failed("Only input_patch_stride == input_patch_size is ported.")
        }
        predictionLength = (chronos["prediction_length"] as? NSNumber)?.intValue ?? predictionLength
        quantiles = (chronos["quantiles"] as? [NSNumber])?.map { $0.doubleValue } ?? quantiles
        useRegToken = chronos["use_reg_token"] as? Bool ?? false
        regTokenId = int("reg_token_id") ?? 1
    }
}

/// Linear → ReLU → Linear plus a linear skip (ResidualBlock in chronos_bolt.py, no layer norm).
final class ResidualBlock: Module {
    @ModuleInfo(key: "hidden_layer") var hiddenLayer: Linear
    @ModuleInfo(key: "output_layer") var outputLayer: Linear
    @ModuleInfo(key: "residual_layer") var residualLayer: Linear
    init(_ input: Int, _ hidden: Int, _ output: Int) {
        _hiddenLayer.wrappedValue = Linear(input, hidden)
        _outputLayer.wrappedValue = Linear(hidden, output)
        _residualLayer.wrappedValue = Linear(input, output)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { outputLayer(relu(hiddenLayer(x))) + residualLayer(x) }
}

/// T5LayerNorm: RMS over the last axis with the variance in float32, then cast and scale.
final class T5LayerNorm: Module {
    let eps: Float
    @ParameterInfo(key: "weight") var weight: MLXArray
    init(_ dims: Int, eps: Float) {
        self.eps = eps
        _weight.wrappedValue = MLXArray.ones([dims])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let variance = (x.asType(.float32) * x.asType(.float32)).mean(axis: -1, keepDims: true)
        return weight * (x.asType(.float32) * rsqrt(variance + eps)).asType(weight.dtype)
    }
}

/// T5Attention: no 1/sqrt(d) scaling; an additive bias [B or 1, heads, q, k]; softmax in float32.
final class T5Attention: Module {
    let heads: Int
    let dKV: Int
    @ModuleInfo(key: "q") var q: Linear
    @ModuleInfo(key: "k") var k: Linear
    @ModuleInfo(key: "v") var v: Linear
    @ModuleInfo(key: "o") var o: Linear
    @ModuleInfo(key: "relative_attention_bias") var relativeAttentionBias: Embedding?

    init(_ c: BoltConfig, relativeBias: Bool) {
        heads = c.heads
        dKV = c.dKV
        let inner = c.heads * c.dKV
        _q.wrappedValue = Linear(c.dModel, inner, bias: false)
        _k.wrappedValue = Linear(c.dModel, inner, bias: false)
        _v.wrappedValue = Linear(c.dModel, inner, bias: false)
        _o.wrappedValue = Linear(inner, c.dModel, bias: false)
        _relativeAttentionBias.wrappedValue = relativeBias ? Embedding(embeddingCount: c.buckets, dimensions: c.heads) : nil
    }

    func callAsFunction(_ x: MLXArray, context: MLXArray, bias: MLXArray?) -> MLXArray {
        let b = x.dim(0), lq = x.dim(1), lk = context.dim(1)
        let queries = q(x).reshaped([b, lq, heads, dKV]).transposed(0, 2, 1, 3)
        let keys = k(context).reshaped([b, lk, heads, dKV]).transposed(0, 2, 1, 3)
        let values = v(context).reshaped([b, lk, heads, dKV]).transposed(0, 2, 1, 3)
        var scores = matmul(queries, keys.transposed(0, 1, 3, 2))
        if let bias { scores = scores + bias.asType(scores.dtype) }
        let weights = softmax(scores.asType(.float32), axis: -1).asType(values.dtype)
        let out = matmul(weights, values).transposed(0, 2, 1, 3).reshaped([b, lq, heads * dKV])
        return o(out)
    }
}

/// T5DenseActDense with ReLU.
final class DenseReluDense: Module {
    @ModuleInfo(key: "wi") var wi: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    init(_ c: BoltConfig) {
        _wi.wrappedValue = Linear(c.dModel, c.dFF, bias: false)
        _wo.wrappedValue = Linear(c.dFF, c.dModel, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { wo(relu(wi(x))) }
}

/// transformers clamps float16 hidden states to ±(max − 1000) when any value overflowed.
private func clampHalf(_ x: MLXArray) -> MLXArray {
    guard x.dtype == .float16 else { return x }
    return clip(x, min: -64_504, max: 64_504)
}

/// Pre-norm residual attention sublayer (T5LayerSelfAttention / T5LayerCrossAttention).
final class AttentionLayer: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: T5LayerNorm
    @ModuleInfo(key: "attention") var attention: T5Attention
    init(_ c: BoltConfig, relativeBias: Bool) {
        _layerNorm.wrappedValue = T5LayerNorm(c.dModel, eps: c.eps)
        _attention.wrappedValue = T5Attention(c, relativeBias: relativeBias)
    }
    func callAsFunction(_ x: MLXArray, context: MLXArray? = nil, bias: MLXArray?) -> MLXArray {
        let normed = layerNorm(x)
        return clampHalf(x + attention(normed, context: context ?? normed, bias: bias))
    }
}

/// Pre-norm residual feed-forward sublayer (T5LayerFF).
final class FFLayer: Module {
    @ModuleInfo(key: "layer_norm") var layerNorm: T5LayerNorm
    @ModuleInfo(key: "DenseReluDense") var dense: DenseReluDense
    init(_ c: BoltConfig) {
        _layerNorm.wrappedValue = T5LayerNorm(c.dModel, eps: c.eps)
        _dense.wrappedValue = DenseReluDense(c)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { clampHalf(x + dense(layerNorm(x))) }
}

final class EncoderBlock: Module {
    @ModuleInfo(key: "attn") var attn: AttentionLayer
    @ModuleInfo(key: "ff") var ff: FFLayer
    init(_ c: BoltConfig, relativeBias: Bool) {
        _attn.wrappedValue = AttentionLayer(c, relativeBias: relativeBias)
        _ff.wrappedValue = FFLayer(c)
    }
}

final class DecoderBlock: Module {
    @ModuleInfo(key: "self_attn") var selfAttn: AttentionLayer
    @ModuleInfo(key: "cross_attn") var crossAttn: AttentionLayer
    @ModuleInfo(key: "ff") var ff: FFLayer
    init(_ c: BoltConfig, relativeBias: Bool) {
        _selfAttn.wrappedValue = AttentionLayer(c, relativeBias: relativeBias)
        _crossAttn.wrappedValue = AttentionLayer(c, relativeBias: false)
        _ff.wrappedValue = FFLayer(c)
    }
}

final class EncoderStack: Module {
    @ModuleInfo(key: "block") var block: [EncoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: T5LayerNorm
    init(_ c: BoltConfig) {
        _block.wrappedValue = (0..<c.encoderLayers).map { EncoderBlock(c, relativeBias: $0 == 0) }
        _finalLayerNorm.wrappedValue = T5LayerNorm(c.dModel, eps: c.eps)
    }
}

final class DecoderStack: Module {
    @ModuleInfo(key: "block") var block: [DecoderBlock]
    @ModuleInfo(key: "final_layer_norm") var finalLayerNorm: T5LayerNorm
    init(_ c: BoltConfig) {
        _block.wrappedValue = (0..<c.decoderLayers).map { DecoderBlock(c, relativeBias: $0 == 0) }
        _finalLayerNorm.wrappedValue = T5LayerNorm(c.dModel, eps: c.eps)
    }
}

final class BoltNetwork: Module {
    @ModuleInfo(key: "shared") var shared: Embedding
    @ModuleInfo(key: "input_patch_embedding") var inputPatchEmbedding: ResidualBlock
    @ModuleInfo(key: "encoder") var encoder: EncoderStack
    @ModuleInfo(key: "decoder") var decoder: DecoderStack
    @ModuleInfo(key: "output_patch_embedding") var outputPatchEmbedding: ResidualBlock
    init(_ c: BoltConfig) {
        _shared.wrappedValue = Embedding(embeddingCount: c.useRegToken ? 2 : 1, dimensions: c.dModel)
        _inputPatchEmbedding.wrappedValue = ResidualBlock(2 * c.patchSize, c.dFF, c.dModel)
        _encoder.wrappedValue = EncoderStack(c)
        _decoder.wrappedValue = DecoderStack(c)
        _outputPatchEmbedding.wrappedValue = ResidualBlock(c.dModel, c.dFF, c.quantiles.count * c.predictionLength)
    }
}

/// T5 relative position bucket (computed in float32 as torch does).
func relativeBucket(_ relative: Int, bidirectional: Bool, buckets: Int, maxDistance: Int) -> Int {
    var numBuckets = buckets
    var bucket = 0
    var position: Int
    if bidirectional {
        numBuckets /= 2
        if relative > 0 { bucket += numBuckets }
        position = abs(relative)
    } else {
        position = -min(relative, 0)
    }
    let maxExact = numBuckets / 2
    if position < maxExact { return bucket + position }
    let scaled = log(Float(position) / Float(maxExact)) / Float(log(Double(maxDistance) / Double(maxExact)))
        * Float(numBuckets - maxExact)
    return bucket + min(maxExact + Int(scaled), numBuckets - 1)
}

final class Model {
    let config: BoltConfig
    let net: BoltNetwork
    let maxHorizon = 1024

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, *.safetensors, tokenizer files) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        config = try BoltConfig(try Files.json(weightsDir))
        net = BoltNetwork(config)
        // The checkpoint's T5 sublayers are numbered (layer.0, layer.1, layer.2); the module names say what they are.
        try Weights.apply(try Weights.load(weightsDir), to: net, rename: { key in
            var parts = key.split(separator: ".").map(String.init)
            guard parts.count > 4, parts[1] == "block", parts[3] == "layer", let index = Int(parts[4]) else { return key }
            let names = parts[0] == "encoder" ? ["attn", "ff"] : ["self_attn", "cross_attn", "ff"]
            parts.replaceSubrange(3...4, with: [names[index]])
            if parts[4] == "SelfAttention" || parts[4] == "EncDecAttention" { parts[4] = "attention" }
            return parts.joined(separator: ".")
        })
    }

    /// One request. Return the fields of the op's output schema; throw RunnerError.invalid for bad input.
    func run(op: String, params: Params) throws -> [String: Any] {
        let values = try params.numbers("values")
        let horizon = params.int("horizon", default: 0)
        guard values.count <= 8192 else { throw RunnerError.invalid("values holds at most 8192 numbers.") }
        guard values.allSatisfy({ $0.isFinite }) else { throw RunnerError.invalid("values must be finite numbers.") }
        guard horizon >= 1, horizon <= maxHorizon else { throw RunnerError.invalid("horizon must be from 1 to \(maxHorizon) steps.") }

        // ChronosBoltPipeline.predict: first block, then the quantile rollout for longer horizons.
        let context = Array(values.suffix(config.contextLength))
        var prediction = try forward([context])[0]  // [Q][P]
        var blocks = [prediction]
        var remaining = horizon - config.predictionLength
        var contexts = [[Double]](repeating: context, count: config.quantiles.count)
        while remaining > 0 {
            try params.checkCancelled()
            contexts = contexts.indices.map { q in Array((contexts[q] + prediction[q]).suffix(config.contextLength)) }
            let samples = try forward(contexts)  // [Q rows][Q][P]
            let steps = config.predictionLength
            prediction = config.quantiles.map { level in
                (0..<steps).map { t in torchQuantile(samples.flatMap { row in row.map { $0[t] } }, level) }
            }
            blocks.append(prediction)
            remaining -= steps
        }
        let levels = config.quantiles.count
        let rows: [[Double]] = (0..<horizon).map { step in
            let block = blocks[step / config.predictionLength]
            return (0..<levels).map { block[$0][step % config.predictionLength] }
        }
        let medianIndex = config.quantiles.firstIndex { abs($0 - 0.5) < 1e-9 }
        return [
            "horizon": horizon,
            "quantile_levels": config.quantiles,
            "median": medianIndex.map { index in rows.map { $0[index] } } ?? [],
            "quantiles": rows,
        ]
    }

    /// torch.quantile with linear interpolation.
    private func torchQuantile(_ values: [Double], _ q: Double) -> Double {
        let sorted = values.sorted()
        let position = q * Double(sorted.count - 1)
        let lower = Int(position.rounded(.down)), upper = min(lower + 1, sorted.count - 1)
        return sorted[lower] + (sorted[upper] - sorted[lower]) * (position - Double(lower))
    }

    /// ChronosBoltModelForForecasting.forward for a batch of equal-length contexts → [B][Q][prediction_length] in data units.
    private func forward(_ contexts: [[Double]]) throws -> [[[Double]]] {
        let c = config
        let batch = contexts.count
        let length = contexts[0].count
        let patches = (length + c.patchSize - 1) / c.patchSize
        let padded = patches * c.patchSize
        let pad = padded - length

        // InstanceNorm (float32 in the maker's code): mean and population std; a zero std becomes eps 1e-5.
        var input = [Float](repeating: 0, count: batch * patches * 2 * c.patchSize)
        var locScale: [(Double, Double)] = []
        for (b, series) in contexts.enumerated() {
            let loc = series.reduce(0, +) / Double(length)
            var scale = (series.reduce(0) { $0 + ($1 - loc) * ($1 - loc) } / Double(length)).squareRoot()
            if scale == 0 { scale = 1e-5 }
            locScale.append((loc, scale))
            // Patch: left-pad with NaN to a whole patch; padded cells are 0 with mask 0. Row = [values ‖ mask].
            for i in 0..<padded {
                let p = i / c.patchSize, k = i % c.patchSize
                let base = (b * patches + p) * 2 * c.patchSize
                if i >= pad {
                    input[base + k] = Float((series[i - pad] - loc) / scale)
                    input[base + c.patchSize + k] = 1
                }
            }
        }
        var embeds = net.inputPatchEmbedding(MLXArray(input, [batch, patches, 2 * c.patchSize]).asType(.float16))
        if c.useRegToken {
            let reg = net.shared(MLXArray([Int32](repeating: Int32(c.regTokenId), count: batch), [batch, 1]))
            embeds = concatenated([embeds, reg], axis: 1)
        }
        // Every patch holds at least one observed value, so the encoder mask is all ones.
        let n = embeds.dim(1)
        var buckets = [Int32](repeating: 0, count: n * n)
        for i in 0..<n { for j in 0..<n {
            buckets[i * n + j] = Int32(relativeBucket(j - i, bidirectional: true, buckets: c.buckets, maxDistance: c.maxDistance))
        } }
        guard let table = net.encoder.block[0].attn.attention.relativeAttentionBias else { throw RunnerError.failed("Missing relative bias.") }
        let bias = table(MLXArray(buckets, [n, n])).transposed(2, 0, 1).expandedDimensions(axis: 0)  // [1, heads, n, n]

        var hidden = embeds
        for block in net.encoder.block { hidden = block.ff(block.attn(hidden, bias: bias)) }
        let encoded = net.encoder.finalLayerNorm(hidden)

        // Decoder: one step from decoder_start_token_id. Self-attention over a single position returns its own
        // values whatever the bias, and cross-attention has no position bias and an all-ones mask.
        var decoded = net.shared(MLXArray([Int32](repeating: Int32(c.decoderStartTokenId), count: batch), [batch, 1]))
        let decoderBias: MLXArray? = net.decoder.block[0].selfAttn.attention.relativeAttentionBias.map { table in
            table(MLXArray([Int32(relativeBucket(0, bidirectional: false, buckets: c.buckets, maxDistance: c.maxDistance))], [1, 1]))
                .transposed(2, 0, 1).expandedDimensions(axis: 0)
        }
        for block in net.decoder.block {
            decoded = block.selfAttn(decoded, bias: decoderBias)
            decoded = block.crossAttn(decoded, context: encoded, bias: nil)
            decoded = block.ff(decoded)
        }
        decoded = net.decoder.finalLayerNorm(decoded)
        let raw = net.outputPatchEmbedding(decoded).asType(.float32)  // [B, 1, Q * P]
        eval(raw)
        let flat = raw.asArray(Float.self)
        let q = c.quantiles.count, p = c.predictionLength
        return (0..<batch).map { b in
            let (loc, scale) = locScale[b]
            return (0..<q).map { qi in
                (0..<p).map { t in Double(flat[b * q * p + qi * p + t]) * scale + loc }
            }
        }
    }
}
