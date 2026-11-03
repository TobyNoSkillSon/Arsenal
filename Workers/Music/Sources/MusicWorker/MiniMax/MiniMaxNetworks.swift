// SPDX-License-Identifier: MIT
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXLMCommon

/// MiniMax Music 3's five networks, from the maker's diffusers modules (transformers' Qwen3 for the global language
/// model, `MiniMaxMusic3RVQDepthDecoder`, `MiniMaxMusic3ConditionEncoder`, `MiniMaxMusic3Transformer1DModel`,
/// `MiniMaxMusic3Vocoder`). Parameter names equal the checkpoint's within each component directory. Sequences run
/// channels-last (batch, time, channels); Get stores convolution kernels in MLX's (out, kernel, in) layout.
struct MiniMaxConfiguration: Equatable {
    var hidden = 4096, layers = 36, intermediate = 12_288, heads = 32, kvHeads = 8, headDim = 128, vocabulary = 200_000
    var ropeTheta: Float = 1_000_000, eps: Float = 1e-6
    var codebooks = 8, audioVocabulary = 1024, depthLayers = 4, depthHeads = 16, depthIntermediate = 6144, depthPositions = 16
    var conditionOut = 2048, conditionLayers = 8
    var inputRate = 24_000, inputHop = 960, outputRate = 44_100, outputHop = 512
    var latent = 128, ditLayers = 36, ditHeads = 32, ditHeadDim = 64, ditFF = 8192, ditRotary = 32, fourier = 256
    var vocoderInput = 1024, vocoderHidden = 1536, upsampling = [8, 8, 4, 2], sampleRate = 44_100

    var frameRate: Double { Double(inputRate) / Double(inputHop) }
    /// Latent frames for `frames` language-model frames, as the condition encoder computes it.
    func latentLength(_ frames: Int) -> Int {
        Swift.max(1, Int(Double(frames) * Double(outputRate) / Double(inputRate) * Double(inputHop) / Double(outputHop)))
    }

    static let tiny = MiniMaxConfiguration(
        hidden: 64, layers: 2, intermediate: 128, heads: 4, kvHeads: 2, headDim: 16, vocabulary: 168_064, ropeTheta: 10_000,
        codebooks: 8, audioVocabulary: 32, depthLayers: 2, depthHeads: 4, depthIntermediate: 128, conditionOut: 32,
        latent: 16, ditLayers: 2, ditHeads: 4, ditHeadDim: 16, ditFF: 64, ditRotary: 8, fourier: 16,
        vocoderInput: 16, vocoderHidden: 32, upsampling: [8, 8, 8])

    /// Reads and cross-checks the five component configs of the snapshot.
    static func load(_ root: URL) throws -> MiniMaxConfiguration {
        func json(_ path: String) throws -> [String: Any] {
            guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: root.appendingPathComponent(path))) as? [String: Any] else {
                throw MusicSupport.failure("Unreadable MiniMax Music 3 config: " + path)
            }
            return object
        }
        func int(_ object: [String: Any], _ key: String) throws -> Int {
            guard let value = object[key] as? NSNumber else { throw MusicSupport.failure("MiniMax Music 3 config lacks \(key)") }
            return value.intValue
        }
        let lm = try json("language_model/config.json"), depth = try json("rvq_depth_decoder/config.json")
        let condition = try json("condition_encoder/config.json"), dit = try json("transformer/config.json"), vocoder = try json("vocoder/config.json")
        var c = MiniMaxConfiguration()
        c.hidden = try int(lm, "hidden_size"); c.layers = try int(lm, "num_hidden_layers"); c.intermediate = try int(lm, "intermediate_size")
        c.heads = try int(lm, "num_attention_heads"); c.kvHeads = try int(lm, "num_key_value_heads"); c.headDim = try int(lm, "head_dim")
        c.vocabulary = try int(lm, "vocab_size")
        c.eps = (lm["rms_norm_eps"] as? NSNumber)?.floatValue ?? 1e-6
        c.ropeTheta = ((lm["rope_parameters"] as? [String: Any])?["rope_theta"] as? NSNumber)?.floatValue ?? (lm["rope_theta"] as? NSNumber)?.floatValue ?? 1e6
        c.codebooks = try int(depth, "num_codebooks"); c.audioVocabulary = try int(depth, "audio_vocab_size"); c.depthLayers = try int(depth, "num_layers")
        c.depthHeads = try int(depth, "num_attention_heads"); c.depthIntermediate = try int(depth, "intermediate_size")
        c.depthPositions = try int(depth, "max_position_embeddings")
        c.conditionOut = try int(condition, "out_dim"); c.conditionLayers = try int(condition, "num_condition_layers")
        c.inputRate = try int(condition, "input_sampling_rate"); c.inputHop = try int(condition, "input_hop_length")
        c.outputRate = try int(condition, "output_sampling_rate"); c.outputHop = try int(condition, "output_hop_length")
        c.latent = try int(dit, "in_channels"); c.ditLayers = try int(dit, "num_layers"); c.ditHeads = try int(dit, "num_attention_heads")
        c.ditHeadDim = try int(dit, "attention_head_dim"); c.ditFF = try int(dit, "ff_inner_dim"); c.ditRotary = try int(dit, "rotary_dim")
        c.fourier = try int(dit, "fourier_embedding_dim")
        c.vocoderInput = try int(vocoder, "decoder_input_dim"); c.vocoderHidden = try int(vocoder, "decoder_hidden_dim")
        c.upsampling = (vocoder["upsampling_ratios"] as? [NSNumber])?.map(\.intValue) ?? []
        c.sampleRate = try int(vocoder, "sampling_rate")
        guard try int(depth, "hidden_size") == c.hidden, try int(condition, "condition_hidden_dim") == c.hidden,
            try int(dit, "condition_dim") == c.conditionOut, try int(vocoder, "latent_channels") == c.latent,
            c.conditionLayers == c.codebooks, c.codebooks == 8, c.vocabulary > MiniMaxPrompt.audioCodeOffset + MiniMaxPrompt.semanticVocabulary,
            !c.upsampling.isEmpty, c.upsampling.reduce(1, *) == c.outputHop, c.sampleRate == c.outputRate, c.latent % 2 == 0,
            c.hidden % c.depthHeads == 0, c.ditRotary <= c.ditHeadDim
        else { throw MusicSupport.failure("Unsupported MiniMax Music 3 component configuration.") }
        return c
    }
}

// MARK: Global language model (Qwen3)

final class MiniMaxLMAttention: Module {
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float
    let rope: RoPE
    init(_ c: MiniMaxConfiguration) {
        heads = c.heads; kvHeads = c.kvHeads; headDim = c.headDim; scale = 1 / Float(c.headDim).squareRoot()
        _qProj.wrappedValue = Linear(c.hidden, c.heads * c.headDim, bias: false)
        _kProj.wrappedValue = Linear(c.hidden, c.kvHeads * c.headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hidden, c.kvHeads * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(c.heads * c.headDim, c.hidden, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.eps)
        rope = RoPE(dimensions: c.headDim, traditional: false, base: c.ropeTheta)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        var q = qNorm(qProj(x).reshaped(b, l, heads, headDim)).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(b, l, kvHeads, headDim)).transposed(0, 2, 1, 3)
        let v = vProj(x).reshaped(b, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        q = rope(q, offset: cache.offset); k = rope(k, offset: cache.offset)
        let (keys, values) = cache.update(keys: k, values: v)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, scale: scale, mask: l > 1 ? .causal : .none)
        return oProj(out.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class MiniMaxLMMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ dimensions: Int, _ hidden: Int) {
        _gate.wrappedValue = Linear(dimensions, hidden, bias: false); _up.wrappedValue = Linear(dimensions, hidden, bias: false)
        _down.wrappedValue = Linear(hidden, dimensions, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class MiniMaxLMLayer: Module {
    @ModuleInfo(key: "self_attn") var attention: MiniMaxLMAttention
    @ModuleInfo(key: "mlp") var mlp: MiniMaxLMMLP
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    init(_ c: MiniMaxConfiguration) {
        _attention.wrappedValue = MiniMaxLMAttention(c); _mlp.wrappedValue = MiniMaxLMMLP(c.hidden, c.intermediate)
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.eps); _postNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.eps)
    }
    func callAsFunction(_ x: MLXArray, cache: KVCache) -> MLXArray {
        let h = x + attention(inputNorm(x), cache: cache)
        return h + mlp(postNorm(h))
    }
}

final class MiniMaxLMInner: Module {
    @ModuleInfo(key: "embed_tokens") var embedTokens: Embedding
    let layers: [MiniMaxLMLayer]
    let norm: RMSNorm
    init(_ c: MiniMaxConfiguration) {
        _embedTokens.wrappedValue = Embedding(embeddingCount: c.vocabulary, dimensions: c.hidden)
        layers = (0..<c.layers).map { _ in MiniMaxLMLayer(c) }
        norm = RMSNorm(dimensions: c.hidden, eps: c.eps)
    }
}

/// Qwen3ForCausalLM driven by input embeddings, returning final-norm hidden states (transformers' `last_hidden_state`).
final class MiniMaxLanguageModel: Module {
    let model: MiniMaxLMInner
    @ModuleInfo(key: "lm_head") var lmHead: Linear
    init(_ c: MiniMaxConfiguration) {
        model = MiniMaxLMInner(c)
        _lmHead.wrappedValue = Linear(c.hidden, c.vocabulary, bias: false)
    }
    func makeCache() -> [KVCache] { model.layers.map { _ in KVCacheSimple() } }
    func hidden(_ embeddings: MLXArray, cache: [KVCache]) -> MLXArray {
        var h = embeddings
        for (layer, c) in zip(model.layers, cache) { h = layer(h, cache: c) }
        return model.norm(h)
    }
}

// MARK: Local language model (RVQ depth decoder)

final class MiniMaxDepthAttention: Module {
    @ModuleInfo(key: "to_q") var q: Linear
    @ModuleInfo(key: "to_k") var k: Linear
    @ModuleInfo(key: "to_v") var v: Linear
    @ModuleInfo(key: "to_out") var out: Linear
    let heads: Int, headDim: Int
    init(_ dimensions: Int, heads: Int) {
        self.heads = heads; headDim = dimensions / heads
        _q.wrappedValue = Linear(dimensions, dimensions, bias: false); _k.wrappedValue = Linear(dimensions, dimensions, bias: false)
        _v.wrappedValue = Linear(dimensions, dimensions, bias: false); _out.wrappedValue = Linear(dimensions, dimensions, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3) }
        let y = MLXFast.scaledDotProductAttention(
            queries: split(q(x)), keys: split(k(x)), values: split(v(x)), scale: 1 / Float(headDim).squareRoot(), mask: l > 1 ? .causal : .none)
        return out(y.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class MiniMaxDepthBlock: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "attn") var attention: MiniMaxDepthAttention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: MiniMaxConfiguration) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: 1e-6); _attention.wrappedValue = MiniMaxDepthAttention(c.hidden, heads: c.depthHeads)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: 1e-6)
        _gate.wrappedValue = Linear(c.hidden, c.depthIntermediate, bias: false); _up.wrappedValue = Linear(c.hidden, c.depthIntermediate, bias: false)
        _down.wrappedValue = Linear(c.depthIntermediate, c.hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x))
        let n = postNorm(h)
        return h + down(silu(gate(n)) * up(n))
    }
}

final class MiniMaxDepthDecoder: Module {
    @ModuleInfo(key: "audio_embeddings") var audioEmbeddings: Embedding
    @ModuleInfo(key: "projection") var projection: Linear
    @ModuleInfo(key: "pos_embedding") var positions: Embedding
    let layers: [MiniMaxDepthBlock]
    let norm: RMSNorm
    @ModuleInfo(key: "audio_heads") var heads: [Linear]
    init(_ c: MiniMaxConfiguration) {
        _audioEmbeddings.wrappedValue = Embedding(embeddingCount: c.audioVocabulary * (c.codebooks - 1), dimensions: c.hidden)
        _projection.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
        _positions.wrappedValue = Embedding(embeddingCount: c.depthPositions, dimensions: c.hidden)
        layers = (0..<c.depthLayers).map { _ in MiniMaxDepthBlock(c) }
        norm = RMSNorm(dimensions: c.hidden, eps: 1e-6)
        _heads.wrappedValue = (0..<(c.codebooks - 1)).map { _ in Linear(c.hidden, c.audioVocabulary, bias: false) }
    }
    /// Projected depth-sequence embeddings (batch, steps, hidden) → normalized hidden states.
    func callAsFunction(_ embeddings: MLXArray) -> MLXArray {
        var h = embeddings + positions(MLXArray(Int32(0)..<Int32(embeddings.dim(1))))[.newAxis]
        for layer in layers { h = layer(h) }
        return norm(h)
    }
}

// MARK: Hidden-state fusion (condition encoder)

final class MiniMaxConditionEncoder: Module {
    @ParameterInfo(key: "layer_weight_logits") var logits: MLXArray
    @ParameterInfo(key: "layer_scale") var scale: MLXArray
    @ModuleInfo(key: "proj") var proj: Conv1d
    let c: MiniMaxConfiguration
    init(_ c: MiniMaxConfiguration) {
        self.c = c
        _logits.wrappedValue = MLXArray.zeros([c.conditionLayers]); _scale.wrappedValue = MLXArray.ones([1])
        _proj.wrappedValue = Conv1d(inputChannels: c.hidden, outputChannels: c.conditionOut, kernelSize: 3, padding: 1)
    }
    /// Frame hidden states (batch, frames, layers·hidden) → latent-aligned conditioning (batch, latents, out).
    func callAsFunction(_ hidden: MLXArray) -> MLXArray {
        let (b, frames) = (hidden.dim(0), hidden.dim(1))
        let stacked = hidden.reshaped(b, frames, c.conditionLayers, c.hidden)
        let weights = softmax(logits.asType(.float32), axis: 0).asType(hidden.dtype)
        var h = (stacked * weights.reshaped(1, 1, -1, 1)).sum(axis: 2)
        h = scale.asType(h.dtype) * h
        h = proj(h)
        let target = c.latentLength(frames)
        guard target != frames else { return h }
        return h.take(MLXArray(Self.nearestIndices(frames: frames, target: target).map(Int32.init)), axis: 1)
    }
    /// torch's `F.interpolate(mode="nearest")` source indices: `min(floorf(i · (float)in / out), in - 1)` in float32,
    /// which differs from exact arithmetic at some lengths (an index one lower).
    static func nearestIndices(frames: Int, target: Int) -> [Int] {
        if target == frames { return Array(0..<frames) }
        if target == 2 * frames { return (0..<target).map { $0 >> 1 } }
        let scale = Float(frames) / Float(target)
        return (0..<target).map { Swift.min(Int((Float($0) * scale).rounded(.down)), frames - 1) }
    }
}

// MARK: Flow-matching transformer

final class MiniMaxTimestepEmbedding: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ input: Int, _ output: Int) { _linear1.wrappedValue = Linear(input, output); _linear2.wrappedValue = Linear(output, output) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

final class MiniMaxFourier: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    init(_ dimensions: Int) { _weight.wrappedValue = MLXArray.zeros([dimensions / 2, 1]) }
    /// Angles are computed in FP32 (a numerical intermediate, like RoPE angles); the projection is a stored half weight.
    func callAsFunction(_ t: MLXArray) -> MLXArray {
        let angles = (2 * Float.pi) * matmul(t.asType(.float32).reshaped(-1, 1), weight.asType(.float32).T)
        return concatenated([cos(angles), sin(angles)], axis: -1)
    }
}

final class MiniMaxDiTAttention: Module {
    @ModuleInfo(key: "to_q") var q: Linear
    @ModuleInfo(key: "to_k") var k: Linear
    @ModuleInfo(key: "to_v") var v: Linear
    @ModuleInfo(key: "to_out") var out: [Linear]
    let heads: Int, headDim: Int
    let rope: RoPE
    init(_ c: MiniMaxConfiguration) {
        heads = c.ditHeads; headDim = c.ditHeadDim
        let inner = c.ditHeads * c.ditHeadDim
        _q.wrappedValue = Linear(inner, inner, bias: false); _k.wrappedValue = Linear(inner, inner, bias: false)
        _v.wrappedValue = Linear(inner, inner, bias: false); _out.wrappedValue = [Linear(inner, inner, bias: false)]
        rope = RoPE(dimensions: c.ditRotary, traditional: false, base: 10_000)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        func split(_ y: MLXArray) -> MLXArray { y.reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3) }
        let y = MLXFast.scaledDotProductAttention(
            queries: rope(split(q(x)), offset: 0), keys: rope(split(k(x)), offset: 0), values: split(v(x)),
            scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return out[0](y.transposed(0, 2, 1, 3).reshaped(b, l, -1))
    }
}

final class MiniMaxDiTBlock: Module {
    let norm1: LayerNorm, norm2: LayerNorm
    @ModuleInfo(key: "attn") var attention: MiniMaxDiTAttention
    @ModuleInfo(key: "ff_in") var ffIn: Linear
    @ModuleInfo(key: "ff_out") var ffOut: Linear
    init(_ c: MiniMaxConfiguration) {
        let inner = c.ditHeads * c.ditHeadDim
        norm1 = LayerNorm(dimensions: inner); norm2 = LayerNorm(dimensions: inner)
        _attention.wrappedValue = MiniMaxDiTAttention(c)
        _ffIn.wrappedValue = Linear(inner, c.ditFF * 2); _ffOut.wrappedValue = Linear(c.ditFF, inner)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = x + attention(norm1(x))
        let parts = ffIn(norm2(h)).split(parts: 2, axis: -1)
        return h + ffOut(parts[0] * silu(parts[1]))
    }
}

final class MiniMaxFlowTransformer: Module {
    @ModuleInfo(key: "time_proj") var timeProj: MiniMaxFourier
    @ModuleInfo(key: "time_embed") var timeEmbed: MiniMaxTimestepEmbedding
    @ModuleInfo(key: "preprocess_conv") var preprocess: Conv1d
    @ModuleInfo(key: "proj_in") var projIn: Linear
    @ModuleInfo(key: "transformer_blocks") var blocks: [MiniMaxDiTBlock]
    @ModuleInfo(key: "proj_out") var projOut: Linear
    @ModuleInfo(key: "postprocess_conv") var postprocess: Conv1d
    init(_ c: MiniMaxConfiguration) {
        let inner = c.ditHeads * c.ditHeadDim, concat = 2 * c.latent + c.conditionOut
        _timeProj.wrappedValue = MiniMaxFourier(c.fourier); _timeEmbed.wrappedValue = MiniMaxTimestepEmbedding(c.fourier, inner)
        _preprocess.wrappedValue = Conv1d(inputChannels: concat, outputChannels: concat, kernelSize: 1, bias: false)
        _projIn.wrappedValue = Linear(concat, inner, bias: false)
        _blocks.wrappedValue = (0..<c.ditLayers).map { _ in MiniMaxDiTBlock(c) }
        _projOut.wrappedValue = Linear(inner, c.latent, bias: false)
        _postprocess.wrappedValue = Conv1d(inputChannels: c.latent, outputChannels: c.latent, kernelSize: 1, bias: false)
    }
    /// Latents (batch, length, channels), flow time (batch) in [0, 1] with 0 = noise, condition (batch, length, dim)
    /// → velocity (batch, length, channels).
    func callAsFunction(_ latents: MLXArray, time: MLXArray, condition: MLXArray) -> MLXArray {
        var x = concatenated([latents, MLXArray.zeros(like: latents), condition.asType(latents.dtype)], axis: -1)
        x = preprocess(x) + x
        let temb = timeEmbed(timeProj(time).asType(latents.dtype))
        x = concatenated([temb[0..., .newAxis, 0...], projIn(x)], axis: 1)
        for block in blocks { x = block(x) }
        let y = projOut(x[0..., 1..., 0...])
        return postprocess(y) + y
    }
}

// MARK: Flow-VAE decoder (vocoder)

final class MiniMaxSnake: Module {
    @ParameterInfo(key: "alpha") var alpha: MLXArray
    init(_ channels: Int) { _alpha.wrappedValue = MLXArray.ones([channels]) }
    /// In float32: in FP16 `alpha + 1e-9` drops the epsilon, so a zero or tiny scale gives inf or NaN.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = alpha.asType(.float32), y = x.asType(.float32)
        let s = sin(a * y)
        return (y + reciprocal(a + 1e-9) * (s * s)).asType(x.dtype)
    }
}

final class MiniMaxResidualUnit: Module {
    let snake1: MiniMaxSnake, snake2: MiniMaxSnake, conv1: Conv1d, conv2: Conv1d
    init(_ dimensions: Int, dilation: Int) {
        snake1 = MiniMaxSnake(dimensions); snake2 = MiniMaxSnake(dimensions)
        conv1 = Conv1d(inputChannels: dimensions, outputChannels: dimensions, kernelSize: 7, padding: 3 * dilation, dilation: dilation)
        conv2 = Conv1d(inputChannels: dimensions, outputChannels: dimensions, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { x + conv2(snake2(conv1(snake1(x)))) }
}

final class MiniMaxVocoderBlock: Module {
    let snake1: MiniMaxSnake
    @ModuleInfo(key: "conv_t1") var up: ConvTransposed1d
    @ModuleInfo(key: "res_unit1") var unit1: MiniMaxResidualUnit
    @ModuleInfo(key: "res_unit2") var unit2: MiniMaxResidualUnit
    @ModuleInfo(key: "res_unit3") var unit3: MiniMaxResidualUnit
    init(_ input: Int, _ output: Int, stride: Int) {
        snake1 = MiniMaxSnake(input)
        _up.wrappedValue = ConvTransposed1d(inputChannels: input, outputChannels: output, kernelSize: 2 * stride, stride: stride, padding: (stride + 1) / 2)
        _unit1.wrappedValue = MiniMaxResidualUnit(output, dilation: 1); _unit2.wrappedValue = MiniMaxResidualUnit(output, dilation: 3)
        _unit3.wrappedValue = MiniMaxResidualUnit(output, dilation: 9)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { unit3(unit2(unit1(up(snake1(x))))) }
}

final class MiniMaxVocoder: Module {
    @ModuleInfo(key: "dec_in_proj") var inProj: Conv1d
    @ModuleInfo(key: "conv_in") var convIn: Conv1d
    let blocks: [MiniMaxVocoderBlock]
    @ModuleInfo(key: "snake_out") var snakeOut: MiniMaxSnake
    @ModuleInfo(key: "conv_out") var convOut: Conv1d
    let half: Int
    init(_ c: MiniMaxConfiguration) {
        half = c.latent / 2
        _inProj.wrappedValue = Conv1d(inputChannels: half, outputChannels: c.vocoderInput, kernelSize: 1)
        _convIn.wrappedValue = Conv1d(inputChannels: c.vocoderInput, outputChannels: c.vocoderHidden, kernelSize: 7, padding: 3)
        var blocks: [MiniMaxVocoderBlock] = [], output = c.vocoderHidden
        for (index, stride) in c.upsampling.enumerated() {
            let input = c.vocoderHidden >> index
            output = c.vocoderHidden >> (index + 1)
            blocks.append(MiniMaxVocoderBlock(input, output, stride: stride))
        }
        self.blocks = blocks
        _snakeOut.wrappedValue = MiniMaxSnake(output)
        _convOut.wrappedValue = Conv1d(inputChannels: output, outputChannels: 1, kernelSize: 7, padding: 3)
    }
    /// Latents (batch, length, channels) → stereo waveform (batch, 2, samples) in [-1, 1]. The first half of the latent
    /// channels decodes the left channel, the second half the right.
    func callAsFunction(_ latents: MLXArray) -> MLXArray {
        let (b, l) = (latents.dim(0), latents.dim(1))
        var h = latents.reshaped(b, l, 2, half).transposed(0, 2, 1, 3).reshaped(b * 2, l, half)
        h = convIn(inProj(h))
        for block in blocks { h = block(h) }
        let wave = tanh(convOut(snakeOut(h)))
        return wave.reshaped(b, 2, -1)
    }
}
