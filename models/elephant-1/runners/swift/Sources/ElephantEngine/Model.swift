// swiftlint:disable identifier_name
// SPDX-License-Identifier: MIT
// elephant-1's decoder: the Qwen3.5 text model (Qwen3_5ForCausalLM; 24 Gated DeltaNet layers, 8 gated full-attention
// layers), adapted from mlx-swift-lm Libraries/MLXLLM/Models/Qwen35.swift and MLXLMCommon/GatedDelta.swift
// (@bd4b7434, MIT, ports of mlx-lm qwen3_5.py/gated_delta.py). Differences: the checkpoint's `modeling_overrides`
// attention (full-attention layers block-bidirectional, DeltaNet layers causal) and caches that are values, so every
// question branch reads the same shared prefix state without copying or mutating it.
//
// Under the maker's sequential decision path (qev engine `_run_sequential` with readout.blocks) a forward is either the
// shared prefix alone (block 0) or one question suffix after the cached prefix (block 1, which sees all of block 0).
// In both, every query may attend to every key, so the full-attention mask is none. Pictures are part of the prefix
// block (the makers' train.common.batch_scores), so the same holds with image tokens.
import Foundation
import MLX
import MLXFast
import MLXNN

public struct ElephantConfig: Sendable {
    public let hidden: Int, layers: Int, intermediate: Int, heads: Int, kvHeads: Int, headDim: Int
    public let linearKeyHeads: Int, linearValueHeads: Int, linearKeyDim: Int, linearValueDim: Int, convKernel: Int
    public let eps: Float, vocabulary: Int, ropeTheta: Float, ropeDims: Int, fullAttentionInterval: Int
    public let maxPositions: Int
    /// M-RoPE sections (t, h, w) of the 32 rotary frequencies.
    public let mropeSection: [Int]

    /// Reads `text_config` and `modeling_overrides` of the checkpoint's config.json, refusing anything this port does
    /// not implement (a readout head, another mask, MoE, another M-RoPE layout).
    public init(json: [String: Any]) throws {
        guard let text = json["text_config"] as? [String: Any] else { throw PromptError("config.json lacks text_config") }
        func int(_ k: String) throws -> Int {
            guard let v = text[k] as? Int else { throw PromptError("config.json text_config.\(k) missing") }
            return v
        }
        hidden = try int("hidden_size"); layers = try int("num_hidden_layers"); intermediate = try int("intermediate_size")
        heads = try int("num_attention_heads"); kvHeads = try int("num_key_value_heads"); headDim = try int("head_dim")
        linearKeyHeads = try int("linear_num_key_heads"); linearValueHeads = try int("linear_num_value_heads")
        linearKeyDim = try int("linear_key_head_dim"); linearValueDim = try int("linear_value_head_dim")
        convKernel = try int("linear_conv_kernel_dim"); vocabulary = try int("vocab_size"); fullAttentionInterval = try int("full_attention_interval")
        maxPositions = try int("max_position_embeddings")
        eps = Float(text["rms_norm_eps"] as? Double ?? 1e-6)
        let rope = text["rope_parameters"] as? [String: Any] ?? [:]
        ropeTheta = Float(rope["rope_theta"] as? Double ?? 1e7)
        let partial = rope["partial_rotary_factor"] as? Double ?? text["partial_rotary_factor"] as? Double ?? 0.25
        ropeDims = Int(Double(headDim) * partial)
        mropeSection = rope["mrope_section"] as? [Int] ?? [11, 11, 10]
        guard rope["mrope_interleaved"] as? Bool ?? true, mropeSection.count == 3, mropeSection.reduce(0, +) == ropeDims / 2 else {
            throw PromptError("Unexpected M-RoPE layout in config.json.")
        }
        guard (text["num_experts"] as? Int ?? 0) == 0, text["attn_output_gate"] as? Bool ?? true else {
            throw PromptError("Only the dense Qwen3.5 decoder with gated attention is implemented.")
        }
        if let types = text["layer_types"] as? [String] {
            let expected = (0..<layers).map { ($0 + 1) % fullAttentionInterval == 0 ? "full_attention" : "linear_attention" }
            guard types == expected else { throw PromptError("Unexpected layer_types in config.json.") }
        }
        let overrides = json["modeling_overrides"] as? [String: Any]
        let attention = overrides?["attention"] as? [String: String]
        guard attention?["full_attention"] == "block_bidirectional", attention?["linear_attention"] ?? "causal" == "causal",
            overrides?["readout"] as? String ?? "slot_lm" == "slot_lm"
        else { throw PromptError("This runner serves the block-bidirectional slot_lm decision readout only.") }
    }
}

/// One layer's state after a forward: keys/values for attention, conv tail and recurrent state for DeltaNet.
public enum LayerState: @unchecked Sendable {
    case attention(keys: MLXArray, values: MLXArray)
    case delta(conv: MLXArray, state: MLXArray)
}

/// All layers' states plus the tokens they cover. Values: a branch extends a copy, never the shared prefix.
public struct DecoderState: @unchecked Sendable {
    public var layers: [LayerState?]
    /// Tokens covered.
    public var offset: Int
    /// The rotary position the next token takes: `offset` for text; after images, offset + the M-RoPE delta.
    public var ropeOffset: Int
}

/// Rotary positions of one forward: a scalar start (text, all three axes equal) or explicit [3][L] M-RoPE positions.
public enum RotaryPositions {
    case offset(Int)
    case multimodal([[Int32]])
}

final class GatedRMSNorm: Module {
    @ParameterInfo(key: "weight") var weight: MLXArray
    let eps: Float
    init(_ dims: Int, eps: Float) { self.eps = eps; _weight.wrappedValue = MLXArray.ones([dims]) }
    func callAsFunction(_ x: MLXArray, gate: MLXArray) -> MLXArray {
        let normed = MLXFast.rmsNorm(x, weight: weight, eps: eps)
        return (silu(gate.asType(.float32)) * normed.asType(.float32)).asType(x.dtype)
    }
}

final class GatedDeltaNet: Module {
    let numV: Int, numK: Int, dk: Int, dv: Int, keyDim: Int, valueDim: Int, kernel: Int, convDim: Int
    @ModuleInfo(key: "conv1d") var conv1d: Conv1d
    @ModuleInfo(key: "in_proj_qkv") var inProjQKV: Linear
    @ModuleInfo(key: "in_proj_z") var inProjZ: Linear
    @ModuleInfo(key: "in_proj_b") var inProjB: Linear
    @ModuleInfo(key: "in_proj_a") var inProjA: Linear
    @ParameterInfo(key: "dt_bias") var dtBias: MLXArray
    @ParameterInfo(key: "A_log") var aLog: MLXArray
    @ModuleInfo(key: "norm") var norm: GatedRMSNorm
    @ModuleInfo(key: "out_proj") var outProj: Linear

    init(_ c: ElephantConfig) {
        numV = c.linearValueHeads; numK = c.linearKeyHeads; dk = c.linearKeyDim; dv = c.linearValueDim
        keyDim = dk * numK; valueDim = dv * numV; kernel = c.convKernel; convDim = keyDim * 2 + valueDim
        _conv1d.wrappedValue = Conv1d(inputChannels: convDim, outputChannels: convDim, kernelSize: kernel, groups: convDim, bias: false)
        _inProjQKV.wrappedValue = Linear(c.hidden, convDim, bias: false)
        _inProjZ.wrappedValue = Linear(c.hidden, valueDim, bias: false)
        _inProjB.wrappedValue = Linear(c.hidden, numV, bias: false)
        _inProjA.wrappedValue = Linear(c.hidden, numV, bias: false)
        _dtBias.wrappedValue = MLXArray.ones([numV])
        _aLog.wrappedValue = MLXArray.zeros([numV])
        _norm.wrappedValue = GatedRMSNorm(dv, eps: c.eps)
        _outProj.wrappedValue = Linear(valueDim, c.hidden, bias: false)
    }

    func callAsFunction(_ x: MLXArray, previous: LayerState?) -> (MLXArray, LayerState) {
        let B = x.dim(0), S = x.dim(1)
        let qkv = inProjQKV(x)
        let z = inProjZ(x).reshaped(B, S, numV, dv)
        let b = inProjB(x), a = inProjA(x)
        var convState = MLXArray.zeros([B, kernel - 1, convDim], dtype: x.dtype)
        var state = MLXArray.zeros([B, numV, dv, dk], dtype: .float32)
        if case .delta(let c, let s)? = previous { convState = c; state = s }
        let convInput = concatenated([convState, qkv], axis: 1)
        let tail = contiguous(convInput[0..., (-(kernel - 1))..., 0...])
        let convOut = silu(conv1d(convInput))
        let parts = split(convOut, indices: [keyDim, 2 * keyDim], axis: -1)
        let q = parts[0].reshaped(B, S, numK, dk), k = parts[1].reshaped(B, S, numK, dk), v = parts[2].reshaped(B, S, numV, dv)
        // HF's l2norm(q) / sqrt(dk) and l2norm(k), written as weightless RMS norms (mlx-lm).
        let inv = Foundation.pow(Float(dk), -0.5)
        let qn = MLXArray(inv * inv).asType(q.dtype) * MLXFast.rmsNorm(q, weight: MLXArray.mlxNone, eps: 1e-6)
        let kn = MLXArray(inv).asType(k.dtype) * MLXFast.rmsNorm(k, weight: MLXArray.mlxNone, eps: 1e-6)
        let beta = sigmoid(b).asType(.float32)
        // HF: g = -exp(A_log.float()) * softplus(a.float() + dt_bias), the decay exp(g), all float32.
        let g = exp(-exp(aLog.asType(.float32)) * softplus(a.asType(.float32) + dtBias.asType(.float32)))
        let (y, newState) = GatedDelta.update(q: qn, k: kn, v: v, g: g, beta: beta, state: state)
        let out = outProj(norm(y, gate: z).reshaped(B, S, -1))
        return (out, .delta(conv: tail, state: newState))
    }
}

final class Attention: Module {
    /// Query rows per attention block (see callAsFunction).
    var queryBlock = 1024
    let heads: Int, kvHeads: Int, headDim: Int, scale: Float, ropeDims: Int, ropeBase: Float, section: [Int]
    @ModuleInfo(key: "q_proj") var qProj: Linear
    @ModuleInfo(key: "k_proj") var kProj: Linear
    @ModuleInfo(key: "v_proj") var vProj: Linear
    @ModuleInfo(key: "o_proj") var oProj: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm

    init(_ c: ElephantConfig) {
        heads = c.heads; kvHeads = c.kvHeads; headDim = c.headDim; scale = Foundation.pow(Float(c.headDim), -0.5)
        ropeDims = c.ropeDims; ropeBase = c.ropeTheta; section = c.mropeSection
        _qProj.wrappedValue = Linear(c.hidden, c.heads * c.headDim * 2, bias: false)
        _kProj.wrappedValue = Linear(c.hidden, c.kvHeads * c.headDim, bias: false)
        _vProj.wrappedValue = Linear(c.hidden, c.kvHeads * c.headDim, bias: false)
        _oProj.wrappedValue = Linear(c.heads * c.headDim, c.hidden, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.eps)
        _kNorm.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.eps)
    }

    /// Qwen3.5's interleaved M-RoPE (mrope_section [11, 11, 10]) takes each of the 32 rotary frequencies from the t, h
    /// or w position of the token. Text tokens have the same position on all three axes, so a text forward (and every
    /// question suffix, which is text after the prefix's rotary delta) is ordinary RoPE (rotate-half) over the first
    /// `ropeDims` (64) channels with a scalar offset; a prefix holding images uses `multimodalRoPE`.
    /// Queries are processed in blocks of `queryBlock` rows against all keys: exact because the mask is none, and it
    /// bounds the score matrix of a long prefix (16 heads x block x keys) instead of 16 x L x L.
    /// Qwen3.5 M-RoPE for explicit positions (HF Qwen3_5TextRotaryEmbedding + apply_rotary_pos_emb): frequency j uses
    /// the h position when j % 3 == 1 and j < 3 * section[1], the w position when j % 3 == 2 and j < 3 * section[2], else
    /// t; cos and sin are cast to the activation dtype; the channels past `ropeDims` pass through.
    func multimodalRoPE(_ x: MLXArray, _ positions: [[Int32]]) -> MLXArray {
        let half = ropeDims / 2
        let invFreq = (0..<half).map { 1 / Foundation.pow(Double(ropeBase), Double(2 * $0) / Double(ropeDims)) }
        let L = positions[0].count
        var angles = [Float](repeating: 0, count: L * half)
        for j in 0..<half {
            let axis = (j % 3 == 1 && j < 3 * section[1]) ? 1 : (j % 3 == 2 && j < 3 * section[2]) ? 2 : 0
            for i in 0..<L { angles[i * half + j] = Float(Double(positions[axis][i]) * invFreq[j]) }
        }
        let f = MLXArray(angles, [L, half])
        let emb = concatenated([f, f], axis: -1)
        let c = cos(emb).asType(x.dtype), s = sin(emb).asType(x.dtype)
        let rot = x[.ellipsis, 0..<ropeDims], pass = x[.ellipsis, ropeDims...]
        let rotated = rot * c + VisionOps.rotateHalf(rot) * s
        return concatenated([rotated, pass], axis: -1)
    }

    func callAsFunction(_ x: MLXArray, previous: LayerState?, rope: RotaryPositions) -> (MLXArray, LayerState) {
        let B = x.dim(0), L = x.dim(1)
        let qg = qProj(x).reshaped(B, L, heads, -1).split(parts: 2, axis: -1)
        let gate = qg[1].reshaped(B, L, -1)
        var q = qNorm(qg[0]).transposed(0, 2, 1, 3)
        var k = kNorm(kProj(x).reshaped(B, L, kvHeads, -1)).transposed(0, 2, 1, 3)
        var v = vProj(x).reshaped(B, L, kvHeads, -1).transposed(0, 2, 1, 3)
        switch rope {
        case .offset(let offset):
            q = MLXFast.RoPE(q, dimensions: ropeDims, traditional: false, base: ropeBase, scale: 1, offset: offset)
            k = MLXFast.RoPE(k, dimensions: ropeDims, traditional: false, base: ropeBase, scale: 1, offset: offset)
        case .multimodal(let positions):
            q = multimodalRoPE(q, positions)
            k = multimodalRoPE(k, positions)
        }
        if case .attention(let pk, let pv)? = previous {
            k = concatenated([pk, k], axis: 2); v = concatenated([pv, v], axis: 2)
        }
        var blocks: [MLXArray] = []
        for start in stride(from: 0, to: L, by: queryBlock) {
            let rows = q[0..., 0..., start..<Swift.min(start + queryBlock, L), 0...]
            let block = MLXFast.scaledDotProductAttention(queries: rows, keys: k, values: v, scale: scale, mask: .none)
            if L > queryBlock { eval(block) }
            blocks.append(block)
        }
        let o = (blocks.count == 1 ? blocks[0] : concatenated(blocks, axis: 2)).transposed(0, 2, 1, 3).reshaped(B, L, -1)
        return (oProj(o * sigmoid(gate)), .attention(keys: k, values: v))
    }
}

final class MLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ c: ElephantConfig) {
        _gate.wrappedValue = Linear(c.hidden, c.intermediate, bias: false)
        _up.wrappedValue = Linear(c.hidden, c.intermediate, bias: false)
        _down.wrappedValue = Linear(c.intermediate, c.hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class DecoderLayer: Module {
    let isLinear: Bool
    @ModuleInfo(key: "self_attn") var attention: Attention?
    @ModuleInfo(key: "linear_attn") var delta: GatedDeltaNet?
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: MLP
    init(_ c: ElephantConfig, index: Int) {
        isLinear = (index + 1) % c.fullAttentionInterval != 0
        if isLinear { _delta.wrappedValue = GatedDeltaNet(c) } else { _attention.wrappedValue = Attention(c) }
        _inputNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.eps)
        _postNorm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.eps)
        _mlp.wrappedValue = MLP(c)
    }
    func callAsFunction(_ x: MLXArray, previous: LayerState?, rope: RotaryPositions) -> (MLXArray, LayerState) {
        let (r, state) = isLinear ? delta!(inputNorm(x), previous: previous) : attention!(inputNorm(x), previous: previous, rope: rope)
        let h = x + r
        return (h + mlp(postNorm(h)), state)
    }
}

final class LanguageModel: Module {
    @ModuleInfo(key: "embed_tokens") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [DecoderLayer]
    @ModuleInfo(key: "norm") var norm: RMSNorm
    init(_ c: ElephantConfig) {
        _embed.wrappedValue = Embedding(embeddingCount: c.vocabulary, dimensions: c.hidden)
        _layers.wrappedValue = (0..<c.layers).map { DecoderLayer(c, index: $0) }
        _norm.wrappedValue = RMSNorm(dimensions: c.hidden, eps: c.eps)
    }
}

/// Probe points for the component check: called with a name and the last token's activation.
public typealias Probe = (String, MLXArray) -> Void

public final class ElephantModel: Module {
    @ModuleInfo(key: "language_model") var language: LanguageModel
    @ModuleInfo(key: "visual") public var visual: VisionModel
    public let config: ElephantConfig
    /// The serving dtype of the loaded decoder.
    public var dtype: DType { language.embed.weight.dtype }
    /// The LM head rows of the answer slots (qev upcasts the head to float32; only these rows are read).
    public private(set) var slotRows: MLXArray

    public init(_ config: ElephantConfig, vision: VisionConfig) {
        self.config = config
        _language.wrappedValue = LanguageModel(config)
        _visual.wrappedValue = VisionModel(vision)
        slotRows = MLXArray.zeros([1, config.hidden], dtype: .float32)
    }

    /// One forward of `ids` after `state` (nil = from the start): the final-norm hidden state of the last token and the
    /// extended state. `cancelled` is checked between layers.
    public func forward(
        _ ids: [Int], after state: DecoderState?, vision: [(features: MLXArray, tokenId: Int)] = [], positions: (axes: [[Int32]], delta: Int)? = nil,
        probe: Probe? = nil, cancelled: () -> Bool = { false }
    ) throws -> (MLXArray, DecoderState) {
        var h = language.embed(MLXArray(ids.map(Int32.init)).reshaped(1, -1))
        for part in vision {
            // Image features replace the image-pad embeddings in order, video features the video pads (HF _embed:
            // get_placeholder_mask + masked_scatter, images then videos).
            let slots = ids.indices.filter { ids[$0] == part.tokenId }
            guard slots.count == part.features.dim(0) else { throw PromptError("Vision features do not match their tokens.") }
            h[0, MLXArray(slots.map(Int32.init))] = part.features.asType(h.dtype)
        }
        probe?("embed", h[0, -1])
        let offset = state?.offset ?? 0
        let ropeStart = state?.ropeOffset ?? 0
        let rope: RotaryPositions = positions.map { .multimodal($0.axes) } ?? .offset(ropeStart)
        var next: [LayerState?] = []
        next.reserveCapacity(language.layers.count)
        for (i, layer) in language.layers.enumerated() {
            if cancelled() { throw Cancelled() }
            let (out, s) = layer(h, previous: state?.layers[i], rope: rope)
            h = out
            next.append(s)
            if let probe { probe("layer\(i)", h[0, -1]) }
            // Evaluate every layer: a long prompt never holds more than one layer's intermediates.
            eval([h] + Self.arrays(s))
        }
        let last = language.norm(h[0..., (-1)...])[0, 0]
        probe?("norm", last)
        let nextRope = positions.map { ids.count + $0.delta + ropeStart } ?? ropeStart + ids.count
        return (last, DecoderState(layers: next, offset: offset + ids.count, ropeOffset: nextRope))
    }

    static func arrays(_ s: LayerState) -> [MLXArray] {
        switch s {
        case .attention(let k, let v): return [k, v]
        case .delta(let c, let st): return [c, st]
        }
    }

    /// Slot logits in float32: <h, w_slot> for the first `n` slots.
    public func slotLogits(_ hidden: MLXArray, count n: Int) -> MLXArray {
        matmul(slotRows[0..<n], hidden.asType(.float32).reshaped(-1, 1)).reshaped(-1)
    }

    /// Checkpoint keys to module keys: the multimodal checkpoint's text decoder, with Qwen3.5's offset norms made
    /// direct scales and the depthwise conv moved to MLX's layout (mlx-lm `sanitize`). The vision tower is not loaded;
    /// `lm_head` is kept apart for the slot rows. Returns nil for keys this graph does not take.
    public static func map(_ key: String, _ value: MLXArray) -> (String, MLXArray)? {
        guard key.hasPrefix("model.language_model.") else { return nil }
        let name = "language_model." + key.dropFirst("model.language_model.".count)
        var v = value
        if name.hasSuffix("conv1d.weight"), v.dim(-1) != 1 { v = v.movedAxis(source: 2, destination: 1) }
        let offsetNorms = [".input_layernorm.weight", ".post_attention_layernorm.weight", "language_model.norm.weight", ".q_norm.weight", ".k_norm.weight"]
        if v.ndim == 1, offsetNorms.contains(where: { name.hasSuffix($0) }) {
            // HF computes (1 + w) in float32; the sum is formed in float32 and rounded once to the serving dtype.
            v = (v.asType(.float32) + 1).asType(value.dtype)
        }
        return (name, v)
    }

    public func setSlotRows(_ head: MLXArray, slots: [Int]) {
        slotRows = head[MLXArray(slots.map(Int32.init))].asType(.float32)
        eval(slotRows)
    }
}

public struct Cancelled: Error {}

enum GatedDelta {
    /// The recurrence over time steps (mlx-lm gated_delta kernel), state in float32. q,k: [B,T,Hk,Dk]; v: [B,T,Hv,Dv];
    /// g, beta: [B,T,Hv] float32; state: [B,Hv,Dv,Dk] float32.
    static func update(q: MLXArray, k: MLXArray, v: MLXArray, g: MLXArray, beta: MLXArray, state: MLXArray) -> (MLXArray, MLXArray) {
        let B = k.dim(0), T = k.dim(1), Hk = k.dim(2), Dk = k.dim(3), Hv = v.dim(2), Dv = v.dim(3)
        let outputs = kernel(
            [q, k, v, g, beta, state, MLXArray(T)],
            template: [("InT", q.dtype), ("StT", state.dtype), ("Dk", Dk), ("Dv", Dv), ("Hk", Hk), ("Hv", Hv)],
            grid: (32, Dv, B * Hv), threadGroup: (32, 4, 1),
            outputShapes: [[B, T, Hv, Dv], state.shape], outputDTypes: [q.dtype, state.dtype])
        return (outputs[0], outputs[1])
    }

    nonisolated(unsafe) static let kernel = MLXFast.metalKernel(
        name: "elephant_gated_delta_step",
        inputNames: ["q", "k", "v", "g", "beta", "state_in", "T"],
        outputNames: ["y", "state_out"],
        source: """
            auto n = thread_position_in_grid.z;
            auto b_idx = n / Hv;
            auto hv_idx = n % Hv;
            auto hk_idx = hv_idx / (Hv / Hk);
            constexpr int n_per_t = Dk / 32;
            auto q_ = q + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto k_ = k + b_idx * T * Hk * Dk + hk_idx * Dk;
            auto v_ = v + b_idx * T * Hv * Dv + hv_idx * Dv;
            y += b_idx * T * Hv * Dv + hv_idx * Dv;
            auto dk_idx = thread_position_in_threadgroup.x;
            auto dv_idx = thread_position_in_grid.y;
            auto g_ = g + b_idx * T * Hv;
            auto beta_ = beta + b_idx * T * Hv;
            auto i_state = state_in + (n * Dv + dv_idx) * Dk;
            auto o_state = state_out + (n * Dv + dv_idx) * Dk;
            float state[n_per_t];
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              state[i] = static_cast<float>(i_state[s_idx]);
            }
            for (int t = 0; t < T; ++t) {
              float kv_mem = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] * g_[hv_idx];
                kv_mem += state[i] * k_[s_idx];
              }
              kv_mem = simd_sum(kv_mem);
              auto delta = (v_[dv_idx] - kv_mem) * beta_[hv_idx];
              float out = 0.0f;
              for (int i = 0; i < n_per_t; ++i) {
                auto s_idx = n_per_t * dk_idx + i;
                state[i] = state[i] + k_[s_idx] * delta;
                out += state[i] * q_[s_idx];
              }
              out = simd_sum(out);
              if (thread_index_in_simdgroup == 0) {
                y[dv_idx] = static_cast<InT>(out);
              }
              q_ += Hk * Dk;
              k_ += Hk * Dk;
              v_ += Hv * Dv;
              y += Hv * Dv;
              g_ += Hv;
              beta_ += Hv;
            }
            for (int i = 0; i < n_per_t; ++i) {
              auto s_idx = n_per_t * dk_idx + i;
              o_state[s_idx] = static_cast<StT>(state[i]);
            }
            """)
}

// swiftlint:enable identifier_name
