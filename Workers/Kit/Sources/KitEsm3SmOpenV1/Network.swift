// SPDX-License-Identifier: MIT
// The ESM3 networks in MLX, one Swift module per maker module (esm/layers/*, esm/models/{esm3,vqvae,function_decoder}.py,
// esm commit 8f0834c); property keys follow the checkpoint. Weights and activations are float16. Float32 is kept only
// where the maker computes in float32 or a value leaves float16's range: the frame rotations and translations of
// geometric attention and its attention logits (distances in Angstrom), the structure head's frame maths, the
// structure codebook distances, the pLDDT/PAE/pTM expectations and the logits handed to sampling.
import Foundation
import MLX
import MLXFast
import MLXNN

/// nn.Sequential(LayerNorm, Linear): checkpoint keys "0" and "1" (renamed at load, `CheckpointKeys`).
final class NormLinear: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ d: Int, _ out: Int, bias: Bool) {
        _norm.wrappedValue = LayerNorm(dimensions: d)
        _linear.wrappedValue = Linear(d, out, bias: bias)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear(norm(x)) }
}

/// blocks.MultiHeadAttention with rotary embeddings (non-interleaved, base 10000) and optional q/k LayerNorms.
final class Attention: Module {
    @ModuleInfo(key: "layernorm_qkv") var qkv: NormLinear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ModuleInfo(key: "q_ln") var qLN: LayerNorm?
    @ModuleInfo(key: "k_ln") var kLN: LayerNorm?
    let heads: Int, headDim: Int
    let rope: RoPE

    init(_ d: Int, heads: Int, bias: Bool, qkLayerNorm: Bool) {
        self.heads = heads
        headDim = d / heads
        _qkv.wrappedValue = NormLinear(d, 3 * d, bias: bias)
        _outProj.wrappedValue = Linear(d, d, bias: bias)
        _qLN.wrappedValue = qkLayerNorm ? LayerNorm(dimensions: d, bias: bias) : nil
        _kLN.wrappedValue = qkLayerNorm ? LayerNorm(dimensions: d, bias: bias) : nil
        rope = RoPE(dimensions: d / heads, traditional: false, base: 10000)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (B, L) = (x.dim(0), x.dim(1))
        let parts = split(qkv(x), parts: 3, axis: -1)
        var q = parts[0], k = parts[1]
        if let qLN, let kLN { q = qLN(q); k = kLN(k) }
        func heads(_ t: MLXArray) -> MLXArray { t.reshaped(B, L, self.heads, headDim).transposed(0, 2, 1, 3) }
        let out = MLXFast.scaledDotProductAttention(
            queries: rope(heads(q)), keys: rope(heads(k)), values: heads(parts[2]), scale: 1 / Float(headDim).squareRoot(), mask: nil)
        return outProj(out.transposed(0, 2, 1, 3).reshaped(B, L, -1))
    }
}

/// Rigid frames for a batch of residues, float32: rotation [..., 3, 3] and translation [..., 3].
struct Frames {
    var rotation: MLXArray
    var translation: MLXArray
    /// Residues with a backbone frame (keys elsewhere are masked out of geometric attention).
    var mask: MLXArray

    /// Frames from host-computed backbone frames ([N][E] each).
    init(_ frames: [[Frame]], mask: [[Bool]]) {
        let n = frames.count, e = frames.first?.count ?? 0
        rotation = MLXArray(frames.flatMap { $0.flatMap { $0.rotation.map(Float.init) } }, [n, e, 3, 3])
        translation = MLXArray(frames.flatMap { $0.flatMap { [Float($0.translation.0), Float($0.translation.1), Float($0.translation.2)] } }, [n, e, 3])
        self.mask = MLXArray(mask.flatMap { $0 }, [n, e])
    }
}

/// geom_attention.GeometricReasoningOriginalImpl (one vector message per head).
final class GeometricAttention: Module {
    @ModuleInfo(key: "s_norm") var sNorm: LayerNorm
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "out_proj") var outProj: Linear
    @ParameterInfo(key: "distance_scale_per_head") var distanceScale: MLXArray
    @ParameterInfo(key: "rotation_scale_per_head") var rotationScale: MLXArray
    let vHeads: Int, zeroFrameless: Bool

    init(_ d: Int, vHeads: Int, bias: Bool, zeroFrameless: Bool) {
        self.vHeads = vHeads
        self.zeroFrameless = zeroFrameless
        _sNorm.wrappedValue = LayerNorm(dimensions: d, bias: bias)
        _proj.wrappedValue = Linear(d, 4 * vHeads * 3 + vHeads * 3, bias: bias)
        _outProj.wrappedValue = Linear(vHeads * 3, d, bias: bias)
        _distanceScale.wrappedValue = MLXArray.zeros([vHeads])
        _rotationScale.wrappedValue = MLXArray.zeros([vHeads])
    }

    /// s [N, E, d]; frames over [N, E].
    func callAsFunction(_ s: MLXArray, _ frames: Frames) -> MLXArray {
        let (N, E, H) = (s.dim(0), s.dim(1), vHeads)
        let p = proj(sNorm(s)).asType(.float32)
        let vecRot = p[0..., 0..., 0 ..< (H * 9)].reshaped(N, E, 3 * H, 3)
        let vecDist = p[0..., 0..., (H * 9)...].reshaped(N, E, 2 * H, 3)
        // Every 3-wide contraction is an elementwise product and sum: Metal's float32 matmul here measured about 1e-3
        // relative error on these Angstrom-scale vectors (log/002-parity.md), enough to move the softmax by 10%.
        let r = frames.rotation  // [N, E, 3, 3]
        /// R v for vectors [N, E, h, 3]: out_i = sum_j R_ij v_j.
        func rotate(_ v: MLXArray) -> MLXArray { (r.expandedDimensions(axis: 2) * v.expandedDimensions(axis: 3)).sum(axis: -1) }
        /// R^T v: out_i = sum_j R_ji v_j.
        func unrotate(_ v: MLXArray) -> MLXArray { (r.expandedDimensions(axis: 2) * v.expandedDimensions(axis: -1)).sum(axis: -2) }
        let rotated = rotate(vecRot)
        let moved = rotate(vecDist) + frames.translation.expandedDimensions(axis: 2)
        func perHead(_ t: MLXArray) -> MLXArray { t.transposed(0, 2, 1, 3) }  // [N, h, E, 3]
        let qRot = perHead(rotated[0..., 0..., 0 ..< H]), kRot = perHead(rotated[0..., 0..., H ..< (2 * H)])
        let value = perHead(rotated[0..., 0..., (2 * H)...])
        let qDist = perHead(moved[0..., 0..., 0 ..< H]), kDist = perHead(moved[0..., 0..., H...])
        let third = 1 / Float(3).squareRoot()
        let wr = softplus(rotationScale.asType(.float32)).reshaped(1, H, 1, 1)
        let wd = softplus(distanceScale.asType(.float32)).reshaped(1, H, 1, 1)
        // Keys without a frame get float32's most negative value, as the maker's mask: a query with no valid key at all
        // then attends uniformly.
        let bias = MLX.where(frames.mask, MLXArray(Float(0)), MLXArray(-Float.greatestFiniteMagnitude)).reshaped(N, 1, 1, E)
        // Heads in chunks so the [N, h, E, E, 3] differences stay within about 256 MB.
        let chunk = Swift.max(1, Swift.min(H, (64 << 20) / Swift.max(1, N * E * E * 3)))
        var outs: [MLXArray] = []
        for h0 in stride(from: 0, to: H, by: chunk) {
            let hs = h0 ..< Swift.min(H, h0 + chunk)
            let qd = qDist[0..., hs], kd = kDist[0..., hs]
            let diff = qd.expandedDimensions(axis: 3) - kd.expandedDimensions(axis: 2)
            let distance = sqrt((diff * diff).sum(axis: -1)) * third
            let rotation = (qRot[0..., hs].expandedDimensions(axis: 3) * kRot[0..., hs].expandedDimensions(axis: 2)).sum(axis: -1) * third
            let weights = softmax(rotation * wr[0..., hs] - distance * wd[0..., hs] + bias, axis: -1)
            // weights [N, h, Eq, Ek] x value [N, h, Ek, 3], summed over keys.
            outs.append((weights.expandedDimensions(axis: -1) * value[0..., hs].expandedDimensions(axis: 2)).sum(axis: 3))
            eval(outs[outs.count - 1])
        }
        let out = concatenated(outs, axis: 1).transposed(0, 2, 1, 3)  // [N, E, h, 3]
        var back = unrotate(out).reshaped(N, E, 3 * H)  // rotate back into the residue frame
        if zeroFrameless { back = back * frames.mask.expandedDimensions(axis: -1).asType(.float32) }
        return outProj(back.asType(s.dtype))
    }
}

/// The feed-forward of a block: nn.Sequential(LayerNorm, Linear, SwiGLU | GELU, Linear) with keys "0", "1", "3".
final class FeedForward: Module {
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "up") var up: Linear
    @ModuleInfo(key: "down") var down: Linear
    let swiglu: Bool

    init(_ d: Int, hidden: Int, bias: Bool, swiglu: Bool) {
        self.swiglu = swiglu
        _norm.wrappedValue = LayerNorm(dimensions: d)
        _up.wrappedValue = Linear(d, swiglu ? hidden * 2 : hidden, bias: bias)
        _down.wrappedValue = Linear(hidden, d, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = up(norm(x))
        if swiglu {
            let parts = split(h, parts: 2, axis: -1)
            return down(silu(parts[0]) * parts[1])
        }
        return down(gelu(h))
    }
}

/// blocks.UnifiedTransformerBlock.
final class Block: Module {
    @ModuleInfo(key: "attn") var attn: Attention?
    @ModuleInfo(key: "geom_attn") var geom: GeometricAttention?
    @ModuleInfo(key: "ffn") var ffn: FeedForward
    let scale: Float

    init(attn: Attention?, geom: GeometricAttention?, ffn: FeedForward, scale: Float) {
        _attn.wrappedValue = attn
        _geom.wrappedValue = geom
        _ffn.wrappedValue = ffn
        self.scale = scale
    }

    func callAsFunction(_ x: MLXArray, frames: Frames?) -> MLXArray {
        var x = x
        if let attn { x = x + attn(x) / scale }
        // Without any frame the maker's output is zero (masked queries, mask_and_zero_frameless), so it is skipped.
        if let geom, let frames, !geom.zeroFrameless || frames.mask.any().item(Bool.self) { x = x + geom(x, frames) / scale }
        return x + ffn(x) / scale
    }
}

/// swiglu_correction_fn: the expanded width rounded up to a multiple of 256.
func swigluHidden(_ ratio: Double, _ d: Int) -> Int { Int((ratio * Double(d) + 255) / 256) * 256 }

/// transformer_stack.TransformerStack: blocks, then a LayerNorm without bias (absent in the structure encoder).
final class Stack: Module {
    @ModuleInfo(key: "blocks") var blocks: [Block]
    @ModuleInfo(key: "norm") var norm: LayerNorm?

    init(blocks: [Block], finalNorm: Int?) {
        _blocks.wrappedValue = blocks
        _norm.wrappedValue = finalNorm.map { LayerNorm(dimensions: $0, bias: false) }
    }

    /// (normed, pre-norm); `taps` receives the output of the listed blocks.
    func callAsFunction(_ x: MLXArray, frames: Frames?, taps: Set<Int>, tap: (Int, MLXArray) -> Void, cancelled: () throws -> Void) throws -> (MLXArray, MLXArray) {
        var x = x
        for (i, block) in blocks.enumerated() {
            x = block(x, frames: frames)
            if taps.contains(i) { tap(i, x) }
            if i % 8 == 7 { try cancelled() }
        }
        return (norm.map { $0(x) } ?? x, x)
    }

    func callAsFunction(_ x: MLXArray, frames: Frames?) -> (MLXArray, MLXArray) {
        var x = x
        for block in blocks { x = block(x, frames: frames) }
        return (norm.map { $0(x) } ?? x, x)
    }
}

/// layers.regression_head.RegressionHead: nn.Sequential(Linear, GELU, LayerNorm, Linear), keys "0", "2", "3".
final class Head: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "output") var output: Linear
    init(_ d: Int, _ out: Int, hidden: Int? = nil) {
        _dense.wrappedValue = Linear(d, hidden ?? d)
        _norm.wrappedValue = LayerNorm(dimensions: hidden ?? d)
        _output.wrappedValue = Linear(hidden ?? d, out)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(norm(gelu(dense(x)))) }
}

/// vqvae.RegressionHead: dense, GELU, norm, output.
final class NamedHead: Module {
    @ModuleInfo(key: "dense") var dense: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "output") var output: Linear
    init(_ d: Int, _ out: Int) {
        _dense.wrappedValue = Linear(d, d)
        _norm.wrappedValue = LayerNorm(dimensions: d)
        _output.wrappedValue = Linear(d, out)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(norm(gelu(dense(x)))) }
}

/// MLX reads numeric key segments as list indices, so the maker's nn.Sequential positions get names at load.
enum CheckpointKeys {
    static let rules: [(String, String)] = [
        (#"\.layernorm_qkv\.0\."#, ".layernorm_qkv.norm."), (#"\.layernorm_qkv\.1\."#, ".layernorm_qkv.linear."),
        (#"\.ffn\.0\."#, ".ffn.norm."), (#"\.ffn\.1\."#, ".ffn.up."), (#"\.ffn\.3\."#, ".ffn.down."),
        (#"(_head|heads\.[a-z_]+)\.0\."#, "$1.dense."), (#"(_head|heads\.[a-z_]+)\.2\."#, "$1.norm."), (#"(_head|heads\.[a-z_]+)\.3\."#, "$1.output."),
    ]
    static func rename(_ key: String) -> String {
        rules.reduce(key) { $0.replacingOccurrences(of: $1.0, with: $1.1, options: .regularExpression) }
    }
}

// MARK: ESM3

/// The input tracks of one protein, [1, L] each (function [1, L, 8], residue annotations [1, L, 16]).
struct Tracks {
    var sequence: [Int]
    var structure: [Int]
    var ss8: [Int]
    var sasa: [Int]
    var function: [[Int]]
    var residue: [[Int]]
    /// N, CA, C per position (BOS/EOS rows nil): conditioning coordinates, or all nil.
    var coordinates: [[Vec3?]]
    var length: Int { sequence.count }
    var hasCoordinates: Bool { coordinates.contains { row in row.allSatisfy { $0 != nil } } }

    /// Whether residue position `i` (BOS and EOS excluded) is still masked in `track`, as iterative_sampling_tokens reads it.
    func isMasked(_ track: String, _ i: Int) -> Bool {
        guard i > 0, i < length - 1 else { return false }
        switch track {
        case "sequence": return sequence[i] == ESM3Vocab.sequenceMask
        case "structure": return structure[i] == ESM3Vocab.structureMask
        case "secondary_structure": return ss8[i] == 0
        case "sasa": return sasa[i] == 0
        default: return function[i].allSatisfy { $0 == 0 }
        }
    }

    /// ESM3.forward defaults: missing tracks as the maker fills them (all-mask sequence/structure, <pad> elsewhere).
    static func empty(length: Int) -> Tracks {
        let L = length + 2
        var seq = [Int](repeating: ESM3Vocab.sequenceMask, count: L)
        seq[0] = ESM3Vocab.sequenceBOS; seq[L - 1] = ESM3Vocab.sequenceEOS
        var structure = [Int](repeating: ESM3Vocab.structureMask, count: L)
        structure[0] = ESM3Vocab.structureBOS; structure[L - 1] = ESM3Vocab.structureEOS
        return Tracks(
            sequence: seq, structure: structure, ss8: [Int](repeating: 0, count: L), sasa: [Int](repeating: 0, count: L),
            function: [[Int]](repeating: [Int](repeating: 0, count: 8), count: L), residue: [[Int]](repeating: [Int](repeating: 0, count: 16), count: L),
            coordinates: [[Vec3?]](repeating: [nil, nil, nil], count: L))
    }

    /// The structure track as ESM3.forward feeds it: BOS/EOS/PAD/chain-break follow the sequence track.
    var structureInput: [Int] {
        zip(sequence, structure).map { s, t in
            switch s {
            case ESM3Vocab.sequenceBOS: return ESM3Vocab.structureBOS
            case ESM3Vocab.sequencePAD: return ESM3Vocab.structurePAD
            case ESM3Vocab.sequenceEOS: return ESM3Vocab.structureEOS
            case ESM3Vocab.sequenceChainBreak: return ESM3Vocab.structureChainBreak
            default: return t < 0 ? ESM3Vocab.structureMask : t
            }
        }
    }
}

final class EncodeInputs: Module {
    @ModuleInfo(key: "sequence_embed") var sequenceEmbed: Embedding
    @ModuleInfo(key: "plddt_projection") var plddtProjection: Linear
    @ModuleInfo(key: "structure_per_res_plddt_projection") var perResPlddtProjection: Linear
    @ModuleInfo(key: "structure_tokens_embed") var structureEmbed: Embedding
    @ModuleInfo(key: "ss8_embed") var ss8Embed: Embedding
    @ModuleInfo(key: "sasa_embed") var sasaEmbed: Embedding
    @ModuleInfo(key: "function_embed") var functionEmbed: [Embedding]
    @ModuleInfo(key: "residue_embed") var residueEmbed: Embedding

    init(_ d: Int) {
        _sequenceEmbed.wrappedValue = Embedding(embeddingCount: 64, dimensions: d)
        _plddtProjection.wrappedValue = Linear(16, d)
        _perResPlddtProjection.wrappedValue = Linear(16, d)
        _structureEmbed.wrappedValue = Embedding(embeddingCount: 4096 + 5, dimensions: d)
        _ss8Embed.wrappedValue = Embedding(embeddingCount: 11, dimensions: d)
        _sasaEmbed.wrappedValue = Embedding(embeddingCount: 19, dimensions: d)
        _functionEmbed.wrappedValue = (0 ..< 8).map { _ in Embedding(embeddingCount: 260, dimensions: d / 8) }
        _residueEmbed.wrappedValue = Embedding(embeddingCount: 1478, dimensions: d)
    }

    /// misc.rbf(values, 0, 1, 16) in float32.
    static func rbf(_ values: MLXArray) -> MLXArray {
        let centers = MLXArray((0 ..< 16).map { Float($0) / 15 })
        let z = (values.asType(.float32).expandedDimensions(axis: -1) - centers) * 16
        return exp(-(z * z))
    }

    func callAsFunction(_ t: Tracks, perResPlddt: [Float]) -> MLXArray {
        let L = t.length
        func ids(_ v: [Int]) -> MLXArray { MLXArray(v.map(Int32.init), [1, L]) }
        let dtype = sequenceEmbed.weight.dtype
        var x = sequenceEmbed(ids(t.sequence))
        x = x + plddtProjection(Self.rbf(MLXArray(Float(1))).asType(dtype))  // average pLDDT 1, as ESM3.logits
        x = x + perResPlddtProjection(Self.rbf(MLXArray(perResPlddt, [1, L])).asType(dtype))
        x = x + structureEmbed(ids(t.structureInput)) + ss8Embed(ids(t.ss8)) + sasaEmbed(ids(t.sasa))
        let function = MLXArray(t.function.flatMap { $0.map(Int32.init) }, [1, L, 8])
        x = x + concatenated((0 ..< 8).map { functionEmbed[$0](function[0..., 0..., $0]) }, axis: -1)
        // nn.EmbeddingBag(mode="sum", padding_idx=0): token 0 adds nothing.
        let residue = MLXArray(t.residue.flatMap { $0.map(Int32.init) }, [1, L, 16])
        let bag = (residueEmbed(residue) * (residue .!= MLXArray(Int32(0))).expandedDimensions(axis: -1).asType(dtype)).sum(axis: 2)
        return x + bag
    }
}

final class OutputHeads: Module {
    @ModuleInfo(key: "sequence_head") var sequence: Head
    @ModuleInfo(key: "structure_head") var structure: Head
    @ModuleInfo(key: "ss8_head") var ss8: Head
    @ModuleInfo(key: "sasa_head") var sasa: Head
    @ModuleInfo(key: "function_head") var function: Head
    @ModuleInfo(key: "residue_head") var residue: Head
    init(_ d: Int) {
        _sequence.wrappedValue = Head(d, 64)
        _structure.wrappedValue = Head(d, 4096)
        _ss8.wrappedValue = Head(d, 11)
        _sasa.wrappedValue = Head(d, 19)
        _function.wrappedValue = Head(d, 260 * 8)
        _residue.wrappedValue = Head(d, 1478)
    }
}

/// Per-track logits of one forward pass, float32, without the batch dimension.
struct Logits {
    var sequence: MLXArray  // [L, 64]
    var structure: MLXArray  // [L, 4096]
    var ss8: MLXArray  // [L, 11]
    var sasa: MLXArray  // [L, 19]
    var function: MLXArray  // [L, 8, 260]
    var residue: MLXArray  // [L, 1478]
    var embeddings: MLXArray  // [L, d], before the final LayerNorm (ESMOutput.embeddings)
}

final class ESM3Net: Module {
    @ModuleInfo(key: "encoder") var encoder: EncodeInputs
    @ModuleInfo(key: "transformer") var transformer: Stack
    @ModuleInfo(key: "output_heads") var heads: OutputHeads

    /// esm3-sm-open-v1: d_model 1536, 24 heads, 256 geometric heads, 48 layers (the first with geometric attention).
    static let dimensions = 1536

    override init() {
        let d = Self.dimensions, layers = 48
        _encoder.wrappedValue = EncodeInputs(d)
        let scale = Float((Double(layers) / 36).squareRoot())
        _transformer.wrappedValue = Stack(
            blocks: (0 ..< layers).map { i in
                Block(
                    attn: Attention(d, heads: 24, bias: false, qkLayerNorm: true),
                    geom: i == 0 ? GeometricAttention(d, vHeads: 256, bias: false, zeroFrameless: true) : nil,
                    ffn: FeedForward(d, hidden: swigluHidden(8.0 / 3, d), bias: false, swiglu: true), scale: scale)
            }, finalNorm: d)
        _heads.wrappedValue = OutputHeads(d)
    }

    /// ESM3.logits for one protein. `taps` collects named intermediates for the parity check.
    func callAsFunction(_ t: Tracks, taps: Set<Int> = [], tap: ((String, MLXArray) -> Void)? = nil, cancelled: () throws -> Void = {}) throws -> Logits {
        // Per-residue pLDDT conditioning: 1 where the prompt has coordinates (any atom finite), else 0.
        let perRes = t.coordinates.map { row in row.contains { $0 != nil } ? Float(1) : 0 }
        let x0 = encoder(t, perResPlddt: perRes)
        tap?("embed_in", x0)
        let (bb, mask) = Geometry.backboneFrames(t.coordinates)
        let frames = Frames([bb], mask: [mask])
        let (x, pre) = try transformer(x0, frames: frames, taps: taps, tap: { i, v in tap?("block.\(i)", v) }, cancelled: cancelled)
        func f32(_ v: MLXArray) -> MLXArray { v[0].asType(.float32) }
        let L = t.length
        return Logits(
            sequence: f32(heads.sequence(x)), structure: f32(heads.structure(x)), ss8: f32(heads.ss8(x)), sasa: f32(heads.sasa(x)),
            function: f32(heads.function(x)).reshaped(L, 8, 260), residue: f32(heads.residue(x)), embeddings: f32(pre))
    }
}

// MARK: structure tokenizer (VQ-VAE)

final class RelativePosition: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    init(bins: Int, _ d: Int) { _embedding.wrappedValue = Embedding(embeddingCount: 2 * bins + 2, dimensions: d) }
}

final class Codebook: Module {
    @ParameterInfo(key: "embeddings") var embeddings: MLXArray
    init(codes: Int, _ d: Int) { _embeddings.wrappedValue = MLXArray.zeros([codes, d]) }
}

/// vqvae.StructureTokenEncoder: 2 fully geometric blocks over each residue's 16 nearest neighbours.
final class StructureEncoder: Module {
    @ModuleInfo(key: "transformer") var transformer: Stack
    @ModuleInfo(key: "pre_vq_proj") var preVQ: Linear
    @ModuleInfo(key: "codebook") var codebook: Codebook
    @ModuleInfo(key: "relative_positional_embedding") var relative: RelativePosition
    let k = 16, bins = 32

    override init() {
        let d = 1024
        _transformer.wrappedValue = Stack(
            blocks: (0 ..< 2).map { _ in
                Block(attn: nil, geom: GeometricAttention(d, vHeads: 128, bias: true, zeroFrameless: false),
                      ffn: FeedForward(d, hidden: swigluHidden(4, d), bias: true, swiglu: true), scale: 1)
            }, finalNorm: nil)
        _preVQ.wrappedValue = Linear(d, 128)
        _codebook.wrappedValue = Codebook(codes: 4096, 128)
        _relative.wrappedValue = RelativePosition(bins: 32, d)
    }

    /// Structure tokens of a chain's N, CA, C (no BOS/EOS), and the pre-quantisation vectors [L, 128].
    func encode(_ atoms: [[Vec3?]], tap: ((String, MLXArray) -> Void)? = nil) -> (tokens: [Int], latent: MLXArray) {
        let L = atoms.count
        let (frames, mask) = Geometry.backboneFrames(atoms)
        // Missing residues sit at the origin for the neighbour search, as the maker zeroes their coordinates.
        let ca: [Vec3?] = zip(atoms, mask).map { row, ok in ok ? row[1] : (0, 0, 0) }
        let edges = Geometry.knn(ca, mask: mask, k: k)
        let E = edges.first?.count ?? 0
        let group = Frames(edges.map { $0.map { frames[$0] } }, mask: edges.map { $0.map { mask[$0] } })
        // Residue index 1 ... L (ProteinChain.from_atom37), so the relative position is the edge's index offset.
        let offsets = edges.enumerated().flatMap { i, row in row.map { Int32(Swift.min(Swift.max($0 - row[0], -bins), bins) + bins + 1) } }
        let z0 = relative.embedding(MLXArray(offsets, [L, E]))
        tap?("edges", MLXArray(edges.flatMap { $0.map(Int32.init) }, [L, E]))
        tap?("z0", z0)
        var z = z0
        for (i, block) in transformer.blocks.enumerated() {
            z = block(z, frames: group)
            tap?("block.\(i)", z)
        }
        var out = z[0..., 0, 0...]
        out = out * MLXArray(mask).expandedDimensions(axis: -1).asType(out.dtype)
        let latent = preVQ(out).asType(.float32)
        let code = codebook.embeddings.asType(.float32)
        // Exact float32 squared distances to every code (elementwise, in blocks of residues; see GeometricAttention on
        // float32 matmul), then the nearest code.
        var tokens: [Int] = []
        for start in stride(from: 0, to: L, by: 32) {
            let block = latent[start ..< Swift.min(L, start + 32)]
            let diff = block.expandedDimensions(axis: 1) - code.expandedDimensions(axis: 0)
            tokens += (diff * diff).sum(axis: -1).argMin(axis: -1).asArray(Int32.self).map(Int.init)
        }
        return (tokens, latent)
    }
}

final class StructureHead: Module {
    @ModuleInfo(key: "ffn1") var ffn1: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ d: Int) {
        _ffn1.wrappedValue = Linear(d, d)
        _norm.wrappedValue = LayerNorm(dimensions: d)
        _proj.wrappedValue = Linear(d, 9 + 7 * 2)
    }
}

final class PairwiseHead: Module {
    @ModuleInfo(key: "downproject") var downproject: Linear
    @ModuleInfo(key: "linear1") var linear1: Linear
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear2") var linear2: Linear
    init(_ d: Int) {
        _downproject.wrappedValue = Linear(d, 128, bias: false)
        _linear1.wrappedValue = Linear(128, 128, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: 128)
        _linear2.wrappedValue = Linear(128, 64 + 16 * 6 + 64, bias: false)
    }
}

struct DecodedStructure {
    var backbone: MLXArray  // [L, 3, 3] N, CA, C, float32
    var plddt: MLXArray  // [L]
    var ptm: Float
    var pae: MLXArray  // [L, L]
}

/// vqvae.StructureTokenDecoder: 30 plain blocks, the 6D-rotation backbone head, pLDDT and the PAE bins (for pTM).
final class StructureDecoder: Module {
    @ModuleInfo(key: "embed") var embed: Embedding
    @ModuleInfo(key: "decoder_stack") var stack: Stack
    @ModuleInfo(key: "affine_output_projection") var head: StructureHead
    @ModuleInfo(key: "pairwise_classification_head") var pairwise: PairwiseHead
    @ModuleInfo(key: "plddt_head") var plddtHead: NamedHead

    override init() {
        let d = 1280
        _embed.wrappedValue = Embedding(embeddingCount: 4096 + 5, dimensions: d)
        _stack.wrappedValue = Stack(
            blocks: (0 ..< 30).map { _ in
                Block(attn: Attention(d, heads: 20, bias: false, qkLayerNorm: true), geom: nil,
                      ffn: FeedForward(d, hidden: swigluHidden(8.0 / 3, d), bias: false, swiglu: true), scale: 1)
            }, finalNorm: d)
        _head.wrappedValue = StructureHead(d)
        _pairwise.wrappedValue = PairwiseHead(d)
        _plddtHead.wrappedValue = NamedHead(d, 50)
    }

    /// Gram-Schmidt rotation (columns x, e1, x cross e1) of batched vectors [..., 3], float32.
    static func gramSchmidt(_ xAxis: MLXArray, _ xyPlane: MLXArray, eps: Float = 1e-10) -> MLXArray {
        let x = xAxis / sqrt((xAxis * xAxis).sum(axis: -1, keepDims: true) + eps)
        var e1 = xyPlane - x * (x * xyPlane).sum(axis: -1, keepDims: true)
        e1 = e1 / sqrt((e1 * e1).sum(axis: -1, keepDims: true) + eps)
        let e2 = cross(x, e1, axis: -1)
        return stacked([x, e1, e2], axis: -1)
    }

    /// tokens with BOS/EOS -> backbone, pLDDT and pTM over the residues; `stackOut` gets the normed stack output.
    func decode(_ tokens: [Int], stackOut: ((MLXArray) -> Void)? = nil) -> DecodedStructure {
        let L = tokens.count
        let (x, _) = stack(embed(MLXArray(tokens.map(Int32.init), [1, L])), frames: nil)
        stackOut?(x[0])
        // Dim6RotStructureHead from the identity frame: translation x 10, two normalised vectors, Gram-Schmidt.
        let p = head.proj(head.norm(gelu(head.ffn1(x))))[0].asType(.float32)
        let trans = p[0..., 0 ..< 3] * 10
        var vx = p[0..., 3 ..< 6], vy = p[0..., 6 ..< 9]
        vx = vx / (sqrt((vx * vx).sum(axis: -1, keepDims: true)) + 1e-5)
        vy = vy / (sqrt((vy * vy).sum(axis: -1, keepDims: true)) + 1e-5)
        // from_graham_schmidt(neg_x = x + t, origin = t, xy = y + t): x axis -x, xy plane y.
        let rot = Self.gramSchmidt(-vx, vy)  // [L, 3, 3]
        let local = MLXArray(Geometry.localBackbone.flatMap { [Float($0.0), Float($0.1), Float($0.2)] }, [3, 3])
        // atom_a = R local_a + t, elementwise in float32 (no float32 matmul).
        let backbone = (rot.expandedDimensions(axis: 1) * local.reshaped(1, 3, 1, 3)).sum(axis: -1) + trans.expandedDimensions(axis: 1)
        // pLDDT: expectation over 50 bins on [0, 1].
        let plddtLogits = plddtHead(x)[0].asType(.float32)
        let centers = MLXArray((0 ..< 50).map { (Float($0) + 0.5) / 50 })
        let plddt = (softmax(plddtLogits, axis: -1) * centers).sum(axis: -1)
        // PAE logits only (the last 64 pairwise bins): x_ij = [q_j * k_i, q_j - k_i].
        let down = pairwise.downproject(x)[0]
        let q = down[0..., 0 ..< 64], k = down[0..., 64...]
        let pair = concatenated([q.expandedDimensions(axis: 0) * k.expandedDimensions(axis: 1), q.expandedDimensions(axis: 0) - k.expandedDimensions(axis: 1)], axis: -1)
        let hidden = pairwise.norm(gelu(pairwise.linear1(pair)))
        let paeWeight = pairwise.linear2.weight[160...]
        let paeLogits = matmul(hidden, paeWeight.transposed()).asType(.float32)  // [L, L, 64]
        let aa = MLXArray(tokens.map { $0 < ESM3Vocab.structureCodes })
        let square = aa.expandedDimensions(axis: 1) .&& aa.expandedDimensions(axis: 0)
        let masked = MLX.where(square.expandedDimensions(axis: -1), paeLogits, MLXArray(-Float.greatestFiniteMagnitude))
        let probs = softmax(masked, axis: -1)
        let step: Float = 31 / 62
        var binValues = (0 ..< 63).map { Float($0) * 31 / 62 + step / 2 }
        binValues.append(binValues.last! + step)
        let bins = MLXArray(binValues)
        let pae = (probs * bins).sum(axis: -1)
        let n = Float(tokens.filter { $0 < ESM3Vocab.structureCodes }.count)
        let d0 = 1.24 * Foundation.pow(Swift.max(n, 19) - 15, 1.0 / 3) - 1.8
        let fd = 1 / (1 + (bins / d0) * (bins / d0))
        let perPair = (probs * fd).sum(axis: -1)
        let squareF = square.asType(.float32)
        let rowMean = (perPair * squareF).sum(axis: -1) / (squareF.sum(axis: -1) + 1e-10)
        let ptm = rowMean.max().item(Float.self)
        return DecodedStructure(backbone: backbone, plddt: plddt, ptm: ptm, pae: pae)
    }
}

// MARK: function decoder

/// function_decoder.FunctionTokenDecoder: LSH bits -> 3 blocks -> mean pool -> InterPro and keyword heads.
final class FunctionDecoder: Module {
    @ModuleInfo(key: "embedding") var embedding: Embedding
    @ModuleInfo(key: "decoder") var decoder: Stack
    @ModuleInfo(key: "heads") var heads: FunctionHeads

    override init() {
        let d = 1024
        _embedding.wrappedValue = Embedding(embeddingCount: 2 * 8 * 8, dimensions: d)
        _decoder.wrappedValue = Stack(
            blocks: (0 ..< 3).map { _ in
                Block(attn: Attention(d, heads: 8, bias: true, qkLayerNorm: false), geom: nil,
                      ffn: FeedForward(d, hidden: 4 * d, bias: true, swiglu: false), scale: 1)
            }, finalNorm: d)
        _heads.wrappedValue = FunctionHeads(d)
    }

    /// The decoder input of one position's 8 function tokens: bit b of depth k becomes token 2 (8 b + k) + bit;
    /// a row holding any special token (< 4) is all zeros.
    static func bitTokens(_ row: [Int]) -> [Int] {
        if row.contains(where: { $0 < 4 }) { return [Int](repeating: 0, count: 64) }
        var out: [Int] = []
        for bit in 0 ..< 8 { for (k, id) in row.enumerated() { out.append(2 * (bit * 8 + k) + (((id - 4) >> bit) & 1)) } }
        return out
    }

    /// (InterPro logits [N, 29026], keyword logits [N, 58641]) for rows of 8 function tokens, float32.
    func callAsFunction(_ rows: [[Int]]) -> (MLXArray, MLXArray) {
        let N = rows.count
        let inputs = MLXArray(rows.flatMap { Self.bitTokens($0).map(Int32.init) }, [N, 64])
        let (x, _) = decoder(embedding(inputs), frames: nil)
        let pooled = x.mean(axis: 1)
        return (heads.interpro(pooled).asType(.float32), heads.keywords(pooled).asType(.float32))
    }
}

final class FunctionHeads: Module {
    @ModuleInfo(key: "keyword_logits") var keywords: Head
    @ModuleInfo(key: "interpro_logits") var interpro: Head
    init(_ d: Int) {
        _keywords.wrappedValue = Head(d, 58641, hidden: 4 * d)
        _interpro.wrappedValue = Head(d, 29026, hidden: 4 * d)
    }
}
