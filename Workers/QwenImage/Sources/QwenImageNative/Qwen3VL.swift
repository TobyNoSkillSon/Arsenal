// SPDX-License-Identifier: MIT
// Qwen3-VL as Qwen-Image-2.1's prompt encoder: the vision tower (patch embedding, interpolated learned positions, 2D
// RoPE, merger and deep-stack mergers) and the text decoder with interleaved M-RoPE. It returns the last decoder layer's
// output before the final norm (the pipeline neutralises that norm), so the final norm and the LM head are not loaded.
// Vision tower and rope-index layout adapted from ml-explore/mlx-swift-lm `Libraries/MLXVLM/Models/Qwen3VL.swift`
// (MIT, LICENSE-upstream-mlx-swift-lm) and checked against Transformers `modeling_qwen3_vl.py`; the block MLP uses the
// configured tanh GELU (the upstream tower hard-codes x·σ(1.702x)).
import Foundation
import MLX
import MLXFast
import MLXNN
import MLXRandom

/// One condition image's patch grid in vision patches (t = 1 for still images).
public struct VisionGrid: Equatable, Sendable {
    public let t: Int, h: Int, w: Int
    public init(t: Int = 1, h: Int, w: Int) { self.t = t; self.h = h; self.w = w }
    public var patches: Int { t * h * w }
}

final class VisionPatchEmbed: Module {
    /// Stored Conv3d kernel [hidden, C, T, P, P]; the stride equals the kernel, so it is one matmul per flattened patch.
    @ModuleInfo(key: "proj") var proj: VisionPatchProjection
    init(_ v: Qwen3VLConfig.Vision) { _proj.wrappedValue = VisionPatchProjection(v) }
    func callAsFunction(_ patches: MLXArray) -> MLXArray {
        let w = proj.weight.reshaped(proj.weight.dim(0), -1)
        return matmul(patches.asType(w.dtype), w.transposed()) + proj.bias
    }
}
final class VisionPatchProjection: Module {
    let weight: MLXArray, bias: MLXArray
    init(_ v: Qwen3VLConfig.Vision) {
        // Conv3d's default initialisation (uniform ±1/√fan-in), so random-weight graphs depend on the pixels.
        let fanIn = v.inChannels * v.temporalPatch * v.patch * v.patch, bound = 1 / Float(fanIn).squareRoot()
        weight = MLXRandom.uniform(low: -bound, high: bound, [v.hidden, v.inChannels, v.temporalPatch, v.patch, v.patch])
        bias = MLXRandom.uniform(low: -bound, high: bound, [v.hidden])
    }
}

final class VisionAttention: Module {
    let heads: Int, headDim: Int
    @ModuleInfo(key: "qkv") var qkv: Linear
    @ModuleInfo(key: "proj") var proj: Linear
    init(_ v: Qwen3VLConfig.Vision) {
        heads = v.heads; headDim = v.hidden / v.heads
        _qkv.wrappedValue = Linear(v.hidden, 3 * v.hidden, bias: true)
        _proj.wrappedValue = Linear(v.hidden, v.hidden, bias: true)
    }
    /// `spans` are each image's token ranges; attention never crosses images.
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, spans: [Range<Int>]) -> MLXArray {
        let s = x.dim(0)
        let parts = qkv(x).reshaped(s, 3, heads, headDim)
        func rotate(_ t: MLXArray) -> MLXArray {
            let f = t.asType(.float32), half = headDim / 2
            let rotated = concatenated([-f[.ellipsis, half...], f[.ellipsis, 0..<half]], axis: -1)
            return (f * cos + rotated * sin).asType(t.dtype)
        }
        let q = rotate(parts[0..., 0]).transposed(1, 0, 2), k = rotate(parts[0..., 1]).transposed(1, 0, 2), v = parts[0..., 2].transposed(1, 0, 2)
        let scale = 1 / Float(headDim).squareRoot()
        let outputs = spans.map { r -> MLXArray in
            let queries = expandedDimensions(q[0..., r, 0...], axis: 0), keys = expandedDimensions(k[0..., r, 0...], axis: 0)
            let values = expandedDimensions(v[0..., r, 0...], axis: 0)
            return MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: scale, mask: .none)[0]
        }
        let attended = (outputs.count == 1 ? outputs[0] : concatenated(outputs, axis: 1)).transposed(1, 0, 2).reshaped(s, -1)
        return proj(attended)
    }
}

final class VisionMLP: Module {
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear
    init(_ v: Qwen3VLConfig.Vision) {
        _fc1.wrappedValue = Linear(v.hidden, v.intermediate, bias: true); _fc2.wrappedValue = Linear(v.intermediate, v.hidden, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(geluApproximate(fc1(x))) }
}

final class VisionBlock: Module {
    @ModuleInfo(key: "norm1") var norm1: LayerNorm
    @ModuleInfo(key: "norm2") var norm2: LayerNorm
    @ModuleInfo(key: "attn") var attn: VisionAttention
    @ModuleInfo(key: "mlp") var mlp: VisionMLP
    init(_ v: Qwen3VLConfig.Vision) {
        _norm1.wrappedValue = LayerNorm(dimensions: v.hidden, eps: 1e-6); _norm2.wrappedValue = LayerNorm(dimensions: v.hidden, eps: 1e-6)
        _attn.wrappedValue = VisionAttention(v); _mlp.wrappedValue = VisionMLP(v)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, spans: [Range<Int>]) -> MLXArray {
        let h = x + attn(norm1(x), cos: cos, sin: sin, spans: spans)
        return h + mlp(norm2(h))
    }
}

/// LayerNorm → Linear → GELU(exact) → Linear over each 2×2 group of patches. The deep-stack variant normalises after
/// grouping (`use_postshuffle_norm`).
final class VisionMerger: Module {
    let grouped: Int, postShuffle: Bool
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "linear_fc1") var fc1: Linear
    @ModuleInfo(key: "linear_fc2") var fc2: Linear
    init(_ v: Qwen3VLConfig.Vision, postShuffle: Bool) {
        grouped = v.hidden * v.merge * v.merge; self.postShuffle = postShuffle
        _norm.wrappedValue = LayerNorm(dimensions: postShuffle ? grouped : v.hidden, eps: 1e-6)
        _fc1.wrappedValue = Linear(grouped, grouped, bias: true); _fc2.wrappedValue = Linear(grouped, v.outHidden, bias: true)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = postShuffle ? norm(x.reshaped(-1, grouped)) : norm(x).reshaped(-1, grouped)
        return fc2(gelu(fc1(y)))
    }
}

final class VisionTower: Module {
    let config: Qwen3VLConfig.Vision
    @ModuleInfo(key: "patch_embed") var patchEmbed: VisionPatchEmbed
    @ModuleInfo(key: "pos_embed") var posEmbed: Embedding
    @ModuleInfo(key: "blocks") var blocks: [VisionBlock]
    @ModuleInfo(key: "merger") var merger: VisionMerger
    @ModuleInfo(key: "deepstack_merger_list") var deepstack: [VisionMerger]
    init(_ v: Qwen3VLConfig.Vision) {
        config = v
        _patchEmbed.wrappedValue = VisionPatchEmbed(v)
        _posEmbed.wrappedValue = Embedding(embeddingCount: v.positions, dimensions: v.hidden)
        _blocks.wrappedValue = (0..<v.depth).map { _ in VisionBlock(v) }
        _merger.wrappedValue = VisionMerger(v, postShuffle: false)
        _deepstack.wrappedValue = v.deepstack.map { _ in VisionMerger(v, postShuffle: true) }
    }

    /// fast_pos_embed_interpolate: bilinear over the learned √N×√N grid, then reordered into 2×2 merge groups.
    func positions(_ grids: [VisionGrid]) -> MLXArray {
        let side = Int(Double(config.positions).squareRoot()), maxIndex = side - 1, m = config.merge
        let dtype = posEmbed.weight.dtype
        var result: [MLXArray] = []
        for g in grids {
            func axis(_ n: Int) -> (floor: [Int32], ceil: [Int32], delta: [Float]) {
                var f = [Int32](), c = [Int32](), d = [Float]()
                for i in 0..<n {
                    let value = Float(i * maxIndex) / Float(Swift.max(1, n - 1))
                    let lo = Int32(value)
                    f.append(lo); c.append(Swift.min(lo + 1, Int32(maxIndex))); d.append(value - Float(lo))
                }
                return (f, c, d)
            }
            let (hf, hc, dh) = axis(g.h), (wf, wc, dw) = axis(g.w)
            var corners = [[Int32]](repeating: [], count: 4), weights = [[Float]](repeating: [], count: 4)
            for i in 0..<g.h {
                for j in 0..<g.w {
                    corners[0].append(hf[i] * Int32(side) + wf[j]); weights[0].append((1 - dh[i]) * (1 - dw[j]))
                    corners[1].append(hf[i] * Int32(side) + wc[j]); weights[1].append((1 - dh[i]) * dw[j])
                    corners[2].append(hc[i] * Int32(side) + wf[j]); weights[2].append(dh[i] * (1 - dw[j]))
                    corners[3].append(hc[i] * Int32(side) + wc[j]); weights[3].append(dh[i] * dw[j])
                }
            }
            var embed: MLXArray?
            for k in 0..<4 {
                let term = posEmbed(MLXArray(corners[k])) * MLXArray(weights[k]).asType(dtype).reshaped(-1, 1)
                embed = embed.map { $0 + term } ?? term
            }
            let e = tiled(embed!, repetitions: [g.t, 1]).reshaped(g.t, g.h / m, m, g.w / m, m, -1).transposed(0, 1, 3, 2, 4, 5)
            result.append(e.reshaped(-1, config.hidden))
        }
        return concatenated(result, axis: 0)
    }

    /// 2D rotary angles per patch in merge order: (row, column) each over head_dim/4 frequencies, then doubled.
    func rotary(_ grids: [VisionGrid]) -> (MLXArray, MLXArray) {
        let headDim = config.hidden / config.heads, dim = headDim / 2, m = config.merge
        let inv = stride(from: 0, to: dim, by: 2).map { 1 / Float(pow(10000.0, Double($0) / Double(dim))) }
        var angles = [Float]()
        for g in grids {
            for _ in 0..<g.t {
                for bh in 0..<(g.h / m) {
                    for bw in 0..<(g.w / m) {
                        for ih in 0..<m {
                            for iw in 0..<m {
                                let row = Float(bh * m + ih), col = Float(bw * m + iw)
                                angles += inv.map { row * $0 } + inv.map { col * $0 }
                            }
                        }
                    }
                }
            }
        }
        let a = MLXArray(angles, [angles.count / dim, dim])
        let doubled = concatenated([a, a], axis: -1).reshaped(-1, 1, headDim)
        return (MLX.cos(doubled), MLX.sin(doubled))
    }

    /// Patches [N, C·T·P·P] → merged image tokens [N/4, out] and one deep-stack feature set per configured layer.
    func callAsFunction(_ patches: MLXArray, grids: [VisionGrid]) -> (MLXArray, [MLXArray]) {
        var x = patchEmbed(patches) + positions(grids)
        let (cos, sin) = rotary(grids)
        var spans: [Range<Int>] = [], start = 0
        for g in grids { for _ in 0..<g.t { spans.append(start..<(start + g.h * g.w)); start += g.h * g.w } }
        var features: [MLXArray] = []
        for (index, block) in blocks.enumerated() {
            x = block(x, cos: cos, sin: sin, spans: spans)
            if let k = config.deepstack.firstIndex(of: index) { features.append(deepstack[k](x)) }
            eval(x)
        }
        return (merger(x), features)
    }
}

final class TextAttention: Module {
    let heads: Int, kvHeads: Int, headDim: Int
    @ModuleInfo(key: "q_proj") var q: Linear
    @ModuleInfo(key: "k_proj") var k: Linear
    @ModuleInfo(key: "v_proj") var v: Linear
    @ModuleInfo(key: "o_proj") var o: Linear
    @ModuleInfo(key: "q_norm") var qNorm: RMSNorm
    @ModuleInfo(key: "k_norm") var kNorm: RMSNorm
    init(_ t: Qwen3VLConfig.Text) {
        heads = t.heads; kvHeads = t.kvHeads; headDim = t.headDim
        _q.wrappedValue = Linear(t.hidden, heads * headDim, bias: false)
        _k.wrappedValue = Linear(t.hidden, kvHeads * headDim, bias: false)
        _v.wrappedValue = Linear(t.hidden, kvHeads * headDim, bias: false)
        _o.wrappedValue = Linear(heads * headDim, t.hidden, bias: false)
        _qNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: t.eps); _kNorm.wrappedValue = RMSNorm(dimensions: headDim, eps: t.eps)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let l = x.dim(1)
        func rope(_ t: MLXArray) -> MLXArray {
            let half = headDim / 2
            let rotated = concatenated([-t[.ellipsis, half...], t[.ellipsis, 0..<half]], axis: -1)
            return t * cos + rotated * sin
        }
        let queries = rope(qNorm(q(x).reshaped(1, l, heads, headDim)).transposed(0, 2, 1, 3))
        let keys = rope(kNorm(k(x).reshaped(1, l, kvHeads, headDim)).transposed(0, 2, 1, 3))
        let values = v(x).reshaped(1, l, kvHeads, headDim).transposed(0, 2, 1, 3)
        let out = MLXFast.scaledDotProductAttention(queries: queries, keys: keys, values: values, scale: 1 / Float(headDim).squareRoot(), mask: .causal)
        return o(out.transposed(0, 2, 1, 3).reshaped(1, l, heads * headDim))
    }
}

final class TextMLP: Module {
    @ModuleInfo(key: "gate_proj") var gate: Linear
    @ModuleInfo(key: "up_proj") var up: Linear
    @ModuleInfo(key: "down_proj") var down: Linear
    init(_ t: Qwen3VLConfig.Text) {
        _gate.wrappedValue = Linear(t.hidden, t.intermediate, bias: false); _up.wrappedValue = Linear(t.hidden, t.intermediate, bias: false)
        _down.wrappedValue = Linear(t.intermediate, t.hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { down(silu(gate(x)) * up(x)) }
}

final class TextLayer: Module {
    @ModuleInfo(key: "input_layernorm") var inputNorm: RMSNorm
    @ModuleInfo(key: "self_attn") var attention: TextAttention
    @ModuleInfo(key: "post_attention_layernorm") var postNorm: RMSNorm
    @ModuleInfo(key: "mlp") var mlp: TextMLP
    init(_ t: Qwen3VLConfig.Text) {
        _inputNorm.wrappedValue = RMSNorm(dimensions: t.hidden, eps: t.eps); _attention.wrappedValue = TextAttention(t)
        _postNorm.wrappedValue = RMSNorm(dimensions: t.hidden, eps: t.eps); _mlp.wrappedValue = TextMLP(t)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let h = x + attention(inputNorm(x), cos: cos, sin: sin)
        return h + mlp(postNorm(h))
    }
}

final class TextModel: Module {
    @ModuleInfo(key: "embed_tokens") var embed: Embedding
    @ModuleInfo(key: "layers") var layers: [TextLayer]
    init(_ t: Qwen3VLConfig.Text) {
        _embed.wrappedValue = Embedding(embeddingCount: t.vocab, dimensions: t.hidden)
        _layers.wrappedValue = (0..<t.layers).map { _ in TextLayer(t) }
    }
}

/// Holds the checkpoint's `model.visual` and `model.language_model` (loaded as `visual` and `language_model`).
public final class Qwen3VLEncoder: Module {
    public let config: Qwen3VLConfig
    @ModuleInfo(key: "visual") var visual: VisionTower
    @ModuleInfo(key: "language_model") var text: TextModel
    public init(_ c: Qwen3VLConfig) {
        config = c
        _visual.wrappedValue = VisionTower(c.vision); _text.wrappedValue = TextModel(c.text)
    }
    public var dtype: DType { text.layers.first?.inputNorm.weight.dtype ?? .float16 }

    /// Interleaved M-RoPE cos/sin [1, 1, L, headDim] for positions [3][L] (temporal, height, width).
    func mrope(_ positions: [[Int32]]) -> (MLXArray, MLXArray) {
        let t = config.text, half = t.headDim / 2, l = positions[0].count
        let inv = (0..<half).map { 1 / Float(pow(Double(t.ropeTheta), Double(2 * $0) / Double(t.headDim))) }
        var angles = [Float](repeating: 0, count: l * half)
        for i in 0..<l {
            for f in 0..<half {
                var axis = 0
                if f % 3 == 1 && f < 3 * t.mropeSection[1] { axis = 1 } else if f % 3 == 2 && f < 3 * t.mropeSection[2] { axis = 2 }
                angles[i * half + f] = Float(positions[axis][i]) * inv[f]
            }
        }
        let a = MLXArray(angles, [l, half])
        let doubled = concatenated([a, a], axis: -1).reshaped(1, 1, l, t.headDim)
        return (MLX.cos(doubled).asType(dtype), MLX.sin(doubled).asType(dtype))
    }

    /// Hidden states of the last decoder layer before the final norm, [1, L, hidden].
    /// `pixels` are the flattened patches of every condition image in order; `grids` their patch grids.
    public func callAsFunction(tokens: [Int32], pixels: MLXArray?, grids: [VisionGrid]) throws -> MLXArray {
        var x = text.embed(MLXArray(tokens).reshaped(1, -1))
        // Image features replace the `<|image_pad|>` rows, and each deep-stack feature set is added to those rows after
        // the first decoder layers. Both are row gathers (no scatter): row i reads its text embedding or feature k.
        var deepstack: [MLXArray] = []
        if let pixels, !grids.isEmpty {
            let (merged, features) = visual(pixels.asType(dtype), grids: grids)
            var k: Int32 = 0, embedRows = [Int32](), addRows = [Int32]()
            for (i, token) in tokens.enumerated() {
                if token == Int32(config.imageToken) {
                    embedRows.append(Int32(tokens.count) + k); addRows.append(1 + k); k += 1
                } else {
                    embedRows.append(Int32(i)); addRows.append(0)
                }
            }
            guard Int(k) == merged.dim(0) else { throw QwenImageError("The prompt has \(k) image slots for \(merged.dim(0)) image features") }
            x = concatenated([x[0], merged.asType(x.dtype)], axis: 0)[MLXArray(embedRows)].expandedDimensions(axis: 0)
            let rows = MLXArray(addRows), zero = MLXArray.zeros([1, x.dim(2)], dtype: x.dtype)
            deepstack = features.map { concatenated([zero, $0.asType(x.dtype)], axis: 0)[rows].expandedDimensions(axis: 0) }
        }
        let (cos, sin) = mrope(Self.ropeIndex(tokens: tokens, grids: grids, config: config))
        for (i, layer) in text.layers.enumerated() {
            x = layer(x, cos: cos, sin: sin)
            if i < deepstack.count {
                // swiftlint:disable:next shorthand_operator
                x = x + deepstack[i]
            }
            eval(x)
        }
        return x
    }

    /// get_rope_index for still images: text advances all three axes together; an image's tokens take
    /// (s, s + row, s + column) on the merged grid, and the text after it resumes at s + max(rows, columns).
    public static func ropeIndex(tokens: [Int32], grids: [VisionGrid], config: Qwen3VLConfig) -> [[Int32]] {
        var axes: [[Int32]] = [[], [], []]
        var next: Int32 = 0, i = 0, image = 0
        let merge = config.vision.merge
        while i < tokens.count {
            if tokens[i] == Int32(config.imageToken), image < grids.count {
                let g = grids[image], rows = g.h / merge, cols = g.w / merge
                for f in 0..<g.t {
                    for r in 0..<rows {
                        for c in 0..<cols {
                            axes[0].append(next + Int32(f)); axes[1].append(next + Int32(r)); axes[2].append(next + Int32(c))
                        }
                    }
                }
                next += Int32(Swift.max(g.t, rows, cols)); i += g.patches / (merge * merge); image += 1
            } else {
                for a in 0..<3 { axes[a].append(next) }
                next += 1; i += 1
            }
        }
        return axes
    }
}
