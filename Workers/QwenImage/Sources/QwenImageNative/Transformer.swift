// SPDX-License-Identifier: MIT
// QwenImage21Transformer2DModel (diffusers transformer_qwenimage21.py) in MLX. One sequence holds the prompt's text
// tokens, each condition image's latents (where the prompt encoder reserved its image slots, four latents per slot) and
// the target latents last. Attention is block-causal: (q ≥ k) or both in the same image block. With causal_condition
// the prefix (text and condition images) is modulated from t = 0 and never sees the target, so its keys and values
// are computed once per request (`prefill`) and every denoising step runs only the target tokens (`denoise`).
import Foundation
import MLX
import MLXFast
import MLXNN

/// Projection layout of the legacy diffusers Attention, so checkpoint names load unchanged.
final class DiTAttention: Module {
    let heads: Int, headDim: Int
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Linear]
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    init(_ c: TransformerConfig) {
        heads = c.heads; headDim = c.headDim
        _toQ.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
        _toK.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
        _toV.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
        _toOut.wrappedValue = [Linear(c.hidden, c.hidden, bias: false)]
        _normQ.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.eps)
        _normK.wrappedValue = RMSNorm(dimensions: c.headDim, eps: c.eps)
    }
    /// Queries, keys and values [1, heads, L, headDim] with RoPE applied.
    func qkv(_ x: MLXArray, rope: DiTRope.Table) -> (MLXArray, MLXArray, MLXArray) {
        let l = x.dim(1)
        let q = DiTRope.apply(normQ(toQ(x).reshaped(1, l, heads, headDim)), rope)
        let k = DiTRope.apply(normK(toK(x).reshaped(1, l, heads, headDim)), rope)
        let v = toV(x).reshaped(1, l, heads, headDim)
        return (q.transposed(0, 2, 1, 3), k.transposed(0, 2, 1, 3), v.transposed(0, 2, 1, 3))
    }
    func output(_ attended: MLXArray) -> MLXArray {
        toOut[0](attended.transposed(0, 2, 1, 3).reshaped(1, attended.dim(2), heads * headDim))
    }
}

final class DiTFeedForward: Module {
    @ModuleInfo(key: "proj") var proj: Linear
    @ModuleInfo(key: "gate_layer") var gate: Linear
    @ModuleInfo(key: "out") var out: Linear
    init(_ c: TransformerConfig) {
        _proj.wrappedValue = Linear(c.hidden, c.hidden * c.mlpRatio, bias: false)
        _gate.wrappedValue = Linear(c.hidden, c.hidden * c.mlpRatio, bias: false)
        _out.wrappedValue = Linear(c.hidden * c.mlpRatio, c.hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { out(silu(gate(x)) * proj(x)) }
}

/// The four shared modulation vectors one block reads: (1 + scale) and tanh(gate) for attention and MLP.
public struct DiTModulation {
    let scale1: MLXArray, gate1: MLXArray, scale2: MLXArray, gate2: MLXArray
    /// `params` is one row [1, 4·hidden]: mod1.scale, mod1.gate, mod2.scale, mod2.gate.
    init(_ params: MLXArray, hidden: Int) {
        let p = params.reshaped(1, 1, 4 * hidden)
        scale1 = 1 + p[0..., 0..., 0..<hidden]; gate1 = tanh(p[0..., 0..., hidden..<(2 * hidden)])
        scale2 = 1 + p[0..., 0..., (2 * hidden)..<(3 * hidden)]; gate2 = tanh(p[0..., 0..., (3 * hidden)...])
    }
}

final class DiTBlock: Module {
    let eps: Float
    @ModuleInfo(key: "attn") var attn: DiTAttention
    @ModuleInfo(key: "img_mlp") var mlp: DiTFeedForward
    init(_ c: TransformerConfig) {
        eps = c.eps
        _attn.wrappedValue = DiTAttention(c); _mlp.wrappedValue = DiTFeedForward(c)
    }
    func norm(_ x: MLXArray) -> MLXArray { MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps) }
    /// `attend` receives this block's modulated queries/keys/values and returns the attended values [1, heads, L, d].
    func callAsFunction(
        _ x: MLXArray, mod: DiTModulation, rope: DiTRope.Table, attend: (MLXArray, MLXArray, MLXArray) -> MLXArray
    ) -> MLXArray {
        let (q, k, v) = attn.qkv(norm(x) * mod.scale1, rope: rope)
        var h = x + mod.gate1 * attn.output(attend(q, k, v))
        // swiftlint:disable:next shorthand_operator
        h = h + mod.gate2 * mlp(norm(h) * mod.scale2)
        return h
    }
    /// diffusers clips the residual stream at the FP16 range after every block; BF16 passes unchanged.
    static func clipHalf(_ h: MLXArray) -> MLXArray { h.dtype == .float16 ? clip(h, min: -65504, max: 65504) : h }
}

/// 3-axis (frame, height, width) RoPE over the joint sequence (QwenImage21Rope). Pairs are interleaved (complex view).
public enum DiTRope {
    public struct Table { public let cos: MLXArray, sin: MLXArray }
    /// Position triples for one run of tokens.
    public struct Positions: Equatable {
        public var frame: [Int32] = [], height: [Int32] = [], width: [Int32] = []
        public var count: Int { frame.count }
        mutating func append(_ other: Positions) { frame += other.frame; height += other.height; width += other.width }
    }
    /// Per-axis inverse frequencies 1/10000^(2i/dim), float32.
    static func inverse(_ dim: Int) -> [Float] { stride(from: 0, to: dim, by: 2).map { 1 / Float(pow(10000.0, Double($0) / Double(dim))) } }
    public static func table(_ p: Positions, axes: [Int]) -> Table {
        let n = p.count
        var angles = [Float](); angles.reserveCapacity(n * axes.reduce(0, +) / 2)
        let freqs = axes.map(inverse)
        for i in 0..<n {
            for (axis, values) in [p.frame, p.height, p.width].enumerated() {
                let position = Float(values[i])
                for f in freqs[axis] { angles.append(position * f) }
            }
        }
        let a = MLXArray(angles, [n, axes.reduce(0, +) / 2])
        return Table(cos: MLX.cos(a), sin: MLX.sin(a))
    }
    /// x [1, L, H, D] → rotated pairs (x0, x1)·e^{iθ}, computed in float32 like the reference, returned in x's dtype.
    static func apply(_ x: MLXArray, _ t: Table) -> MLXArray {
        let shape = x.shape
        let pairs = x.asType(.float32).reshaped(shape[0], shape[1], shape[2], shape[3] / 2, 2)
        let x0 = pairs[.ellipsis, 0], x1 = pairs[.ellipsis, 1]
        let c = t.cos.reshaped(1, shape[1], 1, shape[3] / 2), s = t.sin.reshaped(1, shape[1], 1, shape[3] / 2)
        return stacked([x0 * c - x1 * s, x0 * s + x1 * c], axis: -1).reshaped(shape).asType(x.dtype)
    }
    /// Image block grid centred on zero: h ∈ [−(H − H/2), H/2), raster order, frame fixed.
    public static func image(frame: Int, height: Int, width: Int) -> Positions {
        var p = Positions()
        for h in -(height - height / 2)..<(height / 2) {
            for w in -(width - width / 2)..<(width / 2) {
                p.frame.append(Int32(frame)); p.height.append(Int32(h)); p.width.append(Int32(w))
            }
        }
        return p
    }
    public static func text(from start: Int, count: Int) -> Positions {
        let values = (start..<(start + count)).map(Int32.init)
        return Positions(frame: values, height: values, width: values)
    }
}

/// The joint-sequence layout of one request: runs of text tokens and image blocks in prompt-encoder order, the target
/// image last. Built from the prompt encoder's image-slot mask and each image's latent grid.
public struct SequenceLayout: Equatable {
    public enum Run: Equatable {
        /// `range` indexes the prompt encoder's output rows (system prompt already dropped).
        case text(range: Range<Int>)
        /// Condition image `index` (its latent grid), or the target.
        case image(index: Int, height: Int, width: Int)
        case target(height: Int, width: Int)
    }
    public let runs: [Run]
    public let positions: DiTRope.Positions
    public let prefixPositions: DiTRope.Positions
    public let targetPositions: DiTRope.Positions
    public var prefixLength: Int { prefixPositions.count }

    /// `imageSlots[i]` is true where the encoder row is an `<|image_pad|>`; condition images take their slot runs in
    /// order, each slot standing for a 2×2 group of latents. `conditions` are latent grids (height, width).
    public init(imageSlots: [Bool], conditions: [(height: Int, width: Int)], target: (height: Int, width: Int)) throws {
        var runs: [Run] = [], i = 0, image = 0
        while i < imageSlots.count {
            if imageSlots[i] {
                guard image < conditions.count else { throw QwenImageError("More image slots than condition images") }
                let (h, w) = conditions[image]
                guard h % 2 == 0, w % 2 == 0 else { throw QwenImageError("Condition latent grid must be even") }
                let slots = h * w / 4
                guard i + slots <= imageSlots.count, imageSlots[i..<(i + slots)].allSatisfy({ $0 }) else {
                    throw QwenImageError("Image slots do not match condition image \(image + 1)'s latent grid")
                }
                runs.append(.image(index: image, height: h, width: w)); i += slots; image += 1
            } else {
                var j = i
                while j < imageSlots.count, !imageSlots[j] { j += 1 }
                runs.append(.text(range: i..<j)); i = j
            }
        }
        guard image == conditions.count else { throw QwenImageError("Condition images without image slots") }
        runs.append(.target(height: target.height, width: target.width))
        var position = 0, all = DiTRope.Positions(), prefix = DiTRope.Positions()
        for run in runs {
            let p: DiTRope.Positions
            switch run {
            case .text(let range):
                p = DiTRope.text(from: position, count: range.count); position += range.count
            case .image(_, let h, let w), .target(let h, let w):
                p = DiTRope.image(frame: position, height: h, width: w); position += Swift.max(h, w)
            }
            all.append(p)
            if case .target = run {} else { prefix.append(p) }
        }
        self.runs = runs; positions = all; prefixPositions = prefix
        targetPositions = DiTRope.image(frame: Self.targetFrame(runs), height: target.height, width: target.width)
    }
    static func targetFrame(_ runs: [Run]) -> Int {
        runs.reduce(0) { position, run in
            switch run {
            case .text(let range): return position + range.count
            case .image(_, let h, let w): return position + Swift.max(h, w)
            case .target: return position
            }
        }
    }
    /// Prefix runs as (start, end, isText) in prefix-token coordinates (QwenImage21AttnProcessor segments).
    public var segments: [(start: Int, end: Int, text: Bool)] {
        var result: [(Int, Int, Bool)] = [], start = 0
        for run in runs {
            switch run {
            case .text(let range): result.append((start, start + range.count, true)); start += range.count
            case .image(_, let h, let w): result.append((start, start + h * w, false)); start += h * w
            case .target: break
            }
        }
        return result.filter { $0.1 > $0.0 }.map { (start: $0.0, end: $0.1, text: $0.2) }
    }
}

/// Step-independent prefix keys/values per block, [1, heads, P, d].
public final class PrefixCache {
    var keys: [MLXArray] = [], values: [MLXArray] = []
    public var length: Int { keys.first?.dim(2) ?? 0 }
    public var bytes: Int { (keys + values).reduce(0) { $0 + $1.nbytes } }
}

public final class QwenImageTransformer: Module {
    public let config: TransformerConfig
    @ModuleInfo(key: "time_text_embed") var timeEmbed: TimeTextEmbed
    @ModuleInfo(key: "txt_in") var textIn: TextProjection
    @ModuleInfo(key: "img_in") var imageIn: Linear
    @ModuleInfo(key: "modulation") var modulation: SharedModulation
    @ModuleInfo(key: "transformer_blocks") var blocks: [DiTBlock]
    @ModuleInfo(key: "norm_out") var normOut: OutputNorm
    @ModuleInfo(key: "proj_out") var projOut: Linear

    /// Lab diagnostics: each block's residual stream before the FP16 clip (overflow would otherwise hide in the clip).
    public var residualObserver: ((String, Int, MLXArray) -> Void)?

    public init(_ c: TransformerConfig) {
        config = c
        _timeEmbed.wrappedValue = TimeTextEmbed(c.hidden)
        _textIn.wrappedValue = TextProjection(c)
        _imageIn.wrappedValue = Linear(c.inChannels, c.hidden, bias: false)
        _modulation.wrappedValue = SharedModulation(c.hidden)
        _blocks.wrappedValue = (0..<c.layers).map { _ in DiTBlock(c) }
        _normOut.wrappedValue = OutputNorm(c)
        _projOut.wrappedValue = Linear(c.hidden, c.outChannels, bias: false)
    }

    /// The model dtype (activations follow the stored half format).
    public var dtype: DType { blocks.first?.attn.normQ.weight.dtype ?? .float16 }

    /// Timestep embedding for t ∈ [0, 1000]; the pipeline rounds t/1000 through the activation dtype.
    func temb(_ t: Float) -> MLXArray {
        let rounded = MLXArray([t]).asType(dtype).asType(.float32) / 1000
        let sigma = rounded.asType(dtype).asType(.float32)
        return timeEmbed(sigma * 1000, dtype: dtype)
    }
    func modulationRow(_ temb: MLXArray) -> MLXArray { modulation.linear(silu(temb)) }

    /// Runs the prefix (text and condition images) once and stores every block's keys and values.
    /// `text` is the prompt encoder output [1, S, contextDim]; `conditions` the packed condition latents per image [1, h·w, 64].
    public func prefill(text: MLXArray, conditions: [MLXArray], layout: SequenceLayout, eval evaluate: Bool = true) -> PrefixCache {
        let cache = PrefixCache()
        let projected = textIn(text.asType(dtype))
        var pieces: [MLXArray] = []
        for run in layout.runs {
            switch run {
            case .text(let range): pieces.append(projected[0..., range, 0...])
            case .image(let index, _, _): pieces.append(imageIn(conditions[index].asType(dtype)))
            case .target: break
            }
        }
        guard !pieces.isEmpty else { return cache }
        var x = concatenated(pieces, axis: 1)
        let mod = DiTModulation(modulationRow(temb(0)), hidden: config.hidden)
        let rope = DiTRope.table(layout.prefixPositions, axes: config.axes)
        let segments = layout.segments
        let scale = 1 / Float(config.headDim).squareRoot()
        for (index, block) in blocks.enumerated() {
            let last = index == blocks.count - 1
            x = block(x, mod: mod, rope: rope) { q, k, v in
                cache.keys.append(k); cache.values.append(v)
                if last { return v } // the prefix output after the last block is never read
                let parts = segments.map { s -> MLXArray in
                    let keys = k[0..., 0..., 0..<s.end], values = v[0..., 0..., 0..<s.end]
                    let queries = q[0..., 0..., s.start..<s.end]
                    // A text run is causal over its own tokens and sees everything before it (bottom-right aligned
                    // causal mask); an image block sees everything before it and all of itself.
                    return MLXFast.scaledDotProductAttention(
                        queries: queries, keys: keys, values: values, scale: scale, mask: s.text ? .causal : .none)
                }
                return parts.count == 1 ? parts[0] : concatenated(parts, axis: 2)
            }
            residualObserver?("prefill", index, x)
            x = DiTBlock.clipHalf(x)
            if evaluate { eval(x, cache.keys[index], cache.values[index]) }
        }
        return cache
    }

    /// Velocity for the target latents [1, T, 64] at timestep t (0…1000), attending to the cached prefix.
    public func denoise(_ latents: MLXArray, t: Float, cache: PrefixCache, layout: SequenceLayout) -> MLXArray {
        var x = imageIn(latents.asType(dtype))
        let e = temb(t)
        let mod = DiTModulation(modulationRow(e), hidden: config.hidden)
        let rope = DiTRope.table(layout.targetPositions, axes: config.axes)
        let scale = 1 / Float(config.headDim).squareRoot()
        for (index, block) in blocks.enumerated() {
            x = block(x, mod: mod, rope: rope) { q, k, v in
                let keys = cache.length > 0 ? concatenated([cache.keys[index], k], axis: 2) : k
                let values = cache.length > 0 ? concatenated([cache.values[index], v], axis: 2) : v
                return MLXFast.scaledDotProductAttention(queries: q, keys: keys, values: values, scale: scale, mask: .none)
            }
            residualObserver?("denoise", index, x)
            x = DiTBlock.clipHalf(x)
        }
        return projOut(normOut(x, temb: e))
    }
}

/// nn.Sequential(SiLU, Linear): the checkpoint's `modulation.1.weight` loads as `modulation.linear.weight`
/// (`QwenImageWeights.transformerName`), since MLX reads a numeric path component as an array index.
final class SharedModulation: Module {
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ hidden: Int) { _linear.wrappedValue = Linear(hidden, 4 * hidden, bias: false) }
}

/// Sinusoidal timestep (cos first) → Linear → SiLU → Linear, no biases (TimestepEmbedding with sample_proj_bias=False).
final class TimeTextEmbed: Module {
    @ModuleInfo(key: "timestep_embedder") var embedder: TimestepEmbedder
    init(_ hidden: Int) { _embedder.wrappedValue = TimestepEmbedder(hidden) }
    /// QwenImage21TemporalTimesteps: [cos(t·f), sin(t·f)], f_k = exp(−ln(10000)·k/128), float32.
    static func sinusoid(_ t: MLXArray) -> MLXArray {
        let half = 128
        let freqs = MLXArray((0..<half).map { Float(exp(-log(10000.0) * Double($0) / Double(half))) })
        let args = t.reshaped(-1, 1) * freqs.reshaped(1, half)
        return concatenated([MLX.cos(args), MLX.sin(args)], axis: -1)
    }
    func callAsFunction(_ t: MLXArray, dtype: DType) -> MLXArray { embedder(Self.sinusoid(t).asType(dtype)) }
}
final class TimestepEmbedder: Module {
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(_ hidden: Int) {
        _linear1.wrappedValue = Linear(256, hidden, bias: false); _linear2.wrappedValue = Linear(hidden, hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(silu(linear1(x))) }
}

/// Zero-centred RMSNorm (scale = weight + 1, float32) → Linear → GELU(tanh) → Linear.
final class TextProjection: Module {
    let eps: Float
    @ModuleInfo(key: "text_norm") var norm: ZeroCenterNorm
    @ModuleInfo(key: "in_layer") var inLayer: Linear
    @ModuleInfo(key: "out_layer") var outLayer: Linear
    init(_ c: TransformerConfig) {
        eps = c.eps
        _norm.wrappedValue = ZeroCenterNorm(c.contextDim)
        _inLayer.wrappedValue = Linear(c.contextDim, c.hidden, bias: false)
        _outLayer.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let normed = MLXFast.rmsNorm(x.asType(.float32), weight: norm.weight.asType(.float32) + 1, eps: eps).asType(x.dtype)
        return outLayer(geluApproximate(inLayer(normed)))
    }
}
final class ZeroCenterNorm: Module {
    let weight: MLXArray
    init(_ dim: Int) { weight = MLXArray.zeros([dim]) }
}

/// Final scale-only adaptive LayerNorm: LN(x) · (1 + Linear(SiLU(temb))).
final class OutputNorm: Module {
    let eps: Float
    @ModuleInfo(key: "linear") var linear: Linear
    init(_ c: TransformerConfig) {
        eps = c.eps; _linear.wrappedValue = Linear(c.hidden, c.hidden, bias: false)
    }
    func callAsFunction(_ x: MLXArray, temb: MLXArray) -> MLXArray {
        let scale = linear(silu(temb).asType(x.dtype)).reshaped(1, 1, -1)
        return MLXFast.layerNorm(x, weight: nil, bias: nil, eps: eps) * (1 + scale)
    }
}
