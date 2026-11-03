// SPDX-License-Identifier: MIT
// The FLUX 3 Klein DiT: per-mode blocks (target image, reference images, text each attend only within themselves), then
// single-stream blocks over [image, references, text] with per-token modulation, each block a parallel attention +
// SwiGLU feed-forward under one gate. Time conditioning is a flow map: the embedding of the current time plus the
// embedding of the target time; references and text see time 0. Parameter names follow the diffusers checkpoint.
// mflux reference: models/flux3/model/flux3_transformer/*.py. Difference, within BF16 rounding: per-head RMSNorm and
// RoPE run in BF16 (the fused RMSNorm accumulates in float32), where mflux casts q/k to float32 for both.
import Foundation
import MLX
import MLXFast
import MLXNN

/// Adjacent-pair rotation: x [B, H, L, D], cos/sin [L, D/2] in x's dtype.
func flux3Rope(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
    let s = x.shape
    let pairs = x.reshaped(s[0], s[1], s[2], s[3] / 2, 2)
    let x0 = pairs[.ellipsis, 0], x1 = pairs[.ellipsis, 1]
    return stacked([cos * x0 - sin * x1, sin * x0 + cos * x1], axis: -1).reshaped(s)
}

public final class Flux3Attention: Module {
    let heads: Int, headDim: Int
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    @ModuleInfo(key: "norm_q") var normQ: RMSNorm
    @ModuleInfo(key: "norm_k") var normK: RMSNorm
    init(_ c: Flux3TransformerConfig) {
        heads = c.heads; headDim = c.headDim
        _toQ.wrappedValue = Linear(c.inner, c.inner, bias: false); _toK.wrappedValue = Linear(c.inner, c.inner, bias: false)
        _toV.wrappedValue = Linear(c.inner, c.inner, bias: false); _toOut.wrappedValue = Linear(c.inner, c.inner, bias: false)
        _normQ.wrappedValue = RMSNorm(dimensions: c.headDim, eps: 1e-6); _normK.wrappedValue = RMSNorm(dimensions: c.headDim, eps: 1e-6)
    }
    func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        func split(_ t: MLXArray) -> MLXArray { t.reshaped(b, l, heads, headDim).transposed(0, 2, 1, 3) }
        let q = flux3Rope(normQ(split(toQ(x))), cos: cos, sin: sin)
        let k = flux3Rope(normK(split(toK(x))), cos: cos, sin: sin)
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: split(toV(x)), scale: 1 / Float(headDim).squareRoot(), mask: .none)
        return toOut(out.transposed(0, 2, 1, 3).reshaped(b, l, heads * headDim))
    }
}

public final class Flux3FeedForward: Module {
    @ModuleInfo(key: "linear_in") var linearIn: Linear
    @ModuleInfo(key: "linear_out") var linearOut: Linear
    init(_ c: Flux3TransformerConfig) {
        _linearIn.wrappedValue = Linear(c.inner, c.mlpHidden * 2, bias: false)
        _linearOut.wrappedValue = Linear(c.mlpHidden, c.inner, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let parts = MLX.split(linearIn(x), parts: 2, axis: -1)
        return linearOut(silu(parts[0]) * parts[1])
    }
}

/// (shift, scale, gate), each [B, 1, dim] or per-token [B, L, dim].
public typealias Flux3Mod = (shift: MLXArray, scale: MLXArray, gate: MLXArray)

public final class Flux3Block: Module {
    @ModuleInfo var norm: LayerNorm
    @ModuleInfo var attn: Flux3Attention
    @ModuleInfo var ff: Flux3FeedForward
    init(_ c: Flux3TransformerConfig) {
        _norm.wrappedValue = LayerNorm(dimensions: c.inner, eps: 1e-6, affine: false)
        _attn.wrappedValue = Flux3Attention(c); _ff.wrappedValue = Flux3FeedForward(c)
    }
    public func callAsFunction(_ x: MLXArray, cos: MLXArray, sin: MLXArray, mod: Flux3Mod) -> MLXArray {
        let h = (1 + mod.scale) * norm(x) + mod.shift
        return x + mod.gate * (attn(h, cos: cos, sin: sin) + ff(h))
    }
}

public final class Flux3Modulation: Module {
    @ModuleInfo var linear: Linear
    init(_ dim: Int) { _linear.wrappedValue = Linear(dim, dim * 3, bias: false) }
    public func callAsFunction(_ vec: MLXArray) -> Flux3Mod {
        let parts = MLX.split(linear(silu(vec))[0..., .newAxis, 0...], parts: 3, axis: -1)
        return (parts[0], parts[1], parts[2])
    }
}

public final class Flux3TimestepEmbedder: Module {
    let channels: Int
    @ModuleInfo(key: "linear_1") var linear1: Linear
    @ModuleInfo(key: "linear_2") var linear2: Linear
    init(channels: Int, dim: Int) {
        self.channels = channels
        _linear1.wrappedValue = Linear(channels, dim, bias: false); _linear2.wrappedValue = Linear(dim, dim, bias: false)
    }
    /// The vendor carries timesteps in BF16: t is rounded to BF16, scaled by 1000 in BF16, then [cos | sin] of
    /// t · exp(−ln(10⁴) · i / half) in float32 (a [B, 256] table), rounded to BF16.
    public static func sinusoid(_ t: MLXArray, channels: Int) -> MLXArray {
        let tb = t.asType(.bfloat16) * MLXArray(Float(1000)).asType(.bfloat16)
        let half = channels / 2
        let freqs = MLX.exp(MLXArray(0..<half).asType(.float32) * Float(-Foundation.log(10000.0)) / Float(half))
        let args = tb.asType(.float32)[0..., .newAxis] * freqs[.newAxis, 0...]
        return concatenated([MLX.cos(args), MLX.sin(args)], axis: -1).asType(.bfloat16)
    }
    public func callAsFunction(_ t: MLXArray) -> MLXArray {
        let e = Self.sinusoid(t, channels: channels).asType(linear1.weight.dtype)
        return linear2(silu(linear1(e)))
    }
}

/// diffusers AdaLayerNormContinuous without bias: (scale, shift) from SiLU(vec).
public final class Flux3NormOut: Module {
    let dim: Int
    @ModuleInfo var linear: Linear
    @ModuleInfo var norm: LayerNorm
    init(_ dim: Int) {
        self.dim = dim
        _linear.wrappedValue = Linear(dim, dim * 2, bias: false)
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6, affine: false)
    }
    func callAsFunction(_ x: MLXArray, _ vec: MLXArray) -> MLXArray {
        let e = linear(silu(vec))
        let scale = e[0..., ..<dim], shift = e[0..., dim...]
        return norm(x) * (1 + scale)[0..., .newAxis, 0...] + shift[0..., .newAxis, 0...]
    }
}

/// Containers named after the checkpoint's per-mode keys.
public final class Flux3ModeBlocks: Module {
    @ModuleInfo public var image: [Flux3Block]
    @ModuleInfo(key: "image_cond") public var imageCond: [Flux3Block]
    init(_ c: Flux3TransformerConfig) {
        _image.wrappedValue = (0..<c.numLayers).map { _ in Flux3Block(c) }
        _imageCond.wrappedValue = (0..<c.numLayers).map { _ in Flux3Block(c) }
    }
}

public final class Flux3Modulations: Module {
    @ModuleInfo public var image: Flux3Modulation
    @ModuleInfo(key: "image_cond") public var imageCond: Flux3Modulation
    @ModuleInfo public var txt: Flux3Modulation
    init(_ dim: Int) {
        _image.wrappedValue = Flux3Modulation(dim); _imageCond.wrappedValue = Flux3Modulation(dim); _txt.wrappedValue = Flux3Modulation(dim)
    }
}

public final class Flux3Embedders: Module {
    @ModuleInfo public var image: Linear
    @ModuleInfo(key: "image_cond") public var imageCond: Linear
    init(_ c: Flux3TransformerConfig) {
        _image.wrappedValue = Linear(c.inChannels, c.inner, bias: false)
        _imageCond.wrappedValue = Linear(c.condInChannels, c.inner, bias: false)
    }
}

/// RoPE tables for one image: per-mode slices of one host table (the single stream's table is their concatenation).
public struct Flux3RopeTables {
    public let image: (cos: MLXArray, sin: MLXArray)
    public let cond: (cos: MLXArray, sin: MLXArray)?
    public let text: (cos: MLXArray, sin: MLXArray)
    public let all: (cos: MLXArray, sin: MLXArray)

    /// ids in single-stream order [image, references…, text]; tables [L, D/2] in `dtype`.
    public init(image: [[Float]], cond: [[Float]], text: [[Float]], config: Flux3TransformerConfig, dtype: DType) {
        let ids = image + cond + text, half = config.headDim / 2
        let (c, s) = Flux3Rope.tables(ids, axes: config.axes, theta: config.ropeTheta)
        let cos = MLXArray(c, [ids.count, half]).asType(dtype), sin = MLXArray(s, [ids.count, half]).asType(dtype)
        let a = image.count, b = a + cond.count
        all = (cos, sin)
        self.image = (cos[..<a], sin[..<a])
        self.cond = cond.isEmpty ? nil : (cos[a..<b], sin[a..<b])
        self.text = (cos[b...], sin[b...])
    }
}

public final class Flux3Transformer: Module {
    public let config: Flux3TransformerConfig
    @ModuleInfo(key: "timestep_embedder") var timeEmbedder: Flux3TimestepEmbedder
    @ModuleInfo(key: "target_timestep_embedder") var targetEmbedder: Flux3TimestepEmbedder
    @ModuleInfo(key: "early_stream_modulations") var early: Flux3Modulations
    @ModuleInfo(key: "single_stream_modulations") var single: Flux3Modulations
    @ModuleInfo(key: "x_embedders") var xEmbedders: Flux3Embedders
    @ModuleInfo(key: "context_embedder") var contextEmbedder: Linear
    @ModuleInfo(key: "mode_blocks") public var modeBlocks: Flux3ModeBlocks
    @ModuleInfo(key: "txt_mode_blocks") public var txtBlocks: [Flux3Block]
    @ModuleInfo(key: "single_transformer_blocks") public var singleBlocks: [Flux3Block]
    @ModuleInfo(key: "norm_out") var normOut: Flux3NormOut
    @ModuleInfo(key: "proj_out") var projOut: Linear

    public init(_ c: Flux3TransformerConfig) {
        config = c
        _timeEmbedder.wrappedValue = Flux3TimestepEmbedder(channels: c.timestepChannels, dim: c.inner)
        _targetEmbedder.wrappedValue = Flux3TimestepEmbedder(channels: c.timestepChannels, dim: c.inner)
        _early.wrappedValue = Flux3Modulations(c.inner); _single.wrappedValue = Flux3Modulations(c.inner)
        _xEmbedders.wrappedValue = Flux3Embedders(c)
        _contextEmbedder.wrappedValue = Linear(c.jointDim, c.inner, bias: false)
        _modeBlocks.wrappedValue = Flux3ModeBlocks(c)
        _txtBlocks.wrappedValue = (0..<c.numLayers).map { _ in Flux3Block(c) }
        _singleBlocks.wrappedValue = (0..<c.numSingleLayers).map { _ in Flux3Block(c) }
        _normOut.wrappedValue = Flux3NormOut(c.inner)
        _projOut.wrappedValue = Linear(c.inner, c.inChannels, bias: false)
    }

    /// Flow-map time vectors: (image: e(t) + e'(t_next), context: e(0) + e'(t_next)), each [B, dim].
    public func timeVectors(t: MLXArray, next: MLXArray) -> (image: MLXArray, context: MLXArray) {
        let target = targetEmbedder(next)
        return (timeEmbedder(t) + target, timeEmbedder(MLXArray.zeros(like: t)) + target)
    }

    public struct Stages {
        public var vec: (image: MLXArray, context: MLXArray)
        public var early: [String: Flux3Mod] = [:], single: [String: Flux3Mod] = [:]
        public var embedded: [String: MLXArray] = [:], modes: [[String: MLXArray]] = [], singles: [MLXArray] = []
    }

    /// One velocity: x [B, N, 128] (float32 state, cast to the weights' dtype), text [B, T, 20480], cond [B, M, 128] or
    /// nil, t and next [B] float32. Returns [B, N, 128] in the weights' dtype. `stages` records every intermediate.
    public func callAsFunction(
        _ x: MLXArray, text: MLXArray, cond: MLXArray?, t: MLXArray, next: MLXArray, rope: Flux3RopeTables,
        stages: ((Stages) -> Void)? = nil
    ) -> MLXArray {
        let dtype = projOut.weight.dtype
        let vec = timeVectors(t: t, next: next)
        let hasCond = cond.map { $0.dim(1) > 0 } ?? false
        var modes: [(name: String, early: Flux3Mod, single: Flux3Mod)] = [("image", early.image(vec.image), single.image(vec.image))]
        if hasCond { modes.append(("image_cond", early.imageCond(vec.context), single.imageCond(vec.context))) }
        modes.append(("txt", early.txt(vec.context), single.txt(vec.context)))

        var streams: [MLXArray] = [xEmbedders.image(x.asType(dtype))]
        var tables: [(cos: MLXArray, sin: MLXArray)] = [rope.image]
        if hasCond, let cond, let c = rope.cond { streams.append(xEmbedders.imageCond(cond.asType(dtype))); tables.append(c) }
        streams.append(contextEmbedder(text.asType(dtype))); tables.append(rope.text)

        var record = stages.map { _ in Stages(vec: vec) }
        if record != nil {
            for (i, m) in modes.enumerated() { record!.early[m.name] = m.early; record!.single[m.name] = m.single; record!.embedded[m.name] = streams[i] }
        }
        for d in 0..<config.numLayers {
            for (i, m) in modes.enumerated() {
                let block = m.name == "txt" ? txtBlocks[d] : m.name == "image" ? modeBlocks.image[d] : modeBlocks.imageCond[d]
                streams[i] = block(streams[i], cos: tables[i].cos, sin: tables[i].sin, mod: m.early)
            }
            record?.modes.append(Dictionary(uniqueKeysWithValues: modes.enumerated().map { ($1.name, streams[$0]) }))
        }

        let numImage = streams[0].dim(1), b = streams[0].dim(0), dim = config.inner
        var h = concatenated(streams, axis: 1)
        let mod: Flux3Mod = (
            concatenated(modes.enumerated().map { broadcast($1.single.shift, to: [b, streams[$0].dim(1), dim]) }, axis: 1),
            concatenated(modes.enumerated().map { broadcast($1.single.scale, to: [b, streams[$0].dim(1), dim]) }, axis: 1),
            concatenated(modes.enumerated().map { broadcast($1.single.gate, to: [b, streams[$0].dim(1), dim]) }, axis: 1)
        )
        for block in singleBlocks {
            h = block(h, cos: rope.all.cos, sin: rope.all.sin, mod: mod)
            record?.singles.append(h)
        }
        if let record { stages?(record) }
        return projOut(normOut(h[0..., ..<numImage], vec.image))
    }
}
