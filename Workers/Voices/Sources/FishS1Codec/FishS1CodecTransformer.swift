// SPDX-License-Identifier: MIT
// Fish S1 DAC transformers. Derived from mlx-audio-swift FishS1DACTransformer.swift at 01dec7c9 (MIT); numerics and the
// streaming step follow mlx-audio Python fish_s1_dac.py at dad2087f. See lab/drama3-port/codec-notes.md.
import Foundation
import MLX
import MLXFast
import MLXNN

struct FishS1TransformerArgs {
    var blockSize: Int
    var nLayer: Int
    var nHead: Int
    var dim: Int
    var intermediateSize: Int
    var nLocalHeads: Int
    var headDim: Int
    var ropeBase: Int = 10_000
    var normEps: Float = 1e-5
}

/// cos/sin table [seqLen, nElem/2, 2]. The checkpoint stores its own copy (`freqs_cis`), which replaces this at load.
func fishS1PrecomputeFreqsCis(seqLen: Int, nElem: Int, base: Int) -> MLXArray {
    let exponents = MLX.arange(0, nElem, step: 2, dtype: .float32) / Float(nElem)
    let freqs = 1.0 / pow(MLXArray(Float(base)), exponents)
    let t = MLX.arange(seqLen, dtype: .float32)
    let angles = outer(t, freqs)
    return stacked([cos(angles), sin(angles)], axis: -1)
}

/// x [B, T, H, D], freqs [T, D/2, 2]; rotated in float32, cast back.
func fishS1ApplyRotaryEmb(_ x: MLXArray, freqsCis: MLXArray) -> MLXArray {
    let shaped = x.asType(.float32).reshaped(Array(x.shape.dropLast()) + [-1, 2])
    let freqs = freqsCis.reshaped(1, shaped.dim(1), 1, shaped.dim(3), 2)
    let x0 = shaped[.ellipsis, 0], x1 = shaped[.ellipsis, 1]
    let f0 = freqs[.ellipsis, 0], f1 = freqs[.ellipsis, 1]
    let out = stacked([x0 * f0 - x1 * f1, x1 * f0 + x0 * f1], axis: -1)
    return out.reshaped(x.shape).asType(x.dtype)
}

final class FishS1TFRMSNorm: Module, UnaryLayer {
    let eps: Float
    var weight: MLXArray

    init(dim: Int, eps: Float = 1e-5) {
        self.eps = eps
        self.weight = MLXArray.ones([dim])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let out = x.asType(.float32)
        let normed = out * rsqrt(mean(out * out, axis: -1, keepDims: true) + eps)
        return normed.asType(x.dtype) * weight
    }
}

final class FishS1LayerScale: Module, UnaryLayer {
    var gamma: MLXArray

    init(dim: Int, initValues: Float = 1e-2) {
        self.gamma = MLXArray.ones([dim]) * initValues
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { x * gamma }
}

typealias FishS1KV = (keys: MLXArray, values: MLXArray)

final class FishS1Attention: Module {
    @ModuleInfo(key: "wqkv") var wqkv: Linear
    @ModuleInfo(key: "wo") var wo: Linear
    let nHead: Int
    let headDim: Int
    let nLocalHeads: Int

    init(_ args: FishS1TransformerArgs) {
        let total = (args.nHead + 2 * args.nLocalHeads) * args.headDim
        _wqkv.wrappedValue = Linear(args.dim, total, bias: false)
        _wo.wrappedValue = Linear(args.headDim * args.nHead, args.dim, bias: false)
        nHead = args.nHead; headDim = args.headDim; nLocalHeads = args.nLocalHeads
    }

    /// As the reference: every slice is n_local_heads * head_dim wide (n_head == n_local_heads in every Fish S1 codec).
    private func project(_ x: MLXArray) -> (MLXArray, MLXArray, MLXArray) {
        let b = x.dim(0), t = x.dim(1), kv = nLocalHeads * headDim
        let qkv = wqkv(x)
        let q = qkv[.ellipsis, 0 ..< kv].reshaped(b, t, nHead, headDim)
        let k = qkv[.ellipsis, kv ..< (2 * kv)].reshaped(b, t, nLocalHeads, headDim)
        let v = qkv[.ellipsis, (2 * kv) ..< (3 * kv)].reshaped(b, t, nLocalHeads, headDim)
        return (q, k, v)
    }

    func callAsFunction(_ x: MLXArray, freqsCis: MLXArray, mask: MLXArray) -> MLXArray {
        let (q, k, v) = project(x)
        return attend(fishS1ApplyRotaryEmb(q, freqsCis: freqsCis), fishS1ApplyRotaryEmb(k, freqsCis: freqsCis), v, mask: mask)
    }

    /// Attends new positions to cached ones. Keys are cached before RoPE and rotated with `freqsCis`, which covers the cached
    /// and new positions.
    func step(_ x: MLXArray, freqsCis: MLXArray, mask: MLXArray, cache: FishS1KV?) -> (MLXArray, FishS1KV) {
        var (q, k, v) = project(x)
        if let cache {
            k = concatenated([cache.keys, k], axis: 1)
            v = concatenated([cache.values, v], axis: 1)
        }
        let newCache: FishS1KV = (k, v)
        let t = q.dim(1), total = freqsCis.dim(0)
        q = fishS1ApplyRotaryEmb(q, freqsCis: freqsCis[(total - t)...])
        let rotated = fishS1ApplyRotaryEmb(k, freqsCis: freqsCis)
        return (attend(q, rotated, v, mask: mask), newCache)
    }

    private func attend(_ q: MLXArray, _ k: MLXArray, _ v: MLXArray, mask: MLXArray) -> MLXArray {
        let b = q.dim(0), t = q.dim(1)
        var keys = k.transposed(0, 2, 1, 3)
        var values = v.transposed(0, 2, 1, 3)
        if nLocalHeads != nHead {
            keys = repeated(keys, count: nHead / nLocalHeads, axis: 1)
            values = repeated(values, count: nHead / nLocalHeads, axis: 1)
        }
        let y = MLXFast.scaledDotProductAttention(
            queries: q.transposed(0, 2, 1, 3), keys: keys, values: values, scale: 1.0 / Float(headDim).squareRoot(), mask: mask)
        return wo(y.transposed(0, 2, 1, 3).reshaped(b, t, headDim * nHead))
    }
}

final class FishS1FeedForward: Module, UnaryLayer {
    @ModuleInfo(key: "w1") var w1: Linear
    @ModuleInfo(key: "w3") var w3: Linear
    @ModuleInfo(key: "w2") var w2: Linear

    init(_ args: FishS1TransformerArgs) {
        _w1.wrappedValue = Linear(args.dim, args.intermediateSize, bias: false)
        _w3.wrappedValue = Linear(args.dim, args.intermediateSize, bias: false)
        _w2.wrappedValue = Linear(args.intermediateSize, args.dim, bias: false)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { w2(silu(w1(x)) * w3(x)) }
}

final class FishS1TransformerBlock: Module {
    @ModuleInfo(key: "attention") var attention: FishS1Attention
    @ModuleInfo(key: "feed_forward") var feedForward: FishS1FeedForward
    @ModuleInfo(key: "ffn_norm") var ffnNorm: FishS1TFRMSNorm
    @ModuleInfo(key: "attention_norm") var attentionNorm: FishS1TFRMSNorm
    @ModuleInfo(key: "attention_layer_scale") var attentionLayerScale: FishS1LayerScale
    @ModuleInfo(key: "ffn_layer_scale") var ffnLayerScale: FishS1LayerScale

    init(_ args: FishS1TransformerArgs) {
        _attention.wrappedValue = FishS1Attention(args)
        _feedForward.wrappedValue = FishS1FeedForward(args)
        _ffnNorm.wrappedValue = FishS1TFRMSNorm(dim: args.dim, eps: args.normEps)
        _attentionNorm.wrappedValue = FishS1TFRMSNorm(dim: args.dim, eps: args.normEps)
        _attentionLayerScale.wrappedValue = FishS1LayerScale(dim: args.dim)
        _ffnLayerScale.wrappedValue = FishS1LayerScale(dim: args.dim)
    }

    func callAsFunction(_ x: MLXArray, freqsCis: MLXArray, mask: MLXArray) -> MLXArray {
        let h = x + attentionLayerScale(attention(attentionNorm(x), freqsCis: freqsCis, mask: mask))
        return h + ffnLayerScale(feedForward(ffnNorm(h)))
    }

    func step(_ x: MLXArray, freqsCis: MLXArray, mask: MLXArray, cache: FishS1KV?) -> (MLXArray, FishS1KV) {
        let (attended, newCache) = attention.step(attentionNorm(x), freqsCis: freqsCis, mask: mask, cache: cache)
        let h = x + attentionLayerScale(attended)
        return (h + ffnLayerScale(feedForward(ffnNorm(h))), newCache)
    }
}

/// Streaming state of a windowed transformer: per-layer pre-RoPE KV of the last window - 1 positions.
final class FishS1TransformerState {
    var cache: [FishS1KV]?
    var position = 0
}

/// Python WindowLimitedTransformer (channels-first, causal, RoPE). Input and output projections and the look-ahead conv are
/// identities in every Fish S1 codec (input_dim == dim), so they are not modelled.
final class FishS1WindowLimitedTransformer: Module, UnaryLayer, FishS1Steppable, FishS1Resettable {
    @ModuleInfo(key: "layers") var layers: [FishS1TransformerBlock]
    @ModuleInfo(key: "norm") var norm: FishS1TFRMSNorm
    /// A checkpoint buffer in the reference, loaded like a parameter (kept float32: it is a table, not a learned weight).
    @ParameterInfo(key: "freqs_cis") var freqsCis: MLXArray
    let windowSize: Int?
    let blockSize: Int
    let state = FishS1TransformerState()

    init(_ args: FishS1TransformerArgs, windowSize: Int?) {
        _layers.wrappedValue = (0 ..< args.nLayer).map { _ in FishS1TransformerBlock(args) }
        _norm.wrappedValue = FishS1TFRMSNorm(dim: args.dim, eps: args.normEps)
        _freqsCis.wrappedValue = fishS1PrecomputeFreqsCis(seqLen: args.blockSize, nElem: args.headDim, base: args.ropeBase)
        self.windowSize = windowSize
        self.blockSize = args.blockSize
    }

    /// Bool mask [1, 1, T, T]: causal, limited to the last `windowSize` positions.
    func windowMask(_ length: Int) -> MLXArray {
        let rows = MLX.arange(length)[0..., .newAxis]
        let cols = MLX.arange(length)[.newAxis, 0...]
        let window = windowSize ?? length
        let start = maximum(rows - window + 1, 0)
        return ((cols .>= start) .&& (cols .<= rows))[.newAxis, .newAxis]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x.swappedAxes(1, 2)
        let t = h.dim(1)
        let freqs = freqsCis[0 ..< t]
        let mask = windowMask(t)
        for layer in layers { h = layer(h, freqsCis: freqs, mask: mask) }
        return norm(h).swappedAxes(1, 2)
    }

    func step(_ x: MLXArray) -> MLXArray {
        var h = x.swappedAxes(1, 2)
        let n = h.dim(1)
        let cached = state.cache?.first?.keys.dim(1) ?? 0
        let total = cached + n
        // Same absolute RoPE rows as a full decode while the stream fits the table; past that, rebase onto the cached window.
        let start = min(state.position - cached, blockSize - total)
        precondition(start >= 0, "Streaming chunk of \(n) frames exceeds the transformer block size \(blockSize).")
        state.position += n
        let rows = MLX.arange(cached, total)[0..., .newAxis]
        let cols = MLX.arange(total)[.newAxis, 0...]
        var mask = cols .<= rows
        if let windowSize { mask = mask .&& (cols .> (rows - windowSize)) }
        mask = mask[.newAxis, .newAxis]
        let freqs = freqsCis[start ..< (start + total)]

        var newCaches: [FishS1KV] = []
        for (index, layer) in layers.enumerated() {
            var cache: FishS1KV
            (h, cache) = layer.step(h, freqsCis: freqs, mask: mask, cache: state.cache?[index])
            if let windowSize {
                let keep = windowSize - 1, length = cache.keys.dim(1)
                let from = max(0, length - keep)
                cache = (cache.keys[0..., from...], cache.values[0..., from...])
            }
            newCaches.append(cache)
        }
        state.cache = newCaches
        return norm(h).swappedAxes(1, 2)
    }

    func resetState() {
        state.cache = nil
        state.position = 0
    }
}
