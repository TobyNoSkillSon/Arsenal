// SPDX-License-Identifier: MIT
// TIGER (Time-frequency Interleaved Gain Extraction and Reconstruction), ported from the maker's
// look2hear/models/tiger.py and tiger_dnr.py (github.com/JusperLee/TIGER @ 9f18d4a). Plain MLX, FP16 weights.
// Layout: PyTorch keeps [B, C, L]; here every sequence is channels-last [B, L, C]. Norm statistics, the STFT,
// attention scores and the mask arithmetic run in float32; the convolutions run in the weights' float16.
import Foundation
import MLX
import MLXNN

struct TigerConfig {
    let outChannels: Int, inChannels: Int, numBlocks: Int, depth: Int, nHead: Int, hidChan: Int
    let win: Int, hop: Int, numSources: Int, sampleRate: Int
    /// "speech" uses the 16 kHz band plan (tiger.py); "dnr" the 44.1 kHz one (tiger_dnr.py).
    let bandPlan: String

    init(_ json: [String: Any], bandPlan: String) throws {
        func int(_ key: String) throws -> Int {
            guard let value = (json[key] as? NSNumber)?.intValue else { throw RunnerError.failed("config.json has no \(key).") }
            return value
        }
        outChannels = try int("out_channels"); inChannels = try int("in_channels"); numBlocks = try int("num_blocks")
        depth = try int("upsampling_depth"); nHead = try int("att_n_head"); hidChan = try int("att_hid_chan")
        win = try int("win"); hop = try int("stride"); numSources = try int("num_sources"); sampleRate = try int("sample_rate")
        self.bandPlan = bandPlan
        guard win % hop == 0, win % 2 == 0 else { throw RunnerError.failed("win must be an even multiple of stride.") }
    }

    /// Sub-band widths in STFT bins, exactly as TIGER.__init__ computes them (np.floor of Hz / Nyquist * bins).
    var bandWidths: [Int] {
        let enc = win / 2 + 1
        func bw(_ hz: Double) -> Int { Int((hz / (Double(sampleRate) / 2.0) * Double(enc)).rounded(.down)) }
        var widths: [Int]
        if bandPlan == "dnr" {
            widths = Array(repeating: bw(50), count: 20) + Array(repeating: bw(100), count: 10) + Array(repeating: bw(250), count: 8)
                + Array(repeating: bw(500), count: 8) + Array(repeating: bw(1000), count: 8) + Array(repeating: bw(2000), count: 2)
        } else {
            widths = Array(repeating: bw(25), count: 40) + Array(repeating: bw(100), count: 10) + Array(repeating: bw(250), count: 8)
                + Array(repeating: bw(500), count: 8)
        }
        widths.append(enc - widths.reduce(0, +))
        return widths
    }
}

// MARK: - Layers

/// nn.PReLU() with one shared slope.
final class PReLU1: Module, UnaryLayer {
    var weight = MLXArray.zeros([1])
    func callAsFunction(_ x: MLXArray) -> MLXArray { maximum(x, 0) + weight.asType(x.dtype) * minimum(x, 0) }
}

/// nn.GroupNorm(1, C): normalise each item over all of [L, C], then a per-channel affine. Statistics in float32.
final class GroupNorm1: Module, UnaryLayer {
    var weight: MLXArray, bias: MLXArray
    let eps: Float
    init(_ channels: Int, eps: Float) { weight = MLXArray.ones([channels]); bias = MLXArray.zeros([channels]); self.eps = eps }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let mean = xf.mean(axes: [1, 2], keepDims: true)
        let variance = ((xf - mean) * (xf - mean)).mean(axes: [1, 2], keepDims: true)
        let y = (xf - mean) * rsqrt(variance + eps) * weight.asType(.float32) + bias.asType(.float32)
        return y.asType(x.dtype)
    }
}

/// LayerNormalization4D with n_freqs = 1: normalise over channels only (the last axis here), eps 1e-5.
final class ChannelNorm: Module, UnaryLayer {
    var gamma: MLXArray, beta: MLXArray
    init(_ channels: Int) { gamma = MLXArray.ones([channels]); beta = MLXArray.zeros([channels]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let xf = x.asType(.float32)
        let mean = xf.mean(axis: -1, keepDims: true)
        let variance = ((xf - mean) * (xf - mean)).mean(axis: -1, keepDims: true)
        return ((xf - mean) * rsqrt(variance + 1e-5) * gamma.asType(.float32) + beta.asType(.float32)).asType(x.dtype)
    }
}

/// nn.Conv1d (and 1x1 nn.Conv2d) on channels-last input; weight [out, k, in/groups].
final class Conv: Module, UnaryLayer {
    var weight: MLXArray
    var bias: MLXArray?
    let stride: Int, padding: Int, groups: Int
    init(_ inC: Int, _ outC: Int, kernel: Int = 1, stride: Int = 1, padding: Int = 0, groups: Int = 1, bias: Bool = true) {
        weight = MLXArray.zeros([outC, kernel, inC / groups])
        self.bias = bias ? MLXArray.zeros([outC]) : nil
        self.stride = stride; self.padding = padding; self.groups = groups
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv1d(x, weight, stride: stride, padding: padding, groups: groups)
        if let bias { return y + bias }
        return y
    }
}

/// ConvNormAct / ConvNorm / DilatedConvNorm: conv, GroupNorm(1, eps 1e-8), optional PReLU.
final class ConvNorm: Module, UnaryLayer {
    var conv: Conv
    var norm: GroupNorm1
    @ModuleInfo var act: PReLU1?
    init(_ inC: Int, _ outC: Int, kernel: Int, stride: Int = 1, groups: Int = 1, bias: Bool = true, act: Bool = false) {
        conv = Conv(inC, outC, kernel: kernel, stride: stride, padding: (kernel - 1) / 2, groups: groups, bias: bias)
        norm = GroupNorm1(outC, eps: 1e-8)
        _act.wrappedValue = act ? PReLU1() : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = norm(conv(x))
        if let act { return act(y) }
        return y
    }
}

final class Mlp: Module, UnaryLayer {
    var fc1: ConvNorm, dwconv: Conv, fc2: ConvNorm
    init(_ c: Int) {
        fc1 = ConvNorm(c, c, kernel: 1, bias: false)
        dwconv = Conv(c, c, kernel: 5, padding: 2, groups: c)
        fc2 = ConvNorm(c, c, kernel: 1, bias: false)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { fc2(relu(dwconv(fc1(x)))) }
}

/// F.interpolate(x, size: n, mode: "nearest") along axis 1 of [B, L, C].
func nearest(_ x: MLXArray, to n: Int) -> MLXArray {
    let length = x.dim(1)
    if length == n { return x }
    let scale = Float(length) / Float(n)
    let index = (0..<n).map { Int32(min(Int((Float($0) * scale).rounded(.down)), length - 1)) }
    return x.take(MLXArray(index), axis: 1)
}

/// F.adaptive_avg_pool1d along axis 1 of [B, L, C] to n positions, as a pooling matrix.
func adaptivePool(_ x: MLXArray, to n: Int) -> MLXArray {
    let length = x.dim(1)
    if length == n { return x }
    var matrix = [Float](repeating: 0, count: n * length)
    for i in 0..<n {
        let start = (i * length) / n, end = ((i + 1) * length + n - 1) / n
        for j in start..<end { matrix[i * length + j] = 1 / Float(end - start) }
    }
    return matmul(MLXArray(matrix, [n, length]).asType(x.dtype), x)
}

final class InjectionMultiSum: Module {
    var local_embedding: ConvNorm, global_embedding: ConvNorm, global_act: ConvNorm
    init(_ c: Int, kernel: Int) {
        local_embedding = ConvNorm(c, c, kernel: kernel, groups: c, bias: false)
        global_embedding = ConvNorm(c, c, kernel: kernel, groups: c, bias: false)
        global_act = ConvNorm(c, c, kernel: kernel, groups: c, bias: false)
    }
    func callAsFunction(_ local: MLXArray, _ global: MLXArray) -> MLXArray {
        let n = local.dim(1)
        let gate = nearest(sigmoid(global_act(global)), to: n)
        let feat = nearest(global_embedding(global), to: n)
        return local_embedding(local) * gate + feat
    }
}

final class UConvBlock: Module, UnaryLayer {
    var proj_1x1: ConvNorm
    var spp_dw: [ConvNorm]
    var loc_glo_fus: [InjectionMultiSum]
    var res_conv: Conv
    var globalatt: Mlp
    var last_layer: [InjectionMultiSum]
    let depth: Int

    init(out: Int, inner: Int, depth: Int) {
        self.depth = depth
        proj_1x1 = ConvNorm(out, inner, kernel: 1, act: true)
        spp_dw = (0..<depth).map { ConvNorm(inner, inner, kernel: 5, stride: $0 == 0 ? 1 : 2, groups: inner) }
        loc_glo_fus = (0..<depth).map { _ in InjectionMultiSum(inner, kernel: 1) }
        res_conv = Conv(inner, out, kernel: 1)
        globalatt = Mlp(inner)
        last_layer = (0..<(depth - 1)).map { _ in InjectionMultiSum(inner, kernel: 5) }
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var outputs = [spp_dw[0](proj_1x1(x))]
        for k in 1..<depth { outputs.append(spp_dw[k](outputs[k - 1])) }
        let n = outputs[depth - 1].dim(1)
        var global = adaptivePool(outputs[0], to: n)
        for fea in outputs.dropFirst() { global = global + adaptivePool(fea, to: n) }
        global = globalatt(global)
        let fused = (0..<depth).map { loc_glo_fus[$0](outputs[$0], global) }
        // The maker's loop: the deepest step takes x_fused[depth-2] with x_fused[depth-3], then climbs to level 0.
        var expanded = last_layer[depth - 2](fused[depth - 2], fused[depth - 3])
        for i in stride(from: depth - 3, through: 0, by: -1) { expanded = last_layer[i](fused[i], expanded) }
        return res_conv(expanded) + x
    }
}

/// ATTConvActNorm with a 1x1 Conv2d: conv, PReLU, channel norm.
final class AttConv: Module, UnaryLayer {
    var conv: Conv, act: PReLU1, norm: ChannelNorm
    init(_ inC: Int, _ outC: Int) { conv = Conv(inC, outC); act = PReLU1(); norm = ChannelNorm(outC) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { norm(act(conv(x))) }
}

/// MultiHeadSelfAttention2D. Input [S, F, C]: attention runs along S; F is folded into each head's embedding.
final class FrameAttention: Module, UnaryLayer {
    var Queries: [AttConv], Keys: [AttConv], Values: [AttConv]
    var attn_concat_proj: AttConv
    init(channels: Int, heads: Int, hid: Int) {
        Queries = (0..<heads).map { _ in AttConv(channels, hid) }
        Keys = (0..<heads).map { _ in AttConv(channels, hid) }
        Values = (0..<heads).map { _ in AttConv(channels, channels / heads) }
        attn_concat_proj = AttConv(channels, channels)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let s = x.dim(0), f = x.dim(1)
        var heads: [MLXArray] = []
        for h in 0..<Queries.count {
            let q = Queries[h](x).reshaped([s, -1]).asType(.float32)
            let k = Keys[h](x).reshaped([s, -1]).asType(.float32)
            let v = Values[h](x)
            let cv = v.dim(2)
            let scores = matmul(q, k.transposed()) / Float(q.dim(1)).squareRoot()
            let attention = softmax(scores, axis: -1).asType(x.dtype)
            let head = matmul(attention, v.reshaped([s, -1])).reshaped([s, f, cv])
            eval(head)  // one head's [S, S] scores alive at a time
            heads.append(head)
        }
        return attn_concat_proj(concatenated(heads, axis: -1)) + x
    }
}

/// One of freq_path / frame_path: ModuleList [UConvBlock, MultiHeadSelfAttention2D, LayerNormalization4D]
/// (checkpoint children 0, 1, 2; Model.swift renames numbered Sequential children to these names).
final class Path: Module {
    var uconv: UConvBlock
    var attention: FrameAttention
    var norm: ChannelNorm
    init(_ c: TigerConfig) {
        uconv = UConvBlock(out: c.outChannels, inner: c.inChannels, depth: c.depth)
        attention = FrameAttention(channels: c.outChannels, heads: c.nHead, hid: c.hidChan)
        norm = ChannelNorm(c.outChannels)
    }
}

final class ConcatBlock: Module, UnaryLayer {
    var conv: Conv
    var act: PReLU1
    init(_ c: Int) { conv = Conv(c, c, groups: c); act = PReLU1() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { act(conv(x)) }
}

/// Recurrent: the same freq/frame block applied num_blocks times. Layout [nband, T, N].
final class Recurrent: Module {
    var freq_path: Path, frame_path: Path
    var concat_block: ConcatBlock
    let iterations: Int
    init(_ c: TigerConfig) {
        freq_path = Path(c); frame_path = Path(c); concat_block = ConcatBlock(c.outChannels); iterations = c.numBlocks
    }
    func callAsFunction(_ mixture: MLXArray, cancelled: () throws -> Void) throws -> MLXArray {
        var x = mixture
        for i in 0..<iterations {
            try cancelled()
            x = step(i == 0 ? mixture : concat_block(mixture + x))
            eval(x)
        }
        return x
    }
    private func step(_ x: MLXArray) -> MLXArray {
        // Frequency path: sequences over bands (batch T), then attention over bands with T folded in.
        var f = batched(x.transposed(1, 0, 2)) { self.freq_path.uconv($0) }.transposed(1, 0, 2)  // [nband, T, N]
        f = freq_path.norm(freq_path.attention(f))
        let y = f + x
        eval(y)
        // Frame path: sequences over time (batch nband), then attention over time with bands folded in.
        var g = batched(y) { self.frame_path.uconv($0) }.transposed(1, 0, 2)  // [T, nband, N]
        g = frame_path.norm(frame_path.attention(g)).transposed(1, 0, 2)
        return g + y
    }
}

/// Runs a per-sequence block over the batch in slices (UConvBlock's norms are per sequence, so this is exact) to bound memory.
func batched(_ x: MLXArray, _ f: (MLXArray) -> MLXArray) -> MLXArray {
    let b = x.dim(0), per = max(1, 65_536 / max(x.dim(1), 1))
    if b <= per { return f(x) }
    var parts: [MLXArray] = []
    for start in stride(from: 0, to: b, by: per) {
        let y = f(x[start..<min(start + per, b)])
        eval(y)
        parts.append(y)
    }
    return concatenated(parts, axis: 0)
}

final class BandIn: Module {
    var norm: GroupNorm1
    var conv: Conv
    init(width: Int, features: Int) {
        norm = GroupNorm1(width * 2, eps: Float.ulpOfOne)  // torch.finfo(torch.float32).eps
        conv = Conv(width * 2, features)
    }
}

final class BandOut: Module {
    var act: PReLU1
    var conv: Conv
    init(width: Int, features: Int, sources: Int) {
        act = PReLU1()
        conv = Conv(features, width * 4 * sources, groups: sources)
    }
}

/// The TIGER network: STFT, band-split bottleneck, recurrent separator, complex masks, iSTFT.
final class Tiger: Module {
    var BN: [BandIn]
    var separator: Recurrent
    var mask: [BandOut]
    let config: TigerConfig
    let widths: [Int]

    init(_ c: TigerConfig) {
        config = c
        widths = c.bandWidths
        BN = widths.map { BandIn(width: $0, features: c.outChannels) }
        separator = Recurrent(c)
        mask = widths.map { BandOut(width: $0, features: c.outChannels, sources: c.numSources) }
    }

    /// Mono float32 samples [n] to [numSources, n] float32.
    func separate(_ samples: MLXArray, cancelled: () throws -> Void) throws -> MLXArray {
        let n = samples.dim(0)
        let (re, im) = stft(samples, nfft: config.win, hop: config.hop)  // [T, F] each
        var features: [MLXArray] = []
        var start = 0
        for (i, w) in widths.enumerated() {
            let band = concatenated([re[0..., start..<(start + w)], im[0..., start..<(start + w)]], axis: -1)  // [T, 2w]
            let normed = BN[i].norm(band.expandedDimensions(axis: 0)).asType(.float16)
            features.append(BN[i].conv(normed))  // [1, T, N]
            start += w
        }
        let sep = try separator(concatenated(features, axis: 0), cancelled: cancelled)  // [nband, T, N]

        let k = config.numSources, frames = re.dim(0)
        var realParts: [MLXArray] = [], imagParts: [MLXArray] = []
        start = 0
        for (i, w) in widths.enumerated() {
            let out = mask[i].conv(mask[i].act(sep[i..<(i + 1)]))[0].asType(.float32).reshaped([frames, 2, 2, k, w])
            let m = out[0..., 0] * sigmoid(out[0..., 1])  // [T, 2, K, w]
            var mr = m[0..., 0], mi = m[0..., 1]  // [T, K, w]
            mr = mr - (mr.sum(axis: 1, keepDims: true) - 1) / Float(k)
            mi = mi - mi.sum(axis: 1, keepDims: true) / Float(k)
            let sr = re[0..., start..<(start + w)].expandedDimensions(axis: 1), si = im[0..., start..<(start + w)].expandedDimensions(axis: 1)
            realParts.append(sr * mr - si * mi)
            imagParts.append(sr * mi + si * mr)
            start += w
        }
        let estRe = concatenated(realParts, axis: -1).transposed(1, 0, 2)  // [K, T, F]
        let estIm = concatenated(imagParts, axis: -1).transposed(1, 0, 2)
        return istft(estRe, estIm, nfft: config.win, hop: config.hop, length: n)
    }
}

/// TIGERDNR: three full TIGER networks; dialog keeps its track 2, effect track 1, music track 0.
final class TigerDnR: Module {
    var dialog: Tiger, effect: Tiger, music: Tiger
    init(_ c: TigerConfig) { dialog = Tiger(c); effect = Tiger(c); music = Tiger(c) }
}

// MARK: - STFT (torch.stft / torch.istft with center=True, reflect padding, periodic Hann window)

func hann(_ n: Int) -> MLXArray {
    MLXArray((0..<n).map { Float(0.5 - 0.5 * cos(2 * Double.pi * Double($0) / Double(n))) })
}

func stft(_ x: MLXArray, nfft: Int, hop: Int) -> (MLXArray, MLXArray) {
    let n = x.dim(0), pad = nfft / 2
    let frames = n / hop + 1
    var index = [Int32](repeating: 0, count: frames * nfft)
    for t in 0..<frames {
        for m in 0..<nfft {
            var p = t * hop + m - pad
            if p < 0 { p = -p }
            if p >= n { p = 2 * (n - 1) - p }
            index[t * nfft + m] = Int32(p)
        }
    }
    let framed = x.asType(.float32).take(MLXArray(index, [frames, nfft])) * hann(nfft)
    let spectrum = rfft(framed, axis: 1)
    return (spectrum.realPart(), spectrum.imaginaryPart())
}

/// [B, T, F] real and imaginary parts to [B, length] samples.
func istft(_ re: MLXArray, _ im: MLXArray, nfft: Int, hop: Int, length: Int) -> MLXArray {
    let b = re.dim(0), frames = re.dim(1), ratio = nfft / hop
    let complexSpec = re.asType(.complex64) + im.asType(.complex64) * MLXArray(real: 0, imaginary: 1)
    let window = hann(nfft)
    let chunks = (irfft(complexSpec, n: nfft, axis: -1) * window).reshaped([b, frames, ratio, hop])
    let envelopeChunks = (window * window).reshaped([1, ratio, hop])
    var signal = MLXArray.zeros([b, frames + ratio - 1, hop])
    var envelope = MLXArray.zeros([frames + ratio - 1, hop])
    for j in 0..<ratio {
        let widths: [IntOrPair] = [.init((0, 0)), .init((j, ratio - 1 - j)), .init((0, 0))]
        signal = signal + padded(chunks[0..., 0..., j], widths: widths)
        envelope = envelope + padded(broadcast(envelopeChunks[0..., j], to: [frames, hop]), widths: Array(widths.dropFirst()))
    }
    let total = (frames + ratio - 1) * hop
    let flat = signal.reshaped([b, total]), env = envelope.reshaped([total])
    let normalised = which(env .> 1e-11, flat / env, flat)
    let start = nfft / 2, end = min(start + length, total)
    var out = normalised[0..., start..<end]
    if end - start < length { out = padded(out, widths: [.init((0, 0)), .init((0, length - (end - start)))]) }
    return out
}
