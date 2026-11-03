// SPDX-License-Identifier: MIT
// UniPASE (Rong et al., IEEE TASLP 2026; Xiaobin-Rong/unipase 857b60ad, MIT code; Apache-2.0 weights) on MLX, written for Arsenal
// after the maker's PyTorch modules: DeWavLM-Omni (a fine-tuned WavLM Large, models/wavlm/WavLM.py and modules.py)
// gives the noisy acoustic layer 1 and the enhanced phonetic layer 24; the Vocos adapter (models/adapter/vocos)
// turns them into an enhanced layer 1; the Vocos vocoder (models/vocoder/vocos) makes 16 kHz speech; the TF-GridNet
// PostNet (models/postnet/tfgrid_cws_res.py) extends it to 48 kHz. Weights and matrix products run in the weights'
// half dtype (FP16); norm statistics, GELU, softmax, the LSTM cell state, FFTs and resampling are float32
// intermediates, rounded once.
import Foundation
import MLX
import MLXFast
import MLXNN

public final class UniPASENetwork: @unchecked Sendable {
    public let weights: [String: MLXArray]
    public let dtype: DType
    /// Optional observer of named intermediate tensors (parity checks).
    public typealias Trace = (String, MLXArray) -> Void

    public init(weights: [String: MLXArray], dtype: DType) { self.weights = weights; self.dtype = dtype }

    func w(_ name: String) -> MLXArray {
        guard let a = weights[name] else { preconditionFailure("UniPASE weight \(name) missing") }
        return a
    }

    // MARK: Shared layers (NLC: [batch, time, channels])

    func linear(_ x: MLXArray, _ p: String, bias: Bool = true) -> MLXArray {
        let y = matmul(x, w(p + ".weight").T)
        return bias ? y + w(p + ".bias") : y
    }
    func layerNorm(_ x: MLXArray, _ p: String, eps: Float) -> MLXArray {
        MLXFast.layerNorm(x.asType(.float32), weight: w(p + ".weight").asType(.float32), bias: w(p + ".bias").asType(.float32), eps: eps).asType(x.dtype)
    }
    /// nn.GroupNorm over [B, T, C] channels-last: statistics per group over time and the group's channels.
    func groupNorm(_ x: MLXArray, groups: Int, weight: MLXArray, bias: MLXArray, eps: Float) -> MLXArray {
        let (b, t, c) = (x.dim(0), x.dim(1), x.dim(2))
        let f = x.asType(.float32).reshaped(b, t, groups, c / groups)
        let mean = f.mean(axes: [1, 3], keepDims: true), variance = (f - mean).square().mean(axes: [1, 3], keepDims: true)
        let n = ((f - mean) * rsqrt(variance + eps)).reshaped(b, t, c)
        return (n * weight.asType(.float32) + bias.asType(.float32)).asType(x.dtype)
    }
    func gelu(_ x: MLXArray) -> MLXArray { MLXNN.gelu(x.asType(.float32)).asType(x.dtype) }
    func swish(_ x: MLXArray) -> MLXArray { let f = x.asType(.float32); return (f * sigmoid(f)).asType(x.dtype) }
    func conv(_ x: MLXArray, _ p: String, padding: Int = 0, stride: Int = 1, groups: Int = 1, bias: Bool = true) -> MLXArray {
        let y = conv1d(x, w(p + ".weight"), stride: stride, padding: padding, groups: groups)
        return bias ? y + w(p + ".bias") : y
    }
    /// Row-wise softmax in float32, rounded once.
    func softmax32(_ x: MLXArray) -> MLXArray { softmax(x.asType(.float32), axis: -1).asType(dtype) }

    // MARK: DeWavLM-Omni (WavLM Large)

    static let cnnStrides = [5, 2, 2, 2, 2, 2, 2]

    /// WavLM relative position bucket (bidirectional, 320 buckets, max distance 800), in the maker's float32 arithmetic.
    public static func relativeBucket(_ relative: Int) -> Int {
        var bucket = relative > 0 ? 160 : 0
        let r = abs(relative)
        if r < 80 { return bucket + r }
        let large = 80 + Int(Float(log(Float(r) / 80)) / Float(log(10.0)) * 80)
        bucket += min(large, 159)
        return bucket
    }

    /// The position bias of layer 0, shared by every layer: [1, 16, T, T].
    func positionBias(_ t: Int) -> MLXArray {
        let table = w("wavlm.encoder.layers.0.self_attn.relative_attention_bias.weight") // [320, 16]
        let buckets = MLXArray((-(t - 1)...(t - 1)).map { Int32(Self.relativeBucket($0)) })
        let perOffset = table[buckets].T // [16, 2T - 1]
        let positions = (0..<t).map { Int32($0) }
        let i = MLXArray(positions).reshaped(t, 1), j = MLXArray(positions).reshaped(1, t)
        let index = j - i + Int32(t - 1)
        return take(perOffset, index, axis: 1).expandedDimensions(axis: 0) // [1, 16, T, T]
    }

    func wavlmLayer(_ x: MLXArray, _ i: Int, bias: MLXArray) -> MLXArray {
        let p = "wavlm.encoder.layers.\(i).", t = x.dim(1)
        let h = layerNorm(x, p + "self_attn_layer_norm", eps: 1e-5)
        func heads(_ a: MLXArray) -> MLXArray { a.reshaped(1, t, 16, 64).transposed(0, 2, 1, 3) }
        let q = heads(linear(h, p + "self_attn.q_proj")), k = heads(linear(h, p + "self_attn.k_proj")), v = heads(linear(h, p + "self_attn.v_proj"))
        // Gated relative position bias: the gate reads the normalised input itself, split into heads.
        let g = sigmoid(linear(heads(h), p + "self_attn.grep_linear").reshaped(1, 16, t, 2, 4).sum(axis: -1))
        let gateA = g[0..., 0..., 0..., 0..<1], gateB = g[0..., 0..., 0..., 1..<2]
        let grepA = w(p + "self_attn.grep_a").reshaped(1, 16, 1, 1)
        let mask = (gateA * (gateB * grepA - 1) + 2) * bias
        let attn = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / 8, mask: mask.asType(q.dtype))
        let y = x + linear(attn.transposed(0, 2, 1, 3).reshaped(1, t, 1024), p + "self_attn.out_proj")
        let f = layerNorm(y, p + "final_layer_norm", eps: 1e-5)
        return y + linear(gelu(linear(f, p + "fc1")), p + "fc2")
    }

    /// The encoder: 16 kHz padded audio and the packet-loss mask (nil: no masking) → layers 1 and 24, each normalised
    /// over the whole utterance ([1, T, 1024]).
    public func encode(_ wav16: [Float], mask: [Bool]?, trace: Trace? = nil) -> (l1: MLXArray, l24: MLXArray) {
        var x = MLXArray(wav16).reshaped(1, -1, 1).asType(dtype)
        for (i, stride) in Self.cnnStrides.enumerated() {
            let p = "wavlm.feature_extractor.conv_layers.\(i)."
            x = conv1d(x, w(p + "0.weight"), stride: stride)
            x = gelu(layerNorm(x, p + "2.1", eps: 1e-5))
        }
        trace?("cnn", x[0])
        x = linear(layerNorm(x, "wavlm.layer_norm", eps: 1e-5), "wavlm.post_extract_proj")
        trace?("proj", x[0])
        let t = x.dim(1)
        if let mask {
            let m = MLXArray((0..<t).map { $0 < mask.count && mask[$0] }).reshaped(1, t, 1)
            x = which(m, w("wavlm.mask_emb").reshaped(1, 1, 1024), x)
        }
        trace?("masked", x[0])
        return encodeLayers(x, trace: trace)
    }

    /// From the masked projection to the two normalised layers (separate for parity on the maker's own input).
    public func encodeLayers(_ input: MLXArray, trace: Trace? = nil) -> (l1: MLXArray, l24: MLXArray) {
        var x = input
        let t = x.dim(1)
        let pos = conv(x, "wavlm.encoder.pos_conv.0", padding: 64, groups: 16)[0..., 0..<t, 0...]
        x += gelu(pos)
        trace?("pos", x[0])
        let bias = positionBias(t)
        var l1 = x
        for i in 0..<24 {
            x = wavlmLayer(x, i, bias: bias)
            if i == 0 { l1 = x }
            if [0, 1, 5, 11, 17, 22, 23].contains(i) { trace?("layer\(i + 1)", x[0]) }
            if i % 6 == 5 { eval(x, l1) }
        }
        return (UniPASEDSP.utteranceNorm(l1), UniPASEDSP.utteranceNorm(x))
    }

    /// One WavLM layer on its own (parity of an isolated layer): `index` 0 computes the bias as in the maker.
    public func layer(_ x: MLXArray, index: Int) -> MLXArray { wavlmLayer(x, index, bias: positionBias(x.dim(1))) }

    // MARK: Vocos backbone, adapter and vocoder

    func resnetBlock(_ x: MLXArray, _ p: String) -> MLXArray {
        var h = swish(groupNorm(x, groups: 32, weight: w(p + "norm1.weight"), bias: w(p + "norm1.bias"), eps: 1e-6))
        h = conv(h, p + "conv1", padding: 1)
        h = swish(groupNorm(h, groups: 32, weight: w(p + "norm2.weight"), bias: w(p + "norm2.bias"), eps: 1e-6))
        return x + conv(h, p + "conv2", padding: 1)
    }

    func attnBlock(_ x: MLXArray, _ p: String) -> MLXArray {
        let h = groupNorm(x, groups: 32, weight: w(p + "norm.weight"), bias: w(p + "norm.bias"), eps: 1e-6)
        let c = x.dim(2)
        let q = linear(h, p + "q") * (1 / Float(c).squareRoot()), k = linear(h, p + "k"), v = linear(h, p + "v")
        let a = softmax32(matmul(q, k.transposed(0, 2, 1)))
        return x + linear(matmul(a, v), p + "proj_out")
    }

    /// VocosBackbone: [1, T, 1024] → [1, T, 1024].
    func backbone(_ input: MLXArray, _ net: String, trace: Trace? = nil) -> MLXArray {
        let d = net + ".decoder."
        var x = conv(input, d + "embed", padding: 3)
        trace?(net + "_embed", x[0])
        x = resnetBlock(x, d + "pos_net.0.")
        x = resnetBlock(x, d + "pos_net.1.")
        x = attnBlock(x, d + "pos_net.2.")
        x = resnetBlock(x, d + "pos_net.3.")
        x = resnetBlock(x, d + "pos_net.4.")
        x = groupNorm(x, groups: 32, weight: w(d + "pos_net.5.weight"), bias: w(d + "pos_net.5.bias"), eps: 1e-6)
        trace?(net + "_posnet", x[0])
        x = layerNorm(x, d + "norm", eps: 1e-6)
        for i in 0..<12 {
            let p = d + "convnext.\(i)."
            var h = conv(x, p + "dwconv", padding: 3, groups: 1024)
            h = linear(gelu(linear(layerNorm(h, p + "norm", eps: 1e-6), p + "pwconv1")), p + "pwconv2")
            x += w(p + "gamma") * h
            if i == 0 || i == 11 { trace?(net + "_convnext\(i)", x[0]) }
            if i % 4 == 3 { eval(x) }
        }
        return layerNorm(x, d + "final_layer_norm", eps: 1e-6)
    }

    /// VocosAdapter: noisy layer 1 + projected enhanced layer 24 → enhanced layer 1, [1, T, 1024].
    public func adapt(l1: MLXArray, l24: MLXArray, trace: Trace? = nil) -> MLXArray {
        let x = linear(backbone(linear(l24, "adapter.proj") + l1, "adapter", trace: trace), "adapter.head")
        return x
    }

    /// VocosVocoder up to its ISTFT: [1, T, 1024] → hidden [T, 1282] (log-magnitude and phase).
    public func vocoderHidden(_ x: MLXArray, trace: Trace? = nil) -> MLXArray {
        let b = backbone(x, "vocoder", trace: trace)
        trace?("vocoder_backbone", b[0])
        return linear(b, "vocoder.head.out")[0]
    }

    // MARK: TF-GridNet PostNet ([1, T, F, C] channels-last)

    /// nn.PReLU with one slope per channel group: `x` [..., groups, per].
    func prelu(_ x: MLXArray, slopes: MLXArray) -> MLXArray { maximum(x, 0) + slopes * minimum(x, 0) }

    /// Bidirectional single-layer LSTM (PyTorch gate order i, f, g, o): [N, L, In] → [N, L, 200]. Both directions run in
    /// one batched loop; the cell state stays float32.
    func biLSTM(_ x: MLXArray, _ p: String) -> MLXArray {
        let (n, l) = (x.dim(0), x.dim(1)), hidden = 100
        func projected(_ suffix: String) -> MLXArray {
            matmul(x, w(p + ".weight_ih_l0" + suffix).T) + (w(p + ".bias_ih_l0" + suffix) + w(p + ".bias_hh_l0" + suffix))
        }
        let forward = projected(""), backward = projected("_reverse")[0..., .stride(by: -1), 0...]
        let inputs = stacked([forward, backward], axis: 0) // [2, N, L, 400]
        let recurrent = stacked([w(p + ".weight_hh_l0").T, w(p + ".weight_hh_l0_reverse").T], axis: 0) // [2, 100, 400]
        var h = MLXArray.zeros([2, n, hidden], dtype: dtype), c = MLXArray.zeros([2, n, hidden], dtype: .float32)
        var outputs: [MLXArray] = []
        outputs.reserveCapacity(l)
        for s in 0..<l {
            let gates = (inputs[0..., 0..., s, 0...] + matmul(h, recurrent)).asType(.float32)
            let i = sigmoid(gates[.ellipsis, 0..<100]), f = sigmoid(gates[.ellipsis, 100..<200])
            let g = tanh(gates[.ellipsis, 200..<300]), o = sigmoid(gates[.ellipsis, 300..<400])
            c = f * c + i * g
            h = (o * tanh(c)).asType(dtype)
            outputs.append(h)
            if s % 64 == 63 { eval(h, c) }
        }
        let all = stacked(outputs, axis: 2) // [2, N, L, 100]
        return concatenated([all[0], all[1][0..., .stride(by: -1), 0...]], axis: -1)
    }

    /// One GridNetV3 block on [1, T, F, 48].
    func gridBlock(_ input: MLXArray, _ b: Int) -> MLXArray {
        let p = "postnet.blocks.\(b).", (oldT, oldQ) = (input.dim(1), input.dim(2)), c = 48, olp = 3
        // Pad time and frequency by 3 on both sides (unfold 4 at hop 1).
        let x = padded(input, widths: [.init(0), .init((olp, olp)), .init((olp, olp)), .init(0)])
        let (t, q) = (oldT + 2 * olp, oldQ + 2 * olp)
        func sequence(_ seq: MLXArray, _ side: String) -> MLXArray {
            // seq [N, L, C] → normalise, unfold 4 (channel-major features c·4 + k), BiLSTM, transposed conv back to L.
            let (n, len) = (seq.dim(0), seq.dim(1))
            let normed = layerNorm(seq, p + side + "_norm", eps: 1e-5)
            let windows = stacked((0..<4).map { normed[0..., $0..<($0 + len - 3), 0...] }, axis: -1).reshaped(n, len - 3, c * 4)
            let rnn = biLSTM(windows, p + side + "_rnn")
            return conv1d(padded(rnn, widths: [.init(0), .init((3, 3)), .init(0)]), w(p + side + "_linear.weight")) + w(p + side + "_linear.bias")
        }
        let intra = sequence(x.reshaped(t, q, c), "intra").reshaped(1, t, q, c) + x
        let swapped = intra.transposed(0, 2, 1, 3) // [1, Q, T, C]
        let inter = (sequence(swapped.reshaped(q, t, c), "inter").reshaped(1, q, t, c) + swapped).transposed(0, 2, 1, 3)
        let batch = inter[0..., olp..<(olp + oldT), olp..<(olp + oldQ), 0...]
        eval(batch)
        // Full-band self-attention across time, 4 heads.
        func project(_ name: String, per: Int) -> MLXArray {
            let y = linear(batch, p + "attn_conv_\(name)").reshaped(1, oldT, oldQ, 4, per)
            let a = prelu(y, slopes: w(p + "attn_norm_\(name).act.weight").reshaped(4, 1)).asType(.float32)
            let mean = a.mean(axis: -1, keepDims: true), variance = (a - mean).square().mean(axis: -1, keepDims: true)
            let n = (a - mean) * rsqrt(variance + 1e-5)
            let gamma = w(p + "attn_norm_\(name).gamma").asType(.float32).reshaped(4, per), beta = w(p + "attn_norm_\(name).beta").asType(.float32).reshaped(4, per)
            return (n * gamma + beta).asType(dtype).transposed(0, 3, 1, 2, 4).reshaped(1, 4, oldT, oldQ * per) // [1, H, T, F·per]
        }
        let qh = project("Q", per: 2), kh = project("K", per: 2), vh = project("V", per: 12)
        let scale = 1 / Float(qh.dim(3)).squareRoot()
        let attn = softmax32(matmul(qh * scale, kh.transposed(0, 1, 3, 2)))
        let mixed = matmul(attn, vh).reshaped(1, 4, oldT, oldQ, 12).transposed(0, 2, 3, 1, 4).reshaped(1, oldT, oldQ, 48)
        var out = linear(mixed, p + "attn_concat_proj.0")
        out = prelu(out, slopes: w(p + "attn_concat_proj.1.weight"))
        let f = out.asType(.float32), mean = f.mean(axis: -1, keepDims: true), variance = (f - mean).square().mean(axis: -1, keepDims: true)
        out =
            (((f - mean) * rsqrt(variance + 1e-5)) * w(p + "attn_concat_proj.2.gamma").asType(.float32)
            + w(p + "attn_concat_proj.2.beta").asType(.float32)).asType(dtype)
        return out + batch
    }

    /// The PostNet network between band split and merge: [T, 257, 6] float32 → [T, 257, 6].
    public func postNetwork(_ bands: MLXArray, trace: Trace? = nil) -> MLXArray {
        var x = conv2d(bands.expandedDimensions(axis: 0).asType(dtype), w("postnet.conv.0.weight"), padding: 1) + w("postnet.conv.0.bias")
        // GroupNorm(1, 48): one mean and variance over time, frequency and channels.
        let f = x.asType(.float32), mean = f.mean(keepDims: true), variance = (f - mean).square().mean(keepDims: true)
        x = (((f - mean) * rsqrt(variance + 1e-5)) * w("postnet.conv.1.weight").asType(.float32) + w("postnet.conv.1.bias").asType(.float32)).asType(dtype)
        trace?("postnet_conv", x[0])
        for b in 0..<5 {
            x = gridBlock(x, b)
            trace?("postnet_block\(b)", x[0])
        }
        let y = conv2d(x, w("postnet.deconv.weight"), padding: 1) + w("postnet.deconv.bias")
        trace?("postnet_deconv", y[0])
        return y[0]
    }

    // MARK: Pipeline

    /// The four stages `enhance` runs, in order; `stage` is called with each name before it starts.
    public static let stages = ["encoder", "adapter", "vocoder", "postnet"]

    /// models/unipase.py forward on one mono segment at `rateIn`: [-1, 1] samples → the enhanced segment at `rateOut`,
    /// ⌊n·rateOut/rateIn⌋ samples (before the inference script's peak matching). `stage` runs before each stage and
    /// may throw (cancellation); "postnet" is reported only when the PostNet runs (rateOut above 16 kHz).
    public func enhance(
        _ x: [Float], rateIn: Int, rateOut: Int, plc: Bool, trace: Trace? = nil, stage: (String) throws -> Void = { _ in }
    ) throws -> [Float] {
        guard UniPASEDSP.sincFits(from: rateIn, to: 16_000), UniPASEDSP.sincFits(from: 16_000, to: rateOut),
            rateOut <= 16_000 || UniPASEDSP.sincFits(from: 48_000, to: rateOut)
        else {
            throw UniPASEError("Convert \(rateIn) Hz audio to a standard rate first")
        }
        try stage("encoder")
        let wav16 = UniPASEDSP.padForEncoder(rateIn == 16_000 ? x : UniPASEDSP.resample(x, from: rateIn, to: 16_000))
        guard wav16.count >= 400 else { throw UniPASEError("Audio shorter than 25 ms at 16 kHz") }
        let (l1, l24) = encode(wav16, mask: plc ? UniPASEDSP.packetLossMask(wav16) : nil, trace: trace)
        eval(l1, l24)
        trace?("L1", l1[0]); trace?("L24", l24[0])
        try stage("adapter")
        let enhanced = adapt(l1: l1, l24: l24, trace: trace)
        eval(enhanced)
        trace?("adapter", enhanced[0])
        try stage("vocoder")
        let hidden = vocoderHidden(enhanced, trace: trace)
        trace?("vocoder_hidden", hidden)
        let wave = UniPASEDSP.vocosISTFT(hidden)
        eval(wave)
        trace?("vocoder_wav", wave)
        let w16 = wave.asArray(Float.self)
        let y: [Float]
        if rateOut < 16_000 {
            y = UniPASEDSP.resample(w16, from: 16_000, to: rateOut)
        } else if rateOut == 16_000 {
            y = w16
        } else {
            try stage("postnet")
            y = UniPASEDSP.postNet(w16, rateOut: rateOut) { bands in
                let out = postNetwork(bands, trace: trace)
                eval(out)
                return out
            }
        }
        return UniPASEDSP.fit(y, UniPASEDSP.outputLength(x.count, rateIn: rateIn, rateOut: rateOut))
    }
}
