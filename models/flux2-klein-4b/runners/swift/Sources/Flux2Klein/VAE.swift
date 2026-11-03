// SPDX-License-Identifier: MIT
// FLUX.2 VAE decoder (diffusers AutoencoderKLFlux2, decode path only), written for Arsenal in NHWC throughout.
// Structure from the vendor config: post_quant_conv 32→32; conv_in 32→512; mid block (resnet, single-head attention,
// resnet); up blocks over channels [512, 512, 256, 128] with 3 resnets each and a nearest-2× upsampler on all but the
// last; GroupNorm(32, eps 1e-6) + SiLU + conv_out 128→3. Latents arrive patchified and batch-normed:
// x·sqrt(var + 1e-4) + mean, then the 2×2 unpatchify, then decode.
import Foundation
import MLX
import MLXFast
import MLXNN

func conv3x3(_ i: Int, _ o: Int) -> Conv2d { Conv2d(inputChannels: i, outputChannels: o, kernelSize: 3, padding: 1) }
func groupNorm(_ c: Int) -> GroupNorm { GroupNorm(groupCount: 32, dimensions: c, eps: 1e-6, pytorchCompatible: true) }
/// GroupNorm keeps half precision arrays; the stock MLX normalization kernel handles its accumulation.
func normSilu(_ norm: GroupNorm, _ x: MLXArray, silu useSilu: Bool = true) -> MLXArray {
    let y = norm(x)
    return useSilu ? silu(y) : y
}

public final class VAEResnet: Module {
    @ModuleInfo var norm1: GroupNorm
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var norm2: GroupNorm
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var shortcut: Conv2d?
    public init(_ i: Int, _ o: Int) {
        _norm1.wrappedValue = groupNorm(i); _conv1.wrappedValue = conv3x3(i, o)
        _norm2.wrappedValue = groupNorm(o); _conv2.wrappedValue = conv3x3(o, o)
        _shortcut.wrappedValue = i == o ? nil : Conv2d(inputChannels: i, outputChannels: o, kernelSize: 1)
    }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        (shortcut?(x) ?? x) + conv2(normSilu(norm2, conv1(normSilu(norm1, x))))
    }
}

public final class VAEAttention: Module {
    @ModuleInfo(key: "group_norm") var norm: GroupNorm
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: Linear
    public init(_ c: Int) {
        _norm.wrappedValue = groupNorm(c)
        _toQ.wrappedValue = Linear(c, c); _toK.wrappedValue = Linear(c, c); _toV.wrappedValue = Linear(c, c); _toOut.wrappedValue = Linear(c, c)
    }
    /// Single-head attention over every pixel. The head (512 channels) has no fused kernel, so the scores are built in
    /// full: queries run in chunks of `queryChunk` against all keys (each row's softmax is unchanged), which bounds the
    /// score matrix at chunk × pixels (a 2048² decode would otherwise hold 65,536² scores).
    public func callAsFunction(_ x: MLXArray, queryChunk: Int = 4096) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let n = normSilu(norm, x, silu: false).reshaped(b, 1, h * w, c)
        let q = toQ(n), k = toK(n), v = toV(n), scale = 1 / Float(c).squareRoot()
        let out: MLXArray
        if h * w <= queryChunk {
            out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: scale, mask: .none)
        } else {
            eval(q, k, v)
            out = concatenated(
                stride(from: 0, to: h * w, by: queryChunk).map { start in
                    let part = MLXFast.scaledDotProductAttention(
                        queries: q[0..., 0..., start..<Swift.min(start + queryChunk, h * w)], keys: k, values: v, scale: scale, mask: .none)
                    eval(part)
                    return part
                }, axis: 2)
        }
        return x + toOut(out.reshaped(b, h, w, c))
    }
}

public final class VAEMidBlock: Module {
    @ModuleInfo var resnets: [VAEResnet]
    @ModuleInfo var attentions: [VAEAttention]
    public init(_ c: Int) { _resnets.wrappedValue = [VAEResnet(c, c), VAEResnet(c, c)]; _attentions.wrappedValue = [VAEAttention(c)] }
    public func callAsFunction(_ x: MLXArray) -> MLXArray { resnets[1](attentions[0](resnets[0](x))) }
}

public final class VAEUpsampler: Module {
    @ModuleInfo var conv: Conv2d
    init(_ c: Int) { _conv.wrappedValue = conv3x3(c, c) }
    public func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let up = broadcast(x[0..., 0..., .newAxis, 0..., .newAxis, 0...], to: [b, h, 2, w, 2, c]).reshaped(b, h * 2, w * 2, c)
        return conv(up)
    }
}

public final class VAEUpBlock: Module {
    @ModuleInfo public var resnets: [VAEResnet]
    @ModuleInfo public var upsamplers: [VAEUpsampler]
    init(_ i: Int, _ o: Int, upsample: Bool) {
        _resnets.wrappedValue = (0..<3).map { VAEResnet($0 == 0 ? i : o, o) }
        _upsamplers.wrappedValue = upsample ? [VAEUpsampler(o)] : []
    }
}

public final class VAEDecoder: Module {
    @ModuleInfo(key: "conv_in") public var convIn: Conv2d
    @ModuleInfo(key: "mid_block") public var mid: VAEMidBlock
    @ModuleInfo(key: "up_blocks") public var up: [VAEUpBlock]
    @ModuleInfo(key: "conv_norm_out") public var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") public var convOut: Conv2d
    /// `blockOut` is diffusers' block_out_channels (encoder order); the decoder runs it reversed.
    init(blockOut: [Int] = [128, 256, 512, 512]) {
        let channels = Array(blockOut.reversed())
        _convIn.wrappedValue = conv3x3(32, channels[0])
        _mid.wrappedValue = VAEMidBlock(channels[0])
        _up.wrappedValue = channels.enumerated().map { i, o in VAEUpBlock(i == 0 ? o : channels[i - 1], o, upsample: i < channels.count - 1) }
        _normOut.wrappedValue = groupNorm(channels[channels.count - 1])
        _convOut.wrappedValue = conv3x3(channels[channels.count - 1], 3)
    }
    /// Evaluates per resnet, which bounds the intermediates a 1024² decode keeps alive.
    public var activationTrace: ((String, MLXArray) -> Void)?
    public func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = mid(convIn(z)); eval(h)
        for (i, block) in up.enumerated() {
            for (j, resnet) in block.resnets.enumerated() { h = resnet(h); eval(h); activationTrace?("up-\(i)-resnet-\(j)", h) }
            for upsampler in block.upsamplers { h = upsampler(h); eval(h) }
        }
        return convOut(normSilu(normOut, h))
    }
}

public final class Flux2VAEDecoder: Module {
    @ModuleInfo(key: "post_quant_conv") public var postQuantConv: Conv2d
    @ModuleInfo public var decoder: VAEDecoder
    @ParameterInfo(key: "bn_mean") var bnMean: MLXArray
    @ParameterInfo(key: "bn_var") var bnVar: MLXArray
    public init(blockOut: [Int] = [128, 256, 512, 512]) {
        _postQuantConv.wrappedValue = Conv2d(inputChannels: 32, outputChannels: 32, kernelSize: 1)
        _decoder.wrappedValue = VAEDecoder(blockOut: blockOut)
        _bnMean.wrappedValue = MLXArray.zeros([128], dtype: .float16); _bnVar.wrappedValue = MLXArray.ones([128], dtype: .float16)
    }

    /// Packed latents [1, seq, 128] on a gridH × gridW grid → batch-norm-denormalized, unpatchified NHWC [1, 2gh, 2gw, 32].
    public func latents(_ packed: MLXArray, gridH: Int, gridW: Int) -> MLXArray {
        let x = packed.asType(bnMean.dtype) * MLX.sqrt(bnVar + 1e-4) + bnMean // [1, seq, 128]
        // channel index = c·4 + dy·2 + dx (diffusers _unpatchify_latents)
        return x.reshaped(1, gridH, gridW, 32, 2, 2).transposed(0, 1, 4, 2, 5, 3).reshaped(1, gridH * 2, gridW * 2, 32)
    }

    /// FLUX 3's order (mflux `unpack_packed_latents`): the statistics stay in the VAE dtype (sqrt in that dtype), the
    /// float32 latent state is denormalized in float32, unpatchified, and only then cast to `dtype`. [1, seq, 128] float32
    /// on a gridH × gridW grid → NHWC [1, 2gh, 2gw, 32].
    public func latents(state packed: MLXArray, gridH: Int, gridW: Int, dtype: DType) -> MLXArray {
        let x = packed * MLX.sqrt(bnVar + 1e-4) + bnMean
        return x.reshaped(1, gridH, gridW, 32, 2, 2).transposed(0, 1, 4, 2, 5, 3).reshaped(1, gridH * 2, gridW * 2, 32).asType(dtype)
    }

    /// The latent batch-norm statistics (mean, variance), for the encoder's normalization.
    public var statistics: (mean: MLXArray, variance: MLXArray) { (bnMean, bnVar) }

    /// NHWC latents → image [1, H, W, 3] in [-1, 1] (unclipped), in `dtype`.
    public func decode(_ z: MLXArray, dtype: DType) -> MLXArray {
        decoder(postQuantConv(z.asType(dtype)))
    }
}
