// SPDX-License-Identifier: MIT
// The FLUX.2 VAE's encoder half (diffusers AutoencoderKLFlux2.encode), NHWC, for reference images. The decoder half
// is Flux2Klein's `Flux2VAEDecoder` (same weights in FLUX 3). Latents leave the encoder as the posterior mean, are
// patchified 2×2 (channel c·4 + dy·2 + dx), normalized by the VAE's batch-norm statistics and packed to tokens.
// mflux reference: models/flux2/model/flux2_vae/encoder/*, flux3_latent_creator.py (encode_tokens). Difference, within
// BF16 rounding: the image enters in BF16, so the encoder's residual stream is BF16 (mflux feeds a float32 image and
// its residual stream stays float32 by promotion).
import Flux2Klein
import Foundation
import MLX
import MLXNN

final class Flux3Downsampler: Module {
    @ModuleInfo var conv: Conv2d
    init(_ c: Int) { _conv.wrappedValue = Conv2d(inputChannels: c, outputChannels: c, kernelSize: 3, stride: 2, padding: 0) }
    /// diffusers Downsample2D with padding 0: pad one row and column at the bottom and right, then a stride-2 conv.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv(padded(x, widths: [IntOrPair(0), IntOrPair((0, 1)), IntOrPair((0, 1)), IntOrPair(0)]))
    }
}

final class Flux3DownBlock: Module {
    @ModuleInfo var resnets: [VAEResnet]
    @ModuleInfo var downsamplers: [Flux3Downsampler]
    init(_ i: Int, _ o: Int, downsample: Bool) {
        _resnets.wrappedValue = [VAEResnet(i, o), VAEResnet(o, o)]
        _downsamplers.wrappedValue = downsample ? [Flux3Downsampler(o)] : []
    }
}

public final class Flux3VAEEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var down: [Flux3DownBlock]
    @ModuleInfo(key: "mid_block") var mid: VAEMidBlock
    @ModuleInfo(key: "conv_norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    init(_ c: Flux3VAEConfig) {
        let ch = c.blockOut
        _convIn.wrappedValue = Conv2d(inputChannels: 3, outputChannels: ch[0], kernelSize: 3, padding: 1)
        _down.wrappedValue = ch.enumerated().map { i, o in Flux3DownBlock(i == 0 ? ch[0] : ch[i - 1], o, downsample: i < ch.count - 1) }
        _mid.wrappedValue = VAEMidBlock(ch[ch.count - 1])
        _normOut.wrappedValue = GroupNorm(groupCount: 32, dimensions: ch[ch.count - 1], eps: 1e-6, pytorchCompatible: true)
        _convOut.wrappedValue = Conv2d(inputChannels: ch[ch.count - 1], outputChannels: 64, kernelSize: 3, padding: 1)
    }
    func callAsFunction(_ image: MLXArray) -> MLXArray {
        var h = convIn(image)
        for block in down {
            for resnet in block.resnets { h = resnet(h) }
            for d in block.downsamplers { h = d(h) }
            eval(h)
        }
        return convOut(silu(normOut(mid(h))))
    }
}

/// encoder + quant_conv (the diffusers names under the VAE checkpoint's root).
public final class Flux3VAEEncoderHalf: Module {
    @ModuleInfo public var encoder: Flux3VAEEncoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    public init(_ c: Flux3VAEConfig) {
        _encoder.wrappedValue = Flux3VAEEncoder(c)
        _quantConv.wrappedValue = Conv2d(inputChannels: 64, outputChannels: 64, kernelSize: 1)
    }

    /// image NHWC [1, H, W, 3] in [-1, 1] → normalized tokens [1, (H/16)·(W/16), 128], in the weights' dtype.
    /// `statistics` are the decoder half's batch-norm mean and variance.
    public func tokens(_ image: MLXArray, statistics: (mean: MLXArray, variance: MLXArray)) -> MLXArray {
        let dtype = quantConv.weight.dtype
        let moments = quantConv(encoder(image.asType(dtype)))
        let std = MLX.sqrt(statistics.variance + 1e-4)
        return (Self.patchify(moments[.ellipsis, ..<32]) - statistics.mean) / std
    }

    /// image NHWC [1, H, W, 3] in [-1, 1] → the posterior's moments NHWC [1, H/8, W/8, 64]: mean (channels 0..<32)
    /// and log-variance (32..<64), in the weights' dtype. For callers that sample the posterior (IDOL).
    public func moments(_ image: MLXArray) -> MLXArray {
        quantConv(encoder(image.asType(quantConv.weight.dtype)))
    }

    /// NHWC [B, h, w, C] → tokens [B, (h/2)·(w/2), 4C], channel c·4 + dy·2 + dx (diffusers `_patchify_latents` + pack).
    public static func patchify(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        return x.reshaped(b, h / 2, 2, w / 2, 2, c).transposed(0, 1, 3, 5, 2, 4).reshaped(b, (h / 2) * (w / 2), c * 4)
    }
}
