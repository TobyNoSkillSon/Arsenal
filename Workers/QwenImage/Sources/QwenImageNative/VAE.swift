// SPDX-License-Identifier: MIT
// AutoencoderKLQwenImage21 (diffusers autoencoder_kl_qwenimage21.py): Wan 2.2's residual VAE specialised to one frame,
// RGBA in and out, 64 latent channels, 16× spatial compression. Every "causal 3D" convolution is a Conv2d on the single
// frame; the temporal convolutions (`time_conv`) never run for a first frame and are not loaded. NHWC throughout.
import Foundation
import MLX
import MLXFast
import MLXNN

/// Wan's RMS_norm: F.normalize over channels (float32) × √C × γ. γ is stored [C,1,1(,1)] and loaded as [C].
final class ChannelRMSNorm: Module {
    let gamma: MLXArray
    init(_ dim: Int) { gamma = MLXArray.ones([dim]) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let x32 = x.asType(.float32)
        let norm = maximum(sqrt(sum(x32 * x32, axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
        let normalized = (x32 / norm).asType(x.dtype)
        return normalized * Float(gamma.dim(0)).squareRoot() * gamma
    }
}

final class VAEResidual: Module {
    @ModuleInfo(key: "norm1") var norm1: ChannelRMSNorm
    @ModuleInfo(key: "conv1") var conv1: Conv2d
    @ModuleInfo(key: "norm2") var norm2: ChannelRMSNorm
    @ModuleInfo(key: "conv2") var conv2: Conv2d
    @ModuleInfo(key: "conv_shortcut") var shortcut: Conv2d?
    init(_ input: Int, _ output: Int) {
        _norm1.wrappedValue = ChannelRMSNorm(input)
        _conv1.wrappedValue = Conv2d(inputChannels: input, outputChannels: output, kernelSize: 3, padding: 1)
        _norm2.wrappedValue = ChannelRMSNorm(output)
        _conv2.wrappedValue = Conv2d(inputChannels: output, outputChannels: output, kernelSize: 3, padding: 1)
        _shortcut.wrappedValue = input == output ? nil : Conv2d(inputChannels: input, outputChannels: output, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = shortcut?(x) ?? x
        let y = conv2(silu(norm2(conv1(silu(norm1(x))))))
        return y + h
    }
}

/// Single-head spatial self-attention over all pixels (mid block).
final class VAEAttention: Module {
    @ModuleInfo(key: "norm") var norm: ChannelRMSNorm
    @ModuleInfo(key: "to_qkv") var toQKV: Conv2d
    @ModuleInfo(key: "proj") var proj: Conv2d
    init(_ dim: Int) {
        _norm.wrappedValue = ChannelRMSNorm(dim)
        _toQKV.wrappedValue = Conv2d(inputChannels: dim, outputChannels: 3 * dim, kernelSize: 1)
        _proj.wrappedValue = Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        let qkv = toQKV(norm(x)).reshaped(b, 1, h * w, 3 * c)
        let q = qkv[.ellipsis, 0..<c], k = qkv[.ellipsis, c..<(2 * c)], v = qkv[.ellipsis, (2 * c)...]
        let out = MLXFast.scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(c).squareRoot(), mask: .none)
        return proj(out.reshaped(b, h, w, c)) + x
    }
}

final class VAEMid: Module {
    @ModuleInfo(key: "resnets") var resnets: [VAEResidual]
    @ModuleInfo(key: "attentions") var attentions: [VAEAttention]
    init(_ dim: Int) {
        _resnets.wrappedValue = [VAEResidual(dim, dim), VAEResidual(dim, dim)]
        _attentions.wrappedValue = [VAEAttention(dim)]
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { resnets[1](attentions[0](resnets[0](x))) }
}

/// Resample's spatial half: nearest ×2 then a 3×3 conv, or zero-pad right/bottom then a stride-2 3×3 conv.
/// The checkpoint's `resample.1` loads as `conv`.
final class VAEResample: Module {
    let up: Bool
    @ModuleInfo(key: "conv") var conv: Conv2d
    init(_ dim: Int, up: Bool) {
        self.up = up
        _conv.wrappedValue =
            up
            ? Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, padding: 1)
            : Conv2d(inputChannels: dim, outputChannels: dim, kernelSize: 3, stride: 2)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        if up {
            let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
            let doubled = broadcast(x.reshaped(b, h, 1, w, 1, c), to: [b, h, 2, w, 2, c]).reshaped(b, 2 * h, 2 * w, c)
            return conv(doubled)
        }
        return conv(padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))]))
    }
}

/// AvgDown3D on one frame: a temporal factor of 2 pads a zero frame in front, so its channels interleave zeros.
func vaeAverageDown(_ x: MLXArray, outChannels: Int, temporal: Int, spatial: Int) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let factor = temporal * spatial * spatial, group = c * factor / outChannels
    // [b, h', w', C, fs(a), fs(b)]
    var y = x.reshaped(b, h / spatial, spatial, w / spatial, spatial, c).transposed(0, 1, 3, 5, 2, 4)
    if temporal == 2 { y = stacked([MLXArray.zeros(like: y), y], axis: 4) } else { y = expandedDimensions(y, axis: 4) }
    // channel index ((c·ft + t)·fs + a)·fs + b, then the mean of each run of `group` channels.
    return mean(y.reshaped(b, h / spatial, w / spatial, outChannels, group).asType(.float32), axis: -1).asType(x.dtype)
}

/// DupUp3D on a first chunk: channels repeated, unfolded into (time, 2×2 space), and the last time slice kept.
func vaeDuplicateUp(_ x: MLXArray, outChannels: Int, temporal: Int) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    let repeats = outChannels * temporal * 4 / c
    let y = repeated(x, count: repeats, axis: -1).reshaped(b, h, w, outChannels, temporal, 2, 2)[0..., 0..., 0..., 0..., temporal - 1]
    return y.transposed(0, 1, 4, 2, 5, 3).reshaped(b, 2 * h, 2 * w, outChannels)
}

final class VAEDownBlock: Module {
    let outChannels: Int, temporal: Int, spatial: Int
    @ModuleInfo(key: "resnets") var resnets: [VAEResidual]
    @ModuleInfo(key: "downsampler") var downsampler: VAEResample?
    init(_ input: Int, _ output: Int, blocks: Int, temporal: Bool, down: Bool) {
        outChannels = output; self.temporal = temporal ? 2 : 1; spatial = down ? 2 : 1
        _resnets.wrappedValue = (0..<blocks).map { VAEResidual($0 == 0 ? input : output, output) }
        _downsampler.wrappedValue = down ? VAEResample(output, up: false) : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for r in resnets { y = r(y) }
        if let downsampler { y = downsampler(y) }
        return y + vaeAverageDown(x, outChannels: outChannels, temporal: temporal, spatial: spatial)
    }
}

final class VAEUpBlock: Module {
    let outChannels: Int, temporal: Int, up: Bool
    @ModuleInfo(key: "resnets") var resnets: [VAEResidual]
    @ModuleInfo(key: "upsampler") var upsampler: VAEResample?
    init(_ input: Int, _ output: Int, blocks: Int, temporal: Bool, up: Bool) {
        outChannels = output; self.temporal = temporal ? 2 : 1; self.up = up
        _resnets.wrappedValue = (0...blocks).map { VAEResidual($0 == 0 ? input : output, output) }
        _upsampler.wrappedValue = up ? VAEResample(output, up: true) : nil
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for r in resnets { y = r(y) }
        guard let upsampler else { return y }
        return upsampler(y) + vaeDuplicateUp(x, outChannels: outChannels, temporal: temporal)
    }
}

final class VAEEncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "down_blocks") var downBlocks: [VAEDownBlock]
    @ModuleInfo(key: "mid_block") var mid: VAEMid
    @ModuleInfo(key: "norm_out") var normOut: ChannelRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    init(_ c: VAEConfig) {
        let dims = ([1] + c.dimMult).map { c.baseDim * $0 }
        _convIn.wrappedValue = Conv2d(inputChannels: c.inChannels, outputChannels: dims[0], kernelSize: 3, padding: 1)
        let last = c.dimMult.count - 1
        _downBlocks.wrappedValue = (0...last).map { i in
            VAEDownBlock(dims[i], dims[i + 1], blocks: c.resBlocks, temporal: i != last && c.temporalDownsample[i], down: i != last)
        }
        _mid.wrappedValue = VAEMid(dims[dims.count - 1])
        _normOut.wrappedValue = ChannelRMSNorm(dims[dims.count - 1])
        _convOut.wrappedValue = Conv2d(inputChannels: dims[dims.count - 1], outputChannels: 2 * c.zDim, kernelSize: 3, padding: 1)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = convIn(x)
        for block in downBlocks { y = block(y); eval(y) }
        return convOut(silu(normOut(mid(y))))
    }
}

final class VAEDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo(key: "mid_block") var mid: VAEMid
    @ModuleInfo(key: "up_blocks") var upBlocks: [VAEUpBlock]
    @ModuleInfo(key: "norm_out") var normOut: ChannelRMSNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    init(_ c: VAEConfig) {
        let dims = ([c.dimMult.last!] + c.dimMult.reversed()).map { c.decoderBaseDim * $0 }
        let temporalUp = Array(c.temporalDownsample.reversed())
        _convIn.wrappedValue = Conv2d(inputChannels: c.zDim, outputChannels: dims[0], kernelSize: 3, padding: 1)
        _mid.wrappedValue = VAEMid(dims[0])
        let last = c.dimMult.count - 1
        _upBlocks.wrappedValue = (0...last).map { i in
            VAEUpBlock(dims[i], dims[i + 1], blocks: c.resBlocks, temporal: i != last && temporalUp[i], up: i != last)
        }
        _normOut.wrappedValue = ChannelRMSNorm(dims[dims.count - 1])
        _convOut.wrappedValue = Conv2d(inputChannels: dims[dims.count - 1], outputChannels: c.outChannels, kernelSize: 3, padding: 1)
    }
    var observer: ((String, MLXArray) -> Void)?
    func callAsFunction(_ z: MLXArray) -> MLXArray {
        let input = convIn(z)
        observer?("conv_in", input)
        var y = mid(input); eval(y)
        observer?("mid", y)
        for (i, block) in upBlocks.enumerated() {
            y = block(y); eval(y)
            observer?("up\(i)", y)
        }
        return convOut(silu(normOut(y)))
    }
}

public final class QwenImageVAE: Module {
    public let config: VAEConfig
    @ModuleInfo(key: "encoder") var encoder: VAEEncoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    @ModuleInfo(key: "decoder") var decoder: VAEDecoder
    public init(_ c: VAEConfig) {
        config = c
        _encoder.wrappedValue = VAEEncoder(c)
        _quantConv.wrappedValue = Conv2d(inputChannels: 2 * c.zDim, outputChannels: 2 * c.zDim, kernelSize: 1)
        _postQuantConv.wrappedValue = Conv2d(inputChannels: c.zDim, outputChannels: c.zDim, kernelSize: 1)
        _decoder.wrappedValue = VAEDecoder(c)
    }
    public var dtype: DType { postQuantConv.weight.dtype }
    var mean: MLXArray { MLXArray(config.latentsMean).reshaped(1, 1, 1, config.zDim) }
    var std: MLXArray { MLXArray(config.latentsStd).reshaped(1, 1, 1, config.zDim) }

    /// RGBA in [-1, 1], [1, H, W, 4] → normalised latents [1, H/16, W/16, 64] (posterior mode, float32).
    public func encode(_ image: MLXArray) -> MLXArray {
        let moments = quantConv(encoder(image.asType(dtype)))
        let mode = moments[.ellipsis, 0..<config.zDim].asType(.float32)
        return (mode - mean) / std
    }
    /// Normalised latents [1, h, w, 64] → RGBA in [-1, 1], [1, 16h, 16w, 4] (float32, clamped).
    /// Lab diagnostics: the decoder's activations after each stage.
    public var decoderObserver: ((String, MLXArray) -> Void)? {
        get { decoder.observer }
        set { decoder.observer = newValue }
    }
    public func decode(_ latents: MLXArray) -> MLXArray {
        let z = (latents.asType(.float32) * std + mean).asType(dtype)
        return clip(decoder(postQuantConv(z)).asType(.float32), min: -1, max: 1)
    }
}
