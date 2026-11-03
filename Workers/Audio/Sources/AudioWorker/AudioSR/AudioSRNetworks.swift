// SPDX-License-Identifier: MIT
// AudioSR (haoheliu/versatile_audio_super_resolution, MIT) networks for MLX: the AudioLDM-style KL autoencoder, the
// latent-diffusion UNet and the 48 kHz HiFi-GAN vocoder. The VAE and UNet modules are adapted from soniqo/speech-swift
// `Sources/FlashSR` (Apache-2.0, commit 7baf2781d5c6a5d38b008a96eef3dca61c55f369), whose FlashSR student shares
// AudioSR's architecture; names follow the PyTorch checkpoint so the Get-time conversion only transposes kernels and
// inserts `.layers.` for TimestepEmbedSequential blocks. NHWC throughout ([batch, time, mel, channels]).
import Foundation
import MLX
import MLXFast
import MLXNN

/// Parameterless placeholder keeping nn.Sequential indices (SiLU, Dropout, Tanh).
final class AudioSRNoop: Module {}

@inline(__always) private func silu(_ x: MLXArray) -> MLXArray { x * sigmoid(x) }
private func groupNorm(_ channels: Int, eps: Float) -> GroupNorm {
    GroupNorm(groupCount: 32, dimensions: channels, eps: eps, affine: true, pytorchCompatible: true)
}
private func audioSRConv(_ input: Int, _ output: Int, kernel: Int = 3, stride: Int = 1, padding: Int? = nil) -> Conv2d {
    Conv2d(
        inputChannels: input, outputChannels: output, kernelSize: IntOrPair(kernel), stride: IntOrPair(stride),
        padding: IntOrPair(padding ?? kernel / 2))
}
private func nearest2x(_ x: MLXArray) -> MLXArray {
    let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
    return broadcast(x.reshaped(b, h, 1, w, 1, c), to: [b, h, 2, w, 2, c]).reshaped(b, h * 2, w * 2, c)
}

// MARK: - KL autoencoder (latent_diffusion/modules/diffusionmodules/model.py Encoder/Decoder)

final class AudioSRResnet: Module {
    @ModuleInfo var norm1: GroupNorm
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var norm2: GroupNorm
    @ModuleInfo var conv2: Conv2d
    @ModuleInfo(key: "nin_shortcut") var shortcut: Conv2d?
    init(_ input: Int, _ output: Int) {
        _norm1.wrappedValue = groupNorm(input, eps: 1e-6)
        _conv1.wrappedValue = audioSRConv(input, output)
        _norm2.wrappedValue = groupNorm(output, eps: 1e-6)
        _conv2.wrappedValue = audioSRConv(output, output)
        _shortcut.wrappedValue = input == output ? nil : audioSRConv(input, output, kernel: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = conv2(silu(norm2(conv1(silu(norm1(x))))))
        return (shortcut.map { $0(x) } ?? x) + h
    }
}

final class AudioSRVAEAttention: Module {
    @ModuleInfo var norm: GroupNorm
    @ModuleInfo var q: Conv2d
    @ModuleInfo var k: Conv2d
    @ModuleInfo var v: Conv2d
    @ModuleInfo(key: "proj_out") var projOut: Conv2d
    init(_ channels: Int) {
        _norm.wrappedValue = groupNorm(channels, eps: 1e-6)
        _q.wrappedValue = audioSRConv(channels, channels, kernel: 1)
        _k.wrappedValue = audioSRConv(channels, channels, kernel: 1)
        _v.wrappedValue = audioSRConv(channels, channels, kernel: 1)
        _projOut.wrappedValue = audioSRConv(channels, channels, kernel: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = norm(x)
        let (b, height, width, c) = (h.dim(0), h.dim(1), h.dim(2), h.dim(3))
        func tokens(_ t: MLXArray) -> MLXArray { t.reshaped(b, 1, height * width, c) }
        let out = MLXFast.scaledDotProductAttention(
            queries: tokens(q(h)), keys: tokens(k(h)), values: tokens(v(h)), scale: 1 / Float(c).squareRoot(), mask: .none)
        return x + projOut(out.reshaped(b, height, width, c))
    }
}

final class AudioSRResample: Module {
    @ModuleInfo var conv: Conv2d
    let down: Bool
    init(_ channels: Int, down: Bool) {
        self.down = down
        _conv.wrappedValue = audioSRConv(channels, channels, kernel: 3, stride: down ? 2 : 1, padding: down ? 0 : 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // Downsample pads (0, 1) on both spatial axes before a stride-2, padding-0 convolution.
        down ? conv(padded(x, widths: [.init((0, 0)), .init((0, 1)), .init((0, 1)), .init((0, 0))])) : conv(nearest2x(x))
    }
}

final class AudioSRVAELevel: Module {
    @ModuleInfo var block: [AudioSRResnet]
    @ModuleInfo var downsample: AudioSRResample?
    @ModuleInfo var upsample: AudioSRResample?
    init(blocks: [AudioSRResnet], downsample: AudioSRResample? = nil, upsample: AudioSRResample? = nil) {
        _block.wrappedValue = blocks
        _downsample.wrappedValue = downsample
        _upsample.wrappedValue = upsample
        super.init()
    }
}

final class AudioSRVAEMid: Module {
    @ModuleInfo(key: "block_1") var block1: AudioSRResnet
    @ModuleInfo(key: "attn_1") var attention: AudioSRVAEAttention
    @ModuleInfo(key: "block_2") var block2: AudioSRResnet
    init(_ channels: Int) {
        _block1.wrappedValue = AudioSRResnet(channels, channels)
        _attention.wrappedValue = AudioSRVAEAttention(channels)
        _block2.wrappedValue = AudioSRResnet(channels, channels)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { block2(attention(block1(x))) }
}

/// ch 128, ch_mult [1, 2, 4, 8], two resnet blocks per level, no level attention, 16 latent channels.
final class AudioSREncoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo var down: [AudioSRVAELevel]
    @ModuleInfo var mid: AudioSRVAEMid
    @ModuleInfo(key: "norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    init(ch: Int = 128, mult: [Int] = [1, 2, 4, 8], blocks: Int = 2, latent: Int = 16) {
        _convIn.wrappedValue = audioSRConv(1, ch)
        var levels: [AudioSRVAELevel] = [], input = ch
        for (index, m) in mult.enumerated() {
            let output = ch * m
            var resnets: [AudioSRResnet] = []
            for _ in 0..<blocks { resnets.append(AudioSRResnet(input, output)); input = output }
            levels.append(AudioSRVAELevel(blocks: resnets, downsample: index == mult.count - 1 ? nil : AudioSRResample(output, down: true)))
        }
        _down.wrappedValue = levels
        _mid.wrappedValue = AudioSRVAEMid(input)
        _normOut.wrappedValue = groupNorm(input, eps: 1e-6)
        _convOut.wrappedValue = audioSRConv(input, 2 * latent)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = convIn(x)
        for level in down {
            for block in level.block { h = block(h) }
            if let d = level.downsample { h = d(h) }
        }
        return convOut(silu(normOut(mid(h))))
    }
}

final class AudioSRDecoder: Module {
    @ModuleInfo(key: "conv_in") var convIn: Conv2d
    @ModuleInfo var mid: AudioSRVAEMid
    @ModuleInfo var up: [AudioSRVAELevel]
    @ModuleInfo(key: "norm_out") var normOut: GroupNorm
    @ModuleInfo(key: "conv_out") var convOut: Conv2d
    init(ch: Int = 128, mult: [Int] = [1, 2, 4, 8], blocks: Int = 2, latent: Int = 16) {
        var input = ch * mult[mult.count - 1]
        _convIn.wrappedValue = audioSRConv(latent, input)
        _mid.wrappedValue = AudioSRVAEMid(input)
        var levels: [AudioSRVAELevel] = []
        for index in stride(from: mult.count - 1, through: 0, by: -1) {
            let output = ch * mult[index]
            var resnets: [AudioSRResnet] = []
            for _ in 0...blocks { resnets.append(AudioSRResnet(input, output)); input = output }
            levels.insert(AudioSRVAELevel(blocks: resnets, upsample: index == 0 ? nil : AudioSRResample(output, down: false)), at: 0)
        }
        _up.wrappedValue = levels
        _normOut.wrappedValue = groupNorm(input, eps: 1e-6)
        _convOut.wrappedValue = audioSRConv(input, 1)
        super.init()
    }
    func callAsFunction(_ z: MLXArray) -> MLXArray {
        var h = mid(convIn(z))
        for level in up.reversed() {
            for block in level.block { h = block(h) }
            if let u = level.upsample { h = u(h) }
        }
        return convOut(silu(normOut(h)))
    }
}

final class AudioSRAutoencoder: Module {
    @ModuleInfo var encoder: AudioSREncoder
    @ModuleInfo var decoder: AudioSRDecoder
    @ModuleInfo(key: "quant_conv") var quantConv: Conv2d
    @ModuleInfo(key: "post_quant_conv") var postQuantConv: Conv2d
    override init() {
        _encoder.wrappedValue = AudioSREncoder()
        _decoder.wrappedValue = AudioSRDecoder()
        _quantConv.wrappedValue = audioSRConv(32, 32, kernel: 1)
        _postQuantConv.wrappedValue = audioSRConv(16, 16, kernel: 1)
        super.init()
    }
    /// Posterior mean and standard deviation (logvar clamped to [−30, 20]) of a log-mel [b, t, 256, 1].
    func encode(_ mel: MLXArray) -> (mean: MLXArray, std: MLXArray) {
        let moments = quantConv(encoder(mel))
        let c = moments.dim(-1) / 2
        let logvar = clip(moments[.ellipsis, c...], min: -30, max: 20)
        return (moments[.ellipsis, ..<c], exp(0.5 * logvar))
    }
    func decode(_ z: MLXArray) -> MLXArray { decoder(postQuantConv(z)) }
}

// MARK: - UNet (latent_diffusion/modules/diffusionmodules/openaimodel.py, use_spatial_transformer, extra_sa_layer)

final class AudioSRUNetResBlock: Module {
    @ModuleInfo(key: "in_layers") var inLayers: [Module]
    @ModuleInfo(key: "emb_layers") var embLayers: [Module]
    @ModuleInfo(key: "out_layers") var outLayers: [Module]
    @ModuleInfo(key: "skip_connection") var skip: Conv2d?
    init(_ input: Int, _ output: Int, embedding: Int) {
        _inLayers.wrappedValue = [groupNorm(input, eps: 1e-5), AudioSRNoop(), audioSRConv(input, output)]
        _embLayers.wrappedValue = [AudioSRNoop(), Linear(embedding, output)]
        _outLayers.wrappedValue = [groupNorm(output, eps: 1e-5), AudioSRNoop(), AudioSRNoop(), audioSRConv(output, output)]
        _skip.wrappedValue = input == output ? nil : audioSRConv(input, output, kernel: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray, embedding: MLXArray) -> MLXArray {
        // swiftlint:disable force_cast
        let projected = (embLayers[1] as! Linear)(silu(embedding)).expandedDimensions(axes: [1, 2])
        let h = (outLayers[3] as! Conv2d)(silu((outLayers[0] as! GroupNorm)((inLayers[2] as! Conv2d)(silu((inLayers[0] as! GroupNorm)(x))) + projected)))
        // swiftlint:enable force_cast
        return (skip.map { $0(x) } ?? x) + h
    }
}

final class AudioSRAttention: Module {
    @ModuleInfo(key: "to_q") var toQ: Linear
    @ModuleInfo(key: "to_k") var toK: Linear
    @ModuleInfo(key: "to_v") var toV: Linear
    @ModuleInfo(key: "to_out") var toOut: [Module]
    let heads: Int, dimHead: Int
    init(_ dim: Int, heads: Int, dimHead: Int) {
        self.heads = heads; self.dimHead = dimHead
        _toQ.wrappedValue = Linear(dim, heads * dimHead, bias: false)
        _toK.wrappedValue = Linear(dim, heads * dimHead, bias: false)
        _toV.wrappedValue = Linear(dim, heads * dimHead, bias: false)
        _toOut.wrappedValue = [Linear(heads * dimHead, dim), AudioSRNoop()]
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, n) = (x.dim(0), x.dim(1))
        func split(_ t: MLXArray) -> MLXArray { t.reshaped(b, n, heads, dimHead).transposed(0, 2, 1, 3) }
        let out = MLXFast.scaledDotProductAttention(
            queries: split(toQ(x)), keys: split(toK(x)), values: split(toV(x)), scale: 1 / Float(dimHead).squareRoot(), mask: .none)
        // swiftlint:disable:next force_cast
        return (toOut[0] as! Linear)(out.transposed(0, 2, 1, 3).reshaped(b, n, heads * dimHead))
    }
}

final class AudioSRGEGLU: Module {
    @ModuleInfo var proj: Linear
    init(_ input: Int, _ output: Int) { _proj.wrappedValue = Linear(input, 2 * output); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let h = proj(x), half = h.dim(-1) / 2
        return h[.ellipsis, ..<half] * gelu(h[.ellipsis, half...])
    }
}

final class AudioSRTransformerBlock: Module {
    @ModuleInfo var attn1: AudioSRAttention
    @ModuleInfo var ff: AudioSRFeedForward
    @ModuleInfo var attn2: AudioSRAttention
    @ModuleInfo var norm1: LayerNorm
    @ModuleInfo var norm2: LayerNorm
    @ModuleInfo var norm3: LayerNorm
    init(_ dim: Int, heads: Int, dimHead: Int) {
        _attn1.wrappedValue = AudioSRAttention(dim, heads: heads, dimHead: dimHead)
        _ff.wrappedValue = AudioSRFeedForward(dim)
        _attn2.wrappedValue = AudioSRAttention(dim, heads: heads, dimHead: dimHead)
        _norm1.wrappedValue = LayerNorm(dimensions: dim)
        _norm2.wrappedValue = LayerNorm(dimensions: dim)
        _norm3.wrappedValue = LayerNorm(dimensions: dim)
        super.init()
    }
    /// Without a context, attn2 attends over the same tokens.
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = attn1(norm1(input)) + input
        x = attn2(norm2(x)) + x
        return ff(norm3(x)) + x
    }
}

final class AudioSRFeedForward: Module {
    @ModuleInfo var net: [Module]
    init(_ dim: Int) { _net.wrappedValue = [AudioSRGEGLU(dim, 4 * dim), AudioSRNoop(), Linear(4 * dim, dim)]; super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        // swiftlint:disable:next force_cast
        (net[2] as! Linear)((net[0] as! AudioSRGEGLU)(x))
    }
}

final class AudioSRSpatialTransformer: Module {
    @ModuleInfo var norm: GroupNorm
    @ModuleInfo(key: "proj_in") var projIn: Conv2d
    @ModuleInfo(key: "transformer_blocks") var blocks: [AudioSRTransformerBlock]
    @ModuleInfo(key: "proj_out") var projOut: Conv2d
    init(_ channels: Int, headChannels: Int) {
        _norm.wrappedValue = groupNorm(channels, eps: 1e-6)
        _projIn.wrappedValue = audioSRConv(channels, channels, kernel: 1)
        _blocks.wrappedValue = [AudioSRTransformerBlock(channels, heads: channels / headChannels, dimHead: headChannels)]
        _projOut.wrappedValue = audioSRConv(channels, channels, kernel: 1)
        super.init()
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, h, w, c) = (x.dim(0), x.dim(1), x.dim(2), x.dim(3))
        var t = projIn(norm(x)).reshaped(b, h * w, c)
        for block in blocks { t = block(t) }
        return projOut(t.reshaped(b, h, w, c)) + x
    }
}

final class AudioSRUNetUpsample: Module {
    @ModuleInfo var conv: Conv2d
    init(_ channels: Int) { _conv.wrappedValue = audioSRConv(channels, channels); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { conv(nearest2x(x)) }
}

final class AudioSRUNetDownsample: Module {
    @ModuleInfo var op: Conv2d
    init(_ channels: Int) { _op.wrappedValue = audioSRConv(channels, channels, kernel: 3, stride: 2, padding: 1); super.init() }
    func callAsFunction(_ x: MLXArray) -> MLXArray { op(x) }
}

/// TimestepEmbedSequential: resblocks take the timestep embedding, the rest only the activations.
final class AudioSRUNetStage: Module {
    @ModuleInfo var layers: [Module]
    init(_ layers: [Module]) { _layers.wrappedValue = layers; super.init() }
    func callAsFunction(_ input: MLXArray, embedding: MLXArray) -> MLXArray {
        var x = input
        for layer in layers {
            switch layer {
            case let r as AudioSRUNetResBlock: x = r(x, embedding: embedding)
            case let s as AudioSRSpatialTransformer: x = s(x)
            case let u as AudioSRUNetUpsample: x = u(x)
            case let d as AudioSRUNetDownsample: x = d(x)
            case let c as Conv2d: x = c(x)
            default: preconditionFailure("Unknown UNet stage layer")
            }
        }
        return x
    }
}

/// in 32 (16 noisy + 16 condition), model channels 128, mult [1, 2, 3, 5], two resblocks per level, attention at
/// downsampling 2, 4 and 8 with an extra self-attention transformer, 32 channels per head, 16 out.
final class AudioSRUNet: Module {
    @ModuleInfo(key: "time_embed") var timeEmbed: [Module]
    @ModuleInfo(key: "input_blocks") var inputBlocks: [AudioSRUNetStage]
    @ModuleInfo(key: "middle_block") var middleBlock: AudioSRUNetStage
    @ModuleInfo(key: "output_blocks") var outputBlocks: [AudioSRUNetStage]
    @ModuleInfo var out: [Module]
    let modelChannels: Int

    init(input: Int = 32, output: Int = 16, modelChannels: Int = 128, mult: [Int] = [1, 2, 3, 5], blocks: Int = 2, attention: Set<Int> = [2, 4, 8], headChannels: Int = 32) {
        self.modelChannels = modelChannels
        let embedding = 4 * modelChannels
        _timeEmbed.wrappedValue = [Linear(modelChannels, embedding), AudioSRNoop(), Linear(embedding, embedding)]
        func attentionLayers(_ channels: Int) -> [Module] {
            [AudioSRSpatialTransformer(channels, headChannels: headChannels), AudioSRSpatialTransformer(channels, headChannels: headChannels)]
        }
        var stages = [AudioSRUNetStage([audioSRConv(input, modelChannels)])]
        var skips = [modelChannels], channels = modelChannels, ds = 1
        for (level, m) in mult.enumerated() {
            for _ in 0..<blocks {
                var layers: [Module] = [AudioSRUNetResBlock(channels, m * modelChannels, embedding: embedding)]
                channels = m * modelChannels
                if attention.contains(ds) { layers += attentionLayers(channels) }
                stages.append(AudioSRUNetStage(layers)); skips.append(channels)
            }
            if level != mult.count - 1 {
                stages.append(AudioSRUNetStage([AudioSRUNetDownsample(channels)])); skips.append(channels); ds *= 2
            }
        }
        _inputBlocks.wrappedValue = stages
        _middleBlock.wrappedValue = AudioSRUNetStage(
            [AudioSRUNetResBlock(channels, channels, embedding: embedding)] + attentionLayers(channels) + [AudioSRUNetResBlock(channels, channels, embedding: embedding)])
        var outputs: [AudioSRUNetStage] = []
        for (level, m) in mult.enumerated().reversed() {
            for index in 0...blocks {
                var layers: [Module] = [AudioSRUNetResBlock(channels + skips.removeLast(), m * modelChannels, embedding: embedding)]
                channels = m * modelChannels
                if attention.contains(ds) { layers += attentionLayers(channels) }
                if level != 0 && index == blocks { layers.append(AudioSRUNetUpsample(channels)); ds /= 2 }
                outputs.append(AudioSRUNetStage(layers))
            }
        }
        _outputBlocks.wrappedValue = outputs
        _out.wrappedValue = [groupNorm(channels, eps: 1e-5), AudioSRNoop(), audioSRConv(channels, output)]
        super.init()
    }

    /// x and condition: [b, t, f, 16]; timesteps: [b] integers.
    func callAsFunction(_ x: MLXArray, condition: MLXArray, timesteps: MLXArray) -> MLXArray {
        let half = modelChannels / 2
        let freqs = exp(-log(Float(10_000)) * MLXArray(0..<half).asType(.float32) / Float(half))
        let args = timesteps.asType(.float32).expandedDimensions(axis: 1) * freqs.expandedDimensions(axis: 0)
        let dtype = x.dtype
        // swiftlint:disable force_cast
        var embedding = (timeEmbed[0] as! Linear)(concatenated([cos(args), sin(args)], axis: -1).asType(dtype))
        embedding = (timeEmbed[2] as! Linear)(silu(embedding))
        var h = concatenated([x, condition], axis: -1)
        var skips: [MLXArray] = []
        for stage in inputBlocks { h = stage(h, embedding: embedding); skips.append(h) }
        h = middleBlock(h, embedding: embedding)
        for stage in outputBlocks { h = stage(concatenated([h, skips.removeLast()], axis: -1), embedding: embedding) }
        return (out[2] as! Conv2d)(silu((out[0] as! GroupNorm)(h)))
        // swiftlint:enable force_cast
    }
}

// MARK: - HiFi-GAN generator (audiosr/hifigan/models.py Generator, 48 kHz config, weight norm removed)

final class AudioSRVocoderResBlock: Module {
    @ModuleInfo var convs1: [Conv1d]
    @ModuleInfo var convs2: [Conv1d]
    init(_ channels: Int, kernel: Int, dilations: [Int]) {
        _convs1.wrappedValue = dilations.map {
            Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel * $0 - $0) / 2, dilation: $0)
        }
        _convs2.wrappedValue = dilations.map { _ in Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: kernel, padding: (kernel - 1) / 2) }
        super.init()
    }
    func callAsFunction(_ input: MLXArray) -> MLXArray {
        var x = input
        for (c1, c2) in zip(convs1, convs2) { x = c2(leakyRelu(c1(leakyRelu(x, negativeSlope: 0.1)), negativeSlope: 0.1)) + x }
        return x
    }
}

final class AudioSRVocoder: Module {
    @ModuleInfo(key: "conv_pre") var convPre: Conv1d
    @ModuleInfo var ups: [ConvTransposed1d]
    @ModuleInfo var resblocks: [AudioSRVocoderResBlock]
    @ModuleInfo(key: "conv_post") var convPost: Conv1d
    let kernels: Int

    init(
        mels: Int = 256, initial: Int = 1536, rates: [Int] = [6, 5, 4, 2, 2], upKernels: [Int] = [12, 10, 8, 4, 4], resKernels: [Int] = [3, 7, 11, 15],
        dilations: [Int] = [1, 3, 5]
    ) {
        kernels = resKernels.count
        _convPre.wrappedValue = Conv1d(inputChannels: mels, outputChannels: initial, kernelSize: 7, padding: 3)
        _ups.wrappedValue = zip(rates, upKernels).enumerated().map { index, pair in
            ConvTransposed1d(
                inputChannels: initial >> index, outputChannels: initial >> (index + 1), kernelSize: pair.1, stride: pair.0, padding: (pair.1 - pair.0) / 2)
        }
        _resblocks.wrappedValue = rates.indices.flatMap { index in resKernels.map { AudioSRVocoderResBlock(initial >> (index + 1), kernel: $0, dilations: dilations) } }
        _convPost.wrappedValue = Conv1d(inputChannels: initial >> rates.count, outputChannels: 1, kernelSize: 7, padding: 3)
        super.init()
    }
    /// mel: [b, frames, 256] → [b, frames · 480].
    func callAsFunction(_ mel: MLXArray) -> MLXArray {
        var x = convPre(mel)
        for (index, up) in ups.enumerated() {
            x = up(leakyRelu(x, negativeSlope: 0.1))
            let first = resblocks[index * kernels](x)
            x = (1..<kernels).reduce(first) { $0 + resblocks[index * kernels + $1](x) } / Float(kernels)
            eval(x)
        }
        return tanh(convPost(leakyRelu(x, negativeSlope: 0.01)))[.ellipsis, 0]
    }
}

/// The three networks of AudioSR basic, under the snapshot's top-level names.
final class AudioSRModel: Module {
    @ModuleInfo var vae: AudioSRAutoencoder
    @ModuleInfo var unet: AudioSRUNet
    @ModuleInfo var vocoder: AudioSRVocoder
    override init() {
        _vae.wrappedValue = AudioSRAutoencoder()
        _unet.wrappedValue = AudioSRUNet()
        _vocoder.wrappedValue = AudioSRVocoder()
        super.init()
    }
}
