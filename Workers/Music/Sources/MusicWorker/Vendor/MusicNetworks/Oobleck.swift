//
//  Oobleck.swift
//  MusicNetworks
//
//  Created by David Sherlock on 2026.
//
//  Stable Audio's Oobleck VAE (diffusers AutoencoderOobleck) in MLX, as
//  ACE-Step's acestep/models/mlx/vae_model.py writes it — and YuE2's VAE is
//  the same network at other sizes: channels-last,
//  weight-norm already folded into each convolution's weight at load time,
//  Snake activations with log-scaled alpha and beta. 48 kHz stereo, 1920
//  samples per 64-channel latent frame.
//

import Foundation
import MLX
import MLXNN
import MLXRandom

// `x + 1/(β+1e-9) · sin²(α·x)` with `α = exp(alpha)`, `β = exp(beta)`.
package final class Snake: Module, UnaryLayer {
    package var alpha: MLXArray
    package var beta: MLXArray

    package init(_ channels: Int) {
        alpha = MLXArray.zeros([channels])
        beta = MLXArray.zeros([channels])
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = MLX.exp(alpha), b = MLX.exp(beta)
        return x + MLX.reciprocal(b + 1e-9) * MLX.pow(MLX.sin(a * x), 2)
    }
}

package final class OobleckResidualUnit: Module, UnaryLayer {
    @ModuleInfo package var snake1: Snake
    @ModuleInfo package var conv1: Conv1d
    @ModuleInfo package var snake2: Snake
    @ModuleInfo package var conv2: Conv1d

    package init(_ channels: Int, dilation: Int) {
        snake1 = Snake(channels)
        conv1 = Conv1d(
            inputChannels: channels, outputChannels: channels, kernelSize: 7,
            padding: (6 * dilation) / 2, dilation: dilation)
        snake2 = Snake(channels)
        conv2 = Conv1d(inputChannels: channels, outputChannels: channels, kernelSize: 1)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        let y = conv2(snake2(conv1(snake1(x))))
        let trim = (x.dim(1) - y.dim(1)) / 2
        return (trim > 0 ? x[0..., trim..<(x.dim(1) - trim), 0...] : x) + y
    }
}

package final class OobleckEncoderBlock: Module, UnaryLayer {
    @ModuleInfo(key: "res_unit1") package var resUnit1: OobleckResidualUnit
    @ModuleInfo(key: "res_unit2") package var resUnit2: OobleckResidualUnit
    @ModuleInfo(key: "res_unit3") package var resUnit3: OobleckResidualUnit
    @ModuleInfo package var snake1: Snake
    @ModuleInfo package var conv1: Conv1d

    package init(input: Int, output: Int, stride: Int) {
        _resUnit1.wrappedValue = OobleckResidualUnit(input, dilation: 1)
        _resUnit2.wrappedValue = OobleckResidualUnit(input, dilation: 3)
        _resUnit3.wrappedValue = OobleckResidualUnit(input, dilation: 9)
        snake1 = Snake(input)
        conv1 = Conv1d(
            inputChannels: input, outputChannels: output, kernelSize: 2 * stride, stride: stride,
            padding: (stride + 1) / 2)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        conv1(snake1(resUnit3(resUnit2(resUnit1(x)))))
    }
}

package final class OobleckDecoderBlock: Module, UnaryLayer {
    @ModuleInfo package var snake1: Snake
    @ModuleInfo(key: "conv_t1") package var convT1: ConvTransposed1d
    @ModuleInfo(key: "res_unit1") package var resUnit1: OobleckResidualUnit
    @ModuleInfo(key: "res_unit2") package var resUnit2: OobleckResidualUnit
    @ModuleInfo(key: "res_unit3") package var resUnit3: OobleckResidualUnit

    package init(input: Int, output: Int, stride: Int) {
        snake1 = Snake(input)
        _convT1.wrappedValue = ConvTransposed1d(
            inputChannels: input, outputChannels: output, kernelSize: 2 * stride,
            stride: stride, padding: (stride + 1) / 2)
        _resUnit1.wrappedValue = OobleckResidualUnit(output, dilation: 1)
        _resUnit2.wrappedValue = OobleckResidualUnit(output, dilation: 3)
        _resUnit3.wrappedValue = OobleckResidualUnit(output, dilation: 9)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        resUnit3(resUnit2(resUnit1(convT1(snake1(x)))))
    }
}

package final class OobleckEncoder: Module, UnaryLayer {
    @ModuleInfo package var conv1: Conv1d
    @ModuleInfo package var block: [OobleckEncoderBlock]
    @ModuleInfo package var snake1: Snake
    @ModuleInfo package var conv2: Conv1d

    package init(_ c: OobleckConfiguration) {
        let multiples = [1] + c.channelMultiples
        conv1 = Conv1d(inputChannels: c.audioChannels, outputChannels: c.encoderHiddenSize, kernelSize: 7, padding: 3)
        block = c.downsamplingRatios.enumerated().map { index, stride in
            OobleckEncoderBlock(
                input: c.encoderHiddenSize * multiples[index],
                output: c.encoderHiddenSize * multiples[index + 1], stride: stride)
        }
        let width = c.encoderHiddenSize * multiples.last!
        snake1 = Snake(width)
        conv2 = Conv1d(inputChannels: width, outputChannels: c.encoderHiddenSize, kernelSize: 3, padding: 1)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block { h = b(h) }
        return conv2(snake1(h))
    }
}

package final class OobleckDecoder: Module, UnaryLayer {
    @ModuleInfo package var conv1: Conv1d
    @ModuleInfo package var block: [OobleckDecoderBlock]
    @ModuleInfo package var snake1: Snake
    @ModuleInfo package var conv2: Conv1d

    package init(_ c: OobleckConfiguration) {
        let multiples = [1] + c.channelMultiples
        let strides = Array(c.downsamplingRatios.reversed())
        conv1 = Conv1d(
            inputChannels: c.decoderInputChannels, outputChannels: c.decoderChannels * multiples.last!,
            kernelSize: 7, padding: 3)
        block = strides.enumerated().map { index, stride in
            OobleckDecoderBlock(
                input: c.decoderChannels * multiples[strides.count - index],
                output: c.decoderChannels * multiples[strides.count - index - 1], stride: stride)
        }
        snake1 = Snake(c.decoderChannels)
        conv2 = Conv1d(
            inputChannels: c.decoderChannels, outputChannels: c.audioChannels, kernelSize: 7, padding: 3,
            bias: false)
    }

    package func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = conv1(x)
        for b in block { h = b(h) }
        return conv2(snake1(h))
    }
}

// The VAE: audio `[B, S, 2]` ↔ latents `[B, T, 64]`, channels last.
package final class OobleckVAE: Module {
    @ModuleInfo package var encoder: OobleckEncoder
    @ModuleInfo package var decoder: OobleckDecoder
    package let configuration: OobleckConfiguration

    package init(_ c: OobleckConfiguration) {
        configuration = c
        encoder = OobleckEncoder(c)
        decoder = OobleckDecoder(c)
    }

    // The posterior mean (no sampling) — what upstream's MLX encode path uses for conditioning.
    package func encodeMean(_ audio: MLXArray) -> MLXArray {
        MLX.split(encoder(audio), parts: 2, axis: -1)[0]
    }

    // A sample from the posterior: mean + softplus(scale)·noise.
    package func encodeSample(_ audio: MLXArray, key: MLXArray?) -> MLXArray {
        let halves = MLX.split(encoder(audio), parts: 2, axis: -1)
        let scale = halves[1]
        let std = MLX.where(scale .> 20, scale, MLX.log(1 + MLX.exp(scale))) + 1e-4
        let noise = MLXRandom.normal(halves[0].shape, key: key)
        return halves[0] + std * noise
    }

    package func decode(_ latents: MLXArray) -> MLXArray { decoder(latents) }
}
