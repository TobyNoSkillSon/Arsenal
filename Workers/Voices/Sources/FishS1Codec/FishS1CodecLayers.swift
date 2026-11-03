// SPDX-License-Identifier: MIT
// Fish S1 DAC codec for Drama 3. Derived from mlx-audio-swift MLXAudioCodecs/FishS1DAC at
// 01dec7c9bdce3088a6b6b7ab9f2e403458195efb (MIT), restructured to follow the reference implementation,
// mlx-audio Python mlx_audio/codec/models/fish_s1_dac at dad2087f, including its causal streaming decode.
// Provenance and every deliberate change: lab/drama3-port/codec-notes.md.
import Foundation
import MLX
import MLXNN

// MARK: - Streaming and weight-norm hooks

/// A layer with a streaming forward (Python `step`). Layers without one are pointwise and stream through their call.
protocol FishS1Steppable: AnyObject {
    func step(_ x: MLXArray) -> MLXArray
}

/// A layer that carries streaming state between `step` calls (Python `reset_state`).
protocol FishS1Resettable: AnyObject {
    func resetState()
}

/// A weight-normalised conv whose kernel g * v / |v| is computed once at load (Python `fold_weight_norm`).
protocol FishS1WeightNormed: AnyObject {
    /// Folds the kernel from the current parameters and stores it cast to `dtype`; returns it for evaluation.
    func fold(dtype: DType) -> MLXArray
}

/// Non-parameter state. A plain class, so Module reflection never treats its arrays as parameters.
final class FishS1ConvState {
    var buffer: MLXArray?
    var overlap: MLXArray?
}

/// Folded MLX-layout kernel of a weight-normalised conv.
final class FishS1FoldedKernel {
    var kernel: MLXArray?
}

@inline(__always)
func fishS1Step(_ layer: Module, _ x: MLXArray) -> MLXArray {
    if let steppable = layer as? FishS1Steppable { return steppable.step(x) }
    return (layer as! UnaryLayer)(x)
}

@inline(__always)
func fishS1Call(_ layer: Module, _ x: MLXArray) -> MLXArray {
    (layer as! UnaryLayer)(x)
}

// MARK: - Helpers (Python module-level functions)

/// sqrt(sum(x^2)) over every axis but `exceptDim` (PyTorch weight_norm's norm).
func fishS1NormalizeWeight(_ x: MLXArray, exceptDim: Int = 0) -> MLXArray {
    let axes = (0 ..< x.ndim).filter { $0 != exceptDim }
    return sqrt(sum(pow(x, 2), axes: axes, keepDims: true))
}

/// g * v / |v|, without an epsilon, as the reference folds it.
func fishS1WeightNormKernel(g: MLXArray, v: MLXArray) -> MLXArray {
    g * v / fishS1NormalizeWeight(v, exceptDim: 0)
}

func fishS1ExtraPaddingForConv1d(length: Int, kernelSize: Int, stride: Int, paddingTotal: Int = 0) -> Int {
    let frames = Double(length - kernelSize + paddingTotal) / Double(stride) + 1.0
    let ideal = (ceil(frames) - 1) * Double(stride) + Double(kernelSize - paddingTotal)
    return max(0, Int(ideal - Double(length)))
}

/// conv1d on NCL input with an MLX-layout kernel [out, k, in/groups].
func fishS1Conv1d(_ x: MLXArray, kernel: MLXArray, bias: MLXArray?, stride: Int, padding: Int = 0, dilation: Int, groups: Int) -> MLXArray {
    var y = conv1d(x.swappedAxes(1, 2), kernel, stride: stride, padding: padding, dilation: dilation, groups: groups)
    if let bias { y = y + bias }
    return y.swappedAxes(1, 2)
}

/// conv_transpose1d on NCL input with an MLX-layout kernel [out, k, in/groups]. No output padding, as in the reference.
func fishS1ConvTranspose1d(_ x: MLXArray, kernel: MLXArray, bias: MLXArray?, stride: Int, dilation: Int, groups: Int) -> MLXArray {
    var y = convTransposed1d(x.swappedAxes(1, 2), kernel, stride: stride, padding: 0, dilation: dilation, groups: groups)
    if let bias { y = y + bias }
    return y.swappedAxes(1, 2)
}

/// Causal conv forward: left pad by `padding`, right pad by the extra the stride needs.
func fishS1CausalConvForward(
    _ x: MLXArray, kernel: MLXArray, bias: MLXArray?, kernelSize: Int, stride: Int, padding: Int, dilation: Int, groups: Int
) -> MLXArray {
    let extra = fishS1ExtraPaddingForConv1d(length: x.dim(2), kernelSize: kernelSize, stride: stride, paddingTotal: padding)
    let padded = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((padding, extra))])
    return fishS1Conv1d(padded, kernel: kernel, bias: bias, stride: stride, dilation: dilation, groups: groups)
}

/// Causal conv streaming step: carries the last `padding` input samples between calls (stride 1 only).
func fishS1CausalConvStep(
    _ x: MLXArray, state: FishS1ConvState, kernel: MLXArray, bias: MLXArray?, stride: Int, padding: Int, dilation: Int, groups: Int
) -> MLXArray {
    precondition(stride == 1, "Streaming requires stride 1.")
    var input = x
    if padding > 0 {
        if let buffer = state.buffer {
            input = concatenated([buffer, x], axis: -1)
        } else {
            input = MLX.padded(x, widths: [IntOrPair(0), IntOrPair(0), IntOrPair((padding, 0))])
        }
        let length = input.dim(2)
        state.buffer = input[.ellipsis, (length - padding)...]
    }
    return fishS1Conv1d(input, kernel: kernel, bias: bias, stride: stride, dilation: dilation, groups: groups)
}

/// Causal transposed conv forward: drop the last kernel - stride samples.
func fishS1CausalTransConvForward(
    _ x: MLXArray, kernel: MLXArray, bias: MLXArray?, kernelSize: Int, stride: Int, dilation: Int, groups: Int
) -> MLXArray {
    let y = fishS1ConvTranspose1d(x, kernel: kernel, bias: bias, stride: stride, dilation: dilation, groups: groups)
    let pad = kernelSize - stride
    return y[.ellipsis, 0 ..< (y.dim(2) - pad)]
}

/// Causal transposed conv streaming step: overlap-adds the tail the causal trim drops into the next call.
func fishS1CausalTransConvStep(
    _ x: MLXArray, state: FishS1ConvState, kernel: MLXArray, bias: MLXArray?, stride: Int, dilation: Int, groups: Int
) -> MLXArray {
    var y = fishS1ConvTranspose1d(x, kernel: kernel, bias: bias, stride: stride, dilation: dilation, groups: groups)
    let valid = x.dim(2) * stride
    if var carry = state.overlap {
        // Both halves include the bias; keep it once.
        if let bias { carry = carry - bias[0..., .newAxis] }
        let n = carry.dim(2)
        y = concatenated([y[.ellipsis, 0 ..< n] + carry, y[.ellipsis, n...]], axis: -1)
    }
    state.overlap = y.dim(2) > valid ? y[.ellipsis, valid...] : nil
    return y[.ellipsis, 0 ..< valid]
}

// MARK: - Plain convs (PyTorch parameter layout, NCL input)

final class FishS1Conv1dTorch: Module, UnaryLayer {
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int
    let kernelSize: Int

    var weight: MLXArray
    var bias: MLXArray?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1, bias: Bool = true) {
        self.stride = stride; self.padding = padding; self.dilation = dilation; self.groups = groups; self.kernelSize = kernelSize
        let inPerGroup = inChannels / groups
        let scale = Float(sqrt(1.0 / Double(inPerGroup * kernelSize)))
        self.weight = MLXRandom.uniform(low: -scale, high: scale, [outChannels, inPerGroup, kernelSize])
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    var kernel: MLXArray { weight.transposed(0, 2, 1) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fishS1Conv1d(x, kernel: kernel, bias: bias, stride: stride, padding: padding, dilation: dilation, groups: groups)
    }
}

final class FishS1ConvTranspose1dTorch: Module, UnaryLayer {
    let stride: Int
    let dilation: Int
    let groups: Int
    let kernelSize: Int

    var weight: MLXArray
    var bias: MLXArray?

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1, groups: Int = 1, bias: Bool = true) {
        self.stride = stride; self.dilation = dilation; self.groups = groups; self.kernelSize = kernelSize
        let outPerGroup = outChannels / groups
        let scale = Float(sqrt(1.0 / Double(outPerGroup * kernelSize)))
        self.weight = MLXRandom.uniform(low: -scale, high: scale, [inChannels, outPerGroup, kernelSize])
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    /// PyTorch [in, out/groups, k] -> MLX [out, k, in/groups].
    var kernel: MLXArray { weight.transposed(1, 2, 0) }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fishS1ConvTranspose1d(x, kernel: kernel, bias: bias, stride: stride, dilation: dilation, groups: groups)
    }
}

/// Python CausalConvNet: an unnormalised causal conv, parameters under `conv.`.
final class FishS1CausalConvNet: Module, UnaryLayer, FishS1Steppable, FishS1Resettable {
    @ModuleInfo(key: "conv") var conv: FishS1Conv1dTorch
    let stride: Int
    let kernelSize: Int
    let padding: Int
    let state = FishS1ConvState()

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int = 1, stride: Int = 1, groups: Int = 1, bias: Bool = true) {
        self.stride = stride
        self.kernelSize = (kernelSize - 1) * dilation + 1
        self.padding = self.kernelSize - stride
        _conv.wrappedValue = FishS1Conv1dTorch(
            inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize, stride: stride, padding: 0, dilation: dilation, groups: groups, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fishS1CausalConvForward(
            x, kernel: conv.kernel, bias: conv.bias, kernelSize: kernelSize, stride: stride, padding: padding, dilation: conv.dilation, groups: conv.groups)
    }

    func step(_ x: MLXArray) -> MLXArray {
        fishS1CausalConvStep(x, state: state, kernel: conv.kernel, bias: conv.bias, stride: stride, padding: padding, dilation: conv.dilation, groups: conv.groups)
    }

    func resetState() { state.buffer = nil }
}

/// Python CausalTransConvNet: an unnormalised causal transposed conv, parameters under `conv.`.
final class FishS1CausalTransConvNet: Module, UnaryLayer, FishS1Steppable, FishS1Resettable {
    @ModuleInfo(key: "conv") var conv: FishS1ConvTranspose1dTorch
    let stride: Int
    let kernelSize: Int
    let state = FishS1ConvState()

    init(inChannels: Int, outChannels: Int, kernelSize: Int, dilation: Int = 1, stride: Int = 1, groups: Int = 1, bias: Bool = true) {
        self.stride = stride
        self.kernelSize = kernelSize
        _conv.wrappedValue = FishS1ConvTranspose1dTorch(
            inChannels: inChannels, outChannels: outChannels, kernelSize: kernelSize, stride: stride, dilation: dilation, groups: groups, bias: bias)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fishS1CausalTransConvForward(x, kernel: conv.kernel, bias: conv.bias, kernelSize: kernelSize, stride: stride, dilation: conv.dilation, groups: conv.groups)
    }

    func step(_ x: MLXArray) -> MLXArray {
        fishS1CausalTransConvStep(x, state: state, kernel: conv.kernel, bias: conv.bias, stride: stride, dilation: conv.dilation, groups: conv.groups)
    }

    func resetState() { state.overlap = nil }
}

// MARK: - Weight-normalised convs

/// Python WNConv1d (non-causal; used by the vector quantizers' 1x1 projections). Keys: weight_g, weight_v, bias.
final class FishS1WNConv1d: Module, UnaryLayer, FishS1WeightNormed {
    let stride: Int
    let padding: Int
    let dilation: Int
    let groups: Int

    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray?
    let folded = FishS1FoldedKernel()

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, padding: Int = 0, dilation: Int = 1, groups: Int = 1, bias: Bool = true) {
        self.stride = stride; self.padding = padding; self.dilation = dilation; self.groups = groups
        let inPerGroup = inChannels / groups
        let scale = Float(sqrt(1.0 / Double(inPerGroup * kernelSize)))
        let initial = MLXRandom.uniform(low: -scale, high: scale, [outChannels, inPerGroup, kernelSize])
        let g = fishS1NormalizeWeight(initial, exceptDim: 0)
        self.weight_g = g
        self.weight_v = initial / (g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    var kernel: MLXArray { folded.kernel ?? fishS1WeightNormKernel(g: weight_g, v: weight_v).transposed(0, 2, 1) }

    func fold(dtype: DType) -> MLXArray {
        let k = fishS1WeightNormKernel(g: weight_g, v: weight_v).transposed(0, 2, 1).asType(dtype)
        folded.kernel = k
        return k
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fishS1Conv1d(x, kernel: kernel, bias: bias, stride: stride, padding: padding, dilation: dilation, groups: groups)
    }
}

/// Python CausalWNConv1d. The reference nests CausalConvNet/Conv1dTorch and materialises the kernel into them; here the
/// causal logic is inline, so the parameters are exactly the checkpoint's weight_g, weight_v and bias.
final class FishS1CausalWNConv1d: Module, UnaryLayer, FishS1Steppable, FishS1Resettable, FishS1WeightNormed {
    let stride: Int
    let dilation: Int
    let groups: Int
    let kernelSize: Int
    let padding: Int

    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray?
    let folded = FishS1FoldedKernel()
    let state = FishS1ConvState()

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1, groups: Int = 1, bias: Bool = true) {
        self.stride = stride; self.dilation = dilation; self.groups = groups
        self.kernelSize = (kernelSize - 1) * dilation + 1
        self.padding = self.kernelSize - stride
        let inPerGroup = inChannels / groups
        let scale = Float(sqrt(1.0 / Double(inPerGroup * kernelSize)))
        let initial = MLXRandom.uniform(low: -scale, high: scale, [outChannels, inPerGroup, kernelSize])
        let g = fishS1NormalizeWeight(initial, exceptDim: 0)
        self.weight_g = g
        self.weight_v = initial / (g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    var kernel: MLXArray { folded.kernel ?? fishS1WeightNormKernel(g: weight_g, v: weight_v).transposed(0, 2, 1) }

    func fold(dtype: DType) -> MLXArray {
        let k = fishS1WeightNormKernel(g: weight_g, v: weight_v).transposed(0, 2, 1).asType(dtype)
        folded.kernel = k
        return k
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fishS1CausalConvForward(x, kernel: kernel, bias: bias, kernelSize: kernelSize, stride: stride, padding: padding, dilation: dilation, groups: groups)
    }

    func step(_ x: MLXArray) -> MLXArray {
        fishS1CausalConvStep(x, state: state, kernel: kernel, bias: bias, stride: stride, padding: padding, dilation: dilation, groups: groups)
    }

    func resetState() { state.buffer = nil }
}

/// Python CausalWNConvTranspose1d, flattened the same way as FishS1CausalWNConv1d.
final class FishS1CausalWNConvTranspose1d: Module, UnaryLayer, FishS1Steppable, FishS1Resettable, FishS1WeightNormed {
    let stride: Int
    let dilation: Int
    let groups: Int
    let kernelSize: Int

    var weight_g: MLXArray
    var weight_v: MLXArray
    var bias: MLXArray?
    let folded = FishS1FoldedKernel()
    let state = FishS1ConvState()

    init(inChannels: Int, outChannels: Int, kernelSize: Int, stride: Int = 1, dilation: Int = 1, groups: Int = 1, bias: Bool = true) {
        self.stride = stride; self.dilation = dilation; self.groups = groups; self.kernelSize = kernelSize
        let outPerGroup = outChannels / groups
        let scale = Float(sqrt(1.0 / Double(outPerGroup * kernelSize)))
        let initial = MLXRandom.uniform(low: -scale, high: scale, [inChannels, outPerGroup, kernelSize])
        let g = fishS1NormalizeWeight(initial, exceptDim: 0)
        self.weight_g = g
        self.weight_v = initial / (g + 1e-12)
        self.bias = bias ? MLXArray.zeros([outChannels]) : nil
    }

    /// PyTorch [in, out/groups, k] -> MLX [out, k, in/groups].
    var kernel: MLXArray { folded.kernel ?? fishS1WeightNormKernel(g: weight_g, v: weight_v).transposed(1, 2, 0) }

    func fold(dtype: DType) -> MLXArray {
        let k = fishS1WeightNormKernel(g: weight_g, v: weight_v).transposed(1, 2, 0).asType(dtype)
        folded.kernel = k
        return k
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        fishS1CausalTransConvForward(x, kernel: kernel, bias: bias, kernelSize: kernelSize, stride: stride, dilation: dilation, groups: groups)
    }

    func step(_ x: MLXArray) -> MLXArray {
        fishS1CausalTransConvStep(x, state: state, kernel: kernel, bias: bias, stride: stride, dilation: dilation, groups: groups)
    }

    func resetState() { state.overlap = nil }
}

// MARK: - Pointwise layers and blocks

final class FishS1Identity: Module, UnaryLayer {
    func callAsFunction(_ x: MLXArray) -> MLXArray { x }
}

final class FishS1Snake1d: Module, UnaryLayer {
    var alpha: MLXArray

    init(channels: Int) {
        self.alpha = MLXArray.ones([1, channels, 1])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        x + reciprocal(alpha + 1e-9) * pow(sin(alpha * x), 2)
    }
}

final class FishS1ResidualUnit: Module, UnaryLayer, FishS1Steppable {
    @ModuleInfo(key: "block") var block: [Module]

    init(dim: Int, dilation: Int = 1) {
        _block.wrappedValue = [
            FishS1Snake1d(channels: dim),
            FishS1CausalWNConv1d(inChannels: dim, outChannels: dim, kernelSize: 7, dilation: dilation),
            FishS1Snake1d(channels: dim),
            FishS1CausalWNConv1d(inChannels: dim, outChannels: dim, kernelSize: 1),
        ]
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var y = x
        for layer in block { y = fishS1Call(layer, y) }
        let pad = x.dim(2) - y.dim(2)
        let residual = pad > 0 ? x[.ellipsis, 0 ..< (x.dim(2) - pad)] : x
        return residual + y
    }

    func step(_ x: MLXArray) -> MLXArray {
        var y = x
        for layer in block { y = fishS1Step(layer, y) }
        return x + y
    }
}

final class FishS1ConvNeXtBlock: Module, UnaryLayer, FishS1Steppable {
    @ModuleInfo(key: "dwconv") var dwconv: FishS1CausalConvNet
    @ModuleInfo(key: "norm") var norm: LayerNorm
    @ModuleInfo(key: "pwconv1") var pwconv1: Linear
    @ModuleInfo(key: "pwconv2") var pwconv2: Linear
    let act = GELU()
    var gamma: MLXArray?

    init(dim: Int, layerScaleInitValue: Float = 1e-6, mlpRatio: Float = 4.0, kernelSize: Int = 7, dilation: Int = 1) {
        _dwconv.wrappedValue = FishS1CausalConvNet(inChannels: dim, outChannels: dim, kernelSize: kernelSize, dilation: dilation, groups: dim)
        _norm.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-6)
        let hidden = Int(mlpRatio * Float(dim))
        _pwconv1.wrappedValue = Linear(dim, hidden)
        _pwconv2.wrappedValue = Linear(hidden, dim)
        self.gamma = layerScaleInitValue > 0 ? MLXArray.ones([dim]) * layerScaleInitValue : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { pointwise(x, dwconv(x)) }

    func step(_ x: MLXArray) -> MLXArray { pointwise(x, dwconv.step(x)) }

    private func pointwise(_ input: MLXArray, _ convolved: MLXArray) -> MLXArray {
        var x = convolved.swappedAxes(1, 2)
        x = norm(x)
        x = pwconv1(x)
        x = act(x)
        x = pwconv2(x)
        if let gamma { x = gamma * x }
        return input + x.swappedAxes(1, 2)
    }
}
