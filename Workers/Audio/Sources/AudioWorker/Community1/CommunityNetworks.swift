// SPDX-License-Identifier: MIT
// Community-1 networks, following pyannote.audio 4.0.0 (PyanNet, SincNet, WeSpeakerResNet34, StatsPool) and
// asteroid-filterbanks ParamSincFB, in MLX's channel-last layouts. Learned weights and activations are half;
// normalisation statistics, sinc filter synthesis and pooling statistics are Float32/Double DSP intermediates.
import Foundation
import IntegrationKit
import MLX
import MLXNN

/// InstanceNorm1d(affine=True): per-sample, per-channel statistics over time, biased variance, eps 1e-5.
final class C1InstanceNorm: Module {
    var weight: MLXArray
    var bias: MLXArray
    init(_ channels: Int) {
        weight = MLXArray.ones([channels])
        bias = MLXArray.zeros([channels])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let full = x.asType(.float32)
        let mean = full.mean(axis: 1, keepDims: true), variance = full.variance(axis: 1, keepDims: true)
        return ((full - mean) * rsqrt(variance + 1e-5)).asType(x.dtype) * weight + bias
    }
}

/// ParamSincFB(80 filters, 251 taps, stride 10, min_low_hz 50, min_band_hz 50). The half window and time
/// vector are deterministic buffers recomputed here, not learned parameters.
final class C1Sinc: Module {
    var low_hz_: MLXArray
    var band_hz_: MLXArray
    static let halfKernel = 125
    override init() {
        // Mel-spaced initialisation; trained values replace these on Load.
        func toMel(_ hz: Double) -> Double { 2595 * log10(1 + hz / 700) }
        func toHz(_ mel: Double) -> Double { 700 * (pow(10, mel / 2595) - 1) }
        let low = toMel(30), high = toMel(8000 - 100)
        let hz = (0...40).map { toHz(low + (high - low) * Double($0) / 40) }
        low_hz_ = MLXArray((0..<40).map { Float(hz[$0]) }, [40, 1])
        band_hz_ = MLXArray((0..<40).map { Float(hz[$0 + 1] - hz[$0]) }, [40, 1])
    }
    /// [80, 251, 1] filters: 40 cosine (even) then 40 sine (odd) band-passes, normalised by 2 × bandwidth.
    func filters(dtype: DType) -> MLXArray {
        let lows = low_hz_.asType(.float32).asArray(Float.self), bands = band_hz_.asType(.float32).asArray(Float.self)
        let half = Self.halfKernel, taps = 2 * half + 1
        // np.hamming(251)[:125] and 2π·arange(-125, 0)/16000, as registered by ParamSincFB.
        let window = (0..<half).map { 0.54 - 0.46 * cos(2 * Double.pi * Double($0) / Double(taps - 1)) }
        let n = (0..<half).map { 2 * Double.pi * Double($0 - half) / 16_000 }
        var values = [Float](repeating: 0, count: 80 * taps)
        for filter in 0..<40 {
            let low = 50 + abs(Double(lows[filter]))
            let high = min(max(low + 50 + abs(Double(bands[filter])), 50), 8000)
            let band = high - low
            for i in 0..<half {
                let even = (sin(high * n[i]) - sin(low * n[i])) / (n[i] / 2) * window[i]
                let odd = (cos(low * n[i]) - cos(high * n[i])) / (n[i] / 2) * window[i]
                values[filter * taps + i] = Float(even / (2 * band))
                values[filter * taps + taps - 1 - i] = Float(even / (2 * band))
                values[(40 + filter) * taps + i] = Float(odd / (2 * band))
                values[(40 + filter) * taps + taps - 1 - i] = Float(-odd / (2 * band))
            }
            values[filter * taps + half] = 1
            values[(40 + filter) * taps + half] = 0
        }
        return MLXArray(values, [80, taps, 1]).asType(dtype)
    }
}

/// One direction of one torch.nn.LSTM layer (gate order i, f, g, o; both biases), batched over chunks.
final class C1Recurrent: Module {
    @ModuleInfo(key: "x_proj") var inputProjection: Linear
    @ModuleInfo(key: "h_proj") var recurrentProjection: Linear
    init(_ input: Int, _ hidden: Int) {
        _inputProjection.wrappedValue = Linear(input, 4 * hidden)
        _recurrentProjection.wrappedValue = Linear(hidden, 4 * hidden)
    }
    func callAsFunction(_ x: MLXArray, reverse: Bool, cancellation: NativeCancellation) throws -> MLXArray {
        let projected = inputProjection(x), steps = x.dim(1)
        let hidden = recurrentProjection.weight.dim(0) / 4
        var h = MLXArray.zeros([x.dim(0), hidden], dtype: x.dtype), c = h
        var outputs = [MLXArray](repeating: h, count: steps)
        for step in 0..<steps {
            if step % 64 == 0 {
                try cancellation.check()
                eval(h, c)
            }
            let t = reverse ? steps - 1 - step : step
            let gates = split(projected[0..., t, 0...] + recurrentProjection(h), parts: 4, axis: -1)
            c = sigmoid(gates[1]) * c + sigmoid(gates[0]) * tanh(gates[2])
            h = sigmoid(gates[3]) * tanh(c)
            outputs[t] = h
        }
        return stacked(outputs, axis: 1)
    }
}

/// PyanNet: SincNet → 4-layer BiLSTM(128) → 2 × Linear(128) with leaky ReLU → 7-class powerset log-softmax.
final class C1Segmentation: Module {
    let wav: C1InstanceNorm
    let sinc: C1Sinc
    @ModuleInfo var conv: [Conv1d]
    let norm: [C1InstanceNorm]
    let forward: [C1Recurrent]
    let backward: [C1Recurrent]
    @ModuleInfo var linear: [Linear]
    @ModuleInfo var classifier: Linear
    init(layers: Int = 4) {
        wav = C1InstanceNorm(1)
        sinc = C1Sinc()
        _conv.wrappedValue = [Conv1d(inputChannels: 80, outputChannels: 60, kernelSize: 5), Conv1d(inputChannels: 60, outputChannels: 60, kernelSize: 5)]
        norm = [C1InstanceNorm(80), C1InstanceNorm(60), C1InstanceNorm(60)]
        forward = (0..<layers).map { C1Recurrent($0 == 0 ? 60 : 256, 128) }
        backward = (0..<layers).map { C1Recurrent($0 == 0 ? 60 : 256, 128) }
        _linear.wrappedValue = [Linear(256, 128), Linear(128, 128)]
        _classifier.wrappedValue = Linear(128, 7)
    }
    /// [B, samples, 1] half waveform → [B, frames, 7] powerset log-probabilities.
    func callAsFunction(_ audio: MLXArray, filters: MLXArray, cancellation: NativeCancellation) throws -> MLXArray {
        // MaxPool1d(3, stride 3) over time.
        func pool(_ x: MLXArray) -> MLXArray {
            let n = (x.dim(1) - 3) / 3 + 1
            let trimmed = x[0..., 0..<(n * 3), 0...]
            return trimmed.reshaped(x.dim(0), n, 3, x.dim(2)).max(axis: 2)
        }
        var x = leakyRelu(norm[0](pool(abs(conv1d(wav(audio), filters, stride: 10)))))
        for i in 0..<2 { x = leakyRelu(norm[i + 1](pool(conv[i](x)))) }
        eval(x)
        for layer in forward.indices {
            let a = try forward[layer](x, reverse: false, cancellation: cancellation)
            let b = try backward[layer](x, reverse: true, cancellation: cancellation)
            x = concatenated([a, b], axis: -1)
            eval(x)
        }
        for layer in linear { x = leakyRelu(layer(x)) }
        return logSoftmax(classifier(x), axis: -1)
    }
}

/// BatchNorm in inference mode with the stored running statistics, eps 1e-5.
final class C1BatchNorm: Module {
    var weight: MLXArray
    var bias: MLXArray
    var running_mean: MLXArray
    var running_var: MLXArray
    init(_ n: Int) {
        weight = MLXArray.ones([n])
        bias = MLXArray.zeros([n])
        running_mean = MLXArray.zeros([n])
        running_var = MLXArray.ones([n])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { (x - running_mean) * rsqrt(running_var + 1e-5) * weight + bias }
}

final class C1Block: Module {
    @ModuleInfo var conv1: Conv2d
    @ModuleInfo var conv2: Conv2d
    let bn1: C1BatchNorm
    let bn2: C1BatchNorm
    @ModuleInfo var shortcut: Conv2d?
    let shortcutBN: C1BatchNorm?
    init(_ input: Int, _ output: Int, stride: Int) {
        _conv1.wrappedValue = Conv2d(inputChannels: input, outputChannels: output, kernelSize: 3, stride: IntOrPair(integerLiteral: stride), padding: 1, bias: false)
        _conv2.wrappedValue = Conv2d(inputChannels: output, outputChannels: output, kernelSize: 3, padding: 1, bias: false)
        bn1 = C1BatchNorm(output)
        bn2 = C1BatchNorm(output)
        if stride != 1 || input != output {
            _shortcut.wrappedValue = Conv2d(inputChannels: input, outputChannels: output, kernelSize: 1, stride: IntOrPair(integerLiteral: stride), bias: false)
            shortcutBN = C1BatchNorm(output)
        } else {
            _shortcut.wrappedValue = nil
            shortcutBN = nil
        }
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let residual: MLXArray
        if let shortcut, let shortcutBN { residual = shortcutBN(shortcut(x)) } else { residual = x }
        return relu(bn2(conv2(relu(bn1(conv1(x))))) + residual)
    }
}

/// WeSpeaker ResNet34 (32 base channels, TSTP statistics pooling, one 256-d embedding layer).
final class C1Embedding: Module {
    @ModuleInfo var conv1: Conv2d
    let bn1: C1BatchNorm
    let layer1: [C1Block]
    let layer2: [C1Block]
    let layer3: [C1Block]
    let layer4: [C1Block]
    @ModuleInfo var seg_1: Linear
    init(blocks: [Int] = [3, 4, 6, 3]) {
        _conv1.wrappedValue = Conv2d(inputChannels: 1, outputChannels: 32, kernelSize: 3, padding: 1, bias: false)
        bn1 = C1BatchNorm(32)
        func stage(_ count: Int, _ input: Int, _ output: Int, _ stride: Int) -> [C1Block] {
            (0..<count).map { C1Block($0 == 0 ? input : output, output, stride: $0 == 0 ? stride : 1) }
        }
        layer1 = stage(blocks[0], 32, 32, 1)
        layer2 = stage(blocks[1], 32, 64, 2)
        layer3 = stage(blocks[2], 64, 128, 2)
        layer4 = stage(blocks[3], 128, 256, 2)
        _seg_1.wrappedValue = Linear(5120, 256)
    }
    /// [1, T, 80] fbank → [1, T', 2560] frame features, flattened as (channel, frequency) like TSTP's rearrange.
    func frames(_ fbank: MLXArray, cancellation: NativeCancellation) throws -> MLXArray {
        // torch (B, 1, F, T) is MLX (B, F, T, 1).
        var x = relu(bn1(conv1(fbank.transposed(0, 2, 1).expandedDimensions(axis: -1))))
        for stage in [layer1, layer2, layer3, layer4] {
            for block in stage {
                try cancellation.check()
                x = block(x)
            }
            eval(x)
        }
        // (B, F', T', C) → (B, T', C, F') → (B, T', C·F')
        return x.transposed(0, 2, 3, 1).reshaped(x.dim(0), x.dim(2), -1)
    }
    /// StatsPool with per-speaker weights (nearest-interpolated to T'): weighted mean and unbiased weighted std.
    /// weights: [S, T'] Float32 → [S, 256] embeddings.
    func embed(_ frames: MLXArray, weights: MLXArray) -> MLXArray {
        let x = frames.asType(.float32), w = weights.expandedDimensions(axis: -1)
        let v1 = w.sum(axis: 1) + 1e-8, v2 = (w * w).sum(axis: 1)
        let mean = (x * w).sum(axis: 1) / v1
        let delta = x - mean.expandedDimensions(axis: 1)
        let variance = (delta * delta * w).sum(axis: 1) / (v1 - v2 / v1 + 1e-8)
        return seg_1(concatenated([mean, sqrt(variance)], axis: -1).asType(frames.dtype))
    }
}

/// Stored PLDA/x-vector transform (xvec_transform.npz and plda.npz). Host clustering reads these half values;
/// they are not network matrices and are never packed.
final class C1PLDAWeights: Module {
    var mean1: MLXArray
    var mean2: MLXArray
    var lda: MLXArray
    var mu: MLXArray
    var tr: MLXArray
    var psi: MLXArray
    override init() {
        // Well-conditioned placeholders for random-weight lab runs; trained values replace them on Load.
        mean1 = MLXArray.zeros([256])
        mean2 = MLXArray.zeros([128])
        lda = MLXArray((0..<(256 * 128)).map { Float(sin(Double($0) * 0.7)) * 0.06 }, [256, 128])
        mu = MLXArray.zeros([128])
        tr = MLXArray((0..<(128 * 128)).map { $0 / 128 == $0 % 128 ? Float(1) : Float(0) }, [128, 128])
        psi = MLXArray((0..<128).map { Float(128 - $0) / 16 })
    }
    func value() -> Community1PLDA {
        func vector(_ x: MLXArray) -> [Double] { x.asType(.float32).asArray(Float.self).map(Double.init) }
        let ldaValues = vector(lda), trValues = vector(tr)
        return Community1PLDA(
            mean1: vector(mean1), mean2: vector(mean2), lda: (0..<256).map { Array(ldaValues[($0 * 128)..<(($0 + 1) * 128)]) },
            mu: vector(mu), tr: (0..<128).map { Array(trValues[($0 * 128)..<(($0 + 1) * 128)]) }, psi: vector(psi))
    }
}

final class C1Network: Module {
    let segmentation: C1Segmentation
    let embedding: C1Embedding
    let plda: C1PLDAWeights
    init(layers: Int = 4, blocks: [Int] = [3, 4, 6, 3]) {
        segmentation = C1Segmentation(layers: layers)
        embedding = C1Embedding(blocks: blocks)
        plda = C1PLDAWeights()
    }
}
