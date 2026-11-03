// SPDX-License-Identifier: MIT
// Port of JaesungHuh/voice-gender-classifier (MIT) to MLX. The maker's model.py (ECAPA-TDNN with a two-class head,
// derived from TaoRuijie/ECAPA-TDNN) and its log-mel front end (torchaudio MelSpectrogram, 16 kHz, n_fft 512,
// win 400, hop 160, 80 mels, 20-7600 Hz, Hamming window, centre padding, pre-emphasis 0.97).
// The front end runs in float32; the network runs in float16 (the weights' dtype).
import Foundation
import MLX
import MLXFFT
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "local.gender"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["classify"]

final class Model {
    let net: ECAPAGender

    /// Called once by Load. `weightsDir` holds the pinned files (config.json, model.safetensors) as 16-bit.
    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        net = ECAPAGender(channels: 1024)
        var weights = try Weights.load(weightsDir)
        // PyTorch Conv1d weights [out, in, k] become MLX [out, k, in]. Linear and BatchNorm tensors load unchanged.
        for (key, value) in weights where value.ndim == 3 { weights[key] = value.transposed(0, 2, 1) }
        try Weights.apply(weights, to: net, rename: Model.rename)
    }

    /// Checkpoint keys whose Sequential index or class is not a Swift property name, mapped to the Swift names.
    static func rename(_ key: String) -> String? {
        var name = key
        for (old, new) in [
            ("attention.0.", "attention.conv0."), ("attention.2.", "attention.bn2."), ("attention.4.", "attention.conv4."),
            (".se.se.1.", ".se.se.reduce."), (".se.se.3.", ".se.se.expand."),
        ] {
            name = name.replacingOccurrences(of: old, with: new)
        }
        return name
    }

    /// One request: a recording in, the gender label and its probabilities out.
    func run(op: String, params: Params) throws -> [String: Any] {
        let url = try params.file("audio")
        let samples = try Audio.load(url, sampleRate: 16000)
        let features = try LogMel.features(samples)  // [frames, 80] float32
        let logits = net(features.expandedDimensions(axis: 0).asType(.float16)).asType(.float32)
        let probabilities = list(softmax(logits, axis: -1)[0])
        let male = probabilities[0], female = probabilities[1]
        return [
            "label": female >= male ? "female" : "male",
            "probabilities": ["male": male, "female": female],
            "duration_s": Double(samples.dim(0)) / 16000,
        ]
    }
}

/// The maker's front end: pre-emphasis, centre-padded STFT, 80-band mel filterbank, log, mean over time.
enum LogMel {
    static let nFFT = 512, winLength = 400, hop = 160, nMels = 80, nFreqs = 257
    static let sampleRate = 16000.0

    /// samples: mono float32 at 16 kHz. Returns [frames, 80] log-mel features with the per-bin time mean removed.
    static func features(_ x: MLXArray) throws -> MLXArray {
        let t = x.dim(0)
        guard t >= 512 else { throw RunnerError.invalid("The recording must be at least 512 samples (0.03 s).") }

        // Pre-emphasis: y[n] = x[n] - 0.97 x[n-1], with x[-1] = x[1] (the maker pads by reflection).
        let previous = concatenated([x[1..<2], x[0 ..< (t - 1)]])
        let emphasized = x - previous * 0.97

        // torch.stft(center=True): reflect-pad n_fft/2 on both sides.
        let pad = nFFT / 2
        var reflected = [Int32]()
        reflected.reserveCapacity(t + 2 * pad)
        for j in 0..<(t + 2 * pad) {
            var source = j - pad
            if source < 0 { source = -source }
            if source >= t { source = 2 * (t - 1) - source }
            reflected.append(Int32(source))
        }
        let padded = take(emphasized, MLXArray(reflected), axis: 0)

        // Frames start every `hop` samples: 1 + t / hop frames, as torch counts them.
        let frames = 1 + t / hop
        var frameIndex = [Int32]()
        frameIndex.reserveCapacity(frames * nFFT)
        for f in 0..<frames { for n in 0..<nFFT { frameIndex.append(Int32(f * hop + n)) } }
        let framed = take(padded, MLXArray(frameIndex, [frames, nFFT]), axis: 0)

        // Periodic Hamming window of 400 samples, centred in the 512-point FFT.
        var window = [Float](repeating: 0, count: nFFT)
        let left = (nFFT - winLength) / 2
        for n in 0..<winLength {
            window[left + n] = Float(0.54 - 0.46 * cos(2 * Double.pi * Double(n) / Double(winLength)))
        }
        let spectrum = MLXFFT.rfft(framed * MLXArray(window), n: nFFT, axis: -1)  // [frames, 257] complex
        let magnitude = MLX.abs(spectrum)
        let power = magnitude * magnitude

        let mel = matmul(power, melFilterbank())  // [frames, 80]
        let logMel = MLX.log(mel + 1e-6)
        return logMel - mean(logMel, axis: 0, keepDims: true)
    }

    /// torchaudio.functional.melscale_fbanks(257, 20, 7600, 80, 16000, norm=None, mel_scale="htk"), as [257, 80].
    static func melFilterbank() -> MLXArray {
        func hzToMel(_ f: Double) -> Double { 2595 * log10(1 + f / 700) }
        func melToHz(_ m: Double) -> Double { 700 * (pow(10, m / 2595) - 1) }
        let melMin = hzToMel(20), melMax = hzToMel(7600)
        let points = (0..<(nMels + 2)).map { melToHz(melMin + (melMax - melMin) * Double($0) / Double(nMels + 1)) }
        let freqs = (0..<nFreqs).map { Double($0) * (sampleRate / 2) / Double(nFreqs - 1) }
        var bank = [Float](repeating: 0, count: nFreqs * nMels)
        for k in 0..<nFreqs {
            for m in 0..<nMels {
                let rising = (freqs[k] - points[m]) / (points[m + 1] - points[m])
                let falling = (points[m + 2] - freqs[k]) / (points[m + 2] - points[m + 1])
                bank[k * nMels + m] = Float(max(0, min(rising, falling)))
            }
        }
        return MLXArray(bank, [nFreqs, nMels])
    }
}

/// BatchNorm for [N, L, C] (channels last), in inference mode: the stored running statistics.
/// Property names are the checkpoint's keys.
final class BN: Module, UnaryLayer {
    var weight: MLXArray
    var bias: MLXArray
    var running_mean: MLXArray
    var running_var: MLXArray

    init(_ features: Int) {
        weight = MLXArray.ones([features])
        bias = MLXArray.zeros([features])
        running_mean = MLXArray.zeros([features])
        running_var = MLXArray.ones([features])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        (x - running_mean) * rsqrt(running_var + 1e-5) * weight + bias
    }
}

/// Squeeze-and-excitation gate over time: pool, 1x1 conv down, ReLU, 1x1 conv up, sigmoid. Checkpoint keys se.se.1 and se.se.3 are renamed in Model.rename.
final class SEGate: Module, UnaryLayer {
    var reduce: Conv1d
    var expand: Conv1d

    init(channels: Int, bottleneck: Int) {
        reduce = Conv1d(inputChannels: channels, outputChannels: bottleneck, kernelSize: 1)
        expand = Conv1d(inputChannels: bottleneck, outputChannels: channels, kernelSize: 1)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let pooled = mean(x, axis: 1, keepDims: true)  // [N, 1, C]
        return x * sigmoid(expand(relu(reduce(pooled))))
    }
}

final class SEModule: Module, UnaryLayer {
    var se: SEGate

    init(channels: Int, bottleneck: Int) {
        se = SEGate(channels: channels, bottleneck: bottleneck)
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray { se(x) }
}

/// The maker's Bottle2neck: a Res2Net-style block with `scale` splits and an SE gate. Keys match the checkpoint.
final class Bottle2neck: Module, UnaryLayer {
    let width: Int
    let scale: Int
    var conv1: Conv1d
    var bn1: BN
    var convs: [Conv1d]
    var bns: [BN]
    var conv3: Conv1d
    var bn3: BN
    var se: SEModule

    init(channels: Int, kernel: Int, dilation: Int, scale: Int = 8) {
        let w = channels / scale
        let pad = (kernel / 2) * dilation
        width = w
        self.scale = scale
        conv1 = Conv1d(inputChannels: channels, outputChannels: w * scale, kernelSize: 1)
        bn1 = BN(w * scale)
        convs = (0..<(scale - 1)).map { _ in
            Conv1d(inputChannels: w, outputChannels: w, kernelSize: kernel, padding: pad, dilation: dilation)
        }
        bns = (0..<(scale - 1)).map { _ in BN(w) }
        conv3 = Conv1d(inputChannels: w * scale, outputChannels: channels, kernelSize: 1)
        bn3 = BN(channels)
        se = SEModule(channels: channels, bottleneck: 128)
    }

    func callAsFunction(_ input: MLXArray) -> MLXArray {
        let nums = scale - 1
        var out = bn1(relu(conv1(input)))
        // torch.split(out, width, dim=channels)
        let pieces = (0..<scale).map { i in out[0..., 0..., (i * width)..<((i + 1) * width)] }
        var sp = pieces[0]
        var outs: [MLXArray] = []
        for i in 0..<nums {
            if i > 0 { sp = sp + pieces[i] }
            sp = bns[i](relu(convs[i](sp)))
            outs.append(sp)
        }
        outs.append(pieces[nums])
        out = concatenated(outs, axis: -1)
        out = bn3(relu(conv3(out)))
        out = se(out)
        return out + input
    }
}

/// Attention pooling head. Checkpoint keys attention.0, .2 and .4 (a Sequential) are renamed in Model.rename.
final class AttentionPool: Module, UnaryLayer {
    var conv0: Conv1d
    var bn2: BN
    var conv4: Conv1d

    init(inChannels: Int, hidden: Int, outChannels: Int) {
        conv0 = Conv1d(inputChannels: inChannels, outputChannels: hidden, kernelSize: 1)
        bn2 = BN(hidden)
        conv4 = Conv1d(inputChannels: hidden, outputChannels: outChannels, kernelSize: 1)
    }

    /// Softmax over time (axis 1), computed in float32.
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let hidden = tanh(bn2(relu(conv0(x))))
        return softmax(conv4(hidden).asType(.float32), axis: 1)
    }
}

/// ECAPA-TDNN with a two-class head (male, female). Input [N, frames, 80] log-mel; output [N, 2] logits.
final class ECAPAGender: Module, UnaryLayer {
    var conv1: Conv1d
    var bn1: BN
    var layer1: Bottle2neck
    var layer2: Bottle2neck
    var layer3: Bottle2neck
    var layer4: Conv1d
    var attention: AttentionPool
    var bn5: BN
    var fc6: Linear
    var bn6: BN
    var fc7: Linear

    init(channels c: Int) {
        conv1 = Conv1d(inputChannels: 80, outputChannels: c, kernelSize: 5, padding: 2)
        bn1 = BN(c)
        layer1 = Bottle2neck(channels: c, kernel: 3, dilation: 2)
        layer2 = Bottle2neck(channels: c, kernel: 3, dilation: 3)
        layer3 = Bottle2neck(channels: c, kernel: 3, dilation: 4)
        layer4 = Conv1d(inputChannels: 3 * c, outputChannels: 1536, kernelSize: 1)
        attention = AttentionPool(inChannels: 3 * 1536, hidden: 256, outChannels: 1536)
        bn5 = BN(3072)
        fc6 = Linear(3072, 192)
        bn6 = BN(192)
        fc7 = Linear(192, 2)
    }

    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = bn1(relu(conv1(features)))
        let x1 = layer1(x)
        let x2 = layer2(x + x1)
        let x3 = layer3(x + x1 + x2)
        x = relu(layer4(concatenated([x1, x2, x3], axis: -1)))  // [N, T, 1536]

        // Global context: frame features, then their time mean and std (unbiased), repeated over time.
        let n = x.dim(0), t = x.dim(1)
        let xf = x.asType(.float32)
        let mu = mean(xf, axis: 1, keepDims: true)
        let variance = sum(square(xf - mu), axis: 1, keepDims: true) / Float(t - 1)
        let sd = sqrt(maximum(variance, MLXArray(Float(1e-4))))
        let ones = MLXArray.ones([n, t, 1536])
        let global = concatenated(
            [x, (ones * mu).asType(.float16), (ones * sd).asType(.float16)], axis: -1)  // [N, T, 4608]

        let w = attention(global)  // [N, T, 1536], float32
        let muPooled = sum(xf * w, axis: 1)
        let sgPooled = sqrt(maximum(sum(square(xf) * w, axis: 1) - square(muPooled), MLXArray(Float(1e-4))))
        var pooled = concatenated([muPooled, sgPooled], axis: -1).asType(.float16)  // [N, 3072]

        pooled = bn5(pooled)
        pooled = relu(bn6(fc6(pooled)))
        return fc7(pooled)
    }
}
