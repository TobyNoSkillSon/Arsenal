// SPDX-License-Identifier: MIT
// CLAP HTSAT Fused (LAION): an HTSAT Swin audio encoder with attentional feature fusion, a RoBERTa text encoder
// and two projections into one 512-d space. Ported from transformers' modeling_clap.py and
// feature_extraction_clap.py (truncation "fusion", padding "repeatpad", 48 kHz, 64 HTK mel bins).
import Foundation
import MLX
import MLXNN
import Tokenizers

let modelKind = "local.audio-labels"
let modelOps = ["label"]

final class Model {
    let net: Clap
    let tokenizer: Tokenizer
    let mel: MelFeatures

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        net = Clap()
        var w = try Weights.load(weightsDir)
        for key in Array(w.keys) where key.hasSuffix(".weight") && w[key]!.ndim == 4 {
            w[key] = w[key]!.transposed(0, 2, 3, 1)  // Conv2d [out, in, kh, kw] -> [out, kh, kw, in]
        }
        try Weights.apply(w, to: net, rename: Model.rename)
        tokenizer = try Text.tokenizer(weightsDir)
        mel = MelFeatures()
    }

    static func rename(_ key: String) -> String? {
        if key.hasSuffix("position_ids") || key.hasSuffix("token_type_ids") || key.hasSuffix("relative_position_index") {
            return nil  // buffers recomputed here
        }
        // nn.Sequential indices in the fusion block read as list indices; name them.
        let seq: [String: String] = [
            "local_att.0.": "local_att.conv1.", "local_att.1.": "local_att.bn1.",
            "local_att.3.": "local_att.conv2.", "local_att.4.": "local_att.bn2.",
            "global_att.1.": "global_att.conv1.", "global_att.2.": "global_att.bn1.",
            "global_att.4.": "global_att.conv2.", "global_att.5.": "global_att.bn2.",
        ]
        for (from, to) in seq where key.contains(from) { return key.replacingOccurrences(of: from, with: to) }
        return key
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard let labels = params.raw["labels"] as? [String], !labels.isEmpty else {
            throw RunnerError.invalid("labels must be a nonempty list of short descriptions, e.g. [\"a dog barking\", \"music\"].")
        }
        guard labels.count <= 256 else { throw RunnerError.invalid("At most 256 labels per request.") }
        let clean = labels.map { $0.trimmingCharacters(in: .whitespacesAndNewlines) }
        guard clean.allSatisfy({ !$0.isEmpty }) else { throw RunnerError.invalid("Every label must be nonempty text.") }

        let wav = try AudioInput.mono(try params.file("audio"), sampleRate: 48_000)
        let samples = wav.asArray(Float.self)
        try params.checkCancelled()
        let features = try mel.fusionInput(samples)  // [4, 1001, 64] float32
        let audio = net.audioEmbedding(features)  // [1, 512] float32, unit length
        try params.checkCancelled()

        var textRows: [MLXArray] = []
        for label in clean {
            var ids = tokenizer.encode(text: label)
            if ids.count > 512 { ids = Array(ids.prefix(511)) + [ids.last!] }
            textRows.append(net.textEmbedding(ids))
            try params.checkCancelled()
        }
        let text = concatenated(textRows, axis: 0)  // [N, 512]
        let similarity = matmul(audio, text.transposed())[0]  // cosine
        let logits = similarity * exp(net.logitScaleA.asType(.float32))
        let probabilities = softmax(logits, axis: -1)
        let sims = list(similarity), probs = list(probabilities)
        var rows = (0..<labels.count).map { ["label": labels[$0], "score": probs[$0], "similarity": sims[$0]] as [String: Any] }
        rows.sort { ($0["score"] as! Double) > ($1["score"] as! Double) }
        var out: [String: Any] = ["labels": rows, "top": rows[0]["label"]!, "seconds": Double(samples.count) / 48_000]
        if params.bool("return_embedding", default: false) { out["audio_embedding"] = list(audio[0]) }
        return out
    }
}

// MARK: - Features (ClapFeatureExtractor, truncation "fusion", padding "repeatpad")

final class MelFeatures {
    let nFFT = 1024, hop = 480, maxSamples = 480_000, bins = 64
    let filters: MLXArray  // [513, 64]
    let window: MLXArray  // [1024]

    init() {
        func hz2mel(_ f: Double) -> Double { 2595.0 * log10(1.0 + f / 700.0) }
        func mel2hz(_ m: Double) -> Double { 700.0 * (pow(10.0, m / 2595.0) - 1.0) }
        let nFFT = 1024, bins = 64
        let nBins = nFFT / 2 + 1
        let mMin = hz2mel(50), mMax = hz2mel(14_000)
        let centers = (0..<(bins + 2)).map { mel2hz(mMin + (mMax - mMin) * Double($0) / Double(bins + 1)) }
        var fb = [Float](repeating: 0, count: nBins * bins)
        for k in 0..<nBins {
            let f = 24_000.0 * Double(k) / Double(nBins - 1)  // np.linspace(0, sr // 2, n)
            for m in 0..<bins {
                let down = (f - centers[m]) / (centers[m + 1] - centers[m])
                let up = (centers[m + 2] - f) / (centers[m + 2] - centers[m + 1])
                fb[k * bins + m] = Float(max(0, min(down, up)))
            }
        }
        filters = MLXArray(fb).reshaped([nBins, bins])
        window = MLXArray((0..<nFFT).map { Float(0.5 - 0.5 * cos(2.0 * Double.pi * Double($0) / Double(nFFT))) })
    }

    /// Log-mel in dB, [frames, 64], centred frames with reflect padding (transformers.audio_utils.spectrogram).
    func logMel(_ wave: [Float]) -> MLXArray {
        let half = nFFT / 2
        var padded = [Float](repeating: 0, count: wave.count + nFFT)
        for i in 0..<wave.count { padded[i + half] = wave[i] }
        for i in 0..<half {
            padded[half - 1 - i] = wave[min(i + 1, wave.count - 1)]
            padded[half + wave.count + i] = wave[max(wave.count - 2 - i, 0)]
        }
        let frames = 1 + (padded.count - nFFT) / hop
        let signal = MLXArray(padded)
        var parts: [MLXArray] = []
        let chunk = 2048
        var start = 0
        while start < frames {
            let n = min(chunk, frames - start)
            let offsets = MLXArray((0..<n).map { Int32((start + $0) * hop) }).reshaped([n, 1])
            let taps = MLXArray((0..<nFFT).map { Int32($0) }).reshaped([1, nFFT])
            let framed = signal[offsets + taps] * window  // [n, 1024]
            let spec = abs(rfft(framed, axis: -1))
            let power = spec * spec  // [n, 513]
            let melPower = maximum(matmul(power, filters), MLXArray(Float(1e-10)))
            parts.append(10.0 * log10(melPower))
            eval(parts.last!)
            start += n
        }
        return concatenated(parts, axis: 0)
    }

    /// The four mel channels the fused model takes, [4, 1001, 64].
    func fusionInput(_ wave: [Float]) throws -> MLXArray {
        guard !wave.isEmpty else { throw RunnerError.invalid("The audio file holds no samples.") }
        let chunkFrames = maxSamples / hop + 1
        if wave.count > maxSamples {
            let full = logMel(wave)
            let total = full.dim(0)
            if total == chunkFrames { return stacked([full, full, full, full]) }
            // _random_mel_fusion picks one random start in each third; this port takes the middle of each third.
            let starts = Array(0...(total - chunkFrames))
            let third = splitThree(starts)
            let picks = third.map { $0.isEmpty ? 0 : $0[$0.count / 2] }
            let shrink = bilinearTime(full, to: chunkFrames)
            let crops = picks.map { full[$0..<($0 + chunkFrames)] }
            return stacked([shrink] + crops)
        }
        var padded = wave
        if wave.count < maxSamples {
            let repeats = maxSamples / wave.count
            padded = Array([[Float]](repeating: wave, count: repeats).joined())
            padded += [Float](repeating: 0, count: maxSamples - padded.count)
        }
        let m = logMel(padded)
        return stacked([m, m, m, m])
    }

    /// np.array_split(list, 3)
    func splitThree(_ a: [Int]) -> [[Int]] {
        let n = a.count, base = n / 3, extra = n % 3
        var out: [[Int]] = [], i = 0
        for k in 0..<3 {
            let len = base + (k < extra ? 1 : 0)
            out.append(Array(a[i..<(i + len)])); i += len
        }
        return out
    }

    /// F.interpolate(mode="bilinear", align_corners=False) along time only (64 mel bins map to themselves).
    func bilinearTime(_ x: MLXArray, to out: Int) -> MLXArray {
        let n = x.dim(0)
        let scale = Double(n) / Double(out)
        var i0 = [Int32](), i1 = [Int32](), w1 = [Float]()
        for i in 0..<out {
            let src = max(0, (Double(i) + 0.5) * scale - 0.5)
            let a = min(Int(src), n - 1)
            let b = min(a + 1, n - 1)
            i0.append(Int32(a)); i1.append(Int32(b)); w1.append(Float(src - Double(a)))
        }
        let l = MLXArray(w1).reshaped([out, 1])
        return x[MLXArray(i0)] * (1 - l) + x[MLXArray(i1)] * l
    }
}

// MARK: - Model

final class Clap: Module {
    @ModuleInfo(key: "audio_model") var audioModel: AudioModel
    @ModuleInfo(key: "audio_projection") var audioProjection: Projection
    @ModuleInfo(key: "text_model") var textModel: TextModel
    @ModuleInfo(key: "text_projection") var textProjection: Projection
    @ParameterInfo(key: "logit_scale_a") var logitScaleA: MLXArray
    @ParameterInfo(key: "logit_scale_t") var logitScaleT: MLXArray

    override init() {
        _audioModel.wrappedValue = AudioModel()
        _audioProjection.wrappedValue = Projection(768)
        _textModel.wrappedValue = TextModel()
        _textProjection.wrappedValue = Projection(768)
        _logitScaleA.wrappedValue = MLXArray(Float(0))
        _logitScaleT.wrappedValue = MLXArray(Float(0))
    }

    func audioEmbedding(_ features: MLXArray) -> MLXArray {
        let pooled = audioModel.audioEncoder(features)  // [1, 768]
        return l2normalize(audioProjection(pooled).asType(.float32))
    }

    func textEmbedding(_ ids: [Int]) -> MLXArray {
        let pooled = textModel(MLXArray(ids.map { Int32($0) }).reshaped([1, ids.count]))
        return l2normalize(textProjection(pooled).asType(.float32))
    }
}

func l2normalize(_ x: MLXArray) -> MLXArray {
    x / maximum(sqrt((x * x).sum(axis: -1, keepDims: true)), MLXArray(Float(1e-12)))
}

final class Projection: Module, UnaryLayer {
    let linear1: Linear, linear2: Linear
    init(_ inDim: Int) { linear1 = Linear(inDim, 512); linear2 = Linear(512, 512) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { linear2(relu(linear1(x))) }
}

/// Eval-mode BatchNorm with running statistics, over the last axis.
final class BN: Module, UnaryLayer {
    var weight: MLXArray, bias: MLXArray
    @ParameterInfo(key: "running_mean") var runningMean: MLXArray
    @ParameterInfo(key: "running_var") var runningVar: MLXArray
    init(_ n: Int) {
        weight = MLXArray.ones([n]); bias = MLXArray.zeros([n])
        _runningMean.wrappedValue = MLXArray.zeros([n]); _runningVar.wrappedValue = MLXArray.ones([n])
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let scale = weight.asType(.float32) * rsqrt(runningVar.asType(.float32) + 1e-5)
        let shift = bias.asType(.float32) - runningMean.asType(.float32) * scale
        return (x.asType(.float32) * scale + shift).asType(x.dtype)
    }
}

final class AudioModel: Module {
    @ModuleInfo(key: "audio_encoder") var audioEncoder: AudioEncoder
    override init() { _audioEncoder.wrappedValue = AudioEncoder() }
}

final class AudioEncoder: Module {
    @ModuleInfo(key: "batch_norm") var batchNorm: BN
    @ModuleInfo(key: "patch_embed") var patchEmbed: PatchEmbed
    @ModuleInfo(key: "layers") var layers: [Stage]
    @ModuleInfo(key: "norm") var norm: LayerNorm

    override init() {
        _batchNorm.wrappedValue = BN(64)
        _patchEmbed.wrappedValue = PatchEmbed()
        let depths = [2, 2, 6, 2], heads = [4, 8, 16, 32]
        _layers.wrappedValue = (0..<4).map { i in
            Stage(dim: 96 << i, resolution: 64 >> i, depth: depths[i], heads: heads[i], downsample: i < 3)
        }
        _norm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-5)
    }

    /// features [4, 1001, 64] (channel, time, mel) -> pooled [1, 768]
    func callAsFunction(_ features: MLXArray) -> MLXArray {
        var x = batchNorm(features.asType(.float32))  // normalises each mel bin (last axis)
        x = bicubicTime(x, to: 1024)  // [4, 1024, 64]
        // reshape_mel2img: [1, 4, 1024, 64] -> [1, 16, 256, 64] -> [1, 16, 64, 256] -> [1, 4, 256, 256]
        x = x.reshaped([1, 16, 256, 64]).transposed(0, 1, 3, 2).reshaped([1, 4, 256, 256])
        x = x.transposed(0, 2, 3, 1).asType(.float16)  // channels-last [1, 256, 256, 4]
        var h = patchEmbed(x)  // [1, 4096, 96]
        for stage in layers { h = stage(h) }
        h = norm(h)
        return h.asType(.float32).mean(axis: 1)
    }

    /// F.interpolate(mode="bicubic", align_corners=True) along time; the mel axis keeps its size (identity).
    func bicubicTime(_ x: MLXArray, to out: Int) -> MLXArray {
        let n = x.dim(1)
        if n == out { return x }
        let a = -0.75
        func c1(_ t: Double) -> Double { ((a + 2) * t - (a + 3)) * t * t + 1 }
        func c2(_ t: Double) -> Double { ((a * t - 5 * a) * t + 8 * a) * t - 4 * a }
        let scale = Float(n - 1) / Float(out - 1)
        var m = [Float](repeating: 0, count: out * n)
        for i in 0..<out {
            let src = Double(scale * Float(i))
            let i0 = Int(floor(src))
            let t = src - Double(i0)
            let w = [c2(t + 1), c1(t), c1(1 - t), c2(2 - t)]
            for k in 0..<4 {
                let j = min(max(i0 - 1 + k, 0), n - 1)
                m[i * n + j] += Float(w[k])
            }
        }
        let matrix = MLXArray(m).reshaped([1, out, n])
        return matmul(matrix, x)  // [4, out, 64]
    }
}

final class PatchEmbed: Module {
    let proj: Conv2d
    let norm: LayerNorm
    @ModuleInfo(key: "mel_conv2d") var melConv2d: Conv2d
    @ModuleInfo(key: "fusion_model") var fusionModel: AFF

    override init() {
        proj = Conv2d(inputChannels: 1, outputChannels: 96, kernelSize: 4, stride: 4)
        norm = LayerNorm(dimensions: 96, eps: 1e-5)
        _melConv2d.wrappedValue = Conv2d(inputChannels: 1, outputChannels: 96, kernelSize: [4, 12], stride: [4, 12])
        _fusionModel.wrappedValue = AFF()
    }

    /// x [1, 256, 256, 4]; channel 0 is the global mel, 1-3 the local crops (is_longer = True).
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let global = proj(x[0..., 0..., 0..., 0..<1])  // [1, 64, 64, 96]
        let local = x[0, 0..., 0..., 1..<4].transposed(2, 0, 1).expandedDimensions(axis: -1)  // [3, 256, 256, 1]
        var l = melConv2d(local)  // [3, 64, 21, 96]
        let w = l.dim(2)
        l = l.transposed(1, 0, 2, 3).reshaped([1, 64, 3 * w, 96])  // width index = crop * 21 + column
        l = concatenated([l, MLXArray.zeros([1, 64, 64 - 3 * w, 96], dtype: l.dtype)], axis: 2)  // pad width to 64
        let fused = fusionModel(global, l)
        return norm(fused.reshaped([1, 64 * 64, 96]))
    }
}

/// Attentional feature fusion (2-D), BatchNorm in eval mode.
final class AFF: Module {
    @ModuleInfo(key: "local_att") var localAtt: AttBranch
    @ModuleInfo(key: "global_att") var globalAtt: AttBranch
    override init() { _localAtt.wrappedValue = AttBranch(); _globalAtt.wrappedValue = AttBranch() }
    func callAsFunction(_ x: MLXArray, _ residual: MLXArray) -> MLXArray {
        let input = x + residual
        let pooled = input.asType(.float32).mean(axes: [1, 2], keepDims: true).asType(input.dtype)
        let weight = sigmoid(localAtt(input) + globalAtt(pooled))
        return 2 * x * weight + 2 * residual * (1 - weight)
    }
}

final class AttBranch: Module, UnaryLayer {
    let conv1: Conv2d, bn1: BN, conv2: Conv2d, bn2: BN
    override init() {
        conv1 = Conv2d(inputChannels: 96, outputChannels: 24, kernelSize: 1); bn1 = BN(24)
        conv2 = Conv2d(inputChannels: 24, outputChannels: 96, kernelSize: 1); bn2 = BN(96)
    }
    func callAsFunction(_ x: MLXArray) -> MLXArray { bn2(conv2(relu(bn1(conv1(x))))) }
}

final class Stage: Module {
    let blocks: [SwinBlock]
    @ModuleInfo(key: "downsample") var downsample: PatchMerging?
    let resolution: Int

    init(dim: Int, resolution: Int, depth: Int, heads: Int, downsample: Bool) {
        self.resolution = resolution
        // set_shift_and_window_size: no shift once the map is no larger than the window.
        blocks = (0..<depth).map { i in
            SwinBlock(dim: dim, heads: heads, resolution: resolution, shift: (i % 2 == 1 && resolution > 8) ? 4 : 0)
        }
        _downsample.wrappedValue = downsample ? PatchMerging(dim: dim) : nil
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        var h = x
        for b in blocks { h = b(h) }
        if let downsample { h = downsample(h, resolution: resolution) }
        return h
    }
}

final class PatchMerging: Module {
    let reduction: Linear
    let norm: LayerNorm
    init(dim: Int) {
        reduction = Linear(4 * dim, 2 * dim, bias: false)
        norm = LayerNorm(dimensions: 4 * dim, eps: 1e-5)
    }
    func callAsFunction(_ x: MLXArray, resolution r: Int) -> MLXArray {
        let c = x.dim(2)
        let g = x.reshaped([1, r / 2, 2, r / 2, 2, c])  // [b, h2, row, w2, col, c]
        // torch.cat([x[:, row::2, col::2] for col in 0..1 for row in 0..1]): row varies fastest
        let parts = [(0, 0), (1, 0), (0, 1), (1, 1)].map { rc in g[0..., 0..., rc.0, 0..., rc.1, 0...] }
        let merged = concatenated(parts, axis: -1).reshaped([1, (r / 2) * (r / 2), 4 * c])
        return reduction(norm(merged))
    }
}

final class SwinBlock: Module {
    @ModuleInfo(key: "layernorm_before") var layernormBefore: LayerNorm
    @ModuleInfo(key: "attention") var attention: SwinAttention
    @ModuleInfo(key: "layernorm_after") var layernormAfter: LayerNorm
    @ModuleInfo(key: "intermediate") var intermediate: Dense
    @ModuleInfo(key: "output") var output: Dense

    let resolution: Int, shift: Int, window = 8
    let _mask: MLXArray  // [nW, 1, 64, 64] shift mask (underscore: not a parameter)

    init(dim: Int, heads: Int, resolution: Int, shift: Int) {
        self.resolution = resolution
        self.shift = shift
        _layernormBefore.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
        _attention.wrappedValue = SwinAttention(dim: dim, heads: heads)
        _layernormAfter.wrappedValue = LayerNorm(dimensions: dim, eps: 1e-5)
        _intermediate.wrappedValue = Dense(dim, 4 * dim)
        _output.wrappedValue = Dense(4 * dim, dim)
        _mask = SwinBlock.shiftMask(resolution: resolution, window: 8, shift: shift)
    }

    /// get_attn_mask: -100 between tokens from different cyclic-shift regions, [nW, 1, 64, 64].
    static func shiftMask(resolution r: Int, window ws: Int, shift s: Int) -> MLXArray {
        let nW = (r / ws) * (r / ws)
        guard s > 0 else { return MLXArray.zeros([nW, 1, ws * ws, ws * ws]) }
        func region(_ i: Int) -> Int { (i >= r - ws ? 1 : 0) + (i >= r - s ? 1 : 0) }
        var out = [Float](repeating: 0, count: nW * ws * ws * ws * ws)
        var w = 0
        for wh in 0..<(r / ws) {
            for ww in 0..<(r / ws) {
                var ids = [Int]()
                for y in 0..<ws { for x in 0..<ws { ids.append(region(wh * ws + y) * 3 + region(ww * ws + x)) } }
                for i in 0..<(ws * ws) {
                    for j in 0..<(ws * ws) where ids[i] != ids[j] {
                        out[((w * ws * ws) + i) * ws * ws + j] = -100
                    }
                }
                w += 1
            }
        }
        return MLXArray(out).reshaped([nW, 1, ws * ws, ws * ws])
    }

    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (r, ws, c) = (resolution, window, x.dim(2))
        var h = layernormBefore(x).reshaped([1, r, r, c])
        if shift > 0 { h = roll(roll(h, shift: -shift, axis: 1), shift: -shift, axis: 2) }
        // window_partition
        h = h.reshaped([1, r / ws, ws, r / ws, ws, c]).transposed(0, 1, 3, 2, 4, 5).reshaped([-1, ws * ws, c])
        h = attention(h, shiftMask: _mask)
        // window_reverse
        h = h.reshaped([1, r / ws, r / ws, ws, ws, c]).transposed(0, 1, 3, 2, 4, 5).reshaped([1, r, r, c])
        if shift > 0 { h = roll(roll(h, shift: shift, axis: 1), shift: shift, axis: 2) }
        let y = x + h.reshaped([1, r * r, c])
        return y + output(gelu(intermediate(layernormAfter(y))))
    }
}

final class Dense: Module, UnaryLayer {
    let dense: Linear
    init(_ i: Int, _ o: Int) { dense = Linear(i, o) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { dense(x) }
}

final class SwinAttention: Module {
    @ModuleInfo(key: "self") var selfAttention: SwinSelfAttention
    @ModuleInfo(key: "output") var output: Dense
    init(dim: Int, heads: Int) {
        _selfAttention.wrappedValue = SwinSelfAttention(dim: dim, heads: heads)
        _output.wrappedValue = Dense(dim, dim)
    }
    func callAsFunction(_ x: MLXArray, shiftMask: MLXArray) -> MLXArray { output(selfAttention(x, shiftMask: shiftMask)) }
}

final class SwinSelfAttention: Module {
    let query: Linear, key: Linear, value: Linear
    @ParameterInfo(key: "relative_position_bias_table") var table: MLXArray
    let heads: Int, headDim: Int
    let _index: MLXArray  // [64 * 64] relative_position_index (underscore: not a parameter)

    init(dim: Int, heads: Int) {
        self.heads = heads
        headDim = dim / heads
        query = Linear(dim, dim); key = Linear(dim, dim); value = Linear(dim, dim)
        _table.wrappedValue = MLXArray.zeros([225, heads])
        var idx = [Int32]()
        for i in 0..<64 {
            for j in 0..<64 {
                let dh = (i / 8) - (j / 8) + 7, dw = (i % 8) - (j % 8) + 7
                idx.append(Int32(dh * 15 + dw))
            }
        }
        _index = MLXArray(idx)
    }

    func callAsFunction(_ x: MLXArray, shiftMask: MLXArray) -> MLXArray {
        let (b, n, c) = (x.dim(0), x.dim(1), x.dim(2))
        let q = query(x).reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
        let k = key(x).reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
        let v = value(x).reshaped([b, n, heads, headDim]).transposed(0, 2, 1, 3)
        let bias = table[_index].reshaped([n, n, heads]).transposed(2, 0, 1).expandedDimensions(axis: 0)  // [1, H, 64, 64]
        let mask = (bias.asType(.float32) + shiftMask).asType(x.dtype)
        let o = scaledDotProductAttention(queries: q, keys: k, values: v, scale: 1 / Float(headDim).squareRoot(), mask: mask)
        return o.transposed(0, 2, 1, 3).reshaped([b, n, c])
    }
}

// MARK: - Text (RoBERTa)

final class TextModel: Module {
    let embeddings: TextEmbeddings
    let encoder: TextEncoder
    let pooler: Dense
    override init() { embeddings = TextEmbeddings(); encoder = TextEncoder(); pooler = Dense(768, 768) }
    /// ids [1, L] -> pooled first token [1, 768]
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        var h = embeddings(ids)
        for layer in encoder.layer { h = layer(h) }
        return tanh(pooler(h[0..., 0, 0...]))
    }
}

final class TextEmbeddings: Module {
    @ModuleInfo(key: "word_embeddings") var wordEmbeddings: Embedding
    @ModuleInfo(key: "position_embeddings") var positionEmbeddings: Embedding
    @ModuleInfo(key: "token_type_embeddings") var tokenTypeEmbeddings: Embedding
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    override init() {
        _wordEmbeddings.wrappedValue = Embedding(embeddingCount: 50265, dimensions: 768)
        _positionEmbeddings.wrappedValue = Embedding(embeddingCount: 514, dimensions: 768)
        _tokenTypeEmbeddings.wrappedValue = Embedding(embeddingCount: 1, dimensions: 768)
        _layerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-12)
    }
    func callAsFunction(_ ids: MLXArray) -> MLXArray {
        let l = ids.dim(1)
        // create_position_ids_from_input_ids with no padding: padding_idx (1) + 1 ... L + 1
        let positions = MLXArray((0..<l).map { Int32($0 + 2) }).reshaped([1, l])
        let x = wordEmbeddings(ids) + tokenTypeEmbeddings(MLXArray.zeros([1, l], dtype: .int32)) + positionEmbeddings(positions)
        return layerNorm(x)
    }
}

final class TextEncoder: Module {
    let layer: [TextLayer]
    override init() { layer = (0..<12).map { _ in TextLayer() } }
}

final class TextLayer: Module {
    let attention: TextAttention
    let intermediate: Dense
    let output: AddNorm
    override init() { attention = TextAttention(); intermediate = Dense(768, 3072); output = AddNorm(3072) }
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let a = attention(x)
        return output(gelu(intermediate(a)), residual: a)
    }
}

final class AddNorm: Module {
    let dense: Linear
    @ModuleInfo(key: "LayerNorm") var layerNorm: LayerNorm
    init(_ inDim: Int) {
        dense = Linear(inDim, 768)
        _layerNorm.wrappedValue = LayerNorm(dimensions: 768, eps: 1e-12)
    }
    func callAsFunction(_ x: MLXArray, residual: MLXArray) -> MLXArray { layerNorm(dense(x) + residual) }
}

final class TextAttention: Module {
    @ModuleInfo(key: "self") var selfAttention: TextSelfAttention
    let output: AddNorm
    override init() { _selfAttention.wrappedValue = TextSelfAttention(); output = AddNorm(768) }
    func callAsFunction(_ x: MLXArray) -> MLXArray { output(selfAttention(x), residual: x) }
}

final class TextSelfAttention: Module {
    let query = Linear(768, 768), key = Linear(768, 768), value = Linear(768, 768)
    func callAsFunction(_ x: MLXArray) -> MLXArray {
        let (b, l) = (x.dim(0), x.dim(1))
        let q = query(x).reshaped([b, l, 12, 64]).transposed(0, 2, 1, 3)
        let k = key(x).reshaped([b, l, 12, 64]).transposed(0, 2, 1, 3)
        let v = value(x).reshaped([b, l, 12, 64]).transposed(0, 2, 1, 3)
        let o = scaledDotProductAttention(queries: q, keys: k, values: v, scale: 0.125, mask: nil)
        return o.transposed(0, 2, 1, 3).reshaped([b, l, 768])
    }
}
