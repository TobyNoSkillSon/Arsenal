// SPDX-License-Identifier: MIT
// UniPASE (Xiaobin-Rong/unipase; MIT code, Apache-2.0 weights) as a Sound-family enhancement model: universal speech
// enhancement (noise, reverb, clipping, band limits, packet loss) with output at any of the family's rates, up to
// 48 kHz through its bandwidth-extension PostNet. Weights and matrix products are FP16 (the maker's half run keeps
// every activation far inside FP16's range); norm statistics, softmax, the LSTM cell, FFTs and resampling are float32.
import Foundation
import MLX
import UniPASE
import VireoCore

public enum UniPASEEngine {
    public static let modelID = "unipase"
    /// The only dtype: FP16 (no quantized tiers).
    public static let dtype = "fp16"
    /// Longest clip one request accepts, in seconds of input.
    public static let maxSeconds = 600.0
    /// inference_long.py's suggestion: clips longer than this many seconds go through the long-form path.
    public static let defaultOnePassSeconds = 20

    public static func load(_ source: ModelSource) throws -> UniPASEModel {
        let weights: [String: MLXArray]
        if source.options["random_weights"] == "1" {
            weights = UniPASEWeights.random(dtype: .float16)  // no-weights contract check
        } else {
            weights = try UniPASEWeights.load(source.directory.appendingPathComponent(UniPASEWeights.fileName), dtype: .float16)
        }
        return UniPASEModel(id: source.id, network: UniPASENetwork(weights: weights, dtype: .float16))
    }

    /// `cast:unipase`: the maker's four FP32 PyTorch checkpoints → one FP16 safetensors file in this port's layout. Every
    /// tensor is read without executing the pickle, range-checked against FP16 (finite, |x| ≤ 65504 after the cast) and
    /// shape-checked; the weight norm is folded in float32; the dropped buffers are checked against what they must be.
    public static func convert(from directory: URL, into output: URL) throws {
        var arrays: [String: MLXArray] = [:]
        for (file, _) in UniPASEWeights.checkpoints {
            let checkpoint = try TorchCheckpoint(directory.appendingPathComponent(file))
            let tensors = try checkpoint.tensors(at: ["model"])
            guard tensors.count == UniPASEWeights.sourceTensorCounts[file] else {
                throw EngineError("\(file) holds \(tensors.count) tensors, expected \(UniPASEWeights.sourceTensorCounts[file] ?? 0)")
            }
            for name in tensors.keys.sorted() {
                guard let (key, layout) = UniPASEWeights.map(name, file: file) else { continue }
                guard arrays[key] == nil else { throw EngineError("Two checkpoint tensors map to \(key)") }
                arrays[key] = try half(UniPASEWeights.arrange(try checkpoint.array(tensors[name]!), layout: layout), key)
            }
            if file == "DeWavLM-Omni.pt" {
                guard let g = tensors["encoder.pos_conv.0.weight_g"], let v = tensors["encoder.pos_conv.0.weight_v"] else {
                    throw EngineError("DeWavLM-Omni.pt has no pos_conv weight norm")
                }
                arrays["wavlm.encoder.pos_conv.0.weight"] = try half(UniPASEWeights.foldWeightNorm(g: try checkpoint.array(g), v: try checkpoint.array(v)), "pos_conv")
                guard tensors["encoder.layer_norm.weight"] != nil, tensors["encoder.layer_norm.bias"] != nil else { throw EngineError("DeWavLM-Omni.pt layout changed") }
            }
            if file == "Vocoder_DWO-L1.pt" {
                guard let window = tensors["head.istft.window"] else { throw EngineError("Vocoder_DWO-L1.pt has no ISTFT window") }
                let actual = try checkpoint.array(window).asType(.float32).asArray(Float.self), expected = UniPASEDSP.hann(1280)
                guard actual.count == 1280, zip(actual, expected).allSatisfy({ abs($0 - $1) <= 1e-6 }) else { throw EngineError("Vocoder window is not a 1280-point Hann window") }
            }
        }
        try UniPASEWeights.check(arrays.mapValues(\.shape))
        let url = output.appendingPathComponent(UniPASEWeights.fileName)
        try MLX.save(arrays: arrays, metadata: ["format": "mlx", "source": "Xiaobin-Rong/unipase", "recipe": "cast:" + modelID], url: url)
        _ = try UniPASEWeights.load(url, dtype: .float16)
    }

    /// FP16 cast with the range check the precision rule asks for.
    static func half(_ a: MLXArray, _ name: String) throws -> MLXArray {
        let h = a.asType(.float16)
        let peak = abs(h.asType(.float32)).max().item(Float.self)
        guard peak.isFinite, peak <= 65_504 else { throw EngineError("UniPASE weight \(name) leaves the FP16 range (peak \(peak))") }
        return h
    }

    /// Request options UniPASE reads; any other key is refused.
    public static let planKeys: Set<String> = ["rate", "enable_plc", "one_time_decode_length"]

    /// The request's controls: output rate, PLC and the one-pass limit.
    public struct Plan: Equatable, Sendable {
        public var rateOut: Int
        public var plc: Bool
        public var onePassSeconds: Int
        public init(rateOut: Int, plc: Bool, onePassSeconds: Int) { self.rateOut = rateOut; self.plc = plc; self.onePassSeconds = onePassSeconds }
    }

    /// Modes: denoise and restore are the same universal pass (UniPASE removes noise, reverb, clipping and band limits
    /// together); either keeps the input rate unless `rate` asks for another. Upscale returns `rate` (default 48 kHz),
    /// which must not be below the input rate. Above 16 kHz the PostNet restores the upper band.
    public static func plan(inputRate: Int, mode: String, options: [String: String]) throws -> Plan {
        if let unknown = options.keys.sorted().first(where: { !planKeys.contains($0) }) {
            throw invalidInput("UniPASE does not take \(unknown)")
        }
        let rate = try options["rate"].map { raw -> Int in
            guard let r = Int(raw), [8000, 16000, 22050, 24000, 32000, 44100, 48000].contains(r) else { throw invalidInput("Unsupported output rate \(raw)") }
            return r
        }
        let rateOut: Int
        switch mode {
        case "denoise", "restore": rateOut = rate ?? inputRate
        case "upscale":
            rateOut = rate ?? 48_000
            guard rateOut >= inputRate else { throw invalidInput("upscale rate must be at least the input rate (\(inputRate) Hz)") }
        default: throw invalidInput("UniPASE supports denoise, restore and upscale; not \(mode)")
        }
        var plc = true
        if let raw = options["enable_plc"] {
            guard raw == "1" || raw == "0" else { throw invalidInput("enable_plc must be a boolean") }
            plc = raw == "1"
        }
        var onePass = defaultOnePassSeconds
        if let raw = options["one_time_decode_length"] {
            guard let seconds = Int(raw), (1...60).contains(seconds) else { throw invalidInput("one_time_decode_length must be 1–60 seconds") }
            onePass = seconds
        }
        return Plan(rateOut: rateOut, plc: plc, onePassSeconds: onePass)
    }
}

public final class UniPASEModel: EnhanceModel, @unchecked Sendable {
    public let id: String
    public var sampleRate: Int { 16_000 }
    public var engineReason: String { "stock MLX; FP16 networks, float32 statistics, FFTs and resampling" }
    private(set) var network: UniPASENetwork?

    init(id: String, network: UniPASENetwork) { self.id = id; self.network = network }

    public func unload() async { network = nil }

    /// Each channel is enhanced on its own, as one mono file of the maker's scripts: in one pass up to the one-pass
    /// limit (inference.py), else in 8 s segments at a 4 s hop (inference_long.py); then its peak is matched to the
    /// input channel's peak, as both scripts do.
    public func enhance(_ input: AudioClip, request: EnhanceRequest, control: GenerationControl) async throws -> AudioClip {
        guard let network else { throw EngineError("UniPASE is unloaded") }
        guard input.frames > 0, input.sampleRate > 0 else { throw invalidInput("Empty audio") }
        guard input.seconds <= UniPASEEngine.maxSeconds else { throw invalidInput("Audio longer than 10 minutes; split it first") }
        guard input.seconds >= 0.1 else { throw invalidInput("Audio shorter than 0.1 s") }
        let plan = try UniPASEEngine.plan(inputRate: input.sampleRate, mode: request.mode, options: request.options)
        return try Self.process(input, plan: plan, control: control, model: id) { segment, rateIn, rateOut, plc in
            try network.enhance(segment, rateIn: rateIn, rateOut: rateOut, plc: plc) { _ in
                if control.isCancelled { throw CancellationError() }
            }
        }
    }

    /// The plumbing around the network, with `segment` (mono samples, input rate, output rate, PLC → enhanced samples)
    /// standing for it: a rate the sinc resampler cannot reduce (44,099 Hz) is converted with AVAudioConverter, on the
    /// way in to 16 kHz and on the way out from 16 or 48 kHz; each channel is cut into one pass or the long-form segments, enhanced, stitched, fitted to
    /// the input's length at the output rate and peak-matched to its own input channel.
    public static func process(
        _ input: AudioClip, plan: UniPASEEngine.Plan, control: GenerationControl, model: String = UniPASEEngine.modelID,
        segment: ([Float], Int, Int, Bool) throws -> [Float]
    ) throws -> AudioClip {
        let original = EnhanceAudio.channels(of: input)
        let convert = !UniPASEDSP.sincFits(from: input.sampleRate, to: 16_000)
        let rateIn = convert ? 16_000 : input.sampleRate
        let channels = try convert ? original.map { try EnhanceAudio.resample($0, from: input.sampleRate, to: 16_000) } : original
        let fitsOut = UniPASEDSP.sincFits(from: 16_000, to: plan.rateOut) && (plan.rateOut <= 16_000 || UniPASEDSP.sincFits(from: 48_000, to: plan.rateOut))
        let workOut = fitsOut ? plan.rateOut : plan.rateOut > 16_000 ? 48_000 : 16_000
        let frames = channels.first?.count ?? 0
        let longForm = frames > plan.onePassSeconds * rateIn
        let segments = longForm ? UniPASEDSP.longFormSegments(count: frames, rate: rateIn) : [0..<frames]
        var outputs: [[Float]] = []
        let steps = Double(channels.count * segments.count)
        for (c, channel) in channels.enumerated() {
            var parts: [[Float]] = []
            for (s, range) in segments.enumerated() {
                if control.isCancelled { throw CancellationError() }
                parts.append(try segment(Array(channel[range]), rateIn, workOut, plan.plc))
                control.report(progress: Double(c * segments.count + s + 1) / steps)
            }
            let length = UniPASEDSP.outputLength(original[c].count, rateIn: input.sampleRate, rateOut: plan.rateOut)
            var joined = UniPASEDSP.stitch(parts, rate: workOut)
            if workOut != plan.rateOut { joined = try EnhanceAudio.resample(joined, from: workOut, to: plan.rateOut) }
            let y = UniPASEDSP.matchPeak(UniPASEDSP.fit(joined, length), to: original[c])
            try EnhanceAudio.requireFinite(y, model: model)
            outputs.append(y)
        }
        return AudioClip(samples: EnhanceAudio.interleave(outputs), sampleRate: plan.rateOut, channels: channels.count)
    }

    /// Learned parameters by name with shape and dtype (the no-weights contract check reads this).
    public func parameterInventory() -> [(String, [Int], DType)] {
        (network?.weights ?? [:]).sorted { $0.key < $1.key }.map { ($0.key, $0.value.shape, $0.value.dtype) }
    }
}
