// SPDX-License-Identifier: MIT
// DeepFilterNet3 (Hendrik Schröter, Rikorose/DeepFilterNet; MIT/Apache-2.0) as a Sound-family enhancement model:
// real-time-capable full-band (48 kHz) noise suppression. Weights and matrix products are FP16; the STFT, feature
// normalisation, mask and deep-filter application, post filter and attenuation limit are float32. Everything runs
// on the GPU, including the GRU recurrences (one Metal kernel per layer).
import DeepFilterNet3
import Foundation
import MLX
import VireoCore

public enum DeepFilterNet3Engine {
    public static let modelID = "deepfilternet3"
    /// The pinned snapshot keeps the maker's file names inside the repository's `v3/` folder.
    public static let weightsFile = "v3/model.safetensors"
    public static let convertedFile = "model.safetensors"
    /// FP16: the maker's network runs finite in FP16 on the reference clip; no quantized tiers.
    public static let dtype = "fp16"
    /// The maker checkpoint: 118 tensors (2,167,954 values) once its batch-norm counters are dropped.
    public static let tensorCount = 118
    public static let maxSeconds = 600.0

    public static func load(_ source: ModelSource) throws -> DeepFilterNet3Model {
        let weights: [String: MLXArray]
        if source.options["random_weights"] == "1" {
            weights = try randomWeights()
        } else {
            weights = try MLX.loadArrays(url: source.directory.appendingPathComponent(convertedFile))
            guard weights.count == tensorCount, weights.values.allSatisfy({ $0.dtype == .float16 }) else {
                throw EngineError("DeepFilterNet3 weights must be the \(tensorCount) FP16 tensors made by cast:deepfilternet3")
            }
        }
        return DeepFilterNet3Model(id: source.id, network: try DeepFilterNetModel(weights: weights))
    }

    /// `cast:deepfilternet3`: the pinned FP32 safetensors (byte-identical to the maker's checkpoint) → FP16, with each
    /// tensor checked finite and no larger than FP16's maximum, and the exact count of names. At the small end 51 of the
    /// 2,167,954 values flush to zero and 2,884 become subnormal (2 Nov 2026), mostly running variances of dead channels
    /// (down to 5.6e-45), where batch norm's 1e-5 epsilon dominates, so folding is unchanged.
    public static func convert(from directory: URL, into output: URL) throws {
        let source = try MLX.loadArrays(url: directory.appendingPathComponent(weightsFile))
        guard source.count == tensorCount else { throw EngineError("DeepFilterNet3 source holds \(source.count) tensors, expected \(tensorCount)") }
        var half: [String: MLXArray] = [:]
        for (name, value) in source {
            let wide = value.asType(.float32)
            let peak = abs(wide).max().item(Float.self)
            guard peak.isFinite, peak <= 65504 else { throw EngineError("DeepFilterNet3 tensor \(name) leaves FP16 range") }
            half[name] = value.asType(.float16)
        }
        try MLX.save(arrays: half, metadata: ["format": "mlx", "recipe": "cast:" + modelID], url: output.appendingPathComponent(convertedFile))
        _ = try DeepFilterNetModel(weights: try MLX.loadArrays(url: output.appendingPathComponent(convertedFile)))
    }

    /// The maker's tensor names and shapes with seeded random FP16 values (the no-weights contract check).
    static func randomWeights() throws -> [String: MLXArray] {
        let shapes = DeepFilterNet3Tensors.shapes
        // A fixed key, so the worker's global random state is left alone. Positive values keep variances and filter
        // banks meaningful; the small scale keeps activations small.
        var key = MLXRandom.key(0)
        var out: [String: MLXArray] = [:]
        for name in shapes.keys.sorted() {
            let (next, sub) = MLXRandom.split(key: key)
            key = next
            out[name] = MLXRandom.uniform(low: 0.01, high: 0.1, shapes[name]!, key: sub).asType(.float16)
        }
        return out
    }
}

public final class DeepFilterNet3Model: EnhanceModel, @unchecked Sendable {
    public let id: String
    public var sampleRate: Int { 48_000 }
    public var engineReason: String { "MLX on the GPU; FP16 network, float32 STFT and filtering, GRUs as one Metal kernel per layer" }
    private(set) var network: DeepFilterNetModel?

    init(id: String, network: DeepFilterNetModel) { self.id = id; self.network = network }

    public func unload() async { network = nil }

    /// The request options the maker documents: attenuation limit in dB (0 or absent: none) and the post filter.
    public static func requestOptions(_ options: [String: String]) throws -> (limit: Float?, postFilter: Bool) {
        guard options["rate"] == nil else { throw invalidInput("DeepFilterNet3 returns the input's rate; rate is not supported") }
        var limit: Float?
        if let raw = options["atten_lim_db"] {
            guard let value = Float(raw), value.isFinite, (0...100).contains(value) else { throw invalidInput("atten_lim_db must be 0–100 dB") }
            limit = value > 0 ? value : nil
        }
        return (limit, options["post_filter"] == "1")
    }

    /// Each channel is cleaned on its own at 48 kHz (resampled in and out for other rates); same rate, channels, length.
    public func enhance(_ input: AudioClip, request: EnhanceRequest, control: GenerationControl) async throws -> AudioClip {
        guard let network else { throw EngineError("DeepFilterNet3 is unloaded") }
        guard request.mode == "denoise" else { throw invalidInput("DeepFilterNet3 only removes noise; mode must be denoise") }
        let (limit, postFilter) = try Self.requestOptions(request.options)
        guard input.frames > 0, input.sampleRate > 0 else { throw invalidInput("Empty audio") }
        guard input.seconds <= DeepFilterNet3Engine.maxSeconds else { throw invalidInput("Audio longer than 10 minutes; split it first") }
        let channels = EnhanceAudio.channels(of: input)
        var cleaned: [[Float]] = []
        for (index, channel) in channels.enumerated() {
            if control.isCancelled { throw CancellationError() }
            let at48 = try EnhanceAudio.resample(channel, from: input.sampleRate, to: sampleRate)
            let out = try network.enhance(at48, attenuationLimit: limit, postFilter: postFilter)
            var back = try EnhanceAudio.resample(out, from: sampleRate, to: input.sampleRate)
            if back.count < channel.count { back += [Float](repeating: 0, count: channel.count - back.count) }
            try EnhanceAudio.requireFinite(back, model: id)
            cleaned.append(Array(back.prefix(channel.count)))
            control.report(progress: Double(index + 1) / Double(channels.count))
        }
        return AudioClip(samples: EnhanceAudio.interleave(cleaned), sampleRate: input.sampleRate, channels: input.channels)
    }

    public func parameterInventory() -> [(String, [Int], DType)] {
        (network?.weights ?? [:]).map { ($0.key, $0.value.shape, $0.value.dtype) }.sorted { $0.0 < $1.0 }
    }
}
