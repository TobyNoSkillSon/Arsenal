// SPDX-License-Identifier: MIT
// MossFormer2_SE_48K (Alibaba Tongyi Lab, ClearerVoice-Studio; Apache-2.0 weights) as a Sound-family enhancement
// model. The network runs in BF16 (its FP16 cast overflows inside the gated FSMN, as the maker's own half run does);
// framing, FFTs and overlap-add are float32 numerical intermediates.
import Foundation
import MLX
import MLXNN
import ModelPrecision
import MossFormer2SE
import VireoCore

public enum MossFormer2Engine {
    public static let modelID = "mossformer2-se-48k"
    public static let checkpointFile = "last_best_checkpoint.pt"
    /// The only dtype: BF16 (FP16 fails its range check; no quantized tiers).
    public static let dtype = "bf16"
    /// Longest clip one request accepts, in seconds of input.
    public static let maxSeconds = 600.0

    public static func load(_ source: ModelSource) throws -> MossFormer2Model {
        let network: MossFormer2SE
        if source.options["random_weights"] == "1" {
            // No-weights contract check: the module's own initialisation, cast to the serving dtype.
            network = MossFormer2SE()
            let half = network.parameters().flattened().map { ($0.0, $0.1.asType(.bfloat16)) }
            try network.update(parameters: ModuleParameters.unflattened(Dictionary(uniqueKeysWithValues: half)), verify: .all)
            eval(network)
        } else {
            network = try MossFormer2Weights.load(source.directory.appendingPathComponent(MossFormer2Weights.fileName), dtype: .bfloat16)
        }
        return MossFormer2Model(id: source.id, network: network)
    }

    /// `cast:mossformer2-se-48k`: the maker's FP32 PyTorch checkpoint → one BF16 safetensors file in this port's
    /// layout. Rotary and positional frequency buffers are checked against their formula and recomputed at run time.
    public static func convert(from directory: URL, into output: URL) throws {
        let checkpoint = try TorchCheckpoint(directory.appendingPathComponent(checkpointFile))
        let tensors = try checkpoint.tensors()
        guard tensors.count == MossFormer2Weights.sourceTensorCount else {
            throw EngineError("MossFormer2 checkpoint holds \(tensors.count) tensors, expected \(MossFormer2Weights.sourceTensorCount)")
        }
        var arrays: [String: MLXArray] = [:]
        for name in tensors.keys.sorted() {
            let value = try checkpoint.array(tensors[name]!)
            guard let (key, layout) = MossFormer2Weights.map(name) else {
                try checkBuffer(name, value)
                continue
            }
            guard arrays[key] == nil else { throw EngineError("Two checkpoint tensors map to \(key)") }
            arrays[key] = MossFormer2Weights.arrange(value, layout: layout).asType(.bfloat16)
        }
        guard arrays.count == MossFormer2Weights.parameterCount else { throw EngineError("MossFormer2 conversion produced \(arrays.count) tensors") }
        let url = output.appendingPathComponent(MossFormer2Weights.fileName)
        try MLX.save(arrays: arrays, metadata: ["format": "mlx", "source": "alibabasglab/MossFormer2_SE_48K", "recipe": "cast:" + modelID], url: url)
        _ = try MossFormer2Weights.load(url, dtype: .bfloat16)  // every parameter present with its shape
    }

    /// A dropped buffer must equal what the port recomputes, or the conversion would change the model.
    static func checkBuffer(_ name: String, _ value: MLXArray) throws {
        let expected: [Float]
        if name == "mossformer.pos_enc.inv_freq" {
            expected = stride(from: 0, to: 512, by: 2).map { Float(1.0 / pow(10000.0, Double($0) / 512)) }
        } else if name.hasSuffix(".rotary_pos_emb.freqs") {
            expected = stride(from: 0, to: 32, by: 2).map { Float(1.0 / pow(10000.0, Double($0) / 32)) }
        } else {
            throw EngineError("Unexpected MossFormer2 checkpoint tensor \(name)")
        }
        let actual = value.asType(.float32).asArray(Float.self)
        guard actual.count == expected.count, zip(actual, expected).allSatisfy({ abs($0 - $1) <= 1e-6 * max(1, abs($1)) }) else {
            throw EngineError("MossFormer2 buffer \(name) differs from its formula")
        }
    }
}

public final class MossFormer2Model: EnhanceModel, @unchecked Sendable {
    public let id: String
    public var sampleRate: Int { network?.config.sampleRate ?? 48_000 }
    public var engineReason: String { "stock MLX; BF16 network, float32 framing and FFT" }
    private(set) var network: MossFormer2SE?

    init(id: String, network: MossFormer2SE) { self.id = id; self.network = network }

    public func unload() async { network = nil }

    /// Each channel is cleaned on its own at 48 kHz (resampled in and back out when the input has another rate),
    /// as the maker's reader and writer do; the result has the input's rate, channel count and length.
    public func enhance(_ input: AudioClip, request: EnhanceRequest, control: GenerationControl) async throws -> AudioClip {
        guard let network else { throw EngineError("MossFormer2 is unloaded") }
        guard request.mode == "denoise" else { throw invalidInput("MossFormer2 SE 48K only removes noise; mode must be denoise") }
        guard request.options["rate"] == nil else { throw invalidInput("MossFormer2 SE 48K returns the input's rate; rate is not supported") }
        let config = try Self.decodeConfig(network.config, options: request.options)
        guard input.frames > 0, input.sampleRate > 0 else { throw invalidInput("Empty audio") }
        guard input.seconds <= MossFormer2Engine.maxSeconds else { throw invalidInput("Audio longer than 10 minutes; split it first") }
        let channels = EnhanceAudio.channels(of: input), rate = config.sampleRate
        var cleaned: [[Float]] = []
        for (index, channel) in channels.enumerated() {
            if control.isCancelled { throw CancellationError() }
            let at48 = try EnhanceAudio.resample(channel, from: input.sampleRate, to: rate)
            let out = try MossFormer2DSP.decode(at48, config: config, progress: { fraction in
                control.report(progress: (Double(index) + fraction) / Double(channels.count))
            }) { features in
                if control.isCancelled { throw CancellationError() }
                let mask = network.mask(features.asType(.bfloat16)).asType(.float32)
                eval(mask)
                return mask
            }
            var back = try EnhanceAudio.resample(out, from: rate, to: input.sampleRate)
            if back.count < channel.count { back += [Float](repeating: 0, count: channel.count - back.count) }
            try EnhanceAudio.requireFinite(back, model: id)
            cleaned.append(Array(back.prefix(channel.count)))
        }
        return AudioClip(samples: EnhanceAudio.interleave(cleaned), sampleRate: input.sampleRate, channels: input.channels)
    }

    /// The maker's decode controls from the request: one-pass limit (1–60 s, default 20) and window (1–20 s, default 4).
    public static func decodeConfig(_ base: MossFormer2Config, options: [String: String]) throws -> MossFormer2Config {
        var config = base
        if let raw = options["one_time_decode_length"] {
            guard let seconds = Int(raw), (1...60).contains(seconds) else { throw invalidInput("one_time_decode_length must be 1–60 seconds") }
            config.oneTimeDecodeLength = seconds
        }
        if let raw = options["decode_window"] {
            guard let seconds = Int(raw), (1...20).contains(seconds) else { throw invalidInput("decode_window must be 1–20 seconds") }
            config.decodeWindow = seconds
        }
        return config
    }

    /// Learned parameters by name with shape and dtype (the no-weights contract check reads this).
    public func parameterInventory() -> [(String, [Int], DType)] {
        (network?.parameters().flattened() ?? []).map { ($0.0, $0.1.shape, $0.1.dtype) }
    }
}
