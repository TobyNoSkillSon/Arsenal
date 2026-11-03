// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ModelPrecision
import ArsenalWire

/// Kim's Mel-Band RoFormer vocal model (KimberleyJSN/melbandroformer). The release is one FP32 PyTorch state dict;
/// its configuration lives in the maker's repository (`configs/config_vocals_mel_band_roformer.yaml`) and is
/// `MelBandRoformerConfig()`'s defaults, checked against every checkpoint shape at Get. Get writes an FP16
/// safetensors snapshot with Arsenal's module names.
enum MelBandRoformerConversion {
    static let recipe = "cast:audio-mel-band-roformer-v1"
    static let marker = "arsenal-mel-band-roformer-layout.json"
    static let base = NativeConversion(recipe: recipe, convert: prepare)

    /// Checkpoint name → snapshot name; nil for the rotary frequency buffers (recomputed, verified at conversion).
    static func name(_ source: String) throws -> String? {
        let parts = source.split(separator: ".").map(String.init)
        func fail() -> PrecisionError { PrecisionError("Unexpected Mel-Band RoFormer tensor: " + source) }
        if parts.count == 5, parts[0] == "band_split", parts[1] == "to_features", Int(parts[2]) != nil {
            switch (parts[3], parts[4]) {
            case ("0", "gamma"): return "band_split.\(parts[2]).norm.gamma"
            case ("1", "weight"), ("1", "bias"): return "band_split.\(parts[2]).linear.\(parts[4])"
            default: throw fail()
            }
        }
        if parts.count == 7, parts[0] == "mask_estimators", parts[1] == "0", parts[2] == "to_freqs", Int(parts[3]) != nil, parts[4] == "0",
            let index = Int(parts[5]), index % 2 == 0, ["weight", "bias"].contains(parts[6])
        {
            return "mask_estimator.\(parts[3]).layers.\(index / 2).\(parts[6])"
        }
        // layers.<i>.<0 time | 1 freq>.norm.gamma, or layers.<i>.<axis>.layers.<j>.<0 attention | 1 feed-forward>.…
        guard parts.count >= 5, parts[0] == "layers", Int(parts[1]) != nil, let axis = ["0": "time", "1": "freq"][parts[2]] else { throw fail() }
        let block = "layers.\(parts[1]).\(axis)"
        if parts.count == 5, parts[3] == "norm", parts[4] == "gamma" { return block + ".norm.gamma" }
        guard parts.count >= 7, parts[3] == "layers", Int(parts[4]) != nil else { throw fail() }
        let stem = "\(block).layers.\(parts[4])"
        let rest = parts[6...].joined(separator: ".")
        switch (parts[5], rest) {
        case ("0", "rotary_embed.freqs"): return nil
        case ("0", "norm.gamma"): return stem + ".attn.norm.gamma"
        case ("0", "to_qkv.weight"), ("0", "to_gates.weight"), ("0", "to_gates.bias"): return stem + ".attn." + rest
        case ("0", "to_out.0.weight"): return stem + ".attn.to_out.weight"
        case ("1", "net.0.gamma"): return stem + ".ff.norm.gamma"
        case ("1", "net.1.weight"), ("1", "net.1.bias"): return stem + ".ff.proj_in." + parts[parts.count - 1]
        case ("1", "net.4.weight"), ("1", "net.4.bias"): return stem + ".ff.proj_out." + parts[parts.count - 1]
        default: throw fail()
        }
    }

    static func prepare(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits == nil else { throw PrecisionError("Mel-Band RoFormer ships at 16-bit only") }
        let files = try NativeConversionContext.files(in: context.source).filter { $0.hasSuffix(".ckpt") }
        guard files.count == 1, let ckpt = files.first else { throw PrecisionError("Mel-Band RoFormer source needs exactly one .ckpt") }
        let config = MelBandRoformerConfig()
        let checkpoint = try TorchCheckpoint(context.source.appendingPathComponent(ckpt))
        var mapping: [String: TorchCheckpoint.Tensor] = [:]
        let theta = (0..<(config.dimHead / 2)).map { 1 / pow(Float(10_000), Float(2 * $0) / Float(config.dimHead)) }
        for (source, tensor) in try checkpoint.tensors() {
            try cancellation.check()
            guard let target = try name(source) else {
                let freqs = try checkpoint.array(tensor).asArray(Float.self)
                guard freqs.count == theta.count, zip(freqs, theta).allSatisfy({ abs($0 - $1) <= 1e-6 * max(1, abs($1)) }) else {
                    throw PrecisionError("Mel-Band RoFormer rotary frequencies differ from theta 10 000")
                }
                continue
            }
            guard mapping.updateValue(tensor, forKey: target) == nil else { throw PrecisionError("Duplicate Mel-Band RoFormer tensor " + target) }
        }
        // The band widths come from the mel layout, so equal shapes also confirm the Slaney membership.
        let expected = Dictionary(uniqueKeysWithValues: MelBandRoformerModel(config).parameters().flattened().map { ($0.0, $0.1.shape) })
        guard Set(expected.keys) == Set(mapping.keys), mapping.allSatisfy({ expected[$0.key] == $0.value.shape }) else {
            let missing = Set(expected.keys).subtracting(mapping.keys).sorted().prefix(3), extra = Set(mapping.keys).subtracting(expected.keys).sorted().prefix(3)
            let shapes = mapping.filter { entry in expected[entry.key].map { $0 != entry.value.shape } ?? false }.keys.sorted().prefix(3)
            throw PrecisionError("Mel-Band RoFormer checkpoint does not match its configuration (missing \(Array(missing)), unexpected \(Array(extra)), shape \(Array(shapes)))")
        }
        _ = try StreamedPrecision.convert(
            keys: mapping.keys.sorted(),
            read: { key in
                try cancellation.check()
                return try checkpoint.array(mapping[key]!)
            }, to: context.destination.appendingPathComponent("model.safetensors"), half: context.half, eligible: { _, _ in false })
        try JSONSerialization.data(withJSONObject: ["architecture": "mel_band_roformer", "layout_version": 1], options: [.sortedKeys])
            .write(to: context.destination.appendingPathComponent(marker))
        return try context.receipt(cancellation: cancellation)
    }
}

final class MelBandRoformerAdapter: AudioAdapter {
    static let stems = ["vocals", "instrumental"]
    override var architecture: String { "mel_band_roformer" }
    override var operations: Set<String> { ["separate_audio"] }

    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try prepare(MelBandRoformerModel(MelBandRoformerConfig()), context, cancellation: cancellation, eligible: { _, _ in false })
    }

    /// The maker's `demix_track` (utils.py @25f44ffb): reflect-pad by `border` when the clip is longer than two
    /// borders, chunks of `chunk` samples every `chunk / overlap`, a linear fade of `chunk / 10` at both ends (none
    /// before the first chunk or after the last), a short last chunk reflect-padded when longer than half a chunk
    /// and zero-padded otherwise, then the weighted average and the border cropped.
    struct Demix {
        let chunk: Int, step: Int, fade: Int, border: Int
        init(_ config: MelBandRoformerConfig, overlap: Int? = nil) {
            chunk = config.chunk; step = config.chunk / (overlap ?? config.overlap); fade = config.chunk / 10; border = chunk - step
        }
        func pads(_ frames: Int) -> Bool { frames > 2 * border && border > 0 }
        /// Chunk starts over a padded length.
        func starts(_ total: Int) -> [Int] { Array(stride(from: 0, to: total, by: step)) }
        /// torch `F.pad(mode="reflect")` of one channel.
        static func reflect(_ x: [Float], left: Int, right: Int) -> [Float] {
            let n = x.count
            return (0..<(n + left + right)).map { j in
                var i = j - left
                if i < 0 { i = -i }
                if i >= n { i = 2 * (n - 1) - i }
                return x[i]
            }
        }
        func window(start: Int, total: Int) -> [Float] {
            var w = [Float](repeating: 1, count: chunk)
            let ramp = (0..<fade).map { Float($0) / Float(max(fade - 1, 1)) }
            if start != 0 { for k in 0..<fade { w[k] *= ramp[k] } }
            if start == 0 || start + chunk < total { for k in 0..<fade { w[chunk - fade + k] *= ramp[fade - 1 - k] } }
            return w
        }
        /// One chunk of one channel, padded to `chunk`.
        func part(_ x: [Float], start: Int) -> [Float] {
            let length = min(chunk, x.count - start)
            let slice = Array(x[start..<(start + length)])
            guard length < chunk else { return slice }
            if length > chunk / 2 + 1 { return Self.reflect(slice, left: 0, right: chunk - length) }
            return slice + [Float](repeating: 0, count: chunk - length)
        }
    }

    /// Planar mix [channel][frame] → planar vocals.
    func separate(_ mix: [[Float]], model: MelBandRoformerModel, overlap: Int? = nil, cancellation: NativeCancellation) throws -> [[Float]] {
        let demix = Demix(model.config, overlap: overlap), frames = mix[0].count
        let padded = demix.pads(frames) ? mix.map { Demix.reflect($0, left: demix.border, right: demix.border) } : mix
        let total = padded[0].count, channels = mix.count
        var result = [[Float]](repeating: [Float](repeating: 0, count: total), count: channels)
        var counter = [Float](repeating: 0, count: total)
        for start in demix.starts(total) {
            try cancellation.check()
            let length = min(demix.chunk, total - start)
            let input = MLXArray(padded.flatMap { demix.part($0, start: start) }).reshaped(channels, demix.chunk)
            let output = model(input)
            eval(output)
            let produced = output.dim(1), values = output.asArray(Float.self), window = demix.window(start: start, total: total)
            for k in 0..<min(length, produced) {
                counter[start + k] += window[k]
                for c in 0..<channels { result[c][start + k] += values[c * produced + k] * window[k] }
            }
            Memory.clearCache()
        }
        let offset = demix.pads(frames) ? demix.border : 0
        return result.map { channel in
            (0..<frames).map { k in
                let value = channel[offset + k] / counter[offset + k]
                return value.isFinite ? value : 0
            }
        }
    }

    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "stems", "overlap"])
        guard operation == "separate_audio", let model = graph as? MelBandRoformerModel else { throw audioFailure("Mel-Band RoFormer is not loaded.") }
        let selected = input["stems"]?.array?.compactMap(\.string) ?? Self.stems
        guard !selected.isEmpty, selected.allSatisfy(Self.stems.contains), Set(selected).count == selected.count,
            input["stems"].map({ $0.array?.count == selected.count }) ?? true
        else { throw audioFailure("Mel-Band RoFormer separates vocals and instrumental.") }
        guard let path = input["audio"]?["path"]?.string else { throw audioFailure("Audio is required.") }
        // The maker's `num_overlap`: more overlap smooths chunk seams at proportionally more compute. 1 is refused: with no
        // overlap the maker's fade windows reach 0 on both sides of every seam, so two samples per seam lose their vocals.
        let overlap = try input["overlap"].map { value -> Int in
            guard let n = value.number, n == n.rounded(), (2...8).contains(n) else { throw audioFailure("overlap must be an integer from 2 to 8.") }
            return Int(n)
        }
        let rate = model.config.sampleRate, channels = model.config.channels
        let bound = (NativeFiles.maxBytes - 44 * selected.count) / (2 * channels * selected.count)
        let pcm = try NativeMedia.audio(URL(fileURLWithPath: path), sampleRate: rate, channels: channels, maximumFrames: min(rate * 600, bound))
        let frames = pcm.samples.count / channels
        guard frames > 0 else { throw audioFailure("Audio is empty.") }
        let planar = (0..<channels).map { c in (0..<frames).map { pcm.samples[$0 * channels + c] } }
        let vocals = try separate(planar, model: model, overlap: overlap, cancellation: cancellation)
        var files: [JSONValue] = []
        for stem in Self.stems where selected.contains(stem) {
            var samples = [Float](repeating: 0, count: frames * channels)
            for k in 0..<frames {
                for c in 0..<channels {
                    let v = vocals[c][k]
                    samples[k * channels + c] = stem == "vocals" ? v : planar[c][k] - v
                }
            }
            Self.guardFullScale(&samples)
            files.append(try audioWave(samples, rate: rate, channels: channels, id: stem, input: input, cancellation: cancellation))
        }
        return ["files": .array(files)]
    }

    /// The maker writes float WAV; Arsenal writes 16-bit PCM, so a stem past full scale is scaled to 0.999 instead
    /// of clipping. Stems within full scale are untouched.
    static func guardFullScale(_ samples: inout [Float]) {
        let peak = samples.reduce(Float(0)) { max($0, abs($1)) }
        if peak > 0.999 { let scale = 0.999 / peak; for i in samples.indices { samples[i] *= scale } }
    }
}
