// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ModelPrecision
import ArsenalWire

/// BS-RoFormer vocal separation (viperx 1296 via python-audio-separator / UVR). The pinned release is a PyTorch
/// state dict (`.ckpt`) plus its training YAML; Get converts it once into an FP16 safetensors snapshot with
/// Arsenal's module names, keeping the YAML beside it.
enum BSRoformerConversion {
    static let recipe = "cast:audio-bs-roformer-v1"
    static let marker = "arsenal-bs-roformer-layout.json"
    static let legacyMarker = "mobius-bs-roformer-layout.json" // legacy-compat: read in snapshots written before the rename
    static let base = NativeConversion(recipe: recipe, convert: prepare)

    /// Checkpoint name → snapshot name; nil for the rotary frequency buffers (recomputed, verified at conversion).
    static func name(_ source: String) throws -> String? {
        let parts = source.split(separator: ".").map(String.init)
        func fail() -> PrecisionError { PrecisionError("Unexpected BS-RoFormer tensor: " + source) }
        if parts == ["final_norm", "gamma"] { return source }
        if parts.count == 5, parts[0] == "band_split", parts[1] == "to_features" {
            switch (parts[3], parts[4]) {
            case ("0", "gamma"): return "band_split.\(parts[2]).norm.gamma"
            case ("1", "weight"), ("1", "bias"): return "band_split.\(parts[2]).linear.\(parts[4])"
            default: throw fail()
            }
        }
        if parts.count == 7, parts[0] == "mask_estimators", parts[1] == "0", parts[2] == "to_freqs", parts[4] == "0",
            let index = Int(parts[5]), index % 2 == 0, ["weight", "bias"].contains(parts[6])
        {
            return "mask_estimator.\(parts[3]).layers.\(index / 2).\(parts[6])"
        }
        // layers.<i>.<0 time | 1 freq>.layers.<j>.<0 attention | 1 feed-forward>.…
        guard parts.count >= 8, parts[0] == "layers", parts[3] == "layers", let axis = ["0": "time", "1": "freq"][parts[2]] else { throw fail() }
        let stem = "layers.\(parts[1]).\(axis).\(parts[4])"
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
        guard context.bits == nil else { throw PrecisionError("BS-RoFormer ships at 16-bit only") }
        let files = try NativeConversionContext.files(in: context.source)
        guard let ckpt = files.first(where: { $0.hasSuffix(".ckpt") }), let yaml = files.first(where: { $0.hasSuffix(".yaml") }),
            files.filter({ $0.hasSuffix(".ckpt") || $0.hasSuffix(".yaml") }).count == 2
        else { throw PrecisionError("BS-RoFormer source needs exactly one .ckpt and one .yaml") }
        let config = try BSRoformerConfig.parse(try SimpleYAML.parse(String(contentsOf: context.source.appendingPathComponent(yaml), encoding: .utf8)))
        let checkpoint = try TorchCheckpoint(context.source.appendingPathComponent(ckpt))
        let tensors = try checkpoint.tensors()
        var mapping: [String: TorchCheckpoint.Tensor] = [:]
        let theta = (0..<(config.dimHead / 2)).map { 1 / pow(Float(10_000), Float(2 * $0) / Float(config.dimHead)) }
        for (source, tensor) in tensors {
            try cancellation.check()
            guard let target = try name(source) else {
                let freqs = try checkpoint.array(tensor).asArray(Float.self)
                guard freqs.count == theta.count, zip(freqs, theta).allSatisfy({ abs($0 - $1) <= 1e-6 * max(1, abs($1)) }) else {
                    throw PrecisionError("BS-RoFormer rotary frequencies differ from theta 10 000")
                }
                continue
            }
            guard mapping.updateValue(tensor, forKey: target) == nil else { throw PrecisionError("Duplicate BS-RoFormer tensor " + target) }
        }
        let expected = Dictionary(uniqueKeysWithValues: BSRoformerModel(config).parameters().flattened().map { ($0.0, $0.1.shape) })
        guard Set(expected.keys) == Set(mapping.keys), mapping.allSatisfy({ expected[$0.key] == $0.value.shape }) else {
            throw PrecisionError("BS-RoFormer checkpoint does not match its configuration")
        }
        _ = try StreamedPrecision.convert(
            keys: mapping.keys.sorted(),
            read: { key in
                try cancellation.check()
                return try checkpoint.array(mapping[key]!)
            }, to: context.destination.appendingPathComponent("model.safetensors"), half: context.half, eligible: { _, _ in false })
        try FileManager.default.copyItem(at: context.source.appendingPathComponent(yaml), to: context.destination.appendingPathComponent("config.yaml"))
        try JSONSerialization.data(withJSONObject: ["architecture": "bs_roformer", "layout_version": 1], options: [.sortedKeys])
            .write(to: context.destination.appendingPathComponent(marker))
        return try context.receipt(cancellation: cancellation)
    }
}

final class BSRoformerAdapter: AudioAdapter {
    static let stems = ["vocals", "instrumental"]
    override var architecture: String { "bs_roformer" }
    override var operations: Set<String> { ["separate_audio"] }

    static func configuration(_ weights: URL) throws -> BSRoformerConfig {
        let yaml = weights.appendingPathComponent("config.yaml"), json = weights.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: yaml.path) {
            return try BSRoformerConfig.parse(try SimpleYAML.parse(String(contentsOf: yaml, encoding: .utf8)))
        }
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: json)) as? [String: Any] else {
            throw audioFailure("Invalid BS-RoFormer configuration.")
        }
        return try BSRoformerConfig.parse(object)
    }

    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try prepare(BSRoformerModel(try Self.configuration(context.weights)), context, cancellation: cancellation, eligible: { _, _ in false })
    }

    /// python-audio-separator's MDXC demix for single-target RoFormers (0.45.0, default `mdxc_params`): chunks of
    /// hop × (dim_t − 1) samples (segment 256 frames under 10 s), step = min(overlap 8 s, chunk), Hamming-weighted
    /// overlap-add, the last chunk aligned to the end. A clip shorter than one chunk runs whole.
    static func plan(frames: Int, config: BSRoformerConfig) -> (chunk: Int, starts: [Int]) {
        let seconds = Double(frames) / Double(config.sampleRate)
        let chunk = config.hop * ((seconds < 10 ? 256 : config.dimT) - 1)
        guard frames > chunk else { return (frames, [0]) }
        let step = min(8 * config.sampleRate, chunk)
        var starts: [Int] = []
        for start in stride(from: 0, to: frames, by: step) { starts.append(start + chunk > frames ? frames - chunk : start) }
        return (chunk, starts)
    }

    /// Mix → (vocals, instrumental), interleaved stereo, before the output peak limit.
    func separate(_ mix: [Float], frames: Int, model: BSRoformerModel, cancellation: NativeCancellation) throws -> [Float] {
        let channels = model.config.channels
        var (chunk, starts) = Self.plan(frames: frames, config: model.config)
        let whole = starts == [0] && chunk == frames
        var mix = mix
        if whole {
            // One pass: zero-pad to whole hops and at least one FFT frame, so the STFT never reads past the clip and
            // the last partial hop keeps its vocals (the reference leaves those few samples at zero). Cropped below.
            let hop = model.config.hop
            chunk = max(model.config.nFFT, (frames + hop - 1) / hop * hop)
            mix += [Float](repeating: 0, count: (chunk - frames) * channels)
        }
        let window: [Float] = whole ? Array(repeating: 1, count: chunk) : (0..<chunk).map { Float(0.54 - 0.46 * cos(2 * Double.pi * Double($0) / Double(chunk - 1))) }
        var result = [Float](repeating: 0, count: frames * channels), counter = [Float](repeating: 0, count: frames)
        for start in starts {
            try cancellation.check()
            var planar = [Float](repeating: 0, count: channels * chunk)
            for frame in 0..<chunk { for channel in 0..<channels { planar[channel * chunk + frame] = mix[(start + frame) * channels + channel] } }
            let output = model(MLXArray(planar).reshaped(channels, chunk))
            eval(output)
            let produced = output.dim(1), values = output.asArray(Float.self)
            for frame in 0..<min(chunk, produced, frames - start) {
                let weight = window[frame]
                counter[start + frame] += weight
                for channel in 0..<channels { result[(start + frame) * channels + channel] += values[channel * produced + frame] * weight }
            }
            Memory.clearCache()
        }
        for frame in 0..<frames {
            let scale = 1 / max(counter[frame], 1e-10)
            for channel in 0..<channels { result[frame * channels + channel] *= scale }
        }
        return result
    }

    static func limitPeak(_ samples: inout [Float], to peak: Float = 0.9) {
        let maximum = samples.reduce(Float(0)) { max($0, abs($1)) }
        if maximum > peak { let scale = peak / maximum; for index in samples.indices { samples[index] *= scale } }
    }

    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "stems"])
        guard operation == "separate_audio", let model = graph as? BSRoformerModel else { throw audioFailure("BS-RoFormer is not loaded.") }
        let selected = input["stems"]?.array?.compactMap(\.string) ?? Self.stems
        guard !selected.isEmpty, selected.allSatisfy(Self.stems.contains), Set(selected).count == selected.count,
            input["stems"].map({ $0.array?.count == selected.count }) ?? true
        else { throw audioFailure("BS-RoFormer separates vocals and instrumental.") }
        guard let path = input["audio"]?["path"]?.string else { throw audioFailure("Audio is required.") }
        let rate = model.config.sampleRate, channels = model.config.channels
        let bound = (NativeFiles.maxBytes - 44 * selected.count) / (2 * channels * selected.count)
        let pcm = try NativeMedia.audio(URL(fileURLWithPath: path), sampleRate: rate, channels: channels, maximumFrames: min(rate * 600, bound))
        var mix = pcm.samples
        let frames = mix.count / channels
        guard frames > 0 else { throw audioFailure("Audio is empty.") }
        Self.limitPeak(&mix)
        let vocals = try separate(mix, frames: frames, model: model, cancellation: cancellation)
        var files: [JSONValue] = []
        for stem in Self.stems where selected.contains(stem) {
            var samples = stem == "vocals" ? vocals : zip(mix, vocals).map { $0 - $1 }
            Self.limitPeak(&samples)
            files.append(try audioWave(samples, rate: rate, channels: channels, id: stem, input: input, cancellation: cancellation))
        }
        return ["files": .array(files)]
    }
}

/// The YAML subset of audio-separator/MSST model configs: nested mappings by indentation, scalars, `- item`
/// sequences (optionally tagged `!!python/tuple`) and comments. Anchors, flow collections and multi-line scalars are
/// refused.
enum SimpleYAML {
    static func parse(_ text: String) throws -> [String: Any] {
        struct Line { let indent: Int; let text: String }
        let lines: [Line] = text.split(separator: "\n", omittingEmptySubsequences: false).compactMap { raw in
            var body = String(raw)
            if let hash = body.firstIndex(of: "#"), !body[..<hash].contains("'"), !body[..<hash].contains("\"") { body = String(body[..<hash]) }
            let trimmed = body.trimmingCharacters(in: .whitespaces)
            guard !trimmed.isEmpty else { return nil }
            return Line(indent: body.prefix { $0 == " " }.count, text: trimmed)
        }
        var index = 0
        func scalar(_ raw: String) throws -> Any {
            var value = raw.trimmingCharacters(in: .whitespaces)
            guard !value.hasPrefix("&"), !value.hasPrefix("*"), !value.hasPrefix("{"), !value.hasPrefix("["), !value.hasPrefix("|"), !value.hasPrefix(">") else {
                throw audioFailure("Unsupported YAML in model configuration.")
            }
            if (value.hasPrefix("'") && value.hasSuffix("'")) || (value.hasPrefix("\"") && value.hasSuffix("\"")), value.count >= 2 {
                value = String(value.dropFirst().dropLast()); return value
            }
            switch value.lowercased() {
            case "true": return true
            case "false": return false
            case "null", "~", "none": return NSNull()
            default: break
            }
            if let int = Int(value) { return NSNumber(value: int) }
            if let double = Double(value) { return NSNumber(value: double) }
            return value
        }
        func block(_ indent: Int) throws -> Any {
            if index < lines.count, lines[index].indent == indent, lines[index].text.hasPrefix("- ") || lines[index].text == "-" {
                var items: [Any] = []
                while index < lines.count, lines[index].indent == indent, lines[index].text.hasPrefix("-") {
                    let rest = String(lines[index].text.dropFirst()).trimmingCharacters(in: .whitespaces)
                    guard !rest.isEmpty else { throw audioFailure("Unsupported YAML sequence item.") }
                    items.append(try scalar(rest)); index += 1
                }
                return items
            }
            var table: [String: Any] = [:]
            while index < lines.count, lines[index].indent == indent {
                let text = lines[index].text
                guard let colon = text.firstIndex(of: ":") else { throw audioFailure("Unsupported YAML line.") }
                let key = String(text[..<colon]).trimmingCharacters(in: .whitespaces)
                var value = String(text[text.index(after: colon)...]).trimmingCharacters(in: .whitespaces)
                index += 1
                if value.hasPrefix("!!python/tuple") || value.hasPrefix("!!") { value = "" }
                if value.isEmpty {
                    if index < lines.count, lines[index].indent > indent {
                        table[key] = try block(lines[index].indent)
                    } else if index < lines.count, lines[index].indent == indent, lines[index].text.hasPrefix("-") {
                        table[key] = try block(indent)
                    } else {
                        table[key] = NSNull()
                    }
                } else {
                    table[key] = try scalar(value)
                }
            }
            return table
        }
        guard let root = try block(lines.first?.indent ?? 0) as? [String: Any], index == lines.count else {
            throw audioFailure("Unsupported YAML structure in model configuration.")
        }
        return root
    }
}
