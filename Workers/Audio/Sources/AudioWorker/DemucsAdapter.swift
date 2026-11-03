// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire

final class DemucsBag: Module {
    @ModuleInfo var models: [HTDemucsGraph]
    let weights: [[Float]]
    init(configuration: [String: Any]) throws {
        let count = configuration["num_models"] as? Int ?? 4
        guard count == 4, let weights = configuration["weights"] as? [[Double]], weights.count == count,
            weights.allSatisfy({ $0.count == 4 && $0.allSatisfy { $0.isFinite && $0 >= 0 } })
        else { throw audioFailure("HTDemucs-ft needs all four submodels and their per-stem weights.") }
        let config = try HTDemucsRuntimeConfig.fromJSON(configuration)
        guard config.sources == ["drums", "bass", "other", "vocals"], config.audioChannels == 2 else { throw audioFailure("HTDemucs-ft requires stereo and four fixed stems.") }
        self.weights = weights.map { $0.map(Float.init) }
        _models.wrappedValue = (0..<count).map { _ in HTDemucsGraph(config: config) }
    }
    func separate(_ audio: MLXArray, cancellation: NativeCancellation) throws -> MLXArray {
        var total: MLXArray?
        let totals = (0..<4).map { stem in weights.reduce(Float(0)) { $0 + $1[stem] } }
        guard totals.allSatisfy({ $0 > 0 }) else { throw audioFailure("A stem has no ensemble contribution.") }
        for (index, model) in models.enumerated() {
            try cancellation.check()
            let value = try model.forward(audio, monitor: SeparationMonitor(cancellation))
            let weighted = value * MLXArray(weights[index]).asType(value.dtype).reshaped(1, 4, 1, 1)
            total = total.map { $0 + weighted } ?? weighted
            eval(total!)
        }
        return total! / MLXArray(totals).asType(audio.dtype).reshaped(1, 4, 1, 1)
    }
}

final class DemucsAdapter: AudioAdapter {
    override var architecture: String { "htdemucs_ft" }
    override var operations: Set<String> { ["separate_audio"] }
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        // The pinned release ships its bag configuration as htdemucs_ft_config.json; shrunk lab fixtures provide config.json.
        let shipped = context.weights.appendingPathComponent("htdemucs_ft_config.json")
        let file = FileManager.default.fileExists(atPath: shipped.path) ? shipped : context.weights.appendingPathComponent("config.json")
        guard let object = try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any] else {
            throw audioFailure("Invalid Demucs configuration.")
        }
        try prepare(
            DemucsBag(configuration: object), context, cancellation: cancellation,
            map: { name, tensor in
                for index in 0..<4 where name.hasPrefix("model_\(index).") {
                    return ("models.\(index)." + audioLSTMKey(String(name.dropFirst("model_\(index).".count))), tensor)
                }
                throw audioFailure("HTDemucs-ft checkpoint is missing the four-model namespace.")
            })
    }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "stems", "mode"])
        guard operation == "separate_audio", let model = graph as? DemucsBag else { throw audioFailure("Demucs is not loaded.") }
        let names = ["drums", "bass", "other", "vocals"]
        let selected = input["stems"]?.array?.compactMap(\.string) ?? names
        let mode = input["mode"]?.string ?? "extract"
        guard !selected.isEmpty, selected.allSatisfy(names.contains), Set(selected).count == selected.count,
            ["extract", "remove"].contains(mode), input["prompt"] == nil, input["steps"] == nil
        else { throw audioFailure("Demucs accepts fixed stem names and extract/remove mode.") }
        let kept = mode == "extract" ? selected : names.filter { !selected.contains($0) }
        guard !kept.isEmpty else { throw audioFailure("Removing all stems leaves no output.") }
        let rate = model.models[0].config.samplerate
        guard let path = input["audio"]?["path"]?.string else { throw audioFailure("Audio is required.") }
        let pcm = try NativeMedia.audio(
            URL(fileURLWithPath: path), sampleRate: rate, channels: 2, maximumFrames: min(rate * 600, (NativeFiles.maxBytes - 44 * kept.count) / (4 * kept.count)))
        let frames = pcm.samples.count / 2
        let segment = max(4096, Int(model.models[0].config.segment * Float(rate)))
        let overlap = segment / 4, hop = segment - overlap
        var output = [Float](repeating: 0, count: 4 * frames * 2), normalization = [Float](repeating: 0, count: frames)
        for start in stride(from: 0, to: frames, by: hop) {
            try cancellation.check()
            let count = min(segment, frames - start)
            let chunk = MLXArray(Array(pcm.samples[(start * 2)..<((start + count) * 2)])).reshaped(1, count, 2).transposed(0, 2, 1).asType(.float16)
            let value = try model.separate(chunk, cancellation: cancellation)
            guard value.shape == [1, 4, 2, count] else { throw audioFailure("Demucs returned an unexpected stem shape.") }
            let values = value.asArray(Float.self)
            for frame in 0..<count {
                let weight = max(Float(1e-4), 1 - abs(2 * Float(frame) / Float(max(1, count - 1)) - 1))
                normalization[start + frame] += weight
                for stem in 0..<4 {
                    for channel in 0..<2 { output[(stem * frames + start + frame) * 2 + channel] += values[(stem * 2 + channel) * count + frame] * weight }
                }
            }
            if start + count == frames { break }
        }
        var files: [JSONValue] = []
        for stem in kept {
            let index = names.firstIndex(of: stem)!
            var samples = Array(output[(index * frames * 2)..<((index + 1) * frames * 2)])
            for frame in 0..<frames { for channel in 0..<2 { samples[frame * 2 + channel] /= normalization[frame] } }
            files.append(try audioWave(samples, rate: rate, channels: 2, id: stem, input: input, cancellation: cancellation))
        }
        guard !files.isEmpty else { throw audioFailure("Removing all stems leaves no output.") }
        return ["files": .array(files)]
    }
}
