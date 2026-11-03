// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ArsenalWire

final class YuEAdapter: NativeModel {
    let architecture = "yue2"
    let operations: Set<String> = ["generate_music"]
    private var pipeline: YuE2Pipeline?
    var peakBytes: Int { Memory.peakMemory }
    var parameters: [String: MLXArray] {
        guard let pipeline else { return [:] }
        return MusicSupport.parameters([("transformer.", pipeline.network), ("vae.", pipeline.decoders[.standard]!)])
    }
    func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        try cancellation.check()
        let small = context.randomWeights && context.config["small"]?.bool == true
        var hidden = small ? 256 : 2048, layers = small ? 2 : 28, intermediate = small ? 512 : 6144
        var config = YuE2Pipeline.decoderConfiguration
        if small { config.decoderChannels = 2; config.channelMultiples = [1, 2, 4, 4, 4, 4] }
        if !context.randomWeights {
            let data = try WireJSON.decode(Data(contentsOf: context.weights.appendingPathComponent("config.json")))
            guard let h = data["hidden_size"]?.number, let l = data["num_hidden_layers"]?.number, let i = data["intermediate_size"]?.number,
                h >= 256, h <= 2048, h.rounded() == h, Int(h) % 256 == 0, l >= 1, l <= 28, l.rounded() == l, i >= 64, i <= 6144, i.rounded() == i,
                data["head_dim"]?.number == 128, data["num_attention_heads"]?.number == h / 128, data["num_key_value_heads"]?.number == h / 256,
                data["vocab_size"]?.number == Double(YuE2Protocol.vocabularySize)
            else { throw MusicSupport.failure("Unsupported YuE2 transformer configuration.") }
            hidden = Int(h); layers = Int(l); intermediate = Int(i)
            let raw = try WireJSON.decode(Data(contentsOf: context.weights.appendingPathComponent("vae/config.json")))
            guard let decoder = raw["decoder_config"]?.object, let channels = decoder["channels"]?.number, channels >= 1, channels <= 64,
                let multiples = decoder["c_mults"]?.array?.compactMap({ $0.number.map(Int.init) }), multiples.count == 6,
                decoder["latent_dim"]?.number == 64, decoder["out_channels"]?.number == 2
            else { throw MusicSupport.failure("Unsupported YuE2 decoder configuration.") }
            config.decoderChannels = Int(channels); config.channelMultiples = multiples
        }
        let model = YuE2Network(hidden: hidden, layers: layers, intermediate: intermediate)
        let decoder = OobleckDecoder(config)
        let tokenizer: Tiktoken
        if context.randomWeights {
            try MusicSupport.initialize(model, precision: context.precision); try MusicSupport.initialize(decoder, precision: context.precision)
            tokenizer = try Tiktoken(randomFixture: true)
        } else {
            try MusicSupport.load(model, file: context.weights.appendingPathComponent("model.safetensors"), precision: context.precision)
            try cancellation.check()
            try MusicSupport.load(decoder, file: context.weights.appendingPathComponent("vae/model.safetensors"), precision: context.precision)
            tokenizer = try Tiktoken(file: context.weights.appendingPathComponent("qwen.tiktoken"))
        }
        pipeline = YuE2Pipeline(network: model, decoder: decoder, tokenizer: tokenizer, folder: context.weights)
    }
    func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        guard operation == "generate_music", let pipeline else { throw MusicSupport.failure("YuE2 is not loaded or operation is unsupported.") }
        var request = try MusicRequests.decode(input, maxSeconds: 360)
        guard request.duration == nil || (request.duration!.isFinite && request.duration! > 0 && request.duration! <= 360) else {
            throw MusicSupport.failure("YuE2 duration must be greater than zero and at most 360 seconds.")
        }
        var options = request.yue ?? YuE2Options()
        if let steps = input["steps"]?.number { options.odeSteps = Int(steps) }
        // YuE2 documents no language control: the lyrics carry the vocal language and the style prompt stays as written.
        guard input["language"] == nil, input["time_signature"] == nil else {
            throw MusicSupport.failure("YuE2 takes the vocal language from the lyrics and has no language or time-signature option.")
        }
        request.engine = .yue2; request.yue = options
        do {
            return try withoutActuallyEscaping(emit) { forward in
                if input["mode"]?.string == "score" {
                    guard options.planning != .off else { throw MusicSupport.failure("Score planning requires melody or full planning.") }
                    let result = try YuEGeneration.score(
                        request, options: options, pipeline: pipeline, item: 0, cancelled: { (try? cancellation.check()) == nil },
                        progress: { progress in
                            try? forward(MusicSupport.progress(SongProgress(stage: .planning, fraction: progress.fraction), batch: 1))
                        })
                    guard let directory = input["out_dir"]?.string else { throw MusicSupport.failure("The host must supply out_dir.") }
                    let root = URL(fileURLWithPath: directory); try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true)
                    let file = root.appendingPathComponent("score-" + UUID().uuidString + ".abc"); try Data(result.score.utf8).write(to: file)
                    try cancellation.check()
                    return ["files": [.object(try NativeMedia.file(file, root: root, mime: "text/plain", id: "score"))]]
                }
                var songs: [GeneratedSong] = []
                let batch = request.batchSize
                let seed = request.seed ?? UInt32.random(in: 0...UInt32.max)
                for index in 0..<request.batchSize {
                    try cancellation.check(); var item = request; item.seed = seed &+ UInt32(index)
                    songs.append(
                        try YuEGeneration.run(
                            item, options: options, pipeline: pipeline, item: index, cancelled: { (try? cancellation.check()) == nil },
                            progress: { progress in
                                try? forward(MusicSupport.progress(progress, batch: batch))
                            }))
                }
                return try MusicRequests.files(songs, input: input, cancellation: cancellation)
            }
        } catch {
            try cancellation.check()
            if let failure = error as? ACEStepError {
                switch failure {
                case .invalidRequest(let message), .unreadableAudio(let message): throw MusicSupport.failure(message)
                default: break
                }
            }
            if let failure = error as? YuE2Error, case .invalidRequest(let message) = failure { throw MusicSupport.failure(message) }
            throw error
        }
    }
    func unload() { pipeline = nil }
}
