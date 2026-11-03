// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ArsenalWire

final class GraniteAdapter: AudioAdapter {
    override var architecture: String { "granite_speech5_ctc" }
    override var operations: Set<String> { ["transcribe"] }
    var tokenizer: GraniteTokenizer?
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let configuration = try config(context, as: GraniteModelConfiguration.self)
        tokenizer = try GraniteTokenizer(directory: context.randomWeights ? context.folder.appendingPathComponent("gate/tokenizer") : context.weights)
        try prepare(
            GraniteCTCModel(configuration), context, cancellation: cancellation,
            map: { name, tensor in
                if name.hasSuffix(".num_batches_tracked") { return nil }
                return (name, name.hasSuffix(".conv.depthwise_conv.weight") && tensor.ndim == 3 && tensor.dim(1) == 1 ? tensor.transposed(0, 2, 1) : tensor)
            })
    }
    override func unload() { super.unload(); tokenizer = nil }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        try audioKeys(input, allowed: ["audio", "language", "task", "word_timestamps"])
        guard operation == "transcribe", let model = graph as? GraniteCTCModel, let tokenizer else { throw audioFailure("Granite is not loaded.") }
        guard input["language"] == nil || input["language"]?.string == "en", input["diarize"]?.bool != true else { throw audioFailure("Granite supports English transcription.") }
        guard input["task"] == nil || input["task"]?.string == "transcribe" else { throw audioFailure("Granite transcribes English; translation is unavailable.") }
        let pcm = try audioInput(input, rate: 16000)
        guard pcm.samples.count > 256 else { throw audioFailure("Granite needs more than 16 ms of audio.") }
        // Upstream Granite-MLX GraniteRecognizer.transcribe: 122.88 s cores (twelve 10.24 s attention blocks), each encoded
        // with up to 10.24 s of context on both sides that is evaluated and dropped, then one CTC collapse over every
        // kept frame, so no word is cut or doubled at a core boundary.
        let core = Int(122.88 * 16000), context = Int(10.24 * 16000), frameRate = 12.5
        let chunked = pcm.samples.count > core
        var frames: [Int] = []
        for coreStart in stride(from: 0, to: pcm.samples.count, by: chunked ? core : pcm.samples.count) {
            try cancellation.check()
            let coreEnd = min(coreStart + core, pcm.samples.count)
            let start = chunked ? max(0, coreStart - context) : 0
            let end = chunked ? min(pcm.samples.count, coreEnd + context) : pcm.samples.count
            var samples = Array(pcm.samples[start..<end])
            if samples.count <= 256 { samples += Array(repeating: 0, count: 257 - samples.count) }
            let features = GraniteFeatureExtractor()(samples).asType(.float16)
            let ids = model(features).argMax(axis: -1).reshaped(-1).asArray(Int.self)
            let first = min(ids.count, Int((Double(coreStart - start) / 16000 * frameRate).rounded()))
            let last = min(ids.count, first + Int((Double(coreEnd - coreStart) / 16000 * frameRate).rounded()))
            frames += chunked ? Array(ids[first..<last]) : ids
            Memory.clearCache()
        }
        let text = tokenizer.decode(GraniteCTCDecoder.collapse(frames))
        var words: [JSONValue] = []
        if input["word_timestamps"]?.bool == true {
            let duration = Double(pcm.samples.count) / 16000
            for word in GraniteCTCDecoder.words(from: GraniteCTCDecoder.tokenTimings(frames, frameRate: frameRate, decodeToken: tokenizer.decodeToken)) {
                words.append(["word": .string(word.text), "start": .number(min(duration, word.start)), "end": .number(min(duration, word.end))])
            }
        }
        try cancellation.check()
        var result: JSONObject = ["text": .string(text), "language": "en", "duration": .number(Double(pcm.samples.count) / 16000)]
        if input["word_timestamps"]?.bool == true { result["words"] = .array(words) }
        return result
    }
}
