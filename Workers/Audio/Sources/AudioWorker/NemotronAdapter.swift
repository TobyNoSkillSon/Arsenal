// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import ArsenalWire

final class ArsenalNemotronBackend: Nemotron3InferenceBackend {
    let network: Nemotron3MLXNetwork
    var cancellation = NativeCancellation()
    init(_ network: Nemotron3MLXNetwork) { self.network = network }
    var learnedSilenceEmbedding: [Float] {
        let value = Dictionary(uniqueKeysWithValues: network.parameters().flattened())["sortformer_modules.learnable_sil_emb"]!
        return value.asArray(Float.self)
    }
    func preencode(chunk: [Float]) throws -> [Float] {
        try cancellation.check()
        let value = network.preencode(MLXArray(chunk).reshaped(1, 3040, 128).asType(.float16))
        eval(value)
        return value.asArray(Float.self)
    }
    func predictHead(packedEmbeddings: [Float], validLength: Int) throws -> Nemotron3HeadOutput {
        try cancellation.check()
        let value = network.inferHead(MLXArray(packedEmbeddings).reshaped(1, 684, 512).asType(.float16), validLength: validLength)
        eval(value.high, value.low)
        return Nemotron3HeadOutput(probabilities10ms: value.high.asArray(Float.self), probabilities80ms: value.low.asArray(Float.self))
    }
}

final class NemotronAdapter: AudioAdapter {
    override var architecture: String { "nemotron3_diarization" }
    override var operations: Set<String> { ["diarize", "diarize_session"] }
    var backend: ArsenalNemotronBackend?
    var diarizer: Nemotron3Diarizer?
    private var sessions: [String: Nemotron3StreamingSession] = [:]
    private var sessionSamples: [String: Int] = [:]
    private var sessionUsed: [String: Double] = [:]
    /// The session clock (seconds); tests move it to check that abandoned sessions give up their slot.
    var now: () -> Double = { ProcessInfo.processInfo.systemUptime }
    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        let layers = context.randomWeights ? Int(context.config["lab_layers"]?.number ?? 31) : 31
        guard (1...31).contains(layers) else { throw audioFailure("Invalid diarizer layer count.") }
        let network = Nemotron3MLXNetwork(layerCount: layers)
        try prepare(
            network, context, cancellation: cancellation,
            map: { name, tensor in
                if name.hasPrefix("preprocessor.") { return nil }
                return (name.replacingOccurrences(of: ".ffn.net.0.", with: ".ffn.linear1.").replacingOccurrences(of: ".ffn.net.3.", with: ".ffn.linear2."), tensor)
            })
        let backend = ArsenalNemotronBackend(network)
        self.backend = backend
        diarizer = Nemotron3Diarizer(backend: backend)
    }
    override func unload() { sessions.removeAll(); sessionSamples.removeAll(); sessionUsed.removeAll(); diarizer = nil; backend = nil; super.unload() }
    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        if operation == "diarize_session" { return try session(input, cancellation: cancellation) }
        try audioKeys(input, allowed: ["audio"])
        guard operation == "diarize", let diarizer, let backend else { throw audioFailure("Diarizer is not loaded.") }
        guard input["min_speakers"] == nil && input["max_speakers"] == nil else { throw audioFailure("Nemotron predicts up to eight speakers without a requested speaker count.") }
        backend.cancellation = cancellation
        try cancellation.check()
        let pcm = try audioInput(input, rate: 16000)
        let result = try diarizer.diarize(audio: pcm.samples, sampleRate: pcm.sampleRate)
        try cancellation.check()
        let duration = Double(pcm.samples.count) / 16000
        let segments: [JSONValue] = result.segments.map { item in
            [
                "start": .number(max(0, min(duration, Double(item.startTime)))), "end": .number(max(0, min(duration, Double(item.endTime)))),
                "speaker": .string("speaker-\(item.speakerId)")
            ]
        }
        return ["segments": .array(segments), "speakers": .array(Array(Set(result.segments.map { "speaker-\($0.speakerId)" })).sorted().map(JSONValue.string))]
    }
    private func result(_ result: DiarizationResult, duration: Double) -> JSONObject {
        let segments: [JSONValue] = result.segments.map { item in
            [
                "start": .number(max(0, min(duration, Double(item.startTime)))), "end": .number(max(0, min(duration, Double(item.endTime)))),
                "speaker": .string("speaker-\(item.speakerId)")
            ]
        }
        return ["segments": .array(segments), "speakers": .array(Array(Set(result.segments.map { "speaker-\($0.speakerId)" })).sorted().map(JSONValue.string))]
    }
    private func drop(_ id: String) { sessions.removeValue(forKey: id); sessionSamples.removeValue(forKey: id); sessionUsed.removeValue(forKey: id) }
    private func session(_ input: JSONObject, cancellation: NativeCancellation) throws -> JSONObject {
        try audioKeys(input, allowed: ["action", "session_id", "audio", "chunk_frames", "right_context_frames"])
        guard let diarizer, let backend, let action = input["action"]?.string else { throw audioFailure("A loaded diarizer and session action are required.") }
        backend.cancellation = cancellation
        try cancellation.check()
        if action == "create" {
            // A session untouched for ten minutes was abandoned by its client (as Sortformer v2); it no longer holds a slot.
            let time = now()
            for (id, used) in sessionUsed where SortformerV2Adapter.abandoned(lastUsed: used, now: time) { drop(id) }
            guard sessions.count < 2, input["audio"] == nil else { throw audioFailure("At most two diarization sessions can remain open.") }
            let id = input["session_id"]?.string ?? UUID().uuidString.lowercased()
            guard !id.isEmpty, id.count <= 128, sessions[id] == nil else { throw audioFailure("Session IDs must be unique and contain 1...128 characters.") }
            let chunk = try audioInteger(input, key: "chunk_frames", fallback: 6, range: 1...340),
                right = try audioInteger(input, key: "right_context_frames", fallback: 7, range: 0...40)
            guard (1...340).contains(chunk), (0...40).contains(right), chunk + right <= 380 else { throw audioFailure("Invalid stream chunk geometry.") }
            sessions[id] = try diarizer.makeStreamingSession(coreEncoderFrames: chunk, rightContextEncoderFrames: right)
            sessionSamples[id] = 0; sessionUsed[id] = time
            return ["session_id": .string(id), "done": false, "segments": [], "speakers": []]
        }
        guard let id = input["session_id"]?.string, let stream = sessions[id] else { throw audioFailure("The diarization session is closed or belongs to an unloaded model.") }
        guard input["chunk_frames"] == nil, input["right_context_frames"] == nil else { throw audioFailure("Chunk geometry is set when creating the session.") }
        sessionUsed[id] = now()
        if action == "close" {
            guard input["audio"] == nil else { throw audioFailure("Close accepts a session ID without audio.") }
            drop(id)
            return ["session_id": .string(id), "done": true, "segments": [], "speakers": []]
        }
        let value: DiarizationResult
        if action == "push" {
            let pcm = try audioInput(input, rate: 16000, seconds: 30)
            let count = sessionSamples[id, default: 0] + pcm.samples.count
            guard count <= 16000 * 600 else { throw audioFailure("A diarization session is bounded to ten minutes of audio.") }
            do { value = try stream.push(audio: pcm.samples) } catch { drop(id); throw error }
            sessionSamples[id] = count
        } else if action == "finish", input["audio"] == nil {
            do { value = try stream.finish() } catch { drop(id); throw error }
        } else {
            throw audioFailure("Use create, push, finish or close for a diarization session.")
        }
        var output = result(value, duration: Double(sessionSamples[id, default: 0]) / 16000)
        output["session_id"] = .string(id); output["done"] = .bool(action == "finish")
        if action == "finish" { drop(id) }
        try cancellation.check()
        return output
    }

}
