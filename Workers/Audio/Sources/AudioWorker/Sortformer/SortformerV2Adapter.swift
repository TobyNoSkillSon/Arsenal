// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import MLXNN
import ModelPrecision
import ArsenalWire

extension NativeCancellation: NativeCancellationLike {}

/// Reads members of an uncompressed POSIX tar (a NeMo `.nemo` archive) by name, without extracting them.
enum SortformerNemoArchive {
    struct Member: Equatable { let offset: UInt64; let size: UInt64 }

    static func members(_ file: FileHandle) throws -> [String: Member] {
        let end = try file.seekToEnd()
        var offset: UInt64 = 0, out: [String: Member] = [:]
        while offset + 512 <= end, out.count < 10_000 {
            try file.seek(toOffset: offset)
            guard let header = try file.read(upToCount: 512), header.count == 512 else { throw PrecisionError("Truncated .nemo archive") }
            if header.allSatisfy({ $0 == 0 }) { break }
            func field(_ start: Int, _ length: Int) -> String {
                let bytes = header[header.startIndex + start..<header.startIndex + start + length].prefix { $0 != 0 }
                return String(decoding: bytes, as: UTF8.self)
            }
            let sizeText = field(124, 12).trimmingCharacters(in: .whitespaces)
            guard let size = UInt64(sizeText, radix: 8) else { throw PrecisionError("Invalid .nemo member size") }
            var name = field(0, 100)
            let prefix = field(345, 155)
            if field(257, 5) == "ustar", !prefix.isEmpty { name = prefix + "/" + name }
            if name.hasPrefix("./") { name.removeFirst(2) }
            let type = header[header.startIndex + 156]
            let data = offset + 512
            guard data <= end, size <= end - data else { throw PrecisionError(".nemo member exceeds the archive") }
            if type == 0x30 || type == 0 { out[name] = Member(offset: data, size: size) }
            offset = data + (size + 511) / 512 * 512
        }
        return out
    }
}

/// Get-time conversion of the pinned `.nemo` (NeMo 2.2 tar: model_config.yaml + PyTorch model_weights.ckpt, FP32) into
/// an FP16 safetensors snapshot under NeMo's own tensor names. Conv kernels move to MLX's channels-last layout; the
/// preprocessor's filterbank and window are checked against the Swift frontend and dropped; BatchNorm step counters are
/// dropped. 16-bit only.
enum SortformerV2Conversion {
    static let recipe = "cast:audio-sortformer-v2-v1"
    static let base = NativeConversion(recipe: recipe, convert: prepare)
    static let marker = "arsenal-sortformer-layout.json"

    /// Checkpoint name → (snapshot name, axes to transpose) or nil to drop.
    static func target(_ name: String, shape: [Int]) throws -> (String, [Int]?)? {
        if name.hasSuffix(".num_batches_tracked") || name.hasPrefix("preprocessor.") { return nil }
        guard name.hasPrefix("encoder.") || name.hasPrefix("transformer_encoder.") || name.hasPrefix("sortformer_modules.") else {
            throw PrecisionError("Unexpected Sortformer tensor: " + name)
        }
        if name.hasSuffix(".weight"), shape.count == 4 { return (name, [0, 2, 3, 1]) } // Conv2d [O, I, H, W] → [O, H, W, I]
        if name.hasSuffix(".weight"), shape.count == 3 { return (name, [0, 2, 1]) } // Conv1d [O, I, K] → [O, K, I]
        return (name, nil)
    }

    /// Dimensions read from the checkpoint's shapes; the network built from them must match every tensor.
    static func dimensions(_ shapes: [String: [Int]]) throws -> SortformerV2Dimensions {
        func count(_ prefix: String) -> Int {
            Set(
                shapes.keys.compactMap { key -> Int? in
                    guard key.hasPrefix(prefix) else { return nil }
                    return Int(key.dropFirst(prefix.count).prefix { $0 != "." })
                }
            ).count
        }
        guard let out = shapes["encoder.pre_encode.out.weight"], let conv0 = shapes["encoder.pre_encode.conv.0.weight"],
            let dw = shapes["encoder.layers.0.conv.depthwise_conv.weight"], let bias = shapes["encoder.layers.0.self_attn.pos_bias_u"],
            let inner = shapes["transformer_encoder.layers.0.second_sub_layer.dense_in.weight"],
            let speakers = shapes["sortformer_modules.single_hidden_to_spks.weight"]
        else { throw PrecisionError("Sortformer checkpoint is missing its encoder or head") }
        var d = SortformerV2Dimensions()
        d.conformerWidth = out[0]; d.subsamplingChannels = conv0[0]; d.melBins = out[1] / conv0[0] * 8
        d.conformerKernel = dw[2]; d.conformerHeads = bias[0]
        d.conformerLayers = count("encoder.layers."); d.transformerLayers = count("transformer_encoder.layers.")
        d.transformerWidth = inner[1]; d.transformerInner = inner[0]; d.speakers = speakers[0]
        d.transformerHeads = 8 // NeMo stores fused-free q/k/v; the head count comes from model_config.yaml (num_attention_heads: 8).
        return d
    }

    static func prepare(_ context: NativeConversionContext, _ cancellation: NativeCancellation) throws -> JSONObject {
        guard context.bits == nil else { throw PrecisionError("Sortformer ships at 16-bit only") }
        let files = try NativeConversionContext.files(in: context.source).filter { $0.hasSuffix(".nemo") }
        guard files.count == 1 else { throw PrecisionError("Sortformer source needs exactly one .nemo archive") }
        let handle = try FileHandle(forReadingFrom: context.source.appendingPathComponent(files[0]))
        defer { try? handle.close() }
        let members = try SortformerNemoArchive.members(handle)
        guard let weights = members["model_weights.ckpt"], let yaml = members["model_config.yaml"], yaml.size <= 1 << 20 else {
            throw PrecisionError(".nemo archive lacks model_weights.ckpt or model_config.yaml")
        }
        let checkpoint = try TorchCheckpoint(handle, base: weights.offset, size: weights.size)
        let tensors = try checkpoint.tensors()
        // The frontend's fixed tables must equal what the Swift features compute.
        let frontend = SortformerV2Features()
        if let fb = tensors["preprocessor.featurizer.fb"] {
            let stored = try checkpoint.array(fb).asType(.float32).asArray(Float.self)
            guard stored.count == frontend.filterbank.count,
                zip(stored, frontend.filterbank).allSatisfy({ abs($0 - $1) <= 1e-6 + 1e-4 * abs($1) })
            else { throw PrecisionError("Sortformer mel filterbank differs from librosa Slaney 128×257") }
        }
        if let window = tensors["preprocessor.featurizer.window"] {
            let stored = try checkpoint.array(window).asType(.float32).asArray(Float.self)
            let left = (SortformerV2Features.fft - SortformerV2Features.window) / 2
            guard stored.count == SortformerV2Features.window,
                zip(stored, frontend.hann[left..<left + SortformerV2Features.window]).allSatisfy({ abs($0 - $1) <= 1e-6 })
            else { throw PrecisionError("Sortformer window differs from a 400-point symmetric Hann") }
        }
        var mapping: [String: (TorchCheckpoint.Tensor, [Int]?)] = [:]
        var shapes: [String: [Int]] = [:]
        for (name, tensor) in tensors {
            try cancellation.check()
            guard let (key, axes) = try target(name, shape: tensor.shape) else { continue }
            mapping[key] = (tensor, axes)
            shapes[name] = tensor.shape
        }
        let dims = try dimensions(shapes)
        let expected = Dictionary(uniqueKeysWithValues: SortformerV2Network(dims).parameters().flattened().map { ($0.0, $0.1.shape) })
        let converted = Dictionary(
            uniqueKeysWithValues: mapping.map { key, value in
                (key, value.1.map { axes in axes.map { value.0.shape[$0] } } ?? value.0.shape)
            })
        guard Set(expected.keys) == Set(converted.keys), converted.allSatisfy({ expected[$0.key] == $0.value }) else {
            let missing = Set(expected.keys).subtracting(converted.keys).sorted().prefix(4), extra = Set(converted.keys).subtracting(expected.keys).sorted().prefix(4)
            throw PrecisionError("Sortformer checkpoint does not match the network (missing \(Array(missing)), unexpected \(Array(extra)))")
        }
        _ = try StreamedPrecision.convert(
            keys: mapping.keys.sorted(),
            read: { key in
                try cancellation.check()
                let (tensor, axes) = mapping[key]!
                let array = try checkpoint.array(tensor)
                return axes.map { array.transposed(axes: $0) } ?? array
            }, to: context.destination.appendingPathComponent("model.safetensors"), half: context.half, eligible: { _, _ in false })
        try handle.seek(toOffset: yaml.offset)
        try (handle.read(upToCount: Int(yaml.size)) ?? Data()).write(to: context.destination.appendingPathComponent("model_config.yaml"))
        let config: [String: Any] = [
            "architecture": "sortformer_streaming", "mel_bins": dims.melBins, "conformer_layers": dims.conformerLayers,
            "conformer_width": dims.conformerWidth, "conformer_heads": dims.conformerHeads, "conformer_kernel": dims.conformerKernel,
            "subsampling_channels": dims.subsamplingChannels, "transformer_layers": dims.transformerLayers, "transformer_width": dims.transformerWidth,
            "transformer_heads": dims.transformerHeads, "transformer_inner": dims.transformerInner, "speakers": dims.speakers
        ]
        try JSONSerialization.data(withJSONObject: config, options: [.sortedKeys]).write(to: context.destination.appendingPathComponent("config.json"))
        try JSONSerialization.data(withJSONObject: ["architecture": "sortformer_streaming", "layout_version": 1], options: [.sortedKeys])
            .write(to: context.destination.appendingPathComponent(marker))
        return try context.receipt(cancellation: cancellation)
    }
}

/// Streaming Sortformer 4spk v2: `diarize` (a whole recording, streamed the maker's way) and `diarize_session`
/// (live create / push / finish / close). Output follows the audio kind's diarization contract.
final class SortformerV2Adapter: AudioAdapter {
    override var architecture: String { "sortformer_streaming" }
    override var operations: Set<String> { ["diarize", "diarize_session"] }
    static let maximumSeconds = 3_600
    /// Audio one live session may hold.
    static let sessionSeconds = 600
    /// Idle time after which a live session counts as abandoned (Sortformer v2 and Nemotron); separate from the audio cap.
    static let idleSeconds = 600
    var network: SortformerV2Network?
    let features = SortformerV2Features()
    private var sessions: [String: Session] = [:]
    /// The session clock (seconds); tests move it to check that abandoned sessions give up their slot.
    var now: () -> Double = { ProcessInfo.processInfo.systemUptime }

    final class Session {
        let stream: SortformerV2Stream
        let post: SortformerV2PostProcessing
        let includeProbabilities: Bool
        var pcm: [Float] = []
        var lastUsed = ProcessInfo.processInfo.systemUptime
        init(stream: SortformerV2Stream, post: SortformerV2PostProcessing, includeProbabilities: Bool) {
            self.stream = stream; self.post = post; self.includeProbabilities = includeProbabilities
        }
    }

    override func load(_ context: NativeLoadContext, cancellation: NativeCancellation) throws {
        var config: [String: Any] = [:]
        let file = context.weights.appendingPathComponent("config.json")
        if FileManager.default.fileExists(atPath: file.path) {
            config = (try JSONSerialization.jsonObject(with: Data(contentsOf: file)) as? [String: Any]) ?? [:]
        }
        if context.randomWeights, let layers = context.config["lab_layers"]?.number {
            config["conformer_layers"] = Int(layers); config["transformer_layers"] = Int(layers)
        }
        let network = SortformerV2Network(SortformerV2Dimensions(config))
        try prepare(network, context, cancellation: cancellation, eligible: { _, _ in false })
        self.network = network
    }

    override func unload() { sessions.removeAll(); network = nil; super.unload() }

    /// A live session untouched for ten minutes is dropped when a new session needs its slot.
    static func abandoned(lastUsed: Double, now: Double) -> Bool { now - lastUsed >= Double(idleSeconds) }

    static func geometry(_ input: JSONObject) throws -> SortformerV2Geometry {
        let name = input["latency"]?.string ?? SortformerV2Geometry.defaultPreset
        guard var g = SortformerV2Geometry.presets[name] else { throw audioFailure("latency must be very_high, high, low or ultra_low.") }
        g.chunk = try audioInteger(input, key: "chunk_frames", fallback: g.chunk, range: 1...1_000)
        g.rightContext = try audioInteger(input, key: "right_context_frames", fallback: g.rightContext, range: 0...100)
        g.fifo = try audioInteger(input, key: "fifo_frames", fallback: g.fifo, range: 0...1_000)
        g.updatePeriod = try audioInteger(input, key: "update_period_frames", fallback: g.updatePeriod, range: 1...1_000)
        g.cache = try audioInteger(input, key: "speaker_cache_frames", fallback: g.cache, range: 16...1_000)
        try g.validate()
        return g
    }

    static func postProcessing(_ input: JSONObject) throws -> SortformerV2PostProcessing {
        func number(_ key: String, _ fallback: Float, _ range: ClosedRange<Double>) throws -> Float {
            guard let value = input[key] else { return fallback }
            guard let x = value.number, x.isFinite, range.contains(x) else { throw audioFailure("\(key) must be within \(range.lowerBound)...\(range.upperBound).") }
            return Float(x)
        }
        return SortformerV2PostProcessing(
            onset: try number("onset", 0.5, 0...1), offset: try number("offset", 0.5, 0...1),
            padOnset: try number("pad_onset", 0, 0...10), padOffset: try number("pad_offset", 0, 0...10),
            minDurationOn: try number("min_duration_on", 0, 0...60), minDurationOff: try number("min_duration_off", 0, 0...60))
    }

    static let optionKeys: Set<String> = [
        "latency", "chunk_frames", "right_context_frames", "fifo_frames", "update_period_frames", "speaker_cache_frames",
        "onset", "offset", "pad_onset", "pad_offset", "min_duration_on", "min_duration_off", "include_probabilities"
    ]

    /// The audio kind's diarization output: segments with SPEAKER_0n labels (arrival order), the speakers present, RTTM,
    /// and on request the 80 ms speaker-probability frames.
    static func output(_ probabilities: [Float], post: SortformerV2PostProcessing, duration: Double, includeProbabilities: Bool) -> JSONObject {
        let segments = post.segments(probabilities).compactMap { s -> SortformerV2PostProcessing.Segment? in
            let start = min(max(0, s.start), duration), end = min(s.end, duration)
            return end > start ? .init(start: start, end: end, speaker: s.speaker) : nil
        }
        func label(_ i: Int) -> String { String(format: "SPEAKER_%02d", i) }
        var out: JSONObject = [
            "segments": .array(segments.map { .object(["start": .number($0.start), "end": .number($0.end), "speaker": .string(label($0.speaker))]) }),
            "speakers": .array(Set(segments.map(\.speaker)).sorted().map { .string(label($0)) }),
            "rttm": .string(
                segments.map { String(format: "SPEAKER audio 1 %.3f %.3f <NA> <NA> %@ <NA> <NA>", $0.start, $0.end - $0.start, label($0.speaker)) }
                    .joined(separator: "\n"))
        ]
        if includeProbabilities {
            let frames = probabilities.count / 4
            out["frame_seconds"] = .number(0.08)
            out["probabilities"] = .array((0..<frames).map { f in .array((0..<4).map { .number((Double(probabilities[f * 4 + $0]) * 10_000).rounded() / 10_000) }) })
        }
        return out
    }

    func diarize(_ samples: [Float], geometry: SortformerV2Geometry, cancellation: NativeCancellation?) throws -> [Float] {
        guard let network else { throw audioFailure("Diarizer is not loaded.") }
        let stream = SortformerV2Stream(network: network, geometry: geometry)
        stream.cancellation = cancellation
        let frames = SortformerV2Features.frameCount(samples: samples.count)
        let mel = features.features(samples, frames: 0..<frames)
        try stream.advance(totalFrames: frames, final: true) { range in Array(mel[(range.lowerBound * 128)..<(range.upperBound * 128)]) }
        return stream.probabilities
    }

    override func run(_ operation: String, input: JSONObject, cancellation: NativeCancellation, emit: (WireFrame) throws -> Void) throws -> JSONObject {
        if operation == "diarize_session" { return try session(input, cancellation: cancellation) }
        guard operation == "diarize" else { throw audioFailure("Unsupported audio operation: " + operation) }
        try audioKeys(input, allowed: Self.optionKeys.union(["audio"]))
        let geometry = try Self.geometry(input), post = try Self.postProcessing(input)
        let pcm = try audioInput(input, rate: 16_000, seconds: Self.maximumSeconds)
        try cancellation.check()
        let probabilities = try diarize(pcm.samples, geometry: geometry, cancellation: cancellation)
        return Self.output(probabilities, post: post, duration: Double(pcm.samples.count) / 16_000, includeProbabilities: input["include_probabilities"]?.bool ?? false)
    }

    private func session(_ input: JSONObject, cancellation: NativeCancellation) throws -> JSONObject {
        try audioKeys(input, allowed: Self.optionKeys.union(["action", "session_id", "audio"]))
        guard let network, let action = input["action"]?.string else { throw audioFailure("A loaded diarizer and session action are required.") }
        if action == "create" {
            // A session untouched for ten minutes was abandoned by its client; it no longer holds a slot.
            let time = now()
            sessions = sessions.filter { !Self.abandoned(lastUsed: $0.value.lastUsed, now: time) }
            guard sessions.count < 2, input["audio"] == nil else { throw audioFailure("At most two diarization sessions can remain open.") }
            let id = input["session_id"]?.string ?? UUID().uuidString.lowercased()
            guard !id.isEmpty, id.count <= 128, sessions[id] == nil else { throw audioFailure("Session IDs must be unique and contain 1...128 characters.") }
            let stream = SortformerV2Stream(network: network, geometry: try Self.geometry(input))
            let session = Session(stream: stream, post: try Self.postProcessing(input), includeProbabilities: input["include_probabilities"]?.bool ?? false)
            session.lastUsed = time
            sessions[id] = session
            return ["session_id": .string(id), "done": false, "segments": [], "speakers": [], "rttm": ""]
        }
        guard let id = input["session_id"]?.string, let session = sessions[id] else { throw audioFailure("The diarization session is closed or belongs to an unloaded model.") }
        guard Set(input.keys).isDisjoint(with: Self.optionKeys) else { throw audioFailure("Streaming and post-processing settings are set when creating the session.") }
        session.stream.cancellation = cancellation
        session.lastUsed = now()
        let finishing: Bool
        switch action {
        case "close":
            guard input["audio"] == nil else { throw audioFailure("Close accepts a session ID without audio.") }
            sessions.removeValue(forKey: id)
            return ["session_id": .string(id), "done": true, "segments": [], "speakers": [], "rttm": ""]
        case "push":
            let pcm = try audioInput(input, rate: 16_000, seconds: 30)
            guard session.pcm.count + pcm.samples.count <= 16_000 * Self.sessionSeconds else {
                throw audioFailure("A diarization session is bounded to ten minutes of audio.")
            }
            session.pcm += pcm.samples
            finishing = false
        case "finish":
            guard input["audio"] == nil else { throw audioFailure("Finish accepts a session ID without audio.") }
            finishing = true
        default:
            throw audioFailure("Use create, push, finish or close for a diarization session.")
        }
        // The stream advances a whole chunk at a time, so after a cancelled or failed push its state is still
        // consistent: the session stays open with the audio received so far, and later pushes or finish continue it.
        let pcm = session.pcm
        let total =
            finishing
            ? SortformerV2Features.frameCount(samples: pcm.count)
            : min(SortformerV2Features.frameCount(samples: pcm.count), SortformerV2Features.completeFrames(available: pcm.count))
        try session.stream.advance(totalFrames: total, final: finishing) { range in self.features.features(pcm, frames: range) }
        let confirmedSeconds = Double(session.stream.probabilities.count / 4) * 0.08
        var out = Self.output(
            session.stream.probabilities, post: session.post,
            duration: finishing ? Double(session.pcm.count) / 16_000 : min(confirmedSeconds, Double(session.pcm.count) / 16_000),
            includeProbabilities: session.includeProbabilities)
        out["session_id"] = .string(id)
        out["done"] = .bool(finishing)
        if finishing { sessions.removeValue(forKey: id) }
        return out
    }
}
