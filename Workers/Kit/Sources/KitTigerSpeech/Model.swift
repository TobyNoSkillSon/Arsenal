// SPDX-License-Identifier: MIT
// TIGER speech separation (JusperLee/TIGER-speech) and cinematic dialog/effects/music separation (JusperLee/TIGER-DnR).
// The network is in Tiger.swift; this file is the request path. The checkpoint decides the variant: a DnR checkpoint
// holds three TIGER networks under dialog./effect./music. and runs the maker's wav_chunk_inference (12 s windows, 4 s hop);
// the speech checkpoint runs one TIGER pass over the whole file, as inference_speech.py does.
import AVFoundation
import Foundation
import MLX
import MLXNN

/// The kind in model.toml (an Arsenal kind such as "embeddings", or the folder's "local.<name>").
let modelKind = "audio"
/// The [[op]] names in model.toml. Each one arrives in `run(op:params:)`.
let modelOps = ["separate_audio"]

private struct Stem { let id: String, label: String, aliases: [String] }

final class Model {
    private let config: TigerConfig
    private let speech: Tiger?
    private let dnr: TigerDnR?
    private let stems: [Stem]
    /// Speech: one whole-file pass, so attention memory grows with the square of the length.
    private let maxSeconds: Double

    init(weightsDir: URL, modelDir: URL, settings: [String: Any]) throws {
        // Many differently sized temporaries: without a limit MLX's buffer cache grows to several times the live peak.
        Memory.cacheLimit = 1 << 30
        var weights = try Weights.load(weightsDir)
        let isDnR = weights.keys.contains { $0.hasPrefix("dialog.") }
        config = try TigerConfig(try Files.json(weightsDir), bandPlan: isDnR ? "dnr" : "speech")
        for (key, value) in weights {
            if value.ndim == 3 {
                weights[key] = value.transposed(0, 2, 1)  // Conv1d [out, in, k] to [out, k, in]
            } else if value.ndim == 4 {
                if key.hasSuffix(".gamma") || key.hasSuffix(".beta") {
                    weights[key] = value.reshaped([-1])  // LayerNormalization4D [1, C, 1, 1]
                } else {
                    weights[key] = value.reshaped([value.dim(0), 1, value.dim(1)])  // 1x1 Conv2d [out, in, 1, 1]
                }
            }
        }
        if isDnR {
            let net = TigerDnR(config)
            try Weights.apply(weights, to: net, rename: childNames)
            dnr = net; speech = nil
            stems = [
                Stem(id: "dialog", label: "Dialogue", aliases: ["dialog", "dialogue", "speech", "voice", "vocals"]),
                Stem(id: "effects", label: "Effects", aliases: ["effects", "effect", "sfx", "fx"]),
                Stem(id: "music", label: "Music", aliases: ["music"]),
            ]
            maxSeconds = 300
        } else {
            let net = Tiger(config)
            try Weights.apply(weights, to: net, rename: childNames)
            speech = net; dnr = nil
            stems = (1...config.numSources).map {
                Stem(id: "speaker\($0)", label: "Speaker \($0)", aliases: ["speaker\($0)", "speaker_\($0)", "spk\($0)", "\($0)"])
            }
            maxSeconds = 60
        }
    }

    func run(op: String, params: Params) throws -> [String: Any] {
        guard op == "separate_audio" else { throw RunnerError.invalid("Unknown op \(op).") }
        let url = try params.file("audio")
        let wanted = try chosenStems(params)
        // The file's length from its header, before decoding: an empty file or one over the limit is refused here.
        if let file = try? AVAudioFile(forReading: url) {
            let rate = file.fileFormat.sampleRate
            guard file.length > 0, rate > 0 else { throw RunnerError.invalid("\(url.lastPathComponent) holds no audio frames.") }
            let headerSeconds = Double(file.length) / rate
            guard headerSeconds <= maxSeconds + 1 else {
                throw RunnerError.invalid("The audio is \(Int(headerSeconds)) s; at most \(Int(maxSeconds)) s can be separated in one request.")
            }
        }
        let samples = try Audio.load(url, sampleRate: Double(config.sampleRate))
        let n = samples.dim(0)
        let seconds = Double(n) / Double(config.sampleRate)
        guard n >= config.win else {
            throw RunnerError.invalid("The audio is \(String(format: "%.3f", seconds)) s; at least \(config.win) samples (\(String(format: "%.2f", Double(config.win) / Double(config.sampleRate))) s) are needed.")
        }
        guard seconds <= maxSeconds else {
            throw RunnerError.invalid("The audio is \(Int(seconds)) s; at most \(Int(maxSeconds)) s can be separated in one request.")
        }

        // Each stem's samples, [n] float32.
        var tracks: [String: MLXArray] = [:]
        let needed = Set(wanted.map(\.id))
        if let speech {
            let out = try speech.separate(samples, cancelled: params.checkCancelled)
            for (i, stem) in stems.enumerated() { tracks[stem.id] = out[i] }
        } else if let dnr {
            let plan: [(String, Tiger, Int)] = [("dialog", dnr.dialog, 2), ("effects", dnr.effect, 1), ("music", dnr.music, 0)]
            // In remove mode every other stem is returned, so compute exactly those.
            for (id, net, track) in plan where needed.contains(id) {
                tracks[id] = try chunked(net, samples, track: track, params: params)
            }
        }

        if ProcessInfo.processInfo.environment["ARSENAL_TIGER_MEMORY"] != nil {
            FileHandle.standardError.write("tiger: mlx peak \(Memory.peakMemory / 1_000_000) MB, cache \(Memory.cacheMemory / 1_000_000) MB\n".data(using: .utf8)!)
        }
        Memory.clearCache()
        let dir = try outputDirectory(params)
        var files: [[String: Any]] = []
        for (order, stem) in wanted.enumerated() {
            guard let track = tracks[stem.id] else { continue }
            let file = dir.appendingPathComponent("\(stem.id).wav")
            try writeWAV(track, sampleRate: config.sampleRate, to: file)
            if let dump = ProcessInfo.processInfo.environment["ARSENAL_PARITY_DIR"] {
                let raw = track.asType(.float32).asArray(Float.self)
                try raw.withUnsafeBufferPointer { Data(buffer: $0) }.write(to: URL(fileURLWithPath: dump).appendingPathComponent("\(stem.id).f32"))
            }
            files.append([
                "id": stem.id, "path": file.path, "name": "\(stem.id).wav", "mime": "audio/wav", "label": stem.label, "order": order,
                "bytes": try Files.size(file), "sha256": try Files.sha256(file),
            ])
        }
        return ["files": files]
    }

    /// `stems` picks outputs (default all); `mode: remove` returns every stem not listed.
    private func chosenStems(_ params: Params) throws -> [Stem] {
        let mode = params.string("mode", default: "extract")
        guard mode == "extract" || mode == "remove" else { throw RunnerError.invalid("mode must be extract or remove.") }
        guard params.raw["stems"] != nil else {
            if mode == "remove" { throw RunnerError.invalid("mode remove needs stems: the stems to leave out.") }
            return stems
        }
        let names = try params.strings("stems").map { $0.lowercased().trimmingCharacters(in: .whitespaces) }
        var picked: [Stem] = []
        for name in names {
            guard let stem = stems.first(where: { $0.aliases.contains(name) }) else {
                throw RunnerError.invalid("Unknown stem \(name). This model returns \(stems.map(\.id).joined(separator: ", ")).")
            }
            if !picked.contains(where: { $0.id == stem.id }) { picked.append(stem) }
        }
        if mode == "remove" {
            let rest = stems.filter { s in !picked.contains { $0.id == s.id } }
            guard !rest.isEmpty else { throw RunnerError.invalid("Removing every stem leaves nothing to return.") }
            return rest
        }
        return picked
    }

    /// The maker's TIGERDNR.wav_chunk_inference: 12 s windows every 4 s, zero-padded, summed and divided by 3.
    private func chunked(_ net: Tiger, _ samples: MLXArray, track: Int, params: Params) throws -> MLXArray {
        let sr = config.sampleRate, length = samples.dim(0)
        let session = sr * 12, hop = sr * 4, edge = session - hop
        let paddedSamples = concatenated([MLXArray.zeros([edge]), samples.asType(.float32), MLXArray.zeros([edge])])
        let total = paddedSamples.dim(0)
        let sessions = (total - session) / hop + 2
        var sum = [Float](repeating: 0, count: total)
        for i in 0..<sessions {
            try params.checkCancelled()
            let start = i * hop
            let end = min(start + session, total)
            guard end > start else { continue }
            var input = paddedSamples[start..<end]
            if end - start < session { input = concatenated([input, MLXArray.zeros([session - (end - start)])]) }
            let out = try net.separate(input, cancelled: params.checkCancelled)[track][0..<(end - start)]
            let values = out.asType(.float32).asArray(Float.self)
            for j in 0..<values.count { sum[start + j] += values[j] }
            Memory.clearCache()
        }
        return MLXArray(Array(sum[edge..<(edge + length)])) / 3
    }
}

/// Numbered nn.Sequential / ModuleList children get names (MLX reads numbered keys as arrays).
private func childNames(_ key: String) -> String? {
    var parts = key.split(separator: ".").map(String.init)
    for i in 1..<parts.count {
        let parent = parts[i - 1], grand = i >= 2 ? parts[i - 2] : ""
        switch (grand, parent, parts[i]) {
        case ("BN", _, "0"): parts[i] = "norm"
        case ("BN", _, "1"): parts[i] = "conv"
        case ("mask", _, "0"): parts[i] = "act"
        case ("mask", _, "1"): parts[i] = "conv"
        case (_, "concat_block", "0"): parts[i] = "conv"
        case (_, "concat_block", "1"): parts[i] = "act"
        case (_, "freq_path", "0"), (_, "frame_path", "0"): parts[i] = "uconv"
        case (_, "freq_path", "1"), (_, "frame_path", "1"): parts[i] = "attention"
        case (_, "freq_path", "2"), (_, "frame_path", "2"): parts[i] = "norm"
        default: break
        }
    }
    return parts.joined(separator: ".")
}

// MARK: - Files

/// Arsenal passes `out_dir` (a job folder under <support>/jobs/); outside Arsenal, a new folder under $ARSENAL_SUPPORT_DIR/jobs.
private func outputDirectory(_ params: Params) throws -> URL {
    let dir: URL
    if let given = params.raw["out_dir"] as? String, !given.isEmpty {
        dir = URL(fileURLWithPath: given)
    } else {
        let support = ProcessInfo.processInfo.environment["ARSENAL_SUPPORT_DIR"] ?? NSTemporaryDirectory()
        dir = URL(fileURLWithPath: support).appendingPathComponent("jobs").appendingPathComponent(UUID().uuidString)
    }
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    return dir
}

/// Mono 16-bit PCM WAV, clipped to -1...1.
private func writeWAV(_ samples: MLXArray, sampleRate: Int, to url: URL) throws {
    let values = clip(samples.asType(.float32), min: -1, max: 1).asArray(Float.self)
    let settings: [String: Any] = [
        AVFormatIDKey: kAudioFormatLinearPCM, AVSampleRateKey: Double(sampleRate), AVNumberOfChannelsKey: 1,
        AVLinearPCMBitDepthKey: 16, AVLinearPCMIsFloatKey: false, AVLinearPCMIsBigEndianKey: false,
    ]
    let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: 1, interleaved: false)!
    guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(max(values.count, 1))) else {
        throw RunnerError.failed("Out of memory.")
    }
    buffer.frameLength = AVAudioFrameCount(values.count)
    values.withUnsafeBufferPointer { buffer.floatChannelData![0].update(from: $0.baseAddress!, count: values.count) }
    do {
        let file = try AVAudioFile(forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32, interleaved: false)
        try file.write(from: buffer)
    } catch { throw RunnerError.failed("Could not write \(url.lastPathComponent): \(error.localizedDescription)") }
}
