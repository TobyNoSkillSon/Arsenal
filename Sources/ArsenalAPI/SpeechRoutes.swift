// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func speech(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let body = try object(request), id = try required(body, "model")
        _ = try await resolve(id, kind: "voices", operation: "speak")
        let format = body["response_format"]?.string ?? "mp3"
        guard ["wav", "pcm"].contains(format) else { throw HTTPFailure(400, "Only wav and pcm are supported; request response_format=wav or pcm explicitly.") }
        let params = try speechParams(body)
        let (bytes, rate, channels) = try await renderSpeech(id: id, params: params)
        let pcm = try PCM.convert(bytes, inputRate: rate, inputChannels: channels, outputRate: 24000)
        let output = format == "wav" ? PCM.wav(pcm, rate: 24000) : pcm
        try await connection.respondBytes(output, type: format == "wav" ? "audio/wav" : "audio/pcm")
    }
    /// The kind adapter preserves every optional input declared by speech model folders.
    /// Voice presets are optional: cloning uses reference and design uses instructions.
    func speechParams(_ body: JSONObject) throws -> JSONObject {
        try reject(body, ["stream_format"])
        var params: JSONObject = ["text": .string(try required(body, "input")), "stream": false]
        if let value = body["voice"] {
            guard let voice = value.string ?? value["id"]?.string else { throw HTTPFailure(422, "voice must be a local voice ID.") }
            params["voice"] = .string(voice)
        }
        for key in ["speed", "instructions", "description", "language", "reference", "seed", "max_tokens", "temperature", "top_p", "top_k", "repetition_penalty", "guidance"] {
            params[key] = body[key]
        }
        return params
    }
    func renderSpeech(id: String, params: JSONObject) async throws -> (Data, Int, Int) {
        try await renderAudio(id: id, op: "speak", params: params)
    }
    func singleAudioFilePath(_ result: JSONObject) throws -> String? {
        if let path = result["path"]?.string { return path }
        guard let files = result["files"]?.array else { return nil }
        guard files.count == 1, let path = files[0]["path"]?.string,
            files[0]["mime"]?.string?.hasPrefix("audio/") != false
        else { throw HTTPFailure(502, "This audio profile requires exactly one audio file.") }
        return path
    }
    func renderAudio(id: String, op: String, params: JSONObject) async throws -> (Data, Int, Int) {
        // Consume framed PCM directly. HTTP speech has no durable file consumer,
        // so do not create a job or ask the worker to write a second audio copy.
        var params = params
        let directory = op == "speak" ? nil : try newJob()
        defer { if let directory { removeTransientJob(directory) } }
        if op == "speak" { params["stream"] = true; params["out_dir"] = nil } else { params["out_dir"] = directory.map { .string($0.path) } }
        let collector = AudioCollector()
        let result = try await run(id, op: op, params: params, sink: { frame in try await collector.add(frame) })
        let gathered = await collector.result()
        if let path = try singleAudioFilePath(result) {
            let file = try outputPath(path)
            guard let size = try file.resourceValues(forKeys: [.fileSizeKey]).fileSize, size <= PCM.maxBytes else {
                throw HTTPFailure(500, "Worker audio exceeds the 64 MiB limit.")
            }
            let data = try Data(contentsOf: file)
            if data.starts(with: Data("RIFF".utf8)) { return try PCM.readWAV(data) }
            return (
                data, try jsonInteger(result["sample_rate"], name: "Worker sample rate", range: 8000...192000, default: gathered.1, status: 500),
                try jsonInteger(result["channels"], name: "Worker channels", range: 1...8, default: gathered.2, status: 500)
            )
        }
        guard !gathered.0.isEmpty else { throw HTTPFailure(500, "Runner returned no PCM audio.") }
        return gathered
    }
}

private actor AudioCollector {
    var bytes = Data(), rate = 24000, channels = 1
    func add(_ frame: WireFrame) throws {
        switch frame {
        case .json(let value):
            if value["event"]?.string == "audio_begin" {
                rate = try jsonInteger(value["sample_rate"], name: "Worker sample rate", range: 8000...192000, default: 24000, status: 500)
                channels = try jsonInteger(value["channels"], name: "Worker channels", range: 1...8, default: 1, status: 500)
            }
        case .binary(_, let chunk):
            guard chunk.count <= PCM.maxBytes - bytes.count else { throw HTTPFailure(500, "Worker audio exceeds the 64 MiB limit.") }
            bytes.append(chunk)
        }
    }
    func result() -> (Data, Int, Int) { (bytes, rate, channels) }
}

enum PCM {
    static let maxBytes = 64 * 1024 * 1024
    static func wav(_ data: Data, rate: Int, channels: Int = 1) -> Data {
        var output = Data("RIFF".utf8)
        func u32(_ n: Int) { var n = UInt32(n).littleEndian; withUnsafeBytes(of: &n) { output.append(contentsOf: $0) } }
        func u16(_ n: Int) { var n = UInt16(n).littleEndian; withUnsafeBytes(of: &n) { output.append(contentsOf: $0) } }
        u32(data.count + 36); output += Data("WAVEfmt ".utf8); u32(16); u16(1); u16(channels); u32(rate); u32(rate * channels * 2); u16(channels * 2); u16(16)
        output += Data("data".utf8); u32(data.count); output += data
        return output
    }
    static func readWAV(_ data: Data) throws -> (Data, Int, Int) {
        func u16(_ i: Int) -> Int { Int(data[i]) | Int(data[i + 1]) << 8 }
        func u32(_ i: Int) -> Int { u16(i) | u16(i + 2) << 16 }
        guard data.count >= 44, data[8..<12] == Data("WAVE".utf8) else { throw HTTPFailure(500, "Invalid worker WAV.") }
        var offset = 12, rate = 0, channels = 0
        while offset + 8 <= data.count {
            let size = u32(offset + 4), start = offset + 8
            guard size <= data.count - start else { throw HTTPFailure(500, "Invalid WAV chunk size.") }
            if data[offset..<offset + 4] == Data("fmt ".utf8) {
                guard size >= 16, u16(start) == 1, u16(start + 14) == 16 else { throw HTTPFailure(500, "Worker must return signed 16-bit PCM.") }
                channels = u16(start + 2); rate = u32(start + 4)
                guard (1...8).contains(channels), (8000...192000).contains(rate) else { throw HTTPFailure(500, "Invalid worker WAV rate or channels.") }
            }
            if data[offset..<offset + 4] == Data("data".utf8), rate > 0, channels > 0 { return (Data(data[start..<start + size]), rate, channels) }
            offset = start + size + size % 2
        }
        throw HTTPFailure(500, "WAV has no PCM data.")
    }
    // Linear interpolation is deterministic and does not load an inference or external codec library.
    static func convert(_ data: Data, inputRate: Int, inputChannels: Int, outputRate: Int) throws -> Data {
        guard (8000...192000).contains(inputRate), (8000...192000).contains(outputRate), (1...8).contains(inputChannels), data.count <= maxBytes else {
            throw HTTPFailure(500, "Invalid worker PCM format or size.")
        }
        guard data.count >= inputChannels * 2 else { return Data() }
        let count = data.count / (2 * inputChannels)
        let outputCount = floor(Double(count) * Double(outputRate) / Double(inputRate))
        guard outputCount <= Double(maxBytes / 2), let frames = Int(exactly: outputCount) else { throw HTTPFailure(500, "Converted audio exceeds the 64 MiB limit.") }
        let samples: [Double] = (0..<count).map { frame in
            (0..<inputChannels).reduce(0.0) { sum, channel in
                let i = (frame * inputChannels + channel) * 2
                return sum + Double(Int16(bitPattern: UInt16(data[i]) | UInt16(data[i + 1]) << 8)) / Double(inputChannels)
            }
        }
        var output = Data()
        for i in 0..<frames {
            let x = Double(i) * Double(inputRate) / Double(outputRate), a = min(count - 1, Int(x)), b = min(count - 1, a + 1)
            var n = Int16(max(-32768, min(32767, (samples[a] + (samples[b] - samples[a]) * (x - Double(a))).rounded()))).littleEndian
            withUnsafeBytes(of: &n) { output.append(contentsOf: $0) }
        }
        return output
    }
}
