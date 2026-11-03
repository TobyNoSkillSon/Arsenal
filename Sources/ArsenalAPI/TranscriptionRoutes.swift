// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func transcribe(_ request: HTTPRequest, _ connection: HTTPConnection, translate: Bool = false) async throws {
        guard request.method == "POST" else { throw HTTPFailure(405, "Use POST.") }
        let parts = try Multipart.parse(request.body, contentType: request.headers["content-type"] ?? "")
        func fields(_ name: String) -> [String] { parts.filter { $0.name == name }.compactMap { String(data: request.body[$0.range], encoding: .utf8) } }
        func field(_ name: String) -> String? { fields(name).first }
        guard let file = parts.first(where: { $0.name == "file" }), !file.range.isEmpty else { throw HTTPFailure(422, "file is required.") }
        let model = try await capabilityModel(["model": field("model").map(JSONValue.string) ?? .null], operation: "transcribe")
        let format = field("response_format") ?? "json"
        guard ["json", "text", "verbose_json", "diarized_json", "srt", "vtt"].contains(format) else { throw HTTPFailure(400, "Unsupported transcription response_format.") }
        for key in ["include[]", "known_speaker_names[]", "known_speaker_references[]", "temperature"] where field(key) != nil {
            throw HTTPFailure(400, "\(key) is not supported by this local profile.")
        }
        if let chunking = field("chunking_strategy"), chunking != "auto" { throw HTTPFailure(400, "Only auto chunking_strategy is supported.") }
        let granularities = fields("timestamp_granularities[]")
        guard granularities.allSatisfy({ ["word", "segment"].contains($0) }) else { throw HTTPFailure(422, "Unknown timestamp granularity.") }
        let directory = try newJob(); defer { removeTransientJob(directory) }
        let url = directory.appendingPathComponent("input." + audioExtension(file.filename, contentType: file.contentType))
        try Data(request.body[file.range]).write(to: url)
        var params: JSONObject = ["audio": ["path": .string(url.path)]]
        if translate { params["task"] = "translate" }
        if let language = field("language") { params["language"] = .string(language) }
        if let prompt = field("prompt") { params["prompt"] = .string(prompt) }
        if granularities.contains("word") { params["word_timestamps"] = true }
        if format == "diarized_json" { params["diarize"] = true }
        let streaming = field("stream") == "true"
        if let raw = field("stream"), !["true", "false"].contains(raw) { throw HTTPFailure(422, "stream must be true or false.") }
        if streaming { params["stream"] = true }
        guard let declaration = await service.operations(id: model).first(where: { $0.name == "transcribe" }) else { throw HTTPFailure(422, "Model cannot transcribe.") }
        try OperationSchemaValidator.validate(.object(params), schema: .object(declaration.inputSchema))
        if streaming {
            guard declaration.stream, ["json", "diarized_json"].contains(format) else { throw HTTPFailure(400, "Streaming requires a streaming model and json or diarized_json.") }
            try await connection.beginStream()
            let result = try await run(
                model, op: "transcribe", params: params,
                sink: { frame in
                    guard case .json(let event) = frame else { return }
                    if let delta = event["delta"]?.string ?? event["text"]?.string, event["event"]?.string != "progress" {
                        try await connection.event(["type": "transcript.text.delta", "delta": delta], name: "transcript.text.delta")
                    }
                    if let segment = event["segment"]?.object {
                        var value = plain(segment); value["type"] = "transcript.text.segment"
                        try await connection.event(value, name: "transcript.text.segment")
                    }
                })
            try await connection.event(["type": "transcript.text.done", "text": result["text"]?.string ?? ""], name: "transcript.text.done")
            try await connection.finishStream(); return
        }
        let result = try await run(model, op: "transcribe", params: params)
        guard let text = result["text"]?.string else { throw HTTPFailure(502, "Runner returned no transcript text.") }
        switch format {
        case "text": try await connection.respondBytes(Data(text.utf8), type: "text/plain; charset=utf-8")
        case "srt", "vtt":
            guard let segments = result["segments"]?.array else { throw HTTPFailure(422, "This model did not return segment timestamps.") }
            try await connection.respondBytes(Data(try TranscriptCaptions.render(segments, vtt: format == "vtt").utf8), type: format == "vtt" ? "text/vtt" : "application/x-subrip")
        case "verbose_json", "diarized_json":
            var value = result; value["task"] = .string(translate ? "translate" : "transcribe")
            if format == "diarized_json" {
                guard let segments = result["segments"]?.array, segments.allSatisfy({ $0["speaker"]?.string != nil }) else {
                    throw HTTPFailure(502, "Diarized output requires a speaker on every segment.")
                }
                value["segments"] = .array(
                    segments.enumerated().map { index, segment in
                        var item = segment.object ?? [:]; item["id"] = .string(String(index)); item["type"] = "transcript.text.segment"; return .object(item)
                    })
            }
            try await connection.respond(plain(value))
        default: try await connection.respond(["text": text])
        }
    }
}

enum TranscriptCaptions {
    static func render(_ segments: [JSONValue], vtt: Bool) throws -> String {
        var output = vtt ? "WEBVTT\n\n" : ""
        func time(_ seconds: Double) -> String {
            let ms = Int((seconds * 1000).rounded())
            return String(format: "%02d:%02d:%02d%@%03d", ms / 3_600_000, ms / 60_000 % 60, ms / 1000 % 60, vtt ? "." : ",", ms % 1000)
        }
        for (index, segment) in segments.enumerated() {
            guard let start = segment["start"]?.number, let end = segment["end"]?.number, start.isFinite, end.isFinite,
                start >= 0, end >= start, end <= 86400, let text = segment["text"]?.string
            else { throw HTTPFailure(502, "Invalid transcript segment.") }
            output += "\(index + 1)\n\(time(start)) --> \(time(end))\n\(text)\n\n"
        }
        return output
    }
}
