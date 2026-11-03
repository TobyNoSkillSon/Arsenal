// SPDX-License-Identifier: MIT
import Foundation
import ImageIO
import ArsenalCore
import MCP

/// A finished run as the agent receives it: text written to be relayed, the non-text blocks, and the raw data for
/// `format:"json"`.
struct MCPRendered: Sendable {
    var text: String
    var extra: [Tool.Content] = []
    var json: JSONObject
}

extension APIRouter {
    /// Runs one operation inside a job and renders its output. Content is never trimmed: every text result is also
    /// saved to `<job directory>/output.txt`, and the header line names that file when there is a body.
    func mcpRun(
        entry: CatalogEntry, operation: ModelOperation, input: JSONObject, requester: String, play: Bool, job: MCPJob, directory: URL
    ) async throws -> MCPRendered {
        var params = input
        if ["audio", "image", "file"].contains(operation.output) { params["out_dir"] = .string(directory.path) }
        if operation.inputSchema["properties"]?["stream"] != nil { params["stream"] = false }
        await job.enter(.running)
        let started = mcpEnvironment.now()
        let result = try await service.run(
            id: entry.id, op: operation.name, params: params,
            sink: { frame in
                if case .json(let event) = frame, event["event"]?.string == "progress" { await job.runProgress(event) }
            })
        try RequestedOutputSchema.validate(result, operation: operation.name, input: input)
        let seconds = mcpEnvironment.now().timeIntervalSince(started)
        if ["audio", "image", "file"].contains(operation.output) {
            var voice = input["voice"]?.string
            if play && voice == nil { voice = (try? await service.settings(id: entry.id))?.first { $0.id == "voice" }?.value.string }
            return try await mcpRenderFiles(result, entry: entry, operation: operation, seconds: seconds, directory: directory, requester: requester, voice: voice, play: play)
        }
        return try mcpRenderText(result, entry: entry, operation: operation, input: input, seconds: seconds, directory: directory)
    }

    nonisolated func mcpRenderText(_ result: JSONObject, entry: CatalogEntry, operation: ModelOperation, input: JSONObject, seconds: Double, directory: URL) throws -> MCPRendered {
        var payload = result; payload["ok"] = nil; payload["id"] = nil
        let id = entry.id, took = "took " + MCPFormat.elapsed(seconds)
        let outputText = directory.appendingPathComponent("output.txt"), outputJSON = directory.appendingPathComponent("output.json")
        func saveJSON() throws { try mcpPretty(.object(payload)).write(to: outputJSON, atomically: true, encoding: .utf8) }
        var header: String, body: [String] = []
        switch operation.name {
        case _ where payload["embeddings"]?.array != nil:
            let vectors = payload["embeddings"]?.array ?? [], dims = vectors.first?.array?.count ?? 0
            try saveJSON()
            header =
                "Embedded \(MCPFormat.plural(vectors.count, "input")) with \(id) (\(took)): \(vectors.count) vectors × \(dims) dims in \(outputJSON.path) (JSON key \"embeddings\")."
        case _ where payload["detections"]?.array != nil:
            let detections = payload["detections"]?.array ?? []
            let system = payload["coordinate_system"]?.string.map { "; coordinates: \($0)" } ?? ""
            let verb = operation.name == "ground_screen" ? "Located" : "Found"
            header = "\(verb) \(MCPFormat.plural(detections.count, operation.name == "ground_screen" ? "target" : "object")) with \(id) (\(took)\(system))."
            body = detections.enumerated().map { index, item in
                var line = "\(index + 1). \(item["label"]?.string ?? "object")"
                if let score = item["score"]?.number { line += " " + MCPFormat.number(score) }
                if let box = item["box"] {
                    func v(_ key: String) -> String { MCPFormat.number(box[key]?.number ?? 0, decimals: 3) }
                    line += " — box (\(v("xmin")), \(v("ymin")))–(\(v("xmax")), \(v("ymax")))"
                }
                if let point = item["point"] {
                    line += " — point (\(MCPFormat.number(point["x"]?.number ?? 0, decimals: 3)), \(MCPFormat.number(point["y"]?.number ?? 0, decimals: 3)))"
                }
                return line
            }
        case _ where payload["prediction"] != nil && payload["probability"]?.number != nil:
            let probability = payload["probability"]?.number ?? 0
            let ended = payload["prediction"]?.bool ?? (payload["prediction"]?.number.map { $0 >= 0.5 } ?? (probability >= 0.5))
            header = "\(ended ? "End of turn" : "Not the end of the turn") (probability \(MCPFormat.number(probability))), from \(id)."
        case _ where payload["results"]?.array?.first?["relevance_score"] != nil:
            let results = payload["results"]?.array ?? [], documents = input["documents"]?.array ?? []
            header = "Ranked \(MCPFormat.plural(results.count, "document")) with \(id) (\(took))."
            body = results.enumerated().map { rank, item in
                let index = Int(item["index"]?.number ?? -1)
                let document = index >= 0 && index < documents.count ? documents[index] : nil
                let text = (document?.string ?? document?["text"]?.string).map { ": " + mcpSnippet($0) } ?? ""
                return "\(rank + 1). document \(index) — \(MCPFormat.number(item["relevance_score"]?.number ?? 0, decimals: 3))\(text)"
            }
        case _ where payload["translations"]?.array != nil:
            let translations = payload["translations"]?.array ?? []
            if translations.count == 1, let only = translations.first {
                let from = MCPFormat.language(only["detected_source_language"]?.string).map { "from \($0), " } ?? ""
                header = "Translation from \(id) (\(from)\(took))."
                body = [only["text"]?.string ?? ""]
            } else {
                header = "\(MCPFormat.plural(translations.count, "translation")) from \(id) (\(took))."
                body = translations.enumerated().map { "\($0.offset + 1). " + ($0.element["text"]?.string ?? "") }
            }
        case _ where payload["pages"]?.array != nil:
            let pages = payload["pages"]?.array ?? []
            let images = pages.reduce(0) { $0 + ($1["images"]?.array?.count ?? 0) }
            try saveJSON()
            header = "Read \(MCPFormat.plural(pages.count, "page")) with \(id) (\(took))."
            if images > 0 { header += " \(MCPFormat.plural(images, "embedded image")) are in \(outputJSON.path)." }
            body = pages.enumerated().flatMap { index, page in ["Page \(Int(page["index"]?.number ?? Double(index)) + 1):", page["markdown"]?.string ?? ""] }
        case _ where operation.name == "transcribe" || (payload["text"]?.string != nil && payload["segments"]?.array != nil && payload["duration"] != nil):
            var facts: [String] = []
            if let duration = payload["duration"]?.number { facts.append("\(MCPFormat.elapsed(duration)) of audio") }
            if let language = MCPFormat.language(payload["language"]?.string) { facts.append(language) }
            facts.append(took)
            header = "Transcript from \(id) (\(facts.joined(separator: ", ")))."
            let timed = input.contains { $0.key.contains("timestamp") && $0.value.bool == true }
            if timed, let segments = payload["segments"]?.array, !segments.isEmpty {
                body = segments.map { mcpTimed($0) }
            } else {
                body = [payload["text"]?.string ?? ""]
            }
        case _ where payload["segments"]?.array?.contains(where: { $0["speaker"] != nil }) == true:
            let segments = payload["segments"]?.array ?? []
            let speakers = Set(segments.compactMap { $0["speaker"]?.string }).count
            header = "Speakers from \(id) (\(MCPFormat.plural(speakers, "speaker")), \(MCPFormat.plural(segments.count, "segment")), \(took))."
            body = segments.map { mcpTimed($0) }
        case _ where payload["answers"]?.object != nil || payload["results"]?.array?.first?["answers"] != nil:
            if let answers = payload["answers"]?.object {
                header = "Decisions from \(id) (\(took))."
                body = answers.keys.sorted().map { "\($0): " + mcpAnswer(answers[$0]!) }
            } else {
                let results = payload["results"]?.array ?? []
                header = "Decisions for \(MCPFormat.plural(results.count, "item")) from \(id) (\(took))."
                body = results.map { item in
                    let index = Int(item["index"]?.number ?? 0) + 1
                    guard item["ok"]?.bool != false, let answers = item["answers"]?.object else { return "\(index). error: \(item["error"]?.string ?? "failed")" }
                    return "\(index). " + answers.keys.sorted().map { "\($0) " + mcpAnswer(answers[$0]!) }.joined(separator: " · ")
                }
            }
        case "extract":
            header = "Extracted data with \(id) (\(took))."
            body = [try mcpPretty(payload["data"] ?? .object(payload))]
        case _ where payload["text"]?.string != nil:
            let tokens = payload["usage"]?["completion_tokens"]?.number ?? payload["usage"]?["output_tokens"]?.number
            header = "Reply from \(id) (\(tokens.map { MCPFormat.plural(Int($0), "token") + ", " } ?? "")\(took))."
            body = [payload["text"]?.string ?? ""]
        case _ where payload.keys.sorted() == ["tokens"] || (operation.name == "count_tokens" && payload["tokens"]?.number != nil):
            header = "\(MCPFormat.plural(Int(payload["tokens"]?.number ?? 0), "token")), counted by \(id)."
        default:
            header = "Result of \(operation.name) from \(id) (\(took))."
            body = [try mcpPretty(.object(payload))]
        }
        if !body.isEmpty {
            header += " Full text: \(outputText.path)"
        }
        let text = ([header] + body).joined(separator: "\n")
        try text.write(to: outputText, atomically: true, encoding: .utf8)
        return MCPRendered(text: text, json: payload)
    }

    func mcpRenderFiles(
        _ result: JSONObject, entry: CatalogEntry, operation: ModelOperation, seconds: Double, directory: URL, requester: String, voice: String?, play: Bool
    ) async throws -> MCPRendered {
        let linked = try linkedFileResult(result, directory: directory)
        var files: [JSONValue] = [], links: [Tool.Content] = [], lines: [String] = [], total: Int64 = 0
        var playback: String?
        for value in linked["files"]?.array ?? [] {
            guard let path = value["path"]?.string else { continue }
            let url = try outputPath(path)
            guard url.path.hasPrefix(directory.resolvingSymlinksInPath().path + "/") else { throw MCPFailure(.failed, "Not run: the runner wrote a file outside its job folder.") }
            let bytes = Int64(value["bytes"]?.number ?? 0), mime = value["mime"]?.string ?? contentType(url)
            total += bytes
            let id = UUID().uuidString.lowercased(), uri = "arsenal://outputs/" + id
            if mcpOutputs.count >= 1024, !mcpOutputOrder.isEmpty { mcpOutputs.removeValue(forKey: mcpOutputOrder.removeFirst()) }
            mcpOutputs[id] = url; mcpOutputOrder.append(id)
            var metadata = value.object ?? [:]
            metadata["path"] = .string(url.path); metadata["uri"] = .string(uri); metadata["url"] = nil
            files.append(.object(metadata))
            links.append(.resourceLink(uri: uri, name: url.lastPathComponent, mimeType: mime))
            var facts: [String] = []
            if mime.hasPrefix("image/"), let size = mcpPixelSize(url) { facts.append("\(size.0)×\(size.1)") }
            if mime.hasPrefix("audio/"), let duration = result["duration_s"]?.number ?? value["duration_s"]?.number { facts.append(MCPFormat.number(duration, decimals: 1) + " s") }
            facts.append(url.pathExtension.uppercased()); facts.append(MCPFormat.size(bytes))
            lines.append(url.path + " (" + facts.joined(separator: ", ") + ")")
            if mime.hasPrefix("image/"), let thumbnail = mcpThumbnail(url) {
                links.append(.image(data: thumbnail.base64EncodedString(), mimeType: "image/png", annotations: nil, _meta: nil))
            }
            if play, mime.hasPrefix("audio/") {
                do {
                    playback =
                        try await self.playback.enqueue(path: url, requester: requester, model: entry.id, voice: voice) == nil ? "Not played: playback is muted." : "Playing now."
                } catch { playback = "Not played: \(error.localizedDescription) The file is kept." }
            }
        }
        guard !files.isEmpty else { throw MCPFailure(.failed, "Not run: \(entry.name) returned no file.") }
        let images = files.allSatisfy { ($0["mime"]?.string ?? "").hasPrefix("image/") }, audio = files.allSatisfy { ($0["mime"]?.string ?? "").hasPrefix("audio/") }
        let noun = images ? "image" : audio ? "audio file" : "file"
        var header = "Created \(MCPFormat.plural(files.count, noun)) with \(entry.id) (\(MCPFormat.size(total)), took \(MCPFormat.elapsed(seconds)))"
        if let seed = result["seed"]?.number { header += ", seed \(MCPFormat.number(seed, decimals: 0))" }
        var text = ([header + ":"] + lines).joined(separator: "\n")
        if let playback { text += "\n" + playback }
        var json = linked; json["ok"] = nil; json["id"] = nil; json["files"] = .array(files); json["model"] = .string(entry.id)
        if let playback { json["playback"] = .string(playback) }
        try text.write(to: directory.appendingPathComponent("output.txt"), atomically: true, encoding: .utf8)
        return MCPRendered(text: text, extra: links, json: json)
    }
}

/// Pretty JSON with sorted keys: the content of an extract result or an unknown output, exactly as produced.
func mcpPretty(_ value: JSONValue) throws -> String {
    let encoder = JSONEncoder(); encoder.outputFormatting = [.prettyPrinted, .sortedKeys, .withoutEscapingSlashes]
    return String(decoding: try encoder.encode(value), as: UTF8.self)
}
func mcpSnippet(_ text: String) -> String {
    let flat = text.replacingOccurrences(of: "\n", with: " ")
    return flat.count > 100 ? String(flat.prefix(100)) + "…" : flat
}
/// `[00:01.2–00:04.5] SPEAKER_00: text`.
func mcpTimed(_ segment: JSONValue) -> String {
    let start = MCPFormat.clock(segment["start"]?.number ?? 0), end = MCPFormat.clock(segment["end"]?.number ?? 0)
    let speaker = segment["speaker"]?.string.map { $0 + ": " } ?? ""
    let text = segment["text"]?.string?.trimmingCharacters(in: .whitespaces) ?? ""
    return "[\(start)–\(end)] \(speaker)\(text)".trimmingCharacters(in: .whitespaces)
}
/// One typed decision: "yes 0.77", "no 0.81", "frontend 0.62", "score 1.07 (confidence 0.57)".
func mcpAnswer(_ answer: JSONValue) -> String {
    if let yes = answer["noul"]?.number { return yes >= 0.5 ? "yes \(MCPFormat.number(yes))" : "no \(MCPFormat.number(1 - yes))" }
    if let choice = answer["choice"]?.string {
        let probability = answer["probabilities"]?[choice]?.number ?? answer["confidence"]?.number
        return choice + (probability.map { " " + MCPFormat.number($0) } ?? "")
    }
    if let score = answer["score"]?.number { return "score \(MCPFormat.number(score))" + (answer["confidence"]?.number.map { " (confidence \(MCPFormat.number($0)))" } ?? "") }
    return (try? mcpPretty(answer)) ?? ""
}
func mcpPixelSize(_ url: URL) -> (Int, Int)? {
    guard let source = CGImageSourceCreateWithURL(url as CFURL, nil),
        let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
        let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int
    else { return nil }
    return (width, height)
}
