// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func embedding(_ request: HTTPRequest, _ connection: HTTPConnection, cohere: Bool = false) async throws {
        let body = try object(request)
        var inputs: [JSONValue]
        if cohere {
            if let texts = body["texts"]?.array {
                inputs = texts
            } else if let images = body["images"]?.array {
                inputs = try images.map { try embeddingPart(["type": "image_url", "image_url": ["url": $0]]) }
            } else if let parts = body["inputs"]?.array {
                inputs = try parts.map { input in
                    guard let content = input["content"]?.array, !content.isEmpty && content.count <= 32 else {
                        throw HTTPFailure(422, "Each Cohere input needs 1...32 text, image or audio parts.")
                    }
                    return ["type": "content", "content": .array(try content.map { try embeddingPart($0) })]
                }
            } else {
                throw HTTPFailure(422, "Supply texts, images or inputs.")
            }
        } else if let input = body["input"]?.string {
            inputs = [.string(input)]
        } else if let values = body["input"]?.array {
            inputs = try values.map { $0.object == nil ? $0 : try embeddingPart($0) }
        } else {
            throw HTTPFailure(422, "input must be text or an array of text/image/audio parts.")
        }
        var params: JSONObject = ["input": .array(inputs)]
        let task = body["task"]?.string ?? body["input_type"]?.string
        if let task { params["task"] = .string(["search_query": "query", "search_document": "document"].mapValue(task)) }
        for key in ["instruction", "normalize", "truncate"] { params[key] = body[key] }
        params["dimensions"] = body["dimensions"] ?? body["output_dimension"]
        if cohere {
            if let truncation = body["truncate"]?.string {
                guard ["NONE", "END"].contains(truncation) else { throw HTTPFailure(400, "Only NONE and END truncation are supported.") }
                params["truncate"] = .bool(truncation == "END")
            }
            let types = body["embedding_types"]?.array ?? ["float"]
            guard types == [.string("float")] else { throw HTTPFailure(400, "This profile supports float embeddings only.") }
        }
        let result = try await capabilityRun(body, operation: "embed", params: params)
        guard let vectors = result["embeddings"]?.array, vectors.count == inputs.count else { throw HTTPFailure(502, "Embedding count does not match input count.") }
        let model = try await capabilityModel(body, operation: "embed")
        let tokens = result["input_tokens"]?.number ?? 0
        if cohere {
            try await connection.respond([
                "id": UUID().uuidString, "embeddings": ["float": vectors.map(plain)], "texts": body["texts"].map(plain) ?? [],
                "meta": ["api_version": ["version": "2"], "billed_units": ["input_tokens": tokens]]
            ]); return
        }
        let encoding = body["encoding_format"]?.string ?? "float"
        guard ["float", "base64"].contains(encoding) else { throw HTTPFailure(400, "encoding_format must be float or base64.") }
        let data: [[String: Any]] = try vectors.enumerated().map { index, vector in
            guard let numbers = vector.array?.compactMap(\.number), numbers.count == vector.array?.count, !numbers.isEmpty,
                numbers.allSatisfy({ $0.isFinite && Float($0).isFinite })
            else { throw HTTPFailure(502, "Runner returned an invalid embedding.") }
            let encoded: Any
            if encoding == "base64" {
                var bytes = Data()
                for value in numbers { var bits = Float(value).bitPattern.littleEndian; withUnsafeBytes(of: &bits) { bytes.append(contentsOf: $0) } }
                encoded = bytes.base64EncodedString()
            } else {
                encoded = numbers
            }
            return ["object": "embedding", "index": index, "embedding": encoded]
        }
        try await connection.respond(["object": "list", "data": data, "model": model, "usage": ["prompt_tokens": tokens, "total_tokens": tokens]])
    }
    func embeddingPart(_ part: JSONValue) throws -> JSONValue {
        switch part["type"]?.string {
        case "text":
            guard let text = part["text"]?.string else { throw HTTPFailure(422, "Text part needs text.") }; return .string(text)
        case "image_url":
            let url = part["image_url"]?["url"]?.string ?? part["image_url"]?.string
            return ["type": "image", "image": ["path": .string(try inlineMedia(url, category: "image"))]]
        case "input_audio":
            guard let format = part["input_audio"]?["format"]?.string, ["wav", "mp3"].contains(format), let data = part["input_audio"]?["data"]?.string else {
                throw HTTPFailure(422, "Audio part needs data and wav/mp3 format.")
            }
            return ["type": "audio", "audio": ["path": .string(try inlineMedia("data:audio/" + format + ";base64," + data, category: "audio"))]]
        default: throw HTTPFailure(422, "Unsupported embedding content part.")
        }
    }
    func inlineMedia(_ value: String?, category: String) throws -> String {
        guard let value, value.hasPrefix("data:" + category + "/"), let comma = value.firstIndex(of: ","), value[..<comma].hasSuffix(";base64"),
            let data = Data(base64Encoded: String(value[value.index(after: comma)...])), !data.isEmpty, data.count <= 16 * 1024 * 1024
        else {
            throw HTTPFailure(422, "Use an inline base64 \(category) data URL; remote URLs are not fetched.")
        }
        let mime = String(value.dropFirst(5).prefix(while: { $0 != ";" }))
        let extensions = ["image/png": "png", "image/jpeg": "jpg", "audio/wav": "wav", "audio/mp3": "mp3", "audio/mpeg": "mp3", "application/pdf": "pdf"]
        guard let ext = extensions[mime] else { throw HTTPFailure(422, "Unsupported media type \(mime).") }
        guard (try? support.appendingPathComponent("inbox").resourceValues(forKeys: [.isSymbolicLinkKey]).isSymbolicLink) != true else {
            throw HTTPFailure(422, "Inbox must be a directory, not a symlink.")
        }
        let directory = support.appendingPathComponent("inbox/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true, attributes: [.posixPermissions: 0o700])
        let url = directory.appendingPathComponent("input." + ext); try data.write(to: url, options: .atomic)
        return url.path
    }
    func rerank(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let body = try object(request)
        guard let documents = body["documents"]?.array else { throw HTTPFailure(422, "documents must be an array.") }
        let texts = try documents.map { value -> JSONValue in
            if value.string != nil { return value }
            guard let text = value["text"]?.string else { throw HTTPFailure(422, "Each document must be text or {text}.") }; return .string(text)
        }
        var params: JSONObject = ["query": body["query"] ?? .null, "documents": .array(texts)]
        for key in ["top_n", "max_tokens_per_doc", "instruction", "return_embeddings"] { params[key] = body[key] }
        let result = try await capabilityRun(body, operation: "rerank", params: params)
        guard let ranked = result["results"]?.array else { throw HTTPFailure(502, "Runner returned no ranking.") }
        let results: [Any] = try ranked.map { value in
            guard var item = value.object, let index = try? jsonInteger(item["index"], name: "index", range: 0...max(0, texts.count - 1), status: 502),
                texts.indices.contains(index),
                let score = item["relevance_score"]?.number, score.isFinite, (0...1).contains(score)
            else { throw HTTPFailure(502, "Runner returned an invalid ranking index or score.") }
            if body["return_documents"]?.bool == true { item["document"] = ["text": texts[index]] }
            return plain(item)
        }
        try await connection.respond(["id": UUID().uuidString, "results": results, "meta": ["api_version": ["version": "2"], "billed_units": ["search_units": 0]]])
    }
}

private extension Dictionary where Key == String, Value == String {
    func mapValue(_ key: String) -> String { self[key] ?? key }
}
