// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

extension APIRouter {
    func deepl(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        if request.path.hasSuffix("/languages") {
            guard request.method == "GET" else { throw HTTPFailure(405, "Use GET.") }
            var languages = Set<String>()
            for row in await installedModels() {
                guard let id = row["id"] as? String, let operation = await service.operations(id: id).first(where: { $0.name == "translate" }) else { continue }
                let key = request.query["type"] == "source" ? "source_lang" : "target_lang"
                for value in operation.inputSchema["properties"]?[key]?["enum"]?.array ?? [] { if let language = value.string { languages.insert(language) } }
            }
            try await connection.respond(languages.sorted().map { ["language": $0, "name": $0] }); return
        }
        let body: JSONObject
        if request.headers["content-type"]?.hasPrefix("application/x-www-form-urlencoded") == true {
            guard let form = String(data: request.body, encoding: .utf8) else { throw HTTPFailure(422, "Invalid form.") }
            let items = URLComponents(string: "http://localhost/?" + form.replacingOccurrences(of: "+", with: "%20"))?.queryItems ?? []
            var value: JSONObject = [:]
            for item in items {
                if item.name == "text" { value["text"] = .array((value["text"]?.array ?? []) + [.string(item.value ?? "")]) } else { value[item.name] = .string(item.value ?? "") }
            }; body = value
        } else {
            body = try object(request)
        }
        try reject(body, ["glossary_id", "glossary_ids", "tag_handling", "formality", "model_type"])
        var params: JSONObject = ["text": body["text"] ?? .null, "target_lang": body["target_lang"] ?? .null]
        for key in ["source_lang", "context"] { params[key] = body[key] }
        let result = try await capabilityRun(body, operation: "translate", params: params)
        guard let translations = result["translations"]?.array, let texts = params["text"]?.array, translations.count == texts.count else {
            throw HTTPFailure(502, "Translation count does not match input count.")
        }
        let translated: [JSONValue] = translations.enumerated().map { index, value in
            var item = value.object ?? [:]
            item["billed_characters"] = .number(Double(texts[index].string?.unicodeScalars.count ?? 0))
            return .object(item)
        }
        try await connection.respond(["translations": translated.map(plain)])
    }
    func mistralOCR(_ request: HTTPRequest, _ connection: HTTPConnection) async throws {
        let body = try object(request)
        guard let document = body["document"]?.object, let type = document["type"]?.string else { throw HTTPFailure(422, "document must declare document_url or image_url type.") }
        let path: String
        if let local = document["path"]?.string {
            path = local
        } else if type == "image_url" {
            path = try inlineMedia(document["image_url"]?.string, category: "image")
        } else if type == "document_url" {
            path = try inlineMedia(document["document_url"]?.string, category: "application")
        } else {
            throw HTTPFailure(422, "Use an inline image/PDF data URL or an uploaded document path.")
        }
        var params: JSONObject = ["document": ["path": .string(path)]]
        for key in ["pages", "include_image_base64", "table_format", "document_annotation_format"] { params[key] = body[key] }
        let result = try await capabilityRun(body, operation: "read_document", params: params)
        var response = result; response["model"] = .string(try await capabilityModel(body, operation: "read_document"))
        guard let pages = result["pages"]?.array else { throw HTTPFailure(502, "OCR model returned no pages.") }
        response["pages"] = .array(
            pages.map { value in
                var page = value.object ?? [:]
                page["images"] = page["images"] ?? []
                page["dimensions"] = page["dimensions"] ?? .null
                return .object(page)
            })
        response["usage_info"] = result["usage_info"] ?? ["pages_processed": .number(Double(pages.count))]
        try await connection.respond(plain(response))
    }
}
