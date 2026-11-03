// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore

private func addOutputSchema(_ op: ModelOperation, schemaID: String, schemas: inout JSONObject, content: inout JSONObject) {
    if let outputSchema = op.outputSchema {
        var output = outputSchema
        output["$id"] = nil
        func outputRefs(_ value: JSONValue) -> JSONValue {
            if var object = value.object {
                if let ref = object["$ref"]?.string, ref.hasPrefix("#/") {
                    object["$ref"] = .string("#/components/schemas/" + schemaID + "_output" + String(ref.dropFirst()))
                }
                return .object(object.mapValues(outputRefs))
            }
            if let array = value.array { return .array(array.map(outputRefs)) }; return value
        }
        schemas[schemaID + "_output"] = outputRefs(.object(output))
        content["application/json"] = ["schema": ["$ref": .string("#/components/schemas/" + schemaID + "_output")]]
    }
}

extension APIRouter {
    func openAPI() async throws -> JSONObject {
        var paths: JSONObject = [:], schemas: JSONObject = [:]
        for route in dialects.routes {
            var definition: JSONObject = [
                "summary": .string(route.summary), "operationId": .string(route.method + "_" + route.path.replacingOccurrences(of: "/", with: "_")),
                "responses": [
                    "200": ["description": "Successful local response"], "422": ["description": "Invalid input"], "404": ["description": "Model or endpoint not installed"]
                ], "x-arsenal-op": .string(route.operation)
            ]
            if let kind = route.kind { definition["x-arsenal-kind"] = .string(kind) }
            let parameters = route.path.split(separator: "/").filter { $0.hasPrefix("{") }.map { token -> JSONValue in
                ["name": .string(String(token.dropFirst().dropLast())), "in": "path", "required": true, "schema": ["type": "string"]]
            }
            if !parameters.isEmpty { definition["parameters"] = .array(parameters) }
            if route.method == "post" {
                var schema: JSONObject = ["type": "object"]
                if route.operation == "enhance" {
                    schema = [
                        "type": "object", "required": ["audio"],
                        "properties": ["audio": ["type": "string", "format": "binary"], "file_format": ["type": "string", "enum": ["pcm_s16le_16", "other"]]]
                    ]
                }
                if route.requestMediaType == "multipart/form-data" && route.operation == "clone_voice" {
                    schema = [
                        "type": "object", "required": ["name", "files"],
                        "properties": [
                            "name": ["type": "string"], "files": ["type": "array", "items": ["type": "string", "format": "binary"]], "transcript": ["type": "string"],
                            "labels": ["type": "string"]
                        ]
                    ]
                }
                definition["requestBody"] = ["required": true, "content": [route.requestMediaType: ["schema": .object(schema)]]]
            }
            var path = paths[route.path]?.object ?? [:]; path[route.method] = .object(definition); paths[route.path] = .object(path)
        }
        for model in await service.rows() {
            for op in await service.operations(id: model.id) {
                let mime =
                    op.output == "json"
                    ? "application/json"
                    : op.output == "text-stream" ? "text/event-stream" : op.output == "audio" ? "audio/wav" : op.output == "image" ? "image/png" : "application/octet-stream"
                var content: JSONObject = [mime: ["schema": op.output == "json" ? ["type": "object"] : ["type": "string"]]]
                if ["image", "file"].contains(op.output) { content["multipart/mixed"] = ["schema": ["type": "string"]] }
                if op.output == "text-stream" { content["application/json"] = ["schema": ["type": "object"]] }
                if op.output == "audio" && op.stream { content["audio/pcm"] = ["schema": ["type": "string"]] }
                let schemaID = model.id + "_" + op.name
                func rebaseRefs(_ value: JSONValue) -> JSONValue {
                    switch value {
                    case .array(let items): return .array(items.map(rebaseRefs))
                    case .object(var object):
                        if let ref = object["$ref"]?.string, ref == "#" || ref.hasPrefix("#/") {
                            object["$ref"] = .string("#/components/schemas/" + schemaID + String(ref.dropFirst()))
                        }
                        object["$id"] = nil
                        return .object(object.mapValues(rebaseRefs))
                    default: return value
                    }
                }
                addOutputSchema(op, schemaID: schemaID, schemas: &schemas, content: &content)
                if op.stream { content["text/event-stream"] = ["schema": ["type": "string"]] }
                schemas[schemaID] = rebaseRefs(.object(op.inputSchema))
                paths["/arsenal/v1/models/\(model.id)/ops/\(op.name)"] = [
                    "post": [
                        "summary": .string(op.title), "operationId": .string("op_" + model.id + "_" + op.name), "x-arsenal-model": .string(model.id),
                        "x-arsenal-op": .string(op.name), "x-arsenal-stream": .bool(op.stream),
                        "x-arsenal-output": .string(op.output),
                        "x-arsenal-policy": try op.policy.map { try JSONDecoder().decode(JSONValue.self, from: JSONEncoder().encode($0)) } ?? .null,
                        "x-arsenal-annotations": op.annotations.map(JSONValue.object) ?? .object([:]),
                        "x-arsenal-output-schema": op.outputSchema.map(JSONValue.object) ?? .null,
                        "requestBody": ["required": true, "content": ["application/json": ["schema": ["$ref": .string("#/components/schemas/" + schemaID)]]]],
                        "responses": [
                            "200": ["description": "Declared operation output", "content": .object(content)], "422": ["description": "Input does not match the model's schema"],
                            "404": ["description": .string("Model not installed; call \(MCPNames.prefix)download(model: MODEL)")]
                        ]
                    ]
                ]
            }
        }
        schemas["ModelRow"] = [
            "type": "object", "required": ["id", "needsReload", "codeConsentRequired", "contentDigest"],
            "properties": [
                "id": ["type": "string"], "needsReload": ["type": "boolean"], "codeConsentRequired": ["type": "boolean"],
                "contentDigest": ["type": ["string", "null"], "description": "Current local-folder digest; approve only this exact content"]
            ]
        ]
        for (method, path, summary) in controls {
            var operation: JSONObject = ["summary": .string(summary), "responses": ["200": ["description": "Successful response"]]]
            if path == "/arsenal/v1/models" {
                operation["responses"] = [
                    "200": ["description": "Your model table", "content": ["application/json": ["schema": ["type": "array", "items": ["$ref": "#/components/schemas/ModelRow"]]]]]
                ]
            }
            if path.contains("{id}") { operation["parameters"] = [["name": "id", "in": "path", "required": true, "schema": ["type": "string"]]] }
            if path.hasSuffix("/get-plan") {
                operation["parameters"] = [
                    ["name": "id", "in": "path", "required": true, "schema": ["type": "string"]], ["name": "dtype", "in": "query", "schema": ["type": "string"]]
                ]
            }
            if path.hasSuffix("/reference") {
                operation["parameters"] = [
                    ["name": "id", "in": "path", "required": true, "schema": ["type": "string"]],
                    ["name": "dtype", "in": "query", "schema": ["type": "string"]],
                    ["name": "mode", "in": "query", "schema": ["type": "string", "enum": ["fast", "exact"]]]
                ]
            }
            if method == "post", path != "/arsenal/v1/refresh" { operation["requestBody"] = ["content": ["application/json": ["schema": ["type": "object"]]]] }
            if method == "post", path.hasSuffix("/runner") {
                operation["requestBody"] = [
                    "required": true,
                    "content": [
                        "application/json": [
                            "schema": [
                                "type": "object", "required": ["runner"],
                                "properties": ["runner": ["type": ["string", "null"], "description": "Runner name; null restores the accepted folder default"]]
                            ]
                        ]
                    ]
                ]
            }
            if path.hasSuffix("/approve") {
                operation["requestBody"] = [
                    "required": true,
                    "content": [
                        "application/json": [
                            "schema": [
                                "type": "object", "required": ["content_digest"],
                                "properties": ["content_digest": ["type": "string", "minLength": 1], "approved_by": ["type": "string", "enum": ["cli", "api"], "default": "api"]]
                            ]
                        ]
                    ]
                ]
                operation["responses"] = [
                    "200": ["description": "Approved this digest"], "409": ["description": "Folder digest changed; inspect and approve its new content"],
                    "422": ["description": "Missing content_digest"]
                ]
            }
            var entry = paths[path]?.object ?? [:]; entry[method] = .object(operation); paths[path] = .object(entry)
        }
        // Force is request metadata, outside each model's unchanged input schema.
        let forceParameter: JSONValue = [
            "name": "X-Arsenal-Force-Load", "in": "header", "schema": ["type": "boolean", "default": false],
            "description": "Allow a load that exceeds free RAM; macOS will use swap. JSON requests may instead send top-level force:true."
        ]
        for path in Array(paths.keys) {
            guard var item = paths[path]?.object, var post = item["post"]?.object else { continue }
            var parameters = post["parameters"]?.array ?? []; parameters.append(forceParameter); post["parameters"] = .array(parameters)
            var responses = post["responses"]?.object ?? [:]
            var conflict = responses["409"]?.object ?? [:]
            let existing = conflict["description"]?.string.map { $0 + "; " } ?? ""
            conflict["description"] = .string(
                existing + "insufficient_memory: details.needed_gb and details.free_gb in decimal GB; retry with force:true only after accepting swap.")
            responses["409"] = .object(conflict)
            post["responses"] = .object(responses); item["post"] = .object(post); paths[path] = .object(item)
        }
        return [
            "openapi": "3.1.0", "info": ["title": "Arsenal", "version": "2.0"], "jsonSchemaDialect": "https://json-schema.org/draft/2020-12/schema",
            "servers": [["url": .string(origin)]], "paths": .object(paths), "components": ["schemas": .object(schemas)]
        ]
    }
    private var controls: [(String, String, String)] {
        [
            ("get", "/arsenal/v1/models/{id}/get-plan", "Get download/dependency plan"), ("post", "/arsenal/v1/refresh", "Re-read model folders, config and verified catalog"),
            ("get", "/arsenal/v1/models/{id}/runner", "Runner choice and recorded alternatives"), ("post", "/arsenal/v1/models/{id}/runner", "Set runner override"),
            ("get", "/arsenal/v1/memory/settings", "Memory admission policy"), ("post", "/arsenal/v1/memory/settings", "Set memory admission policy"),
            ("post", "/arsenal/v1/models/{id}/tools", "Show or hide model tools"),
            ("get", "/arsenal/v1/models/{id}/reference", "Reference figures (controlled measurements or labelled quick checks)"),
            ("get", "/arsenal/v1/models", "Your model table"),
            ("get", "/arsenal/v1/activity", "Live model activity"), ("get", "/arsenal/v1/activity/stream", "Activity event stream"),
            ("get", "/arsenal/v1/catalog/browse", "Get more models"), ("get", "/arsenal/v1/catalog", "Offline model browser (compatibility)"),
            ("get", "/arsenal/v1/status", "Status"),
            ("get", "/arsenal/v1/catalog/status", "Catalog trust status"),
            ("get", "/arsenal/v1/models/{id}/settings", "Model settings"), ("post", "/arsenal/v1/models/{id}/settings", "Set model settings"),
            ("post", "/arsenal/v1/models/{id}/add", "Add to your table"), ("post", "/arsenal/v1/models/{id}/remove", "Remove from your table"),
            ("post", "/arsenal/v1/models/{id}/approve", "Approve this exact local code digest"),
            ("post", "/arsenal/v1/models/{id}/delete-withdrawn", "Delete files of precisions the model no longer offers")
        ]
    }
}
