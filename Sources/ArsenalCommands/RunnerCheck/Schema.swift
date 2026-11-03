// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire
import ArsenalCore

/// Intentionally bounded vocabulary. Unknown validation keywords fail closed;
/// schema support is not silently confused with a successful op check.
enum CheckSchema {
    static let keywords: Set<String> = [
        "$schema", "$id", "$defs", "$ref", "$comment", "title", "description", "examples", "default", "deprecated", "readOnly", "writeOnly", "type", "enum", "const", "properties",
        "required", "additionalProperties", "items", "minItems", "maxItems", "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum",
        "multipleOf", "allOf", "anyOf", "oneOf", "not", "uniqueItems", "propertyNames", "minProperties", "maxProperties", "if", "then", "else", "format", "contentMediaType",
        "contentEncoding"
    ]
    static func rule(_ schema: JSONObject, root: JSONObject) throws -> JSONObject {
        for key in schema.keys where !keywords.contains(key) { throw CheckFailure("unsupported schema keyword \(key); checker cannot certify this input") }
        if let reference = schema["$ref"]?.string {
            guard reference.hasPrefix("#/") else { throw CheckFailure("only local JSON-pointer $ref is supported") }
            var value = JSONValue.object(root)
            for part in reference.dropFirst(2).split(separator: "/") {
                let key = String(part).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")
                guard let child = value[key] else { throw CheckFailure("unresolved schema $ref \(reference)") }; value = child
            }
            guard var target = value.object else { throw CheckFailure("schema $ref is not an object") }
            for (key, value) in schema where key != "$ref" { target[key] = value }
            return target
        }
        return schema
    }
    static func validate(_ value: JSONValue, schema: JSONObject) throws {
        let encoded = try JSONEncoder().encode(value)
        let plain = try JSONSerialization.jsonObject(with: encoded, options: [.fragmentsAllowed])
        let errors = try ModelSchemaValidation.validate(value: plain, schemaData: WireJSON.encode(schema))
        guard errors.isEmpty else { throw CheckFailure(errors.joined(separator: "; ")) }
    }
    static func synthesize(_ schema: JSONObject, root: JSONObject? = nil, depth: Int = 0) throws -> JSONValue {
        guard depth < 32 else { throw CheckFailure("recursive schema requires an example") }
        let root = root ?? schema
        let s = try rule(schema, root: root)
        if let value = s["examples"]?.array?.first ?? s["default"] ?? s["const"] ?? s["enum"]?.array?.first { return value }
        if s["$ref"] != nil { return try synthesize(s, root: root, depth: depth + 1) }
        if s["oneOf"] != nil || s["anyOf"] != nil || s["allOf"] != nil || s["not"] != nil || s["if"] != nil {
            throw CheckFailure("schema alternatives require gate/runner-check.json examples")
        }
        switch s["type"]?.string ?? "object" {
        case "object":
            let fields = s["properties"]?.object ?? [:]
            var result: JSONObject = [:]
            for key in s["required"]?.array?.compactMap(\.string) ?? [] {
                guard let field = fields[key]?.object else { throw CheckFailure("required property \(key) has no synthesis rule") }
                result[key] = try synthesize(field, root: root, depth: depth + 1)
            }
            return .object(result)
        case "array":
            let n = Int(s["minItems"]?.number ?? 0)
            guard n >= 0 && n <= 16 else { throw CheckFailure("large array requires an example") }
            return .array(try (0..<n).map { _ in try synthesize(s["items"]?.object ?? [:], root: root, depth: depth + 1) })
        case "string":
            guard s["pattern"] == nil else { throw CheckFailure("pattern requires an example") }
            let n = Int(s["minLength"]?.number ?? 1)
            guard n >= 0 && n <= 4096 else { throw CheckFailure("large string requires an example") }
            return .string(String(repeating: "x", count: n))
        case "integer", "number": return .number(s["minimum"]?.number ?? 1)
        case "boolean": return .bool(false)
        case "null": return .null
        default: throw CheckFailure("schema needs an explicit example")
        }
    }
}
