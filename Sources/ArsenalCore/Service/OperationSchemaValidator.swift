// SPDX-License-Identifier: MIT
import Foundation

/// Deliberately bounded draft-2020-12 subset. No dependency or remote references; unknown assertions fail closed.
public enum OperationSchemaValidator {
    public static func declares(path: String, in schema: JSONObject) -> Bool {
        let root = JSONValue.object(schema)
        var budget = 20_000
        var visited = Set<String>()
        func targets(_ rule: JSONValue, depth: Int) -> [JSONObject] {
            budget -= 1
            guard budget >= 0, depth < 24, let object = rule.object else { return [] }
            var result = [object]
            if let ref = object["$ref"]?.string, ref.hasPrefix("#/"), visited.insert(ref).inserted {
                var target = root
                for part in ref.dropFirst(2).split(separator: "/") {
                    target = target.object?[String(part).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")] ?? .null
                }
                result += targets(target, depth: depth + 1)
            }
            for key in ["allOf", "anyOf", "oneOf"] { for rule in object[key]?.array ?? [] { result += targets(rule, depth: depth + 1) } }
            return result
        }
        var rules = [root]
        for part in path.split(separator: ".").map(String.init) {
            visited.removeAll()
            rules = rules.flatMap { targets($0, depth: 0).compactMap { $0["properties"]?.object?[part] } }
            if rules.isEmpty { return false }
        }
        return !rules.isEmpty
    }
    public static func validate(_ value: JSONValue, schema: JSONValue) throws {
        if let allowed = schema.bool { if !allowed { throw WorkerFailure(code: "invalid_request", message: "Input is not allowed.") }; return }
        guard let object = schema.object else { throw WorkerFailure(code: "op_schema", message: "Operation schema must be an object or boolean.") }
        let encoded = try JSONEncoder().encode(value)
        let plain = try JSONSerialization.jsonObject(with: encoded, options: [.fragmentsAllowed])
        let errors: [String]
        do { errors = try ModelSchemaValidation.validate(value: plain, schemaData: WireJSON.encode(object)) } catch {
            throw WorkerFailure(code: "op_schema", message: error.localizedDescription)
        }
        guard errors.isEmpty else { throw WorkerFailure(code: "invalid_request", message: errors.joined(separator: "; ")) }
    }
}
