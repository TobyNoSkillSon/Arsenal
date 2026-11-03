// SPDX-License-Identifier: MIT
import Foundation
import CoreFoundation

/// Validates the JSON-schema vocabulary used by schemas/model.schema.json. References are local-only, never fetched.
public enum ModelSchemaValidation {
    private final class EvaluationBudget { var remaining = 20_000 }
    public static func validate(document: [String: TOMLValue], schemaData: Data) throws -> [String] {
        let budget = EvaluationBudget()
        try checkSchema(schemaData: schemaData, budget: budget)
        guard let schema = try JSONSerialization.jsonObject(with: schemaData) as? [String: Any] else { throw ManifestError.invalid("model schema is not an object") }
        return evaluate(document.mapValues(\.json), schema: schema, path: "model", budget: budget)
    }
    public static func validate(value: Any, schemaData: Data) throws -> [String] {
        let budget = EvaluationBudget()
        try checkSchema(schemaData: schemaData, budget: budget)
        guard let schema = try JSONSerialization.jsonObject(with: schemaData) as? [String: Any] else { throw ManifestError.invalid("schema must be an object") }
        return evaluate(value, schema: schema, path: "input", budget: budget)
    }
    private static func evaluate(_ value: Any, schema: [String: Any], path: String, budget: EvaluationBudget) -> [String] {
        // The body limit bounds request size; the budget bounds schema amplification per input value.
        // Count values once, without charging this walk as schema work. Admission still uses the fixed base.
        var pending: [Any] = [value]
        while let input = pending.popLast() {
            budget.remaining += 32
            if let array = input as? [Any] { pending.append(contentsOf: array) } else if let object = input as? [String: Any] { pending.append(contentsOf: object.values) }
        }
        let errors = check(value, schema: schema, root: schema, path: path, budget: budget)
        if budget.remaining < 0 { return ["\(path): schema evaluation budget exceeded"] }
        return Array(errors.prefix(64))
    }
    public static func checkSchema(schemaData: Data) throws { try checkSchema(schemaData: schemaData, budget: EvaluationBudget()) }
    private static func checkSchema(schemaData: Data, budget: EvaluationBudget) throws {
        guard schemaData.count <= 1_048_576 else { throw ManifestError.invalid("schema exceeds 1 MiB") }
        guard let root = try JSONSerialization.jsonObject(with: schemaData) as? [String: Any] else { throw ManifestError.invalid("schema must be an object") }
        let allowed: Set<String> = [
            "$schema", "$id", "$defs", "$ref", "$comment", "title", "description", "default", "examples", "deprecated", "readOnly", "writeOnly", "type", "const", "enum",
            "minLength", "maxLength", "pattern", "minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf", "minItems", "maxItems", "uniqueItems", "items",
            "properties", "required", "additionalProperties", "propertyNames", "minProperties", "maxProperties", "allOf", "anyOf", "oneOf", "not", "if", "then", "else", "format",
            "contentMediaType", "contentEncoding", "x-arsenal-file", "x-arsenal-maxUTF8Bytes", "x-arsenal-control", "x-arsenal-image", "x-arsenal-coordinates",
            "x-arsenal-ordered"
        ]
        let admissionBudget = budget
        func spend() throws { admissionBudget.remaining -= 1; guard admissionBudget.remaining >= 0 else { throw ManifestError.invalid("schema admission budget exceeded") } }
        func visit(_ rule: [String: Any], depth: Int = 0) throws {
            try spend()
            guard depth < 12 else { throw ManifestError.invalid("schema nesting exceeds 12") }
            for key in rule.keys where !allowed.contains(key) { throw ManifestError.invalid("unsupported JSON Schema keyword \(key)") }
            func require(_ ok: Bool, _ key: String) throws { if !ok { throw ManifestError.invalid("malformed JSON Schema keyword \(key)") } }
            if let raw = rule["type"] {
                let types = (raw as? String).map { [$0] } ?? (raw as? [String]) ?? []
                try require(
                    !types.isEmpty && Set(types).count == types.count && Set(types).isSubset(of: ["object", "array", "string", "number", "integer", "boolean", "null"]), "type")
            }
            for key in ["$ref", "pattern", "$schema", "$id", "format"] where rule[key] != nil { try require(rule[key] is String, key) }
            for key in ["required"] where rule[key] != nil { let fields = rule[key] as? [String]; try require(fields != nil && Set(fields!).count == fields!.count, key) }
            if let raw = rule["enum"] { try require((raw as? [Any]).map { !$0.isEmpty } ?? false, "enum") }
            for key in ["minLength", "maxLength", "minItems", "maxItems", "minProperties", "maxProperties", "x-arsenal-maxUTF8Bytes"] where rule[key] != nil {
                try require(matches(rule[key]!, type: "integer") && (rule[key] as? NSNumber).map { $0.doubleValue >= 0 } == true, key)
            }
            for key in ["minimum", "maximum", "exclusiveMinimum", "exclusiveMaximum", "multipleOf"] where rule[key] != nil {
                try require(matches(rule[key]!, type: "number") && (rule[key] as? NSNumber)?.doubleValue.isFinite == true, key)
            }
            if let n = rule["multipleOf"] as? NSNumber { try require(n.doubleValue > 0, "multipleOf") }
            for key in ["$defs", "properties"] where rule[key] != nil { try require(rule[key] is [String: [String: Any]], key) }
            for key in ["items", "additionalProperties"] where rule[key] != nil { try require(rule[key] is [String: Any] || matches(rule[key]!, type: "boolean"), key) }
            for key in ["propertyNames", "not", "if", "then", "else"] where rule[key] != nil { try require(rule[key] is [String: Any], key) }
            for key in ["allOf", "anyOf", "oneOf"] where rule[key] != nil { try require((rule[key] as? [[String: Any]]).map { !$0.isEmpty } ?? false, key) }
            if let raw = rule["x-arsenal-file"] {
                let annotation = raw as? [String: Any]
                let accepted = annotation?["accept"] as? [String]
                try require(
                    annotation != nil && Set(annotation!.keys) == ["accept"] && accepted?.isEmpty == false
                        && accepted!.count <= 32 && accepted!.allSatisfy { $0.range(of: "^[a-z0-9.+-]+/[a-z0-9.+*-]+$", options: .regularExpression) != nil }, "x-arsenal-file")
            }
            for key in ["x-arsenal-control", "x-arsenal-image", "x-arsenal-coordinates"] where rule[key] != nil { try require(rule[key] is String, key) }
            if let raw = rule["x-arsenal-ordered"] { try require(matches(raw, type: "boolean"), "x-arsenal-ordered") }
            if let raw = rule["uniqueItems"] { try require(matches(raw, type: "boolean"), "uniqueItems") }
            if let ref = rule["$ref"] as? String, !ref.hasPrefix("#/") || lookup(ref, root: root) == nil {
                throw ManifestError.invalid("schema reference must resolve within this document")
            }
            if let pattern = rule["pattern"] as? String { _ = try NSRegularExpression(pattern: pattern) }
            for key in ["$defs", "properties"] {
                for child in (rule[key] as? [String: [String: Any]] ?? [:]).values { try visit(child, depth: depth + 1) }
            }
            for key in ["items", "additionalProperties", "propertyNames", "not", "if", "then", "else"] {
                if let child = rule[key] as? [String: Any] { try visit(child, depth: depth + 1) }
            }
            for key in ["allOf", "anyOf", "oneOf"] { for child in rule[key] as? [[String: Any]] ?? [] { try visit(child, depth: depth + 1) } }
        }
        try visit(root)
        // Memoize graph heights, not just a visited bit: a cached node must still respect the remaining depth.
        var heights: [String: Int] = [:]
        func refs(_ rule: [String: Any], active: Set<String>, depth: Int = 0) throws -> Int {
            try spend()
            guard depth < 12 else { throw ManifestError.invalid("schema reference nesting exceeds 12") }
            var height = 0
            if let ref = rule["$ref"] as? String {
                guard !active.contains(ref), let target = lookup(ref, root: root) else { throw ManifestError.invalid("recursive schema references are not supported") }
                let childHeight: Int
                if let cached = heights[ref] {
                    childHeight = cached
                } else {
                    childHeight = try refs(target, active: active.union([ref]), depth: depth + 1); heights[ref] = childHeight
                }
                height = max(height, 1 + childHeight)
            }
            var children: [[String: Any]] = []
            for key in ["$defs", "properties"] { children += Array((rule[key] as? [String: [String: Any]] ?? [:]).values) }
            for key in ["items", "additionalProperties", "propertyNames", "not", "if", "then", "else"] { if let child = rule[key] as? [String: Any] { children.append(child) } }
            for key in ["allOf", "anyOf", "oneOf"] { children += rule[key] as? [[String: Any]] ?? [] }
            for child in children { height = max(height, 1 + (try refs(child, active: active, depth: depth + 1))) }
            guard depth + height < 12 else { throw ManifestError.invalid("schema reference nesting exceeds 12") }
            return height
        }
        _ = try refs(root, active: [])
    }
    private static func charge(_ value: Any, budget: EvaluationBudget, depth: Int = 0) {
        budget.remaining -= 1
        if let text = value as? String { budget.remaining -= text.utf8.count / 64 }
        guard budget.remaining >= 0, depth < 64 else { budget.remaining = -1; return }
        if let array = value as? [Any] {
            for child in array { if budget.remaining < 0 { break }; charge(child, budget: budget, depth: depth + 1) }
        } else if let object = value as? [String: Any] {
            for child in object.values { if budget.remaining < 0 { break }; charge(child, budget: budget, depth: depth + 1) }
        }
    }
    private static func check(_ value: Any, schema: [String: Any], root: [String: Any], path: String, depth: Int = 0, budget: EvaluationBudget = EvaluationBudget()) -> [String] {
        budget.remaining -= 1
        guard budget.remaining >= 0 else { return [] }
        guard depth < 24 else { return ["\(path): schema evaluation nesting exceeds 24"] }
        if let ref = schema["$ref"] as? String {
            guard ref.hasPrefix("#/"), let target = lookup(ref, root: root) else { return ["\(path): invalid local schema reference"] }
            var sibling = schema; sibling.removeValue(forKey: "$ref")
            return check(value, schema: target, root: root, path: path, depth: depth + 1, budget: budget)
                + check(value, schema: sibling, root: root, path: path, depth: depth + 1, budget: budget)
        }
        if let type = schema["type"] {
            let allowed = (type as? [String]) ?? (type as? String).map { [$0] } ?? []
            if !allowed.contains(where: { matches(value, type: $0) }) { return ["\(path): expected \(allowed.joined(separator: "|"))"] }
        }
        return scalarProblems(value, schema: schema, path: path, budget: budget)
            + containerProblems(value, schema: schema, root: root, path: path, depth: depth, budget: budget)
            + combinationProblems(value, schema: schema, root: root, path: path, depth: depth, budget: budget)
    }
    private static func scalarProblems(_ value: Any, schema: [String: Any], path: String, budget: EvaluationBudget) -> [String] {
        var errors: [String] = []
        if let constant = schema["const"] { charge(value, budget: budget); charge(constant, budget: budget) }
        if let constant = schema["const"], !equal(value, constant) { errors.append("\(path): must equal schema constant") }
        if let choices = schema["enum"] as? [Any],
            !choices.contains(where: {
                charge(value, budget: budget); charge($0, budget: budget); return budget.remaining >= 0 && equal(value, $0)
            })
        {
            errors.append("\(path): value is outside allowed enum")
        }
        if let string = value as? String {
            if let min = schema["minLength"] as? Int, string.unicodeScalars.count < min { errors.append("\(path): string is empty") }
            if let max = schema["maxLength"] as? Int, string.unicodeScalars.count > max { errors.append("\(path): string too long") }
            if let max = schema["x-arsenal-maxUTF8Bytes"] as? Int, string.utf8.count > max { errors.append("\(path): UTF-8 string too long") }
            if let pattern = schema["pattern"] as? String, string.range(of: pattern, options: .regularExpression) == nil { errors.append("\(path): invalid string format") }
        }
        if let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID(), let minimum = schema["minimum"] as? NSNumber, number.doubleValue < minimum.doubleValue {
            errors.append("\(path): below minimum")
        }
        if let n = value as? NSNumber, CFGetTypeID(n) != CFBooleanGetTypeID() {
            for (key, fails) in [("maximum", { (v: Double) in n.doubleValue > v }), ("exclusiveMinimum", { n.doubleValue <= $0 }), ("exclusiveMaximum", { n.doubleValue >= $0 })] {
                if let limit = schema[key] as? NSNumber, fails(limit.doubleValue) { errors.append("\(path): outside \(key)") }
            }
            if let multiple = schema["multipleOf"] as? NSNumber {
                let units = n.doubleValue / multiple.doubleValue; if multiple.doubleValue <= 0 || abs(units - units.rounded()) > 1e-8 { errors.append("\(path): not a multiple") }
            }
        }
        return errors
    }
    private static func containerProblems(_ value: Any, schema: [String: Any], root: [String: Any], path: String, depth: Int, budget: EvaluationBudget) -> [String] {
        var errors: [String] = []
        if let array = value as? [Any] {
            if let max = schema["maxItems"] as? Int, array.count > max { errors.append("\(path): too many items") }
            if schema["items"] as? Bool == false && !array.isEmpty { errors.append("\(path): items forbidden") }
            if let min = schema["minItems"] as? Int, array.count < min { errors.append("\(path): too few items") }
            if schema["uniqueItems"] as? Bool == true {
                var seen = Set<Data>()
                for item in array {
                    charge(item, budget: budget)
                    budget.remaining -= 1; if budget.remaining < 0 { break }
                    guard let canonical = try? JSONSerialization.data(withJSONObject: [item], options: [.sortedKeys, .fragmentsAllowed]) else { continue }
                    if !seen.insert(canonical).inserted { errors.append("\(path): duplicate array item"); break }
                }
            }
            if let items = schema["items"] as? [String: Any] {
                for (i, item) in array.enumerated() {
                    if budget.remaining < 0 { break }; errors += check(item, schema: items, root: root, path: "\(path)[\(i)]", depth: depth + 1, budget: budget)
                }
            }
        }
        if let object = value as? [String: Any] {
            if let min = schema["minProperties"] as? Int, object.count < min { errors.append("\(path): too few properties") }
            if let max = schema["maxProperties"] as? Int, object.count > max { errors.append("\(path): too many properties") }
            for key in schema["required"] as? [String] ?? [] where object[key] == nil { errors.append("\(path).\(key): required") }
            let properties = schema["properties"] as? [String: [String: Any]] ?? [:]
            for (key, field) in object {
                if budget.remaining < 0 { break }
                if let rule = properties[key] {
                    errors += check(field, schema: rule, root: root, path: "\(path).\(key)", depth: depth + 1, budget: budget)
                } else if let rule = schema["additionalProperties"] as? [String: Any] {
                    errors += check(field, schema: rule, root: root, path: "\(path).\(key)", depth: depth + 1, budget: budget)
                } else if schema["additionalProperties"] as? Bool == false {
                    errors.append("\(path).\(key): unexpected property")
                }
                if let rule = schema["propertyNames"] as? [String: Any] { errors += check(key, schema: rule, root: root, path: "\(path).\(key)", depth: depth + 1, budget: budget) }
            }
        }
        return errors
    }
    private static func combinationProblems(_ value: Any, schema: [String: Any], root: [String: Any], path: String, depth: Int, budget: EvaluationBudget) -> [String] {
        var errors: [String] = []
        for rule in schema["allOf"] as? [[String: Any]] ?? [] { errors += check(value, schema: rule, root: root, path: path, depth: depth + 1, budget: budget) }
        if let alternatives = schema["oneOf"] as? [[String: Any]],
            alternatives.filter({ check(value, schema: $0, root: root, path: path, depth: depth + 1, budget: budget).isEmpty }).count != 1
        {
            errors.append("\(path): must match one schema alternative")
        }
        if let alternatives = schema["anyOf"] as? [[String: Any]],
            !alternatives.contains(where: { check(value, schema: $0, root: root, path: path, depth: depth + 1, budget: budget).isEmpty })
        {
            errors.append("\(path): no schema alternative matched")
        }
        if let negation = schema["not"] as? [String: Any], check(value, schema: negation, root: root, path: path, depth: depth + 1, budget: budget).isEmpty {
            errors.append("\(path): forbidden value")
        }
        if let condition = schema["if"] as? [String: Any] {
            let branch = check(value, schema: condition, root: root, path: path, depth: depth + 1, budget: budget).isEmpty ? "then" : "else"
            if let rule = schema[branch] as? [String: Any] { errors += check(value, schema: rule, root: root, path: path, depth: depth + 1, budget: budget) }
        }
        return errors
    }
    private static func matches(_ value: Any, type: String) -> Bool {
        switch type {
        case "null": return value is NSNull
        case "string": return value is String
        case "object": return value is [String: Any]
        case "array": return value is [Any]
        case "boolean": return (value as? NSNumber).map { CFGetTypeID($0) == CFBooleanGetTypeID() } ?? false
        case "number", "integer":
            guard let number = value as? NSNumber, CFGetTypeID(number) != CFBooleanGetTypeID() else { return false }
            return type == "number" || number.doubleValue.rounded() == number.doubleValue
        default: return false
        }
    }
    private static func equal(_ a: Any, _ b: Any) -> Bool {
        // JSONSerialization wraps scalars so canonical comparison preserves bool versus numeric values.
        let x = try? JSONSerialization.data(withJSONObject: [a], options: [.sortedKeys, .fragmentsAllowed])
        let y = try? JSONSerialization.data(withJSONObject: [b], options: [.sortedKeys, .fragmentsAllowed])
        return x != nil && x == y
    }
    private static func lookup(_ ref: String, root: [String: Any]) -> [String: Any]? {
        var current: Any = root
        for part in ref.dropFirst(2).split(separator: "/") {
            guard let table = current as? [String: Any], let next = table[String(part).replacingOccurrences(of: "~1", with: "/").replacingOccurrences(of: "~0", with: "~")] else {
                return nil
            }; current = next
        }
        return current as? [String: Any]
    }
}
