// SPDX-License-Identifier: MIT
import Foundation

/// Request members whose object key order is meaningful to the model (NuExtract's extraction template is written back
/// in its keys' order). JSONObject is a Swift dictionary, so a decoded request loses that order. An input property
/// declaring `"x-arsenal-ordered": true` is forwarded to the worker as its exact JSON text under `ordered_json.<name>`
/// beside the decoded value, which the service validates as usual. HTTP routes take that text from the request body's
/// own bytes. A property typed `"object"` may also be sent as the JSON text of its object (how MCP clients keep the
/// order); any other ordered property (one that may hold any JSON value, such as a decision state, whose string is
/// text in its own right) keeps a string as its value and gets ordered text only from HTTP or the transport member.
public enum OrderedJSONMembers {
    public static let keyword = "x-arsenal-ordered"
    /// The worker request member carrying `{name: exact JSON text}`; adapters prefer it to the decoded object.
    public static let transport = "ordered_json"

    /// The top-level input properties that declare ordered keys.
    public static func names(_ schema: JSONObject) -> [String] {
        (schema["properties"]?.object ?? [:]).filter { $0.value[keyword]?.bool == true }.map(\.key).sorted()
    }
    /// Whether the property is object-typed, so a string given for it is read as the JSON text of its object.
    public static func takesText(_ name: String, schema: JSONObject) -> Bool { schema["properties"]?[name]?["type"]?.string == "object" }
    /// The request with each ordered member as its value (for schema validation) and the exact text of every
    /// member given as text. A previously split request (its transport member) is accepted again unchanged.
    /// Only schema-declared ordered names travel as text, and a transport text must decode to exactly the value the
    /// request validates: the worker reads the text, so it can never carry anything the schema did not check.
    public static func split(_ params: JSONObject, schema: JSONObject) throws -> (request: JSONObject, ordered: [String: String]) {
        let declared = Set(names(schema))
        var request = params, ordered: [String: String] = [:]
        if let given = request.removeValue(forKey: transport) {
            guard let members = given.object else { throw invalid("\(transport) must be an object of JSON texts.") }
            for (name, value) in members {
                guard declared.contains(name), let text = value.string else {
                    throw invalid("\(transport).\(name) is not a declared ordered member given as JSON text.")
                }
                ordered[name] = text
            }
        }
        for name in declared {
            if takesText(name, schema: schema), case .string(let text)? = request[name] {
                guard case .object(let object)? = try? LastKeyJSON.parse(Data(text.utf8)) else {
                    throw invalid("\(name) must be a JSON object or the JSON text of one.")
                }
                request[name] = .object(object); ordered[name] = text
            } else if let text = ordered[name] {
                guard let value = request[name] else { ordered[name] = nil; continue }
                // Read as the workers read it (a repeated key keeps its last value), so the text cannot show the
                // worker a value other than the one validated.
                guard (try? LastKeyJSON.parse(Data(text.utf8))) == value else {
                    throw invalid("\(transport).\(name) does not match the \(name) value in the request.")
                }
            }
        }
        return (request, ordered)
    }
    private static func invalid(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }
    /// The ordered members that hold no file input (a decision state, an extraction template), plus the transport
    /// member: text to the model, so input staging must leave their `"path"` keys alone. Rewriting one would part the
    /// value from its exact text and import a file nothing reads. A property whose schema declares `x-arsenal-file`
    /// anywhere inside, through its local `#/$defs/…` references too, or refers anywhere else, is staged as usual.
    public static func verbatimNames(_ schema: JSONObject) -> Set<String> {
        let defs = schema["$defs"]?.object ?? [:]
        func fileFree(_ value: JSONValue, _ seen: Set<String> = []) -> Bool {
            switch value {
            case .object(let object):
                if object["x-arsenal-file"] != nil { return false }
                if let ref = object["$ref"] {
                    guard let path = ref.string, path.hasPrefix("#/$defs/") else { return false }
                    let name = String(path.dropFirst("#/$defs/".count))
                    if !seen.contains(name) { guard let target = defs[name], fileFree(target, seen.union([name])) else { return false } }
                }
                return object.values.allSatisfy { fileFree($0, seen) }
            case .array(let values): return values.allSatisfy { fileFree($0, seen) }
            default: return true
            }
        }
        let properties = schema["properties"]?.object ?? [:]
        return Set(names(schema).filter { properties[$0].map { fileFree($0) } ?? false }).union([transport])
    }
    /// Adds the transport member to a validated request.
    public static func attach(_ ordered: [String: String], to request: JSONObject) -> JSONObject {
        guard !ordered.isEmpty else { return request }
        var request = request; request[transport] = .object(ordered.mapValues { .string($0) }); return request
    }
}
