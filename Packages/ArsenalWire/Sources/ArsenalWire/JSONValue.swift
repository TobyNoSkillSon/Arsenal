// SPDX-License-Identifier: MIT
import Foundation

public enum JSONValue: Codable, Sendable, Equatable {
    case null, bool(Bool), number(Double), string(String), array([JSONValue]), object([String: JSONValue])
    public init(from decoder: any Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() {
            self = .null
        } else if let v = try? c.decode(Bool.self) {
            self = .bool(v)
        } else if let v = try? c.decode(Double.self) {
            self = .number(v)
        } else if let v = try? c.decode(String.self) {
            self = .string(v)
        } else if let v = try? c.decode([JSONValue].self) {
            self = .array(v)
        } else {
            self = .object(try c.decode([String: JSONValue].self))
        }
    }
    public func encode(to encoder: any Encoder) throws {
        var c = encoder.singleValueContainer()
        switch self {
        case .null: try c.encodeNil()
        case .bool(let v): try c.encode(v)
        case .number(let v): try c.encode(v)
        case .string(let v): try c.encode(v)
        case .array(let v): try c.encode(v)
        case .object(let v): try c.encode(v)
        }
    }
    public var string: String? { if case .string(let v) = self { return v }; return nil }
    public var number: Double? { if case .number(let v) = self { return v }; return nil }
    public var bool: Bool? { if case .bool(let v) = self { return v }; return nil }
    public var object: JSONObject? { if case .object(let v) = self { return v }; return nil }
    public var array: [JSONValue]? { if case .array(let v) = self { return v }; return nil }
    public subscript(_ key: String) -> JSONValue? { object?[key] }
}
public typealias JSONObject = [String: JSONValue]
extension JSONValue: ExpressibleByStringLiteral { public init(stringLiteral value: String) { self = .string(value) } }
extension JSONValue: ExpressibleByIntegerLiteral { public init(integerLiteral value: Int) { self = .number(Double(value)) } }
extension JSONValue: ExpressibleByFloatLiteral { public init(floatLiteral value: Double) { self = .number(value) } }
extension JSONValue: ExpressibleByBooleanLiteral { public init(booleanLiteral value: Bool) { self = .bool(value) } }
extension JSONValue: ExpressibleByArrayLiteral { public init(arrayLiteral elements: JSONValue...) { self = .array(elements) } }
extension JSONValue: ExpressibleByDictionaryLiteral {
    public init(dictionaryLiteral elements: (String, JSONValue)...) { self = .object(Dictionary(uniqueKeysWithValues: elements)) }
}
public enum WireJSON {
    public static func encode(_ object: JSONObject) throws -> Data {
        let e = JSONEncoder(); e.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]; return try e.encode(object)
    }
    public static func decode(_ data: Data) throws -> JSONObject { try JSONDecoder().decode(JSONObject.self, from: data) }
}
