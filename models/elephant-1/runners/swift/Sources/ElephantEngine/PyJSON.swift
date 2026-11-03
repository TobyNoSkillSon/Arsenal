// SPDX-License-Identifier: MIT
// The parser is ported from Workers/Decisions/Sources/DecisionsProtocol/OrderedJSON.swift (from Verdict, Apache-2.0).
import Foundation

public struct JSONError: Error, CustomStringConvertible {
    public let description: String
    public init(_ description: String) { self.description = description }
}

/// A JSON value as Python's `json.loads` sees it: object key order kept (System One option order changes token ids),
/// keys compared by their UTF-8 bytes as Python compares them, numbers kept as written until rendered.
public indirect enum PyJSON: Sendable {
    case object([(String, PyJSON)])
    case array([PyJSON])
    case string(String)
    /// `true`, `false`, `null` or a number literal, as written.
    case scalar(String)

    public subscript(_ key: String) -> PyJSON? {
        if case .object(let fields) = self { return fields.first { $0.0.utf8.elementsEqual(key.utf8) }?.1 }
        return nil
    }
    public var fields: [(String, PyJSON)]? { if case .object(let entries) = self { return entries }; return nil }
    public var array: [PyJSON]? { if case .array(let values) = self { return values }; return nil }
    public var string: String? { if case .string(let value) = self { return value }; return nil }
    public var isNull: Bool { if case .scalar("null") = self { return true }; return false }
    public var isObject: Bool { if case .object = self { return true }; return false }
    public var isArray: Bool { if case .array = self { return true }; return false }
    /// A JSON integer literal (Python int) as Int, when it fits.
    public var int: Int? {
        if case .scalar(let s) = self, Self.isInteger(s) { return Int(s) }
        return nil
    }
    public var bool: Bool? {
        if case .scalar("true") = self { return true }
        if case .scalar("false") = self { return false }
        return nil
    }

    static func isInteger(_ s: String) -> Bool {
        guard let first = s.utf8.first, first == 45 || (48...57).contains(first) else { return false }
        return !s.utf8.contains { $0 == 46 || $0 == 101 || $0 == 69 } // . e E
    }

    /// Python truthiness of the decoded value (`desc or label`, `if q.criteria`).
    public var truthy: Bool {
        switch self {
        case .object(let f): return !f.isEmpty
        case .array(let a): return !a.isEmpty
        case .string(let s): return !s.isEmpty
        case .scalar(let s):
            if s == "true" { return true }
            if s == "false" || s == "null" { return false }
            if Self.isInteger(s) { return s.contains { ("1"..."9").contains($0) } }
            return (Double(s) ?? 1) != 0
        }
    }

    // MARK: - Python's json.dumps(value, ensure_ascii=False, indent=2)

    public func dumps(indent: Int = 2) -> String {
        var out = ""
        dump(&out, level: 0, indent: indent)
        return out
    }

    private func dump(_ out: inout String, level: Int, indent: Int) {
        switch self {
        case .string(let s): Self.jsonString(s, &out)
        case .scalar(let s): out += Self.jsonScalar(s)
        case .array(let values):
            if values.isEmpty { out += "[]"; return }
            let inner = "\n" + String(repeating: " ", count: (level + 1) * indent)
            out += "["
            for (i, v) in values.enumerated() {
                if i > 0 { out += "," }
                out += inner
                v.dump(&out, level: level + 1, indent: indent)
            }
            out += "\n" + String(repeating: " ", count: level * indent) + "]"
        case .object(let fields):
            if fields.isEmpty { out += "{}"; return }
            let inner = "\n" + String(repeating: " ", count: (level + 1) * indent)
            out += "{"
            for (i, (k, v)) in fields.enumerated() {
                if i > 0 { out += "," }
                out += inner
                Self.jsonString(k, &out)
                out += ": "
                v.dump(&out, level: level + 1, indent: indent)
            }
            out += "\n" + String(repeating: " ", count: level * indent) + "}"
        }
    }

    /// Compact JSON, keys in order (replies: legends keep the caller's structure).
    public func compact() -> String {
        var out = ""
        compact(&out)
        return out
    }
    private func compact(_ out: inout String) {
        switch self {
        case .string(let s): Self.jsonString(s, &out)
        case .scalar(let s): out += Self.jsonScalar(s)
        case .array(let values):
            out += "["
            for (i, v) in values.enumerated() {
                if i > 0 { out += "," }
                v.compact(&out)
            }
            out += "]"
        case .object(let fields):
            out += "{"
            for (i, (k, v)) in fields.enumerated() {
                if i > 0 { out += "," }
                Self.jsonString(k, &out)
                out += ":"
                v.compact(&out)
            }
            out += "}"
        }
    }

    public static func jsonString(_ value: String, _ out: inout String) {
        out += "\""
        for scalar in value.unicodeScalars {
            switch scalar.value {
            case 34: out += "\\\""
            case 92: out += "\\\\"
            case 8: out += "\\b"
            case 12: out += "\\f"
            case 10: out += "\\n"
            case 13: out += "\\r"
            case 9: out += "\\t"
            case 0..<32: out += String(format: "\\u%04x", scalar.value)
            default: out.unicodeScalars.append(scalar)
            }
        }
        out += "\""
    }

    /// A literal as Python writes the value json.loads made of it: ints exactly, floats as repr, the rest as is.
    static func jsonScalar(_ s: String) -> String {
        if s == "true" || s == "false" || s == "null" { return s }
        if isInteger(s) { return normalizedInteger(s) }
        guard let d = Double(s) else { return s } // NaN, Infinity: Python writes them back unchanged
        // json.dumps writes Python's non-finite floats as NaN, Infinity and -Infinity (allow_nan=True).
        if d.isNaN { return "NaN" }
        if d == .infinity { return "Infinity" }
        if d == -.infinity { return "-Infinity" }
        return floatRepr(d)
    }

    static func normalizedInteger(_ s: String) -> String {
        let negative = s.hasPrefix("-")
        let digits = negative ? String(s.dropFirst()) : s
        return digits == "0" ? "0" : (negative ? "-" : "") + digits
    }

    /// Python's float repr: the shortest round-trip digits, fixed notation for decimal exponents -4...15, else
    /// scientific with a signed exponent of at least two digits.
    public static func floatRepr(_ d: Double) -> String {
        if d.isNaN { return "nan" }
        if d.isInfinite { return d < 0 ? "-inf" : "inf" }
        // Swift's description gives the shortest round-trip digits; take them apart and lay them out as Python does.
        let printed = "\(d)"
        let negative = printed.hasPrefix("-")
        var body = negative ? String(printed.dropFirst()) : printed
        var exponent = 0
        if let e = body.firstIndex(where: { $0 == "e" || $0 == "E" }) {
            exponent = Int(body[body.index(after: e)...]) ?? 0
            body = String(body[..<e])
        }
        let parts = body.split(separator: ".", omittingEmptySubsequences: false)
        let whole = String(parts[0])
        let fraction = parts.count > 1 ? String(parts[1]) : ""
        var digits = whole + fraction
        var point = whole.count + exponent // decimal point position after `point` digits
        // Strip leading zeros (0.001 -> digits 0001, point 1).
        while digits.count > 1 && digits.hasPrefix("0") {
            digits.removeFirst()
            point -= 1
        }
        while digits.count > 1 && digits.hasSuffix("0") { digits.removeLast() }
        if digits == "0" { return negative ? "-0.0" : "0.0" }
        let sci = point - 1 // exponent of d.ddd form
        var out = negative ? "-" : ""
        if sci < -4 || sci >= 16 {
            out += String(digits.first!)
            if digits.count > 1 { out += "." + digits.dropFirst() }
            out += "e" + (sci < 0 ? "-" : "+") + (abs(sci) < 10 ? "0" : "") + String(abs(sci))
        } else if point <= 0 {
            out += "0." + String(repeating: "0", count: -point) + digits
        } else if point >= digits.count {
            out += digits + String(repeating: "0", count: point - digits.count) + ".0"
        } else {
            let i = digits.index(digits.startIndex, offsetBy: point)
            out += String(digits[..<i]) + "." + String(digits[i...])
        }
        return out
    }

    // MARK: - Python's str() of the decoded value (f-strings in the vendor prompt)

    public func pyStr() -> String {
        if case .string(let s) = self { return s }
        return pyRepr()
    }

    public func pyRepr() -> String {
        switch self {
        case .string(let s): return Self.stringRepr(s)
        case .scalar(let s):
            switch s {
            case "true": return "True"
            case "false": return "False"
            case "null": return "None"
            default:
                if Self.isInteger(s) { return Self.normalizedInteger(s) }
                guard let d = Double(s) else { return s.lowercased() == "nan" ? "nan" : s }
                return Self.floatRepr(d)
            }
        case .array(let values): return "[" + values.map { $0.pyRepr() }.joined(separator: ", ") + "]"
        case .object(let fields): return "{" + fields.map { Self.stringRepr($0.0) + ": " + $0.1.pyRepr() }.joined(separator: ", ") + "}"
        }
    }

    /// Python's `repr(str)`.
    public static func stringRepr(_ s: String) -> String {
        let scalars = s.unicodeScalars
        let quote: Unicode.Scalar = scalars.contains("'") && !scalars.contains("\"") ? "\"" : "'"
        var out = String(quote)
        for c in scalars {
            switch c {
            case quote, "\\": out += "\\" + String(c)
            case "\t": out += "\\t"
            case "\n": out += "\\n"
            case "\r": out += "\\r"
            default:
                if c.value < 0x20 || c.value == 0x7f {
                    out += String(format: "\\x%02x", c.value)
                } else if c.value < 0x7f || printable(c) {
                    out.unicodeScalars.append(c)
                } else if c.value <= 0xff {
                    out += String(format: "\\x%02x", c.value)
                } else if c.value <= 0xffff {
                    out += String(format: "\\u%04x", c.value)
                } else {
                    out += String(format: "\\U%08x", c.value)
                }
            }
        }
        out.unicodeScalars.append(quote)
        return out
    }

    /// Python's `str.isprintable` for one non-ASCII code point.
    static func printable(_ c: Unicode.Scalar) -> Bool {
        switch c.properties.generalCategory {
        case .control, .format, .surrogate, .privateUse, .unassigned, .lineSeparator, .paragraphSeparator, .spaceSeparator: return false
        default: return true
        }
    }
}

/// Parses a JSON document as Python's `json.loads` does (duplicate keys: the last value, at the first key's place).
public struct PyJSONParser {
    private let bytes: [UInt8]
    private var cursor = 0
    public init(_ data: Data) { bytes = Array(data) }
    public static func parse(_ data: Data) throws -> PyJSON {
        var parser = PyJSONParser(data)
        return try parser.parse()
    }
    public mutating func parse() throws -> PyJSON {
        let result = try value(depth: 0)
        space()
        guard cursor == bytes.count else { throw JSONError("Invalid JSON: trailing data") }
        return result
    }
    private mutating func space() { while cursor < bytes.count && [9, 10, 13, 32].contains(bytes[cursor]) { cursor += 1 } }
    private mutating func take(_ byte: UInt8) throws {
        space()
        guard cursor < bytes.count && bytes[cursor] == byte else { throw JSONError("Invalid JSON") }
        cursor += 1
    }
    private mutating func quoted() throws -> String {
        space()
        let begin = cursor
        try take(34)
        var escaped = false
        while cursor < bytes.count {
            let b = bytes[cursor]
            cursor += 1
            if escaped {
                escaped = false
                continue
            }
            if b == 92 {
                escaped = true
                continue
            }
            if b == 34 { return try JSONDecoder().decode(String.self, from: Data(bytes[begin..<cursor])) }
        }
        throw JSONError("Unterminated JSON string")
    }
    private mutating func value(depth: Int) throws -> PyJSON {
        guard depth < 512 else { throw JSONError("JSON nested too deeply") }
        space()
        guard cursor < bytes.count else { throw JSONError("Invalid JSON") }
        switch bytes[cursor] {
        case 34: return .string(try quoted())
        case 123:
            cursor += 1
            space()
            if cursor < bytes.count && bytes[cursor] == 125 {
                cursor += 1
                return .object([])
            }
            var entries: [(String, PyJSON)] = []
            // Keys by their UTF-8 bytes (Swift's String == is canonical equivalence; Python compares code points), so
            // a repeated key is found in constant time: the last value, at the first key's place.
            var index: [[UInt8]: Int] = [:]
            while true {
                let key = try quoted()
                try take(58)
                let element = try value(depth: depth + 1)
                let keyBytes = Array(key.utf8)
                if let old = index[keyBytes] {
                    entries[old].1 = element
                } else {
                    index[keyBytes] = entries.count
                    entries.append((key, element))
                }
                space()
                guard cursor < bytes.count else { throw JSONError("Invalid JSON object") }
                if bytes[cursor] == 125 {
                    cursor += 1
                    return .object(entries)
                }
                try take(44)
            }
        case 91:
            cursor += 1
            space()
            if cursor < bytes.count && bytes[cursor] == 93 {
                cursor += 1
                return .array([])
            }
            var elements: [PyJSON] = []
            while true {
                elements.append(try value(depth: depth + 1))
                space()
                guard cursor < bytes.count else { throw JSONError("Invalid JSON array") }
                if bytes[cursor] == 93 {
                    cursor += 1
                    return .array(elements)
                }
                try take(44)
            }
        default:
            let start = cursor
            while cursor < bytes.count && ![9, 10, 13, 32, 44, 58, 93, 125].contains(bytes[cursor]) { cursor += 1 }
            guard cursor > start else { throw JSONError("Invalid JSON scalar") }
            let literal = String(decoding: bytes[start..<cursor], as: UTF8.self)
            guard Self.validScalar(literal) else { throw JSONError("Invalid JSON scalar") }
            return .scalar(literal)
        }
    }

    static func validScalar(_ s: String) -> Bool {
        if ["true", "false", "null", "NaN", "Infinity", "-Infinity"].contains(s) { return true }
        // JSON number grammar: -?(0|[1-9]\d*)(\.\d+)?([eE][+-]?\d+)?
        var b = Array(s.utf8)[...]
        if b.first == 45 { b = b.dropFirst() }
        guard let f = b.first, (48...57).contains(f) else { return false }
        if f == 48 { b = b.dropFirst() } else { b = b.drop { (48...57).contains($0) } }
        if b.first == 46 {
            b = b.dropFirst()
            guard let d = b.first, (48...57).contains(d) else { return false }
            b = b.drop { (48...57).contains($0) }
        }
        if b.first == 101 || b.first == 69 {
            b = b.dropFirst()
            if b.first == 43 || b.first == 45 { b = b.dropFirst() }
            guard let d = b.first, (48...57).contains(d) else { return false }
            b = b.drop { (48...57).contains($0) }
        }
        return b.isEmpty
    }
}
