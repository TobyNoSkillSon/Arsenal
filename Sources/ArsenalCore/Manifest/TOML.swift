// SPDX-License-Identifier: MIT
import Foundation

public enum ManifestError: Error, LocalizedError, Sendable {
    case invalid(String)
    public var errorDescription: String? { if case .invalid(let text) = self { return text }; return nil }
}

public indirect enum TOMLValue: Sendable, Equatable {
    case string(String), integer(Int64), decimal(Double), bool(Bool), array([TOMLValue]), table([String: TOMLValue])
    var json: Any {
        switch self {
        case .string(let x): return x
        case .integer(let x): return x
        case .decimal(let x): return x
        case .bool(let x): return x
        case .array(let x): return x.map(\.json)
        case .table(let x): return x.mapValues(\.json)
        }
    }
}

/// Deliberately non-executable TOML subset from CONTRACTS §3; dates and multiline strings are not supported.
public struct TOMLParser {
    private var chars: [Character]
    private var index = 0
    private var line = 1
    public init(_ text: String) { chars = Array(text) }
    private var peek: Character? { index < chars.count ? chars[index] : nil }
    @discardableResult private mutating func take() -> Character? {
        guard let c = peek else { return nil }; index += 1; if c.isNewline { line += 1 }; return c
    }
    private func error(_ text: String) -> ManifestError { .invalid("TOML line \(line): \(text)") }
    private mutating func space(newlines: Bool = false) {
        while let c = peek {
            if c == " " || c == "\t" || c == "\r" || (newlines && c.isNewline) {
                take()
            } else if c == "#" {
                while let c = peek, !c.isNewline { take() }; if !newlines { return }
            } else {
                return
            }
        }
    }
    private mutating func expect(_ c: Character) throws { guard take() == c else { throw error("expected \(c)") } }
    private mutating func string() throws -> String {
        let quote = take()!; var value = ""
        while let c = take() {
            if c == quote { return value }
            if c.isNewline { throw error("multiline strings are unsupported") }
            if c == "\\" && quote == "\"" {
                guard let escaped = take() else { throw error("unfinished escape") }
                switch escaped {
                case "n": value.append("\n")
                case "r": value.append("\r")
                case "t": value.append("\t")
                case "b": value.append("\u{8}")
                case "f": value.append("\u{c}")
                case "\"", "\\": value.append(escaped)
                case "u", "U":
                    var hex = ""; for _ in 0..<(escaped == "u" ? 4 : 8) { guard let c = take(), c.isHexDigit else { throw error("invalid Unicode escape") }; hex.append(c) }
                    guard let n = UInt32(hex, radix: 16), let scalar = UnicodeScalar(n) else { throw error("invalid Unicode scalar") }; value.unicodeScalars.append(scalar)
                default: throw error("unknown escape")
                }
            } else {
                value.append(c)
            }
        }
        throw error("unterminated string")
    }
    private mutating func key() throws -> [String] {
        var result: [String] = []
        repeat {
            space(); var part = ""
            if peek == "\"" || peek == "'" {
                part = try string()
            } else {
                while let c = peek, c.isASCII && (c.isLetter || c.isNumber || c == "_" || c == "-") { part.append(take()!) }
            }
            guard !part.isEmpty else { throw error("empty key") }; result.append(part); space()
            guard peek == "." else { break }; take()
        } while true
        return result
    }
    private mutating func value() throws -> TOMLValue {
        space(newlines: true)
        if peek == "\"" || peek == "'" { return .string(try string()) }
        if peek == "[" {
            take(); var list: [TOMLValue] = []; space(newlines: true)
            while peek != "]" {
                guard peek != nil else { throw error("unfinished array") }; list.append(try value()); space(newlines: true)
                if peek == "," { take(); space(newlines: true) } else if peek != "]" { throw error("expected comma") }
            }; take(); return .array(list)
        }
        if peek == "{" {
            take(); var fields: [String: TOMLValue] = [:]; space()
            while peek != "}" {
                let keys = try key(); try expect("="); let v = try value(); try Self.put(&fields, keys: keys, value: v); space()
                if peek == "," { take(); space() } else if peek != "}" { throw error("expected comma") }
            }; take(); return .table(fields)
        }
        var token = ""
        while let c = peek, !c.isWhitespace && !",#]}".contains(c) { token.append(take()!) }
        if token == "true" { return .bool(true) }; if token == "false" { return .bool(false) }
        let number = token.replacingOccurrences(of: "_", with: "")
        if let integer = Int64(number) { return .integer(integer) }
        if let decimal = Double(number), decimal.isFinite { return .decimal(decimal) }
        throw error("unsupported value \(token)")
    }
    private static func put(_ table: inout [String: TOMLValue], keys: [String], value: TOMLValue, append: Bool = false) throws {
        let first = keys[0]
        if keys.count == 1 {
            if append {
                var array: [TOMLValue] = []
                if let old = table[first] { guard case .array(let a) = old else { throw ManifestError.invalid("duplicate table \(first)") }; array = a }
                array.append(value); table[first] = .array(array)
            } else {
                guard table[first] == nil else { throw ManifestError.invalid("duplicate key \(first)") }; table[first] = value
            }
            return
        }
        let rest = Array(keys.dropFirst())
        if case .array(var items) = table[first], case .table(var nested) = items.last {
            try put(&nested, keys: rest, value: value, append: append); items[items.count - 1] = .table(nested); table[first] = .array(items)
        } else {
            var nested: [String: TOMLValue] = [:]
            if let old = table[first] { guard case .table(let t) = old else { throw ManifestError.invalid("not a table: \(first)") }; nested = t }
            try put(&nested, keys: rest, value: value, append: append); table[first] = .table(nested)
        }
    }
    public mutating func parse() throws -> [String: TOMLValue] {
        var root: [String: TOMLValue] = [:]; var context: [String] = []
        while true {
            space(newlines: true); guard peek != nil else { return root }
            if peek == "[" {
                take(); let array = peek == "["; if array { take() }; context = try key(); try expect("]")
                if array { try expect("]"); try Self.put(&root, keys: context, value: .table([:]), append: true) }
            } else {
                let keys = try key(); try expect("="); let v = try value(); try Self.put(&root, keys: context + keys, value: v)
            }
            space(); guard peek == nil || peek?.isNewline == true else { throw error("unexpected trailing text") }; if peek != nil { take() }
        }
    }
    public static func decode<T: Decodable>(_ type: T.Type, from text: String) throws -> T {
        var parser = Self(text); let fields = try parser.parse()
        do { return try JSONDecoder().decode(type, from: JSONSerialization.data(withJSONObject: fields.mapValues(\.json))) } catch {
            throw ManifestError.invalid("manifest fields: \(error.localizedDescription)")
        }
    }
}
