// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import MLXLMCommon
import ArsenalWire

enum NuExamples {
    /// In-context examples as the card's `developer` messages. Example outputs arrive as decoded JSON (key order lost),
    /// so they are written in the template's key order (`template` is the exact template text), keys the template does
    /// not name following in sorted order, with Python json.dumps spacing as in the card's example.
    static func prepare(
        _ examples: [ArsenalWire.JSONValue], template: String = "", loadImage: (String) throws -> CIImage
    ) throws -> (messages: [Message], images: [UserInput.Image]) {
        guard examples.count <= 16 else { throw WorkerFailure(code: "invalid_request", message: "At most 16 extraction examples are supported.") }
        var messages: [Message] = [], images: [UserInput.Image] = []
        let order = KeyOrder.parse(template)
        for example in examples {
            guard let object = example.object, let output = object["output"], object["text"]?.string != nil || object["image"]?.object?["path"]?.string != nil else {
                throw WorkerFailure(code: "invalid_request", message: "Each extraction example requires text or image and its expected JSON output.")
            }
            var parts: [[String: any Sendable]] = []
            if let path = object["image"]?.object?["path"]?.string { images.append(.ciImage(try loadImage(path))); parts.append(["type": "image"]) }
            if let text = object["text"]?.string { parts.append(["type": "text", "text": text]) }
            // The maker template treats the last developer content item as the expected answer.
            parts.append(["type": "text", "text": encode(output, order: order)])
            messages.append(["role": "developer", "content": parts])
        }
        return (messages, images)
    }
    /// json.dumps(value, ensure_ascii=False) with object keys in `order`.
    static func encode(_ value: ArsenalWire.JSONValue, order: KeyOrder?) -> String {
        switch value {
        case .null: return "null"
        case .bool(let flag): return flag ? "true" : "false"
        case .number(let number):
            if number.isFinite, number == number.rounded(), abs(number) < 1e15 { return String(Int64(number)) }
            return number.isFinite ? "\(number)" : "null"
        case .string(let text): return quote(text)
        case .array(let items):
            return "[" + items.map { encode($0, order: order?.element) }.joined(separator: ", ") + "]"
        case .object(let members):
            let known = order?.keys ?? []
            let names = known.map(\.0).filter { members[$0] != nil } + members.keys.filter { key in !known.contains { $0.0 == key } }.sorted()
            return "{"
                + names.map { name in
                    quote(name) + ": " + encode(members[name]!, order: known.first { $0.0 == name }?.1)
                }.joined(separator: ", ") + "}"
        }
    }
    private static func quote(_ text: String) -> String {
        var result = "\""
        for scalar in text.unicodeScalars {
            switch scalar {
            case "\"": result += "\\\""
            case "\\": result += "\\\\"
            case "\n": result += "\\n"
            case "\r": result += "\\r"
            case "\t": result += "\\t"
            case "\u{08}": result += "\\b"
            case "\u{0C}": result += "\\f"
            default:
                if scalar.value < 0x20 { result += String(format: "\\u%04x", scalar.value) } else { result.unicodeScalars.append(scalar) }
            }
        }
        return result + "\""
    }
}

/// Object key order read from JSON text (a template), for writing decoded values back in that order.
indirect enum KeyOrder {
    case object([(String, KeyOrder)])
    case array(KeyOrder?)
    case leaf

    var keys: [(String, KeyOrder)] { if case .object(let keys) = self { return keys }; return [] }
    var element: KeyOrder? { if case .array(let element) = self { return element }; return nil }

    static func parse(_ text: String) -> KeyOrder? {
        var reader = Reader(Array(text.unicodeScalars))
        return reader.value()
    }
    private struct Reader {
        let s: [Unicode.Scalar]
        var i = 0
        init(_ s: [Unicode.Scalar]) { self.s = s }
        mutating func skip() { while i < s.count, [" ", "\n", "\r", "\t"].contains(s[i]) { i += 1 } }
        mutating func value() -> KeyOrder? {
            skip()
            guard i < s.count else { return nil }
            switch s[i] {
            case "{":
                i += 1
                var keys: [(String, KeyOrder)] = []
                while true {
                    skip()
                    guard i < s.count else { return nil }
                    if s[i] == "}" { i += 1; return .object(keys) }
                    if s[i] == "," { i += 1; continue }
                    guard s[i] == "\"", let key = string() else { return nil }
                    skip()
                    guard i < s.count, s[i] == ":" else { return nil }
                    i += 1
                    guard let child = value() else { return nil }
                    keys.append((key, child))
                }
            case "[":
                i += 1
                var first: KeyOrder?
                while true {
                    skip()
                    guard i < s.count else { return nil }
                    if s[i] == "]" { i += 1; return .array(first) }
                    if s[i] == "," { i += 1; continue }
                    guard let item = value() else { return nil }
                    if first == nil { first = item }
                }
            case "\"":
                return string() == nil ? nil : .leaf
            default:
                while i < s.count, !([",", "}", "]", " ", "\n", "\r", "\t"] as [Unicode.Scalar]).contains(s[i]) { i += 1 }
                return .leaf
            }
        }
        /// A JSON string starting at `"`; escapes are decoded.
        mutating func string() -> String? {
            i += 1
            var view = String.UnicodeScalarView()
            while i < s.count {
                let c = s[i]
                i += 1
                if c == "\"" { return String(view) }
                if c == "\\", i < s.count {
                    let e = s[i]; i += 1
                    switch e {
                    case "n": view.append("\n")
                    case "t": view.append("\t")
                    case "r": view.append("\r")
                    case "b": view.append("\u{08}")
                    case "f": view.append("\u{0C}")
                    case "u":
                        guard i + 4 <= s.count, let code = UInt32(String(String.UnicodeScalarView(s[i..<(i + 4)])), radix: 16) else { return nil }
                        i += 4
                        if let scalar = Unicode.Scalar(code) { view.append(scalar) }
                    default: view.append(e)
                    }
                } else {
                    view.append(c)
                }
            }
            return nil
        }
    }
}
