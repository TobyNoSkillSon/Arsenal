// SPDX-License-Identifier: MIT
import Foundation

/// Keep child JSON slices as bytes. Swift dictionaries equate NFC/NFD keys; typed decisions do not.
struct RawJSON: Sendable {
    let data: Data
    init(_ data: Data) throws {
        guard data.count <= 16 * 1024 * 1024 else { throw HTTPFailure(413, "JSON is too large.") }
        _ = try JSONSerialization.jsonObject(with: data, options: [.fragmentsAllowed])
        self.data = Data(data)
    }
    private init(slice: Data) { data = slice }
    var string: String? { try? JSONDecoder().decode(String.self, from: data) }
    var int: Int? { try? JSONDecoder().decode(Int.self, from: data) }
    var bool: Bool? { try? JSONDecoder().decode(Bool.self, from: data) }
    subscript(_ key: String) -> RawJSON? { (try? fields())?.first { $0.key == key }?.value }
    struct Field: Sendable { let key: String; let rawKey: Data; let value: RawJSON }
    func fields() throws -> [Field] {
        var scanner = Scanner(data: data)
        scanner.space(); guard scanner.take(123) else { throw HTTPFailure(400, "JSON must be an object.") }
        scanner.space(); if scanner.take(125) { return [] }
        var result: [Field] = []
        while true {
            scanner.space(); let start = scanner.offset; try scanner.string()
            let rawKey = Data(data[start..<scanner.offset])
            guard let key = try? JSONDecoder().decode(String.self, from: rawKey) else { throw HTTPFailure(400, "Invalid JSON key.") }
            scanner.space(); guard scanner.take(58) else { throw HTTPFailure(400, "JSON needs a colon.") }
            scanner.space(); let valueStart = scanner.offset; try scanner.value()
            result.append(Field(key: key, rawKey: rawKey, value: RawJSON(slice: Data(data[valueStart..<scanner.offset]))))
            scanner.space(); if scanner.take(125) { return result }; guard scanner.take(44) else { throw HTTPFailure(400, "JSON needs a comma.") }
        }
    }
    func elements() throws -> [RawJSON] {
        var scanner = Scanner(data: data)
        scanner.space(); guard scanner.take(91) else { throw HTTPFailure(400, "JSON must be an array.") }
        scanner.space(); if scanner.take(93) { return [] }
        var result: [RawJSON] = []
        while true {
            scanner.space(); let start = scanner.offset; try scanner.value()
            result.append(RawJSON(slice: Data(data[start..<scanner.offset])))
            scanner.space(); if scanner.take(93) { return result }; guard scanner.take(44) else { throw HTTPFailure(400, "Array needs a comma.") }
        }
    }
    func replacing(_ replacements: [String: Data] = [:], removing: Set<String> = []) throws -> Data {
        var pairs: [Data] = [], seen: Set<String> = []
        for field in try fields() where !removing.contains(field.key) {
            pairs.append(field.rawKey + Data(":".utf8) + (replacements[field.key] ?? field.value.data)); seen.insert(field.key)
        }
        for key in replacements.keys.sorted() where !seen.contains(key) { pairs.append(jsonData(key) + Data(":".utf8) + replacements[key]!) }
        return Data("{".utf8) + Self.join(pairs) + Data("}".utf8)
    }
    static func array(_ elements: [Data]) -> Data { Data("[".utf8) + join(elements) + Data("]".utf8) }
    private static func join(_ values: [Data]) -> Data {
        var output = Data()
        for (i, value) in values.enumerated() { if i > 0 { output.append(44) }; output += value }; return output
    }
    private struct Scanner {
        let data: Data
        var offset = 0
        mutating func space() { while offset < data.count, [9, 10, 13, 32].contains(data[offset]) { offset += 1 } }
        mutating func take(_ byte: UInt8) -> Bool { guard offset < data.count, data[offset] == byte else { return false }; offset += 1; return true }
        mutating func string() throws {
            guard take(34) else { throw HTTPFailure(400, "Expected JSON string.") }
            while offset < data.count {
                let byte = data[offset]; offset += 1
                if byte == 34 { return }
                if byte == 92 { guard offset < data.count else { break }; offset += 1 }
            }
            throw HTTPFailure(400, "Unterminated JSON string.")
        }
        mutating func value(depth: Int = 0) throws {
            guard depth < 128, offset < data.count else { throw HTTPFailure(400, "JSON is too deeply nested.") }
            if data[offset] == 34 { try string(); return }
            if data[offset] == 123 || data[offset] == 91 {
                let close: UInt8 = data[offset] == 123 ? 125 : 93
                offset += 1; space(); if take(close) { return }
                while true {
                    try value(depth: depth + 1); space()
                    if take(58) { space(); try value(depth: depth + 1); space() }
                    if take(close) { return }
                    guard take(44) else { throw HTTPFailure(400, "Malformed JSON container.") }; space()
                }
            }
            let start = offset
            while offset < data.count, ![9, 10, 13, 32, 44, 58, 93, 125].contains(data[offset]) { offset += 1 }
            guard offset > start else { throw HTTPFailure(400, "Expected JSON value.") }
        }
    }
}
