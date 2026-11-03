// SPDX-License-Identifier: MIT
import Foundation

/// Reads JSON the way Python's `json` (and the raw-body decision workers) do: a repeated object key keeps its last value.
/// Foundation keeps the first, so a body checked with JSONDecoder could show the check one path and the worker another.
enum LastKeyJSON {
    static func parse(_ data: Data) throws -> JSONValue {
        var reader = Reader(bytes: Array(data))
        let value = try reader.value(depth: 0)
        reader.space()
        guard reader.cursor == reader.bytes.count else { throw failure }
        return value
    }
    private static let failure = WorkerFailure(code: "invalid_request", message: "Body must be valid JSON.")
    private struct Reader {
        let bytes: [UInt8]
        var cursor = 0
        let decoder = JSONDecoder()
        mutating func space() { while cursor < bytes.count, [9, 10, 13, 32].contains(bytes[cursor]) { cursor += 1 } }
        mutating func take(_ byte: UInt8) throws {
            space()
            guard cursor < bytes.count, bytes[cursor] == byte else { throw LastKeyJSON.failure }
            cursor += 1
        }
        mutating func scalar() throws -> Data {
            let begin = cursor
            if cursor < bytes.count, bytes[cursor] == 34 {
                cursor += 1
                var escaped = false
                while cursor < bytes.count {
                    let b = bytes[cursor]; cursor += 1
                    if escaped { escaped = false } else if b == 92 { escaped = true } else if b == 34 { return Data(bytes[begin..<cursor]) }
                }
                throw LastKeyJSON.failure
            }
            while cursor < bytes.count, ![44, 58, 93, 125, 9, 10, 13, 32].contains(bytes[cursor]) { cursor += 1 }
            guard cursor > begin else { throw LastKeyJSON.failure }
            return Data(bytes[begin..<cursor])
        }
        mutating func value(depth: Int) throws -> JSONValue {
            guard depth < 512 else { throw LastKeyJSON.failure }
            space()
            guard cursor < bytes.count else { throw LastKeyJSON.failure }
            switch bytes[cursor] {
            case 123:
                cursor += 1; space()
                var object: JSONObject = [:]
                if cursor < bytes.count, bytes[cursor] == 125 { cursor += 1; return .object(object) }
                while true {
                    space()
                    let key = try decoder.decode(String.self, from: scalar())
                    try take(58)
                    object[key] = try value(depth: depth + 1)
                    space()
                    guard cursor < bytes.count else { throw LastKeyJSON.failure }
                    if bytes[cursor] == 125 { cursor += 1; return .object(object) }
                    try take(44)
                }
            case 91:
                cursor += 1; space()
                var array: [JSONValue] = []
                if cursor < bytes.count, bytes[cursor] == 93 { cursor += 1; return .array(array) }
                while true {
                    array.append(try value(depth: depth + 1))
                    space()
                    guard cursor < bytes.count else { throw LastKeyJSON.failure }
                    if bytes[cursor] == 93 { cursor += 1; return .array(array) }
                    try take(44)
                }
            default:
                let literal = try scalar(), text = String(decoding: literal, as: UTF8.self)
                // Python's json also reads its non-finite floats, and a number past Double's range as infinity.
                switch text {
                case "NaN": return .number(.nan)
                case "Infinity": return .number(.infinity)
                case "-Infinity": return .number(-.infinity)
                default: break
                }
                if let value = try? decoder.decode(JSONValue.self, from: literal) { return value }
                guard text.first.map({ $0 == "-" || $0.isNumber }) == true, let number = Double(text) else { throw LastKeyJSON.failure }
                return .number(number)
            }
        }
    }
}
