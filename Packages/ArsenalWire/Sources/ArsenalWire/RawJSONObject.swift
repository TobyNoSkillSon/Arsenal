// SPDX-License-Identifier: MIT
import Foundation

/// Slices top-level members without decoding nested dictionaries (Swift normalizes Unicode keys).
/// Use only after the framing layer has validated the payload as a JSON object.
public enum RawJSONObject {
    private struct Field { let key: String; let member: Range<Int>; let value: Range<Int> }
    public static func removing(_ keys: Set<String>, from data: Data) throws -> Data {
        let bytes = Array(data); let fields = try fields(bytes).filter { !keys.contains($0.key) }
        var output = Data([123]); for (i, field) in fields.enumerated() { if i > 0 { output.append(44) }; output.append(contentsOf: bytes[field.member]) }; output.append(125)
        return output
    }
    public static func member(_ key: String, from data: Data) throws -> Data? {
        let bytes = Array(data); guard let field = try fields(bytes).first(where: { $0.key == key }) else { return nil }; return Data(bytes[field.value])
    }
    private static func fields(_ bytes: [UInt8]) throws -> [Field] {
        var i = 0; var fields: [Field] = []
        func whitespace(_ b: UInt8) -> Bool { [9, 10, 13, 32].contains(b) }
        func skip() { while i < bytes.count && whitespace(bytes[i]) { i += 1 } }
        func quoted() throws {
            guard i < bytes.count, bytes[i] == 34 else { throw WireError.malformedFrame }; i += 1
            while i < bytes.count { let b = bytes[i]; i += 1; if b == 92 { guard i < bytes.count else { throw WireError.malformedFrame }; i += 1 } else if b == 34 { return } }
            throw WireError.malformedFrame
        }
        skip(); guard i < bytes.count, bytes[i] == 123 else { throw WireError.malformedFrame }; i += 1; skip()
        if i < bytes.count && bytes[i] == 125 { return fields }
        while i < bytes.count {
            skip(); let memberStart = i; let keyStart = i; try quoted(); let key = try JSONDecoder().decode(String.self, from: Data(bytes[keyStart..<i])); skip()
            guard i < bytes.count, bytes[i] == 58 else { throw WireError.malformedFrame }; i += 1; skip(); let valueStart = i
            var depth = 0
            while i < bytes.count {
                let b = bytes[i]
                if b == 34 { try quoted(); continue }
                if b == 123 || b == 91 { depth += 1 }
                if b == 125 || b == 93 { if depth == 0 { break }; depth -= 1 }
                if b == 44 && depth == 0 { break }; i += 1
            }
            var valueEnd = i; while valueEnd > valueStart && whitespace(bytes[valueEnd - 1]) { valueEnd -= 1 }
            guard valueEnd > valueStart else { throw WireError.malformedFrame }
            fields.append(Field(key: key, member: memberStart..<valueEnd, value: valueStart..<valueEnd))
            guard i < bytes.count else { throw WireError.malformedFrame }
            if bytes[i] == 125 { i += 1; skip(); guard i == bytes.count else { throw WireError.malformedFrame }; return fields }
            guard bytes[i] == 44 else { throw WireError.malformedFrame }; i += 1
        }
        throw WireError.malformedFrame
    }
}
