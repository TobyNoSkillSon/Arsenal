// SPDX-License-Identifier: MIT
// ArsenalWire v1 framing for this runner, self-contained so the package has no path dependencies (Workers/Flux links
// to this folder, and relative package paths would resolve against the link). Same format as Packages/ArsenalWire:
// a 4-byte big-endian length, a kind byte (1 = JSON object), then the body.
import Foundation

enum JSONValue: Codable, Sendable, Equatable {
    case null, bool(Bool), number(Double), string(String), array([JSONValue]), object([String: JSONValue])
    init(from decoder: any Decoder) throws {
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
    func encode(to encoder: any Encoder) throws {
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
    var string: String? { if case .string(let v) = self { v } else { nil } }
    var number: Double? { if case .number(let v) = self { v } else { nil } }
    var bool: Bool? { if case .bool(let v) = self { v } else { nil } }
    var object: JSONObject? { if case .object(let v) = self { v } else { nil } }
    var array: [JSONValue]? { if case .array(let v) = self { v } else { nil } }
    subscript(_ key: String) -> JSONValue? { object?[key] }
}
typealias JSONObject = [String: JSONValue]
extension JSONValue: ExpressibleByStringLiteral, ExpressibleByIntegerLiteral, ExpressibleByBooleanLiteral, ExpressibleByArrayLiteral,
    ExpressibleByDictionaryLiteral, ExpressibleByFloatLiteral
{
    init(stringLiteral value: String) { self = .string(value) }
    init(integerLiteral value: Int) { self = .number(Double(value)) }
    init(floatLiteral value: Double) { self = .number(value) }
    init(booleanLiteral value: Bool) { self = .bool(value) }
    init(arrayLiteral elements: JSONValue...) { self = .array(elements) }
    init(dictionaryLiteral elements: (String, JSONValue)...) { self = .object(Dictionary(uniqueKeysWithValues: elements)) }
}

struct WorkerFailure: Error, Sendable, Equatable, LocalizedError {
    var code: String, message: String
    init(code: String, message: String) { self.code = code; self.message = message }
    var errorDescription: String? { message }
}

enum Frames {
    static let maximumLength = 16 * 1024 * 1024
    static func encode(_ object: JSONObject) throws -> Data {
        let e = JSONEncoder(); e.outputFormatting = [.sortedKeys, .withoutEscapingSlashes]
        var body = Data([1]); body.append(try e.encode(object))
        guard body.count <= maximumLength else { throw WorkerFailure(code: "oversized_frame", message: "Frame exceeds 16 MiB.") }
        let n = UInt32(body.count)
        return Data([UInt8(n >> 24), UInt8((n >> 16) & 255), UInt8((n >> 8) & 255), UInt8(n & 255)]) + body
    }
    /// Splits a byte stream into JSON objects (complete frames only).
    static func decode(_ data: Data) throws -> [JSONObject] {
        var out: [JSONObject] = [], i = data.startIndex
        while data.endIndex - i >= 4 {
            let n = data[i..<(i + 4)].reduce(0) { $0 << 8 | Int($1) }
            guard n > 0, n <= maximumLength, data.endIndex - i - 4 >= n else { break }
            let body = data[(i + 4)..<(i + 4 + n)]
            if body.first == 1 { out.append(try JSONDecoder().decode(JSONObject.self, from: Data(body.dropFirst()))) }
            i += 4 + n
        }
        return out
    }
    /// Reads one JSON frame; nil at a clean end of input.
    static func read(from handle: FileHandle) throws -> JSONObject? {
        guard let header = try exact(4, handle, eofOK: true) else { return nil }
        let n = header.reduce(0) { $0 << 8 | Int($1) }
        guard n > 0, n <= maximumLength, let body = try exact(n, handle, eofOK: false), body.first == 1 else {
            throw WorkerFailure(code: "malformed_frame", message: "Malformed frame.")
        }
        return try JSONDecoder().decode(JSONObject.self, from: Data(body.dropFirst()))
    }
    private static func exact(_ count: Int, _ handle: FileHandle, eofOK: Bool) throws -> Data? {
        var data = Data()
        while data.count < count {
            guard let chunk = try handle.read(upToCount: count - data.count), !chunk.isEmpty else {
                if data.isEmpty && eofOK { return nil }
                throw WorkerFailure(code: "truncated_frame", message: "Truncated frame.")
            }
            data.append(chunk)
        }
        return data
    }
}

/// Framed output with the same request-timing replies as Workers/Transport's WorkerChannel.
final class WorkerChannel: @unchecked Sendable {
    private let lock = NSLock(), output: FileHandle
    private struct Timing { var accepted: Double; var scheduled: Double?; var loading = false; var peak: (@Sendable () -> Double)? }
    private var timings: [Double: Timing] = [:]
    init(output: FileHandle) { self.output = output }
    private var now: Double { ProcessInfo.processInfo.systemUptime }
    func send(_ object: JSONObject) throws { let data = try Frames.encode(object); try lock.withLock { try output.write(contentsOf: data) } }
    func accepted(_ id: JSONValue) throws {
        if let n = id.number { lock.withLock { timings[n] = Timing(accepted: now) } }
        try send(["id": id, "event": "progress", "phase": "queued"])
    }
    func scheduled(_ id: JSONValue, phase: String, peak: @escaping @Sendable () -> Double) throws {
        if let n = id.number {
            lock.withLock {
                timings[n]?.scheduled = now; timings[n]?.loading = phase == "loading"; timings[n]?.peak = peak
            }
        }
        try send(["id": id, "event": "progress", "phase": .string(phase)])
    }
    func reply(_ id: JSONValue, _ fields: JSONObject = [:]) throws {
        var reply = fields
        let timing = lock.withLock { id.number.flatMap { timings.removeValue(forKey: $0) } }
        if let timing, let scheduled = timing.scheduled {
            var t: JSONObject = ["queue_s": .number(max(0, scheduled - timing.accepted))]
            if timing.loading { t["load_s"] = .number(now - scheduled) }
            if let peak = timing.peak { t["peak_mb"] = .number(peak()) }
            reply["timing"] = .object(t)
            try send(["id": id, "event": "progress", "phase": "done", "timing": .object(t)])
        }
        reply["id"] = id; reply["ok"] = true
        try send(reply)
    }
    func failure(_ id: JSONValue, code: String, message: String) throws {
        _ = lock.withLock { id.number.flatMap { timings.removeValue(forKey: $0) } }
        try send(["id": id, "ok": false, "error": ["code": .string(code), "message": .string(message.split(whereSeparator: \.isNewline).joined(separator: " "))]])
    }
}
