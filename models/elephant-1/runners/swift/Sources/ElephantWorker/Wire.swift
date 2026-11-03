// SPDX-License-Identifier: MIT
import ElephantEngine
import Darwin
import Foundation

/// ArsenalWire v1 framing on raw payload bytes (docs/RUNNERS.md): requests are parsed in order (System One option order
/// changes token ids), replies are written as the Python runner wrote them (sorted keys, compact, UTF-8).
enum Wire {
    struct Malformed: Error, CustomStringConvertible {
        let description: String
    }

    /// One kind-1 payload from `fd`; nil at a clean EOF before a frame.
    static func read(_ fd: Int32) throws -> Data? {
        guard let head = try exactly(5, fd, eofOK: true) else { return nil }
        let length = Int(head[0]) << 24 | Int(head[1]) << 16 | Int(head[2]) << 8 | Int(head[3])
        guard length >= 1, length <= 16 * 1024 * 1024 else { throw Malformed(description: "frame length \(length)") }
        guard head[4] == 1 else { throw Malformed(description: "unexpected frame kind \(head[4])") }
        guard let payload = try exactly(length - 1, fd, eofOK: false) else { throw Malformed(description: "short frame") }
        return payload
    }

    private static let lock = NSLock()

    /// Writes one JSON frame atomically (header and body under one lock).
    static func write(_ value: PyJSON, _ fd: Int32) {
        let json = Data(sortedCompact(value).utf8)
        let length = UInt32(json.count + 1)
        var frame = Data([UInt8(length >> 24 & 0xff), UInt8(length >> 16 & 0xff), UInt8(length >> 8 & 0xff), UInt8(length & 0xff), 1])
        frame.append(json)
        lock.lock()
        defer { lock.unlock() }
        frame.withUnsafeBytes { raw in
            var offset = 0
            while offset < raw.count {
                let n = Darwin.write(fd, raw.baseAddress! + offset, raw.count - offset)
                if n < 0 {
                    if errno == EINTR { continue }
                    return
                }
                offset += n
            }
        }
    }

    /// `json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))`.
    static func sortedCompact(_ value: PyJSON) -> String {
        var out = ""
        func go(_ v: PyJSON) {
            switch v {
            case .object(let fields):
                out += "{"
                let sorted = fields.sorted { $0.0.unicodeScalars.lexicographicallyPrecedes($1.0.unicodeScalars) { $0.value < $1.value } }
                for (i, (k, x)) in sorted.enumerated() {
                    if i > 0 { out += "," }
                    PyJSON.jsonString(k, &out)
                    out += ":"
                    go(x)
                }
                out += "}"
            case .array(let values):
                out += "["
                for (i, x) in values.enumerated() {
                    if i > 0 { out += "," }
                    go(x)
                }
                out += "]"
            default: out += v.compact()
            }
        }
        go(value)
        return out
    }

    private static func exactly(_ count: Int, _ fd: Int32, eofOK: Bool) throws -> Data? {
        var buffer = [UInt8](repeating: 0, count: count)
        var got = 0
        while got < count {
            let n = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress! + got, count - got) }
            if n == 0 {
                if got == 0 && eofOK { return nil }
                throw Malformed(description: "stream ended inside a frame")
            }
            if n < 0 {
                if errno == EINTR { continue }
                throw Malformed(description: "read error \(errno)")
            }
            got += n
        }
        return Data(buffer)
    }
}

extension PyJSON {
    static func number(_ d: Double) -> PyJSON { .scalar(d.isFinite ? PyJSON.floatRepr(d) : "null") }
    static func int(_ n: Int) -> PyJSON { .scalar(String(n)) }
    static let null = PyJSON.scalar("null")
    static func bool(_ b: Bool) -> PyJSON { .scalar(b ? "true" : "false") }
}
