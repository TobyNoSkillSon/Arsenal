// SPDX-License-Identifier: MIT
import Foundation
import ArsenalWire

/// ArsenalWire v1 framing (docs/CONTRACTS.md §4) on raw payload bytes. ArsenalWire's FrameIO decodes JSON into
/// `[String: JSONValue]`, which loses key order and merges keys that differ only by Unicode normalization; System
/// One question and option order changes token ids, and Verdict keeps NFC/NFD labels apart. So the decisions worker
/// reads and writes kind-1 payloads as bytes (OrderedJSONParser in, ResponseJSON out) with ArsenalWire's limits.
public enum Wire {
    public static let maxFrame = WireFrame.maximumLength
    public static let jsonKind: UInt8 = 1

    public struct Malformed: Error, CustomStringConvertible {
        public let description: String
        public init(description: String) { self.description = description }
    }

    /// Reads one frame from `fd`. nil at a clean EOF before a frame starts; throws on a short, oversized or
    /// non-JSON frame (the caller exits: a malformed stream kills the runner).
    public static func read(_ fd: Int32 = 0) throws -> Data? {
        guard let head = try exactly(5, fd, eofOK: true) else { return nil }
        let length = Int(head[0]) << 24 | Int(head[1]) << 16 | Int(head[2]) << 8 | Int(head[3])
        guard length >= 1, length <= maxFrame else { throw Malformed(description: "frame length \(length)") }
        guard head[4] == jsonKind else { throw Malformed(description: "unexpected frame kind \(head[4])") }
        guard let payload = try exactly(length - 1, fd, eofOK: false) else { throw Malformed(description: "short frame") }
        return payload
    }

    /// Writes one JSON frame from already-serialized bytes.
    public static func write(_ json: Data, _ fd: Int32 = 1) {
        let length = UInt32(json.count + 1)
        var frame = Data([UInt8(length >> 24 & 0xff), UInt8(length >> 16 & 0xff), UInt8(length >> 8 & 0xff), UInt8(length & 0xff), jsonKind])
        frame.append(json)
        lock.lock(); defer { lock.unlock() }
        frame.withUnsafeBytes { raw in
            var offset = 0
            while offset < raw.count {
                let n = Darwin.write(fd, raw.baseAddress! + offset, raw.count - offset)
                if n < 0 { if errno == EINTR { continue }; return }
                offset += n
            }
        }
    }
    private static let lock = NSLock()

    private static func exactly(_ count: Int, _ fd: Int32, eofOK: Bool) throws -> Data? {
        var buffer = [UInt8](repeating: 0, count: count)
        var got = 0
        while got < count {
            let n = buffer.withUnsafeMutableBytes { Darwin.read(fd, $0.baseAddress! + got, count - got) }
            if n == 0 {
                if got == 0 && eofOK { return nil }
                throw Malformed(description: "stream ended inside a frame")
            }
            if n < 0 { if errno == EINTR { continue }; throw Malformed(description: "read error \(errno)") }
            got += n
        }
        return Data(buffer)
    }
}
