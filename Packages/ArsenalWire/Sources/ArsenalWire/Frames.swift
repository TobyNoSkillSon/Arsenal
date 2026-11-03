// SPDX-License-Identifier: MIT
import Foundation

public enum WireError: Error, Equatable { case oversizedFrame, malformedFrame, truncatedFrame, invalidHandshake }
public enum WireFrame: Sendable, Equatable {
    case json(JSONObject)
    case binary(stream: UInt32, bytes: Data)
    public static let maximumLength = 16 * 1024 * 1024
    public func encoded() throws -> Data {
        var body = Data()
        switch self {
        case .json(let object): body.append(1); body.append(try WireJSON.encode(object))
        case .binary(let stream, let bytes): body.append(2); body.appendBE(stream); body.append(bytes)
        }
        guard body.count <= Self.maximumLength else { throw WireError.oversizedFrame }
        var result = Data(); result.appendBE(UInt32(body.count)); result.append(body); return result
    }
    public static func decodeBody(_ body: Data) throws -> WireFrame {
        guard let kind = body.first else { throw WireError.malformedFrame }
        switch kind {
        case 1: do { return .json(try WireJSON.decode(Data(body.dropFirst()))) } catch { throw WireError.malformedFrame }
        case 2: guard body.count >= 5 else { throw WireError.malformedFrame }; return .binary(stream: body.beUInt32(at: 1), bytes: Data(body.dropFirst(5)))
        default: throw WireError.malformedFrame
        }
    }
}
public struct FrameDecoder: Sendable {
    private var buffer = Data()
    public init() {}
    public mutating func append(_ bytes: Data) throws -> [WireFrame] {
        buffer.append(bytes)
        var frames: [WireFrame] = []
        while buffer.count >= 4 {
            let length = Int(buffer.beUInt32(at: 0))
            guard length <= WireFrame.maximumLength else { throw WireError.oversizedFrame }
            guard length > 0 else { throw WireError.malformedFrame }
            guard buffer.count >= 4 + length else { break }
            frames.append(try WireFrame.decodeBody(Data(buffer.dropFirst(4).prefix(length))))
            buffer = Data(buffer.dropFirst(4 + length))
        }
        return frames
    }
    public func finish() throws { if !buffer.isEmpty { throw WireError.truncatedFrame } }
}
public enum FrameIO {
    public static func read(from handle: FileHandle) throws -> WireFrame? { try readPacket(from: handle)?.frame }
    public static func readPacket(from handle: FileHandle) throws -> (frame: WireFrame, rawJSON: Data?)? {
        guard let header = try exact(4, from: handle, allowEOF: true) else { return nil }
        let length = Int(header.beUInt32(at: 0))
        guard length <= WireFrame.maximumLength else { throw WireError.oversizedFrame }
        guard length > 0, let body = try exact(length, from: handle) else { throw WireError.malformedFrame }
        return (try WireFrame.decodeBody(body), body.first == 1 ? Data(body.dropFirst()) : nil)
    }
    private static func exact(_ count: Int, from handle: FileHandle, allowEOF: Bool = false) throws -> Data? {
        var data = Data()
        while data.count < count {
            guard let chunk = try handle.read(upToCount: count - data.count), !chunk.isEmpty else {
                if data.isEmpty && allowEOF { return nil }; throw WireError.truncatedFrame
            }
            data.append(chunk)
        }
        return data
    }
    public static func write(_ frame: WireFrame, to handle: FileHandle) throws { try handle.write(contentsOf: frame.encoded()) }
}
extension Data {
    mutating func appendBE(_ value: UInt32) { append(contentsOf: [UInt8(value >> 24), UInt8((value >> 16) & 255), UInt8((value >> 8) & 255), UInt8(value & 255)]) }
    func beUInt32(at offset: Int) -> UInt32 { let b = Array(self.dropFirst(offset).prefix(4)); return UInt32(b[0]) << 24 | UInt32(b[1]) << 16 | UInt32(b[2]) << 8 | UInt32(b[3]) }
}
