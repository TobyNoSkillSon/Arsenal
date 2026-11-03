// SPDX-License-Identifier: MIT
import Testing
import Foundation
@testable import ArsenalWire
@Test func fragmentedFrames() throws {
    let frames: [WireFrame] = [.json(["id": 7, "text": "µ"]), .binary(stream: 123, bytes: Data([0, 255, 42]))]
    let bytes = try frames.reduce(into: Data()) { $0.append(try $1.encoded()) }
    for chunk in 1...bytes.count {
        var decoder = FrameDecoder(); var out: [WireFrame] = []
        for i in stride(from: 0, to: bytes.count, by: chunk) { out += try decoder.append(Data(bytes.dropFirst(i).prefix(chunk))) }; try decoder.finish(); #expect(out == frames)
    }
}
@Test func hostileFrames() throws {
    for bytes in [Data([2, 0, 0, 0]), Data([0, 0, 0, 0]), Data([0, 0, 0, 1, 99]), Data([0, 0, 0, 2, 1, 91])] {
        var d = FrameDecoder(); #expect(throws: (any Error).self) { try d.append(bytes) }
    }
    var d = FrameDecoder(); _ = try d.append(Data([0, 0])); #expect(throws: WireError.truncatedFrame) { try d.finish() }
}

@Test func rawJSONKeepsByteDistinctLabelsAndNumbers() throws {
    let raw = Data("{\"answers\":{\"é\":1,\"e\\u0301\":2},\"id\":7,\"ok\":true,\"usage\":{\"x\":0.1234567890123456789}}".utf8)
    let body = try RawJSONObject.removing(["id", "ok"], from: raw)
    #expect(String(decoding: body, as: UTF8.self) == "{\"answers\":{\"é\":1,\"e\\u0301\":2},\"usage\":{\"x\":0.1234567890123456789}}")
    #expect(try RawJSONObject.member("answers", from: raw) == Data("{\"é\":1,\"e\\u0301\":2}".utf8))
}
