// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import ArsenalWire

func audioWave(_ samples: [Float], rate: Int, channels: Int, id: String, input: JSONObject, cancellation: NativeCancellation) throws -> JSONValue {
    guard let directory = input["out_dir"]?.string, !samples.isEmpty, samples.count % channels == 0, samples.allSatisfy(\.isFinite),
        samples.count <= (NativeFiles.maxBytes - 44) / 2
    else { throw audioFailure("Audio output is invalid or exceeds the output bound.") }
    let root = URL(fileURLWithPath: directory), url = root.appendingPathComponent(id + "-" + UUID().uuidString.lowercased() + ".wav")
    var data = Data()
    func string(_ value: String) { data.append(contentsOf: value.utf8) }
    func u32(_ value: Int) { var little = UInt32(value).littleEndian; withUnsafeBytes(of: &little) { data.append(contentsOf: $0) } }
    func u16(_ value: Int) { var little = UInt16(value).littleEndian; withUnsafeBytes(of: &little) { data.append(contentsOf: $0) } }
    string("RIFF"); u32(samples.count * 2 + 36); string("WAVEfmt "); u32(16); u16(1); u16(channels)
    u32(rate); u32(rate * channels * 2); u16(channels * 2); u16(16); string("data"); u32(samples.count * 2)
    for (index, sample) in samples.enumerated() {
        if index % 32768 == 0 { try cancellation.check() }
        var value = Int16(max(-32768, min(32767, (sample * 32768).rounded()))).littleEndian
        withUnsafeBytes(of: &value) { data.append(contentsOf: $0) }
    }
    try cancellation.check()
    try data.write(to: url, options: .withoutOverwriting)
    return .object(try NativeMedia.file(url, root: root, mime: "audio/wav", name: id + ".wav", id: id))
}
