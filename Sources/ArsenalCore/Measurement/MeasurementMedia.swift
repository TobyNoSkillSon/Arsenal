// SPDX-License-Identifier: MIT
import Foundation
import ImageIO
import CoreGraphics

/// Decode precisely the usable files Arsenal returns, outside timing and outside the loaded model.
enum MeasurementMedia {
    static func answers(_ reply: JSONObject, batched: Bool) throws -> [Int] {
        let responses = batched ? reply["results"]?.array?.compactMap(\.object) : [reply]
        guard let responses, !responses.isEmpty else { throw ManifestError.invalid("Ordinary decisions returned no answers") }
        return try responses.map { response in
            guard let choice = response["answers"]?["tone"]?["choice"]?.string,
                let index = ["positive", "negative"].firstIndex(of: choice)
            else { throw ManifestError.invalid("Ordinary decision answer missing") }
            return index
        }
    }
    static func files(_ reply: JSONObject) -> [String] {
        (reply["path"]?.string.map { [$0] } ?? []) + ["images", "files"].flatMap { key in reply[key]?.array?.compactMap { $0["path"]?.string } ?? [] }
    }
    static func cleanup(_ reply: JSONObject, jobs: URL) {
        for path in files(reply) {
            let file = URL(fileURLWithPath: path).resolvingSymlinksInPath()
            guard file.path.hasPrefix(jobs.resolvingSymlinksInPath().path + "/") else { continue }
            try? FileManager.default.removeItem(at: file)
            if file.deletingLastPathComponent() != jobs { try? FileManager.default.removeItem(at: file.deletingLastPathComponent()) }
        }
    }
    static func output(kind: ModelStatKind, reply: JSONObject, fixtureID: String) throws -> MeasurementOutput {
        if kind == .decisions {
            let answers = try answers(reply, batched: true)
            guard answers.count == 4 else { throw ManifestError.invalid("Fixed decision fixture needs exactly four ordered answers") }
            return MeasurementOutput(fixtureID: fixtureID, decisions: answers)
        }
        guard let path = files(reply).first else { throw ManifestError.invalid("Ordinary operation returned no usable file") }
        let file = URL(fileURLWithPath: path)
        guard try FileDigest.bytes(file) <= OperationArtifacts.maxBytes else { throw ManifestError.invalid("Measurement output exceeds file budget") }
        if kind == .images {
            guard let source = CGImageSourceCreateWithURL(file as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil), image.width == 1024,
                image.height == 1024
            else {
                throw ManifestError.invalid("Ordinary quality image must be 1024²")
            }
            var bytes = [UInt8](repeating: 0, count: 1024 * 1024 * 4)
            try bytes.withUnsafeMutableBytes { buffer in
                guard
                    let context = CGContext(
                        data: buffer.baseAddress, width: 1024, height: 1024, bitsPerComponent: 8, bytesPerRow: 1024 * 4,
                        space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
                else { throw ManifestError.invalid("Image decoding unavailable") }
                context.draw(image, in: CGRect(x: 0, y: 0, width: 1024, height: 1024))
            }
            let samples = bytes.enumerated().compactMap { index, byte in index % 4 == 3 ? nil : Double(byte) / 255 }
            return MeasurementOutput(fixtureID: fixtureID, samples: samples, width: 1024, height: 1024)
        }
        let data = try Data(contentsOf: file)
        return try waveOutput(data, fixtureID: fixtureID)
    }
    static func waveOutput(_ data: Data, fixtureID: String) throws -> MeasurementOutput {
        guard data.count >= 44, String(decoding: data.prefix(4), as: UTF8.self) == "RIFF", String(decoding: data[8..<12], as: UTF8.self) == "WAVE" else {
            throw ManifestError.invalid("Expected ordinary PCM WAV output")
        }
        func u16(_ offset: Int) -> UInt16 { data.withUnsafeBytes { UInt16(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt16.self)) } }
        func u32(_ offset: Int) -> UInt32 { data.withUnsafeBytes { UInt32(littleEndian: $0.loadUnaligned(fromByteOffset: offset, as: UInt32.self)) } }
        var offset = 12, rate: Int?, samples: [Double]?
        while offset + 8 <= data.count {
            let tag = String(decoding: data[offset..<(offset + 4)], as: UTF8.self), size = Int(u32(offset + 4)), begin = offset + 8
            guard size <= data.count - begin else { throw ManifestError.invalid("Truncated WAV chunk") }
            if tag == "fmt " {
                guard size >= 16, u16(begin) == 1, u16(begin + 2) > 0, u16(begin + 14) == 16 else { throw ManifestError.invalid("Expected PCM16 output") }
                rate = Int(u32(begin + 4))
            } else if tag == "data" {
                guard size > 0, size % 2 == 0 else { throw ManifestError.invalid("Empty or malformed PCM samples") }
                samples = stride(from: begin, to: begin + size, by: 2).map { Double(Int16(bitPattern: u16($0))) / 32767 }
            }
            offset = begin + size + size % 2
        }
        guard let rate, rate > 0, let samples else { throw ManifestError.invalid("WAV sample rate/data missing") }
        return MeasurementOutput(fixtureID: fixtureID, samples: samples, sampleRate: rate)
    }
    static func tone(directory: URL) throws -> URL {
        let file = directory.appendingPathComponent("reference-tone.wav")
        if FileManager.default.fileExists(atPath: file.path) { return file }
        let rate = 24000, count = rate * 3
        var data = Data("RIFF".utf8)
        func u32(_ value: UInt32) { var value = value.littleEndian; withUnsafeBytes(of: &value) { data.append(contentsOf: $0) } }
        func u16(_ value: UInt16) { var value = value.littleEndian; withUnsafeBytes(of: &value) { data.append(contentsOf: $0) } }
        u32(UInt32(count * 2 + 36)); data.append(Data("WAVEfmt ".utf8)); u32(16); u16(1); u16(1); u32(UInt32(rate)); u32(UInt32(rate * 2)); u16(2); u16(16)
        data.append(Data("data".utf8)); u32(UInt32(count * 2))
        for index in 0..<count { u16(UInt16(bitPattern: Int16((0.1 * sin(Double(index) * 440 * 2 * .pi / Double(rate)) * 32767).rounded()))) }
        try data.write(to: file, options: .atomic); return file
    }
}
