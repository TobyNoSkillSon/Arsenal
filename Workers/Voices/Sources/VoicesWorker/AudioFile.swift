// SPDX-License-Identifier: MIT
import AVFoundation
import CryptoKit
import ArsenalWire
import Foundation
import VireoCore

// Saturating PCM16 transport; the model’s generated Float samples are unchanged.
enum PCM {
    static func encode(_ samples: [Float]) throws -> Data {
        var data = Data(capacity: samples.count * 2)
        for value in samples {
            guard value.isFinite else { throw EngineError("Non-finite audio output") }
            var n = Int16((max(-1, min(1, value)) * 32767).rounded()).littleEndian
            withUnsafeBytes(of: &n) { data.append(contentsOf: $0) }
        }
        return data
    }
    static func header(rate: Int, bytes: UInt32) -> Data {
        var d = Data("RIFF".utf8)
        func u32(_ n: UInt32) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
        func u16(_ n: UInt16) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
        u32(bytes + 36); d.append(Data("WAVEfmt ".utf8)); u32(16); u16(1); u16(1)
        u32(UInt32(rate)); u32(UInt32(rate * 2)); u16(2); u16(16); d.append(Data("data".utf8)); u32(bytes)
        return d
    }
    /// A caller's reference clip; anything wrong with the file is the caller's mistake (invalid_request, HTTP 422).
    static func reference(_ url: URL, rate: Int) throws -> AudioClip {
        let f: AVAudioFile
        do { f = try AVAudioFile(forReading: url) } catch {
            throw WorkerFailure(code: "invalid_request", message: "Reference audio cannot be read as an audio file")
        }
        guard f.length > 0, f.length <= AVAudioFramePosition(f.processingFormat.sampleRate * 30),
            let b = AVAudioPCMBuffer(pcmFormat: f.processingFormat, frameCapacity: AVAudioFrameCount(f.length))
        else {
            throw WorkerFailure(code: "invalid_request", message: "Reference must contain between zero and thirty seconds of audio")
        }
        // A read that fails before `length` is a corrupt file, however little is missing; only a clean end may fall short.
        if let failure = readEveryFrame(f, into: b), (failure as NSError).domain != NSOSStatusErrorDomain || (failure as NSError).code != -39 {
            throw WorkerFailure(code: "invalid_request", message: "Reference could not be decoded after \(b.frameLength) of \(f.length) frames; the file is corrupt")
        }
        // `length` can be an estimate for compressed files and over-reports a truncated WAV: a clean read ending within
        // 1% of it, and never more than 4,096 frames short (a cut file can also end cleanly early), is the file's real
        // end; a larger shortfall is a cut or corrupt file.
        guard b.frameLength > 0, AVAudioFramePosition(b.frameLength) >= f.length - min(AVAudioFramePosition(Double(f.length) * 0.01), 4_096) else {
            throw WorkerFailure(code: "invalid_request", message: "Reference ended after \(b.frameLength) of \(f.length) frames")
        }
        let format = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(rate), channels: 1, interleaved: false)!
        guard let converter = AVAudioConverter(from: b.format, to: format),
            let output = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(Double(b.frameLength) * Double(rate) / b.format.sampleRate + 64))
        else { throw EngineError("Audio conversion unavailable") }
        if b.format.channelCount > 1 { converter.downmix = true }
        var consumed = false; var error: NSError?
        converter.convert(to: output, error: &error) { _, status in
            if consumed { status.pointee = .endOfStream; return nil }
            consumed = true; status.pointee = .haveData; return b
        }
        guard error == nil, let samples = output.floatChannelData else { throw WorkerFailure(code: "invalid_request", message: "Reference is not readable PCM") }
        let values = Array(UnsafeBufferPointer(start: samples[0], count: Int(output.frameLength)))
        guard !values.isEmpty, values.allSatisfy(\.isFinite) else { throw WorkerFailure(code: "invalid_request", message: "Reference audio must contain finite PCM samples") }
        return AudioClip(samples: values, sampleRate: rate)
    }
    static func digest(_ url: URL) throws -> String {
        let f = try FileHandle(forReadingFrom: url); defer { try? f.close() }
        var hash = SHA256()
        while true {
            let more = try autoreleasepool { () throws -> Bool in
                guard let data = try f.read(upToCount: 8 << 20), !data.isEmpty else { return false }
                hash.update(data: data); return true
            }
            if !more { break }
        }
        return hash.finalize().map { String(format: "%02x", $0) }.joined()
    }
}

final class WaveSink {
    let file: URL
    private let handle: FileHandle
    private let rate: Int
    private var bytes: UInt32 = 0
    private var finished = false
    init(directory: URL, rate: Int) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.rate = rate
        file = directory.appendingPathComponent(UUID().uuidString + ".wav")
        guard FileManager.default.createFile(atPath: file.path, contents: PCM.header(rate: rate, bytes: 0)) else { throw EngineError("Cannot create audio output") }
        handle = try FileHandle(forWritingTo: file); try handle.seekToEnd()
    }
    func append(_ data: Data) throws {
        guard UInt64(bytes) + UInt64(data.count) < UInt64(UInt32.max) - 36 else { throw EngineError("Audio exceeds WAV size limit") }
        try handle.write(contentsOf: data); bytes += UInt32(data.count)
    }
    func finish() throws { try handle.seek(toOffset: 0); try handle.write(contentsOf: PCM.header(rate: rate, bytes: bytes)); try handle.close(); finished = true }
    deinit { try? handle.close(); if !finished { try? FileManager.default.removeItem(at: file) } }
}

/// Fills `buffer` with every frame of `file`. One `read(into:)` can stop short of `length` (float WAVs written
/// by libsndfile come back hundreds of frames short), so read in blocks and copy until the buffer is full. A read that
/// throws (end of file, a corrupt tail) ends the data and is returned; the caller checks how much arrived.
func readEveryFrame(_ file: AVAudioFile, into buffer: AVAudioPCMBuffer) -> Error? {
    buffer.frameLength = 0
    guard let block = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65_536), let target = buffer.floatChannelData else {
        return nil
    }
    while file.framePosition < file.length, buffer.frameLength < buffer.frameCapacity {
        let room = AVAudioFramePosition(buffer.frameCapacity - buffer.frameLength)
        do { try file.read(into: block, frameCount: AVAudioFrameCount(min(65_536, file.length - file.framePosition, room))) } catch { return error }
        guard block.frameLength > 0, let source = block.floatChannelData else { break }
        for channel in 0..<Int(buffer.format.channelCount) {
            (target[channel] + Int(buffer.frameLength)).update(from: source[channel], count: Int(block.frameLength))
        }
        buffer.frameLength += block.frameLength
    }
    return nil
}
