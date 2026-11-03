// SPDX-License-Identifier: MIT
import AVFoundation
import ArsenalWire
import CryptoKit
import Foundation
import IntegrationKit
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
    static func header(rate: Int, channels: Int = 1, bytes: UInt32) -> Data {
        var d = Data("RIFF".utf8)
        func u32(_ n: UInt32) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
        func u16(_ n: UInt16) { var n = n.littleEndian; withUnsafeBytes(of: &n) { d.append(contentsOf: $0) } }
        u32(bytes + 36); d.append(Data("WAVEfmt ".utf8)); u32(16); u16(1); u16(UInt16(channels))
        u32(UInt32(rate)); u32(UInt32(rate * channels * 2)); u16(UInt16(channels * 2)); u16(16); d.append(Data("data".utf8)); u32(bytes)
        return d
    }
    /// The highest rate an input is decoded at: every Sound engine runs at 48 kHz or below.
    static let maximumInputRate = 48_000

    /// The caller's audio. Anything wrong with the file is the caller's mistake: invalid_request (HTTP 422), not a
    /// server failure. It streams through the shared decoder at its own rate, capped at 48 kHz, so memory is the
    /// decoded output (at most 600 s x 8 channels at 48 kHz) whatever rate the file was written at.
    static func input(_ url: URL) throws -> AudioClip {
        func refuse(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }
        let file = try NativeMedia.openAudio(url)
        let format = file.processingFormat, length = file.length, channels = Int(format.channelCount)
        guard length > 0, format.sampleRate.isFinite, format.sampleRate > 0, (1...8).contains(channels),
            Double(length) <= format.sampleRate * 600
        else {
            throw refuse("Input must contain 0–600 seconds of audio with at most eight channels")
        }
        let rate = min(max(Int(format.sampleRate.rounded()), 8000), maximumInputRate)
        let audio = try NativeMedia.audio(url, sampleRate: rate, channels: channels, maximumFrames: rate * 600)
        return AudioClip(samples: audio.samples, sampleRate: audio.sampleRate, channels: audio.channels)
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
    private let channels: Int
    private var bytes: UInt32 = 0
    private var finished = false
    init(directory: URL, rate: Int, channels: Int = 1) throws {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        self.rate = rate; self.channels = channels
        file = directory.appendingPathComponent(UUID().uuidString + ".wav")
        guard FileManager.default.createFile(atPath: file.path, contents: PCM.header(rate: rate, channels: channels, bytes: 0)) else {
            throw EngineError("Cannot create audio output")
        }
        handle = try FileHandle(forWritingTo: file); try handle.seekToEnd()
    }
    func append(_ data: Data) throws {
        guard UInt64(bytes) + UInt64(data.count) < UInt64(UInt32.max) - 36 else { throw EngineError("Audio exceeds WAV size limit") }
        try handle.write(contentsOf: data); bytes += UInt32(data.count)
    }
    func finish() throws {
        try handle.seek(toOffset: 0); try handle.write(contentsOf: PCM.header(rate: rate, channels: channels, bytes: bytes)); try handle.close(); finished = true
    }
    deinit { try? handle.close(); if !finished { try? FileManager.default.removeItem(at: file) } }
}
