//
//  AudioFiles.swift
//  MusicGeneration
//
//  Created by David Sherlock on 2026.
//
//  Reading audio in (for covers, repaints and timbre references) and writing
//  songs out, through AVFoundation: WAV 16-bit or 32-bit float, FLAC, ALAC,
//  and AAC in an M4A. Any sample rate the converter takes.
//

import AVFoundation
import Foundation

// The file a song is written as.
public enum AudioFileFormat: String, Codable, Sendable, CaseIterable {
    // 16-bit PCM WAV (upstream `wav`).
    case wav
    // 32-bit float WAV (upstream `wav32`).
    case wav32
    // FLAC, lossless (upstream `flac`).
    case flac
    // AAC in an M4A (upstream `aac`).
    case m4a
    // Apple Lossless in an M4A.
    case alac

    // The format a path's extension names; `.m4a` is AAC.
    public static func from(extension ext: String) -> AudioFileFormat? {
        switch ext.lowercased() {
        case "wav", "wave": .wav
        case "flac": .flac
        case "m4a", "aac", "mp4": .m4a
        case "caf": .alac
        default: nil
        }
    }

    public var fileExtension: String {
        switch self {
        case .wav, .wav32: "wav"
        case .flac: "flac"
        case .m4a, .alac: "m4a"
        }
    }
}

// Why an audio file could not be read or written.
public struct AudioFileError: Error, Sendable, LocalizedError {
    public let message: String
    public init(_ message: String) { self.message = message }
    public var errorDescription: String? { message }
}

public enum AudioFiles {

    // Writes `audio` to `url` as `format`, converting to `sampleRate` first when given.
    public static func write(
        _ audio: SongAudio, to url: URL, format: AudioFileFormat,
        sampleRate: Double? = nil, aacBitRate: Int = 256_000
    ) throws(AudioFileError) {
        let source = sampleRate.map { $0 == audio.sampleRate ? audio : resampled(audio, to: $0) } ?? audio
        let rate = source.sampleRate
        let channels = AVAudioChannelCount(source.channels.count)
        var settings: [String: Any] = [AVSampleRateKey: rate, AVNumberOfChannelsKey: channels]
        switch format {
        case .wav:
            settings[AVFormatIDKey] = kAudioFormatLinearPCM
            settings[AVLinearPCMBitDepthKey] = 16
            settings[AVLinearPCMIsFloatKey] = false
            settings[AVLinearPCMIsBigEndianKey] = false
        case .wav32:
            settings[AVFormatIDKey] = kAudioFormatLinearPCM
            settings[AVLinearPCMBitDepthKey] = 32
            settings[AVLinearPCMIsFloatKey] = true
            settings[AVLinearPCMIsBigEndianKey] = false
        case .flac:
            settings[AVFormatIDKey] = kAudioFormatFLAC
        case .alac:
            settings[AVFormatIDKey] = kAudioFormatAppleLossless
            settings[AVEncoderBitDepthHintKey] = 24
        case .m4a:
            settings[AVFormatIDKey] = kAudioFormatMPEG4AAC
            settings[AVEncoderBitRateKey] = aacBitRate
        }
        try? FileManager.default.removeItem(at: url)
        guard let processing = AVAudioFormat(standardFormatWithSampleRate: rate, channels: channels),
            let buffer = AVAudioPCMBuffer(pcmFormat: processing, frameCapacity: AVAudioFrameCount(source.frameCount))
        else { throw AudioFileError("cannot make a \(Int(rate)) Hz buffer") }
        buffer.frameLength = AVAudioFrameCount(source.frameCount)
        for (index, channel) in source.channels.enumerated() {
            channel.withUnsafeBufferPointer { buffer.floatChannelData![index].update(from: $0.baseAddress!, count: channel.count) }
        }
        do {
            let fileType: AVFileType =
                switch format {
                case .wav, .wav32: .wav
                case .flac: AVFileType(rawValue: "org.xiph.flac")
                case .m4a, .alac: .m4a
                }
            _ = fileType
            let file = try AVAudioFile(
                forWriting: url, settings: settings, commonFormat: .pcmFormatFloat32,
                interleaved: false)
            try file.write(from: buffer)
        } catch {
            throw AudioFileError("cannot write \(url.lastPathComponent) as \(format.rawValue): \(error.localizedDescription)")
        }
    }

    // Reads any file AVFoundation opens, as float channels at its own rate.
    public static func read(_ url: URL) throws(AudioFileError) -> SongAudio {
        let file: AVAudioFile
        do { file = try AVAudioFile(forReading: url) } catch {
            throw AudioFileError("cannot open \(url.lastPathComponent): \(error.localizedDescription)")
        }
        let format = file.processingFormat
        guard let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: AVAudioFrameCount(file.length)) else {
            throw AudioFileError("cannot read \(url.lastPathComponent)")
        }
        do { try readEveryFrame(file, into: buffer) } catch {
            throw AudioFileError("cannot read \(url.lastPathComponent): \(error.localizedDescription)")
        }
        guard AVAudioFramePosition(buffer.frameLength) == file.length else {
            throw AudioFileError("\(url.lastPathComponent) ended after \(buffer.frameLength) of \(file.length) frames")
        }
        guard let data = buffer.floatChannelData else { throw AudioFileError("\(url.lastPathComponent) is not float audio") }
        let frames = Int(buffer.frameLength)
        let channels = (0..<Int(format.channelCount)).map { Array(UnsafeBufferPointer(start: data[$0], count: frames)) }
        return SongAudio(channels: channels, sampleRate: format.sampleRate)
    }

    // `audio` at another sample rate (AVAudioConverter, mastering quality).
    public static func resampled(_ audio: SongAudio, to rate: Double) -> SongAudio {
        let count = AVAudioChannelCount(audio.channels.count)
        guard rate != audio.sampleRate, audio.frameCount > 0,
            let from = AVAudioFormat(standardFormatWithSampleRate: audio.sampleRate, channels: count),
            let to = AVAudioFormat(standardFormatWithSampleRate: rate, channels: count),
            let converter = AVAudioConverter(from: from, to: to),
            let input = AVAudioPCMBuffer(pcmFormat: from, frameCapacity: AVAudioFrameCount(audio.frameCount))
        else { return audio }
        converter.sampleRateConverterQuality = AVAudioQuality.max.rawValue
        converter.sampleRateConverterAlgorithm = AVSampleRateConverterAlgorithm_Mastering
        input.frameLength = AVAudioFrameCount(audio.frameCount)
        for (index, channel) in audio.channels.enumerated() {
            channel.withUnsafeBufferPointer { input.floatChannelData![index].update(from: $0.baseAddress!, count: channel.count) }
        }
        let capacity = AVAudioFrameCount(Double(audio.frameCount) * rate / audio.sampleRate) + 4096
        guard let output = AVAudioPCMBuffer(pcmFormat: to, frameCapacity: capacity) else { return audio }
        var fed = false
        var error: NSError?
        _ = converter.convert(to: output, error: &error) { _, status in
            if fed { status.pointee = .endOfStream; return nil }
            fed = true
            status.pointee = .haveData
            return input
        }
        guard error == nil, let data = output.floatChannelData else { return audio }
        let frames = Int(output.frameLength)
        return SongAudio(
            channels: (0..<Int(count)).map { Array(UnsafeBufferPointer(start: data[$0], count: frames)) },
            sampleRate: rate)
    }
}

/// Fills `buffer` with every frame of `file`. One `read(into:)` can stop short of `length` (float WAVs written
/// by libsndfile come back hundreds of frames short), so read in blocks and copy until the buffer is full.
private func readEveryFrame(_ file: AVAudioFile, into buffer: AVAudioPCMBuffer) throws {
    buffer.frameLength = 0
    guard let block = AVAudioPCMBuffer(pcmFormat: file.processingFormat, frameCapacity: 65_536), let target = buffer.floatChannelData else {
        return
    }
    while file.framePosition < file.length, buffer.frameLength < buffer.frameCapacity {
        let room = AVAudioFramePosition(buffer.frameCapacity - buffer.frameLength)
        try file.read(into: block, frameCount: AVAudioFrameCount(min(65_536, file.length - file.framePosition, room)))
        guard block.frameLength > 0, let source = block.floatChannelData else { break }
        for channel in 0..<Int(buffer.format.channelCount) {
            (target[channel] + Int(buffer.frameLength)).update(from: source[channel], count: Int(block.frameLength))
        }
        buffer.frameLength += block.frameLength
    }
}
