// SPDX-License-Identifier: MIT
import Foundation
import AVFoundation
import CoreGraphics
import ImageIO
import CryptoKit
import ArsenalWire

public enum NativeMedia {
    public struct Audio {
        public let samples: [Float]
        public let sampleRate: Int
        public let channels: Int
    }
    /// How a file's channels become the decoded ones.
    public enum ChannelMix: Sendable {
        /// AVAudioConverter maps them, averaging when the target has fewer channels.
        case convert
        /// One source channel on its own (DiariZen reads the first channel only).
        case channel(Int)
        /// The mean of every source channel (librosa's mono=True), for files with more channels than a plain PCM target holds.
        case mean
    }
    /// Most source channels a decode reads. Only one 64k-frame source block is live at a time, so this bounds it at 8 MB.
    public static let maximumSourceChannels = 32
    /// A read that ends this close to `AVAudioFile.length` is the file's real end (length can be an estimate for compressed
    /// files): 1% of the length, but never more than 4,096 source frames (under 0.1 s at 44.1 kHz), so a long file cannot
    /// lose a long tail unnoticed. A larger shortfall is a corrupt or cut file and is refused.
    static let endTolerance = 0.01
    static let endToleranceFrames: AVAudioFramePosition = 4_096

    /// Whether a decode that read `read` of `length` frames reached the file's real end. AVAudioFile ends a clean file
    /// with eofErr (-39) or an empty read; any other error before `length` is a decode failure and is refused however
    /// small the shortfall. A cut file can also end in eofErr early, which the frame cap bounds.
    public static func checkEnd(read: AVAudioFramePosition, length: AVAudioFramePosition, failure: Error?) throws {
        if let failure, read < length, !isEndOfFile(failure) {
            throw mediaError("Audio could not be decoded after \(read) of \(length) frames; the file is corrupt.")
        }
        let allowed = min(AVAudioFramePosition(Double(length) * endTolerance), endToleranceFrames)
        guard read > 0, read >= length - allowed else {
            throw mediaError("Audio ended after \(read) of \(length) frames; the file is cut short or corrupt.")
        }
    }

    static func isEndOfFile(_ error: Error) -> Bool {
        let error = error as NSError
        return error.domain == NSOSStatusErrorDomain && error.code == -39
    }

    /// Opens a caller's audio file. One AVFoundation cannot open (wrong format, corrupt, a renamed image) is the caller's
    /// mistake: invalid_request (HTTP 422), never a retryable service failure.
    public static func openAudio(_ url: URL) throws -> AVAudioFile {
        do { return try AVAudioFile(forReading: url) } catch {
            throw mediaError("Audio could not be decoded: the file is not a supported audio format or is corrupt.")
        }
    }

    /// Runs a media decode; any failure that is not already a WorkerFailure comes from the file and is reported as
    /// invalid_request (HTTP 422) rather than `runner` (503).
    public static func decoding<T>(_ kind: String, _ body: () throws -> T) throws -> T {
        do { return try body() } catch let failure as WorkerFailure { throw failure } catch {
            // Media services being reset is the system's temporary fault, not the file's: `runner` (503),
            // so a client retries.
            let error = error as NSError
            if error.domain == AVFoundationErrorDomain, error.code == -11819 { // AVErrorMediaServicesWereReset
                throw WorkerFailure(code: "runner", message: "\(kind) decoding hit a temporary system fault: \(error.localizedDescription)")
            }
            throw mediaError("\(kind) could not be decoded: \(error.localizedDescription)")
        }
    }

    /// AVFoundation decodes supported containers, then converts directly into interleaved Float32 PCM. The file streams
    /// through the converter one 64k-frame block at a time, so memory is the output (bounded by `maximumFrames` ×
    /// `channels`) plus one block, whatever the file's own rate and channel count.
    /// Float32 PCM is host I/O, never learned parameters or a model precision.
    public static func audio(
        _ url: URL, sampleRate: Int, channels: Int = 1, mix: ChannelMix = .convert, maximumFrames: Int = 48_000 * 600
    ) throws -> Audio {
        guard (8000...192000).contains(sampleRate), (1...8).contains(channels), maximumFrames > 0 else { throw mediaError("Invalid PCM format.") }
        let file = try openAudio(url)
        let format = file.processingFormat, sourceChannels = Int(format.channelCount)
        guard (1...maximumSourceChannels).contains(sourceChannels) else {
            throw mediaError("Audio has \(sourceChannels) channels; at most \(maximumSourceChannels) are read.")
        }
        let frames = Double(file.length) * Double(sampleRate) / format.sampleRate
        guard frames.isFinite, frames <= Double(maximumFrames), frames >= 1 else { throw mediaError("Audio exceeds the decode bound or cannot be converted.") }
        if case .channel(let index) = mix, !(0..<sourceChannels).contains(index) { throw mediaError("Audio has no channel \(index).") }
        let block: AVAudioFrameCount = 65_536
        var mixedFormat = format
        if case .convert = mix {
        } else {
            guard let mono = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: format.sampleRate, channels: 1, interleaved: false) else {
                throw mediaError("Audio exceeds the decode bound or cannot be converted.")
            }
            mixedFormat = mono
        }
        guard let target = AVAudioFormat(commonFormat: .pcmFormatFloat32, sampleRate: Double(sampleRate), channels: AVAudioChannelCount(channels), interleaved: false),
            let converter = AVAudioConverter(from: mixedFormat, to: target),
            let raw = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: block),
            let mixed = AVAudioPCMBuffer(pcmFormat: mixedFormat, frameCapacity: block),
            let output = AVAudioPCMBuffer(pcmFormat: target, frameCapacity: block)
        else { throw mediaError("Audio exceeds the decode bound or cannot be converted.") }
        // Fewer output channels than input: average the channels rather than let the converter keep only the first ones.
        if target.channelCount < mixedFormat.channelCount { converter.downmix = true }

        var read: AVAudioFramePosition = 0, ended = false, readFailure: Error?
        // The next source block, mixed; nil at end of data. A read that throws (end of file, a corrupt tail) ends the
        // data, and checkEnd below decides whether what was read is the whole file.
        func next() -> AVAudioPCMBuffer? {
            guard !ended else { return nil }
            do { try file.read(into: raw, frameCount: block) } catch { ended = true; readFailure = error; return nil }
            guard raw.frameLength > 0, let source = raw.floatChannelData else { ended = true; return nil }
            read += AVAudioFramePosition(raw.frameLength)
            let count = Int(raw.frameLength)
            switch mix {
            case .convert: return raw
            case .channel(let index): mixed.floatChannelData![0].update(from: source[index], count: count)
            case .mean:
                let to = mixed.floatChannelData![0], scale = 1 / Float(sourceChannels)
                to.update(from: source[0], count: count)
                for channel in 1..<sourceChannels { for i in 0..<count { to[i] += source[channel][i] } }
                for i in 0..<count { to[i] *= scale }
            }
            mixed.frameLength = raw.frameLength
            return mixed
        }
        // The output bound is the caller's, not the estimate: a length that under-reports cannot let a long file through.
        let limit = maximumFrames + 1024
        var samples = [Float](); samples.reserveCapacity((Int(ceil(frames)) + 1024) * channels)
        while true {
            output.frameLength = 0
            var failure: NSError?
            let status = converter.convert(to: output, error: &failure) { _, state in
                guard let buffer = next() else { state.pointee = .endOfStream; return nil }
                state.pointee = .haveData; return buffer
            }
            if failure != nil || status == .error { throw mediaError("Audio could not be converted to \(sampleRate) Hz PCM.") }
            guard let data = output.floatChannelData else { throw mediaError("PCM conversion failed.") }
            let count = Int(output.frameLength)
            guard samples.count / channels + count <= limit else { throw mediaError("Audio exceeds the decode bound or cannot be converted.") }
            for frame in 0..<count {
                for channel in 0..<channels {
                    let value = data[channel][frame]
                    guard value.isFinite else { throw mediaError("Decoded audio contains non-finite samples.") }
                    samples.append(value)
                }
            }
            if status == .endOfStream || (count == 0 && ended) { break }
        }
        try checkEnd(read: read, length: file.length, failure: readFailure)
        return Audio(samples: samples, sampleRate: sampleRate, channels: channels)
    }
    public static func image(_ url: URL, maxPixels: Int = 16_000_000) throws -> CGImage {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil), let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = properties[kCGImagePropertyPixelWidth] as? Int, let height = properties[kCGImagePropertyPixelHeight] as? Int,
            width > 0, height > 0, width <= maxPixels / height,
            let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary)
        else { throw mediaError("Image is invalid or exceeds the pixel bound.") }
        return image
    }
    public static func file(_ url: URL, root: URL, mime: String, name: String? = nil, id: String = UUID().uuidString) throws -> JSONObject {
        guard !id.isEmpty, id.count <= 128 else { throw mediaError("File IDs must contain 1...128 characters.") }
        let confined = try NativeFiles.confined(url.path, root: root)
        let size = try NativeFiles.bytes(confined)
        guard size <= NativeFiles.maxBytes else { throw mediaError("File exceeds the 4 GiB per-job artifact limit.") }
        var result: JSONObject = [
            "id": .string(id), "path": .string(confined.path), "name": .string(name ?? confined.lastPathComponent), "mime": .string(mime),
            "bytes": .number(Double(size)), "sha256": .string(try NativeFiles.sha256(confined))
        ]
        if mime.hasPrefix("image/") {
            let image = try self.image(confined)
            result["width"] = .number(Double(image.width)); result["height"] = .number(Double(image.height))
        }
        return result
    }
    static func mediaError(_ message: String) -> WorkerFailure { .init(code: "invalid_request", message: message) }
}

/// Video frame I/O keeps one decoded frame live at a time. Timestamp/transform are explicit to prevent orientation and timing drift.
public final class NativeVideoReader {
    public let reader: AVAssetReader
    private let output: AVAssetReaderTrackOutput
    public let transform: CGAffineTransform
    /// A file AVFoundation cannot open or decode is the caller's mistake: invalid_request (HTTP 422), not `runner` (503).
    public init(url: URL) async throws {
        let asset = AVURLAsset(url: url)
        let track: AVAssetTrack
        do {
            guard let found = try await asset.loadTracks(withMediaType: .video).first else { throw NativeMedia.mediaError("Video has no video track.") }
            track = found
            transform = try await track.load(.preferredTransform)
            reader = try AVAssetReader(asset: asset)
        } catch let failure as WorkerFailure { throw failure } catch {
            throw NativeMedia.mediaError("Video could not be decoded: the file is not a supported video format or is corrupt.")
        }
        output = AVAssetReaderTrackOutput(track: track, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw NativeMedia.mediaError("Cannot decode video frames.") }
        reader.add(output)
        guard reader.startReading() else { throw NativeMedia.mediaError("Video could not be decoded: \(reader.error?.localizedDescription ?? "the reader did not start").") }
    }
    public func next() throws -> CMSampleBuffer? {
        let sample = output.copyNextSampleBuffer()
        if sample == nil, reader.status == .failed { throw NativeMedia.mediaError("Video decoding failed: \(reader.error?.localizedDescription ?? "unreadable frame").") }
        return sample
    }
    public func cancel() { reader.cancelReading() }
    deinit { reader.cancelReading() }
}

public final class NativeVideoWriter {
    private let writer: AVAssetWriter
    private let input: AVAssetWriterInput
    private let adaptor: AVAssetWriterInputPixelBufferAdaptor
    public init(url: URL, width: Int, height: Int, transform: CGAffineTransform = .identity) throws {
        guard (1...8192).contains(width), (1...8192).contains(height) else { throw NativeMedia.mediaError("Invalid video size.") }
        writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height])
        input.transform = transform
        adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input, sourcePixelBufferAttributes: [kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32BGRA])
        guard writer.canAdd(input) else { throw NativeMedia.mediaError("Cannot encode video.") }
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? NativeMedia.mediaError("Cannot start video writer.") }
        writer.startSession(atSourceTime: .zero)
    }
    public func append(_ frame: CVPixelBuffer, time: CMTime, cancellation: NativeCancellation) throws {
        let deadline = ProcessInfo.processInfo.systemUptime + 30
        while !input.isReadyForMoreMediaData {
            try cancellation.check()
            guard writer.status == .writing, ProcessInfo.processInfo.systemUptime < deadline else { throw writer.error ?? NativeMedia.mediaError("Video writer stalled.") }
            Thread.sleep(forTimeInterval: 0.01)
        }
        guard adaptor.append(frame, withPresentationTime: time) else { throw writer.error ?? NativeMedia.mediaError("Video frame encoding failed.") }
    }
    public func finish() async throws {
        input.markAsFinished(); await writer.finishWriting()
        guard writer.status == .completed else { throw writer.error ?? NativeMedia.mediaError("Video encoding failed.") }
    }
    public func cancel() { writer.cancelWriting() }
}
