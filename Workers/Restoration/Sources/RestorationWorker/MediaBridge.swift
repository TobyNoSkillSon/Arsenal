// SPDX-License-Identifier: MIT
import Foundation
import IntegrationKit
import MLX
import CoreGraphics
import CoreMedia
import CoreVideo
import AVFoundation
import ImageIO
import UniformTypeIdentifiers
import CoreImage
import ArsenalWire

enum MediaBridge {
    static func error(_ message: String) -> WorkerFailure { .init(code: "invalid_request", message: message) }
    static func path(_ value: JSONValue?) throws -> URL {
        guard let path = value?.object?["path"]?.string, !path.isEmpty else { throw error("A media file path is required.") }
        return URL(fileURLWithPath: path)
    }
    static func output(_ input: JSONObject) throws -> URL {
        guard let path = input["out_dir"]?.string else { throw error("The host must supply out_dir.") }
        let url = URL(fileURLWithPath: path)
        try FileManager.default.createDirectory(at: url, withIntermediateDirectories: true)
        return url
    }
    static func tensor(_ image: CGImage) throws -> MLXArray {
        var pixels = [UInt8](repeating: 0, count: image.width * image.height * 4)
        let ok = pixels.withUnsafeMutableBytes { bytes -> Bool in
            guard
                let context = CGContext(
                    data: bytes.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height)); return true
        }
        guard ok else { throw error("Cannot decode RGB pixels.") }
        var rgb = [Float](); rgb.reserveCapacity(image.width * image.height * 3)
        for i in stride(from: 0, to: pixels.count, by: 4) { for c in 0..<3 { rgb.append(Float(pixels[i + c]) / 255) } }
        return MLXArray(rgb).reshaped(1, image.height, image.width, 3).asType(.float16)
    }
    static func lanczos(_ tensor: MLXArray, width: Int, height: Int) throws -> MLXArray {
        let p = try pixels(tensor)
        let source = CIImage(
            bitmapData: Data(p.bytes), bytesPerRow: p.width * 4, size: CGSize(width: p.width, height: p.height), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
        let scaled = source.applyingFilter(
            "CILanczosScaleTransform",
            parameters: [kCIInputScaleKey: Double(height) / Double(p.height), kCIInputAspectRatioKey: (Double(width) / Double(p.width)) / (Double(height) / Double(p.height))])
        guard let image = CIContext(options: [.cacheIntermediates: false]).createCGImage(scaled, from: CGRect(x: 0, y: 0, width: width, height: height)) else {
            throw error("Lanczos image preparation failed.")
        }
        return try self.tensor(image)
    }
    static func tensor(_ sample: CMSampleBuffer) throws -> MLXArray {
        guard let buffer = CMSampleBufferGetImageBuffer(sample) else { throw error("Missing video frame pixels.") }
        CVPixelBufferLockBaseAddress(buffer, .readOnly); defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        guard let address = CVPixelBufferGetBaseAddress(buffer) else { throw error("Missing pixel buffer storage.") }
        let width = CVPixelBufferGetWidth(buffer), height = CVPixelBufferGetHeight(buffer), row = CVPixelBufferGetBytesPerRow(buffer)
        guard width > 0, height > 0, width <= 16_000_000 / height else { throw error("Video frame exceeds 16 megapixels.") }
        let bytes = address.assumingMemoryBound(to: UInt8.self)
        var rgb = [Float](); rgb.reserveCapacity(width * height * 3)
        for y in 0..<height {
            for x in 0..<width { let p = y * row + x * 4; rgb.append(Float(bytes[p + 2]) / 255); rgb.append(Float(bytes[p + 1]) / 255); rgb.append(Float(bytes[p]) / 255) }
        }
        return MLXArray(rgb).reshaped(1, height, width, 3).asType(.float16)
    }
    static func pixels(_ tensor: MLXArray) throws -> (bytes: [UInt8], width: Int, height: Int) {
        guard tensor.ndim == 4, tensor.dim(0) == 1, tensor.dim(3) == 3,
            !MLX.any(.!MLX.isFinite(tensor)).item(Bool.self)
        else { throw error("Model returned invalid image pixels.") }
        // Round to the nearest 8-bit level (a plain cast truncates, darkening every channel by half a level on average).
        let rgb = MLX.round(MLX.clip(tensor.asType(.float32), min: 0, max: 1) * 255).asType(.uint8).asArray(UInt8.self)
        var rgba = [UInt8](); rgba.reserveCapacity(tensor.dim(1) * tensor.dim(2) * 4)
        for i in stride(from: 0, to: rgb.count, by: 3) { rgba.append(contentsOf: [rgb[i], rgb[i + 1], rgb[i + 2], 255]) }
        return (rgba, tensor.dim(2), tensor.dim(1))
    }
    static func png(_ tensor: MLXArray, at url: URL) throws {
        let p = try pixels(tensor), data = Data(p.bytes)
        guard let provider = CGDataProvider(data: data as CFData),
            let image = CGImage(
                width: p.width, height: p.height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: p.width * 4, space: CGColorSpaceCreateDeviceRGB(),
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.premultipliedLast.rawValue), provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
            let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw error("Cannot encode PNG.") }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw error("PNG encoding failed.") }
    }
    static func buffer(_ tensor: MLXArray) throws -> CVPixelBuffer {
        let p = try pixels(tensor); var buffer: CVPixelBuffer?
        guard CVPixelBufferCreate(nil, p.width, p.height, kCVPixelFormatType_32BGRA, nil, &buffer) == kCVReturnSuccess, let buffer else {
            throw error("Cannot allocate output video frame.")
        }
        CVPixelBufferLockBaseAddress(buffer, []); defer { CVPixelBufferUnlockBaseAddress(buffer, []) }
        let dst = CVPixelBufferGetBaseAddress(buffer)!.assumingMemoryBound(to: UInt8.self), row = CVPixelBufferGetBytesPerRow(buffer)
        for y in 0..<p.height {
            for x in 0..<p.width {
                let s = (y * p.width + x) * 4, d = y * row + x * 4; dst[d] = p.bytes[s + 2]; dst[d + 1] = p.bytes[s + 1]; dst[d + 2] = p.bytes[s]; dst[d + 3] = 255
            }
        }
        return buffer
    }
    // NativeModel executes on a serial utility queue; AVFoundation's async operations run on Swift's pool.
    static func wait<T>(_ body: @escaping () async throws -> T) throws -> T {
        let semaphore = DispatchSemaphore(value: 0)
        var result: Result<T, Error>!
        Task {
            do { result = .success(try await body()) } catch { result = .failure(error) }; semaphore.signal()
        }
        semaphore.wait(); return try result.get()
    }
    /// Frame progress as the service reads it (Activity, tool windows, job records): step/total frames under `label`,
    /// plus the original `frames` count. `total` is an estimate from the container; it never falls below `done`.
    static func progress(_ done: Int, total: Int? = nil, label: String = "Frames", emit: (WireFrame) throws -> Void) throws {
        var event: JSONObject = ["event": "progress", "frames": .number(Double(done)), "step": .number(Double(done)), "label": .string(label), "unit": "frames"]
        if let total, total > 0 { event["total"] = .number(Double(Swift.max(total, done))) }
        try emit(.json(event))
    }
    /// Frame 0 of the clip, sent once the video opens: Activity and the tool windows count the request as running from here,
    /// not from the first finished batch (a short clip is one batch, so it read as waiting for the whole job).
    static func begin(total: Int?, label: String, emit: (WireFrame) throws -> Void) throws { try progress(0, total: total, label: label, emit: emit) }
    /// One pass's stage (image restoration has no frames to count): step of total under `label`.
    static func stage(_ step: Int, of total: Int, _ label: String, emit: (WireFrame) throws -> Void) throws {
        try emit(.json(["event": "progress", "step": .number(Double(step)), "total": .number(Double(total)), "label": .string(label)]))
    }
    /// Frames a video holds (or would hold at `fps`), from its duration and nominal rate; nil when unknown.
    static func frameEstimate(_ url: URL, fps: Double? = nil) -> Int? {
        try? wait { () async throws -> Int? in
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { return nil }
            let seconds = try await asset.load(.duration).seconds
            var rate = fps ?? 0
            if fps == nil { rate = Double(try await track.load(.nominalFrameRate)) }
            guard seconds.isFinite, seconds > 0, rate.isFinite, rate > 0 else { return nil }
            return Int((seconds * rate).rounded(.up))
        }
    }

    /// Copies the source's audio onto a rendered MP4 whose timeline starts at the source's first video frame (`origin` seconds).
    /// Returns false and leaves the video untouched when the source has no audio inside the rendered span.
    @discardableResult
    static func keepAudio(from source: URL, video: URL, origin: Double, cancellation: NativeCancellation) throws -> Bool {
        try cancellation.check()
        guard origin.isFinite else { throw error("Invalid source video timestamps.") }
        let muxed = video.deletingLastPathComponent().appendingPathComponent(".audio-" + UUID().uuidString + ".mp4")
        defer { try? FileManager.default.removeItem(at: muxed) }
        let composed = try wait { () async throws -> Bool in
            // An AVAssetTrack holds its asset weakly; a composition built from tracks of a released asset fails export
            // with -11800/-12780, so both assets live until the export has finished.
            let input = AVURLAsset(url: source), rendered = AVURLAsset(url: video)
            defer { withExtendedLifetime((input, rendered)) {} }
            let sounds = try await input.loadTracks(withMediaType: .audio)
            guard !sounds.isEmpty else { return false }
            guard let frames = try await rendered.loadTracks(withMediaType: .video).first else { throw MediaBridge.error("Rendered video has no video track.") }
            // The video track's own range, not the asset duration: inserting past the track's end fails the export.
            let span = try await frames.load(.timeRange)
            guard span.duration > .zero else { throw MediaBridge.error("Rendered video has no frames.") }
            let composition = AVMutableComposition()
            guard let picture = composition.addMutableTrack(withMediaType: .video, preferredTrackID: kCMPersistentTrackID_Invalid) else {
                throw MediaBridge.error("Cannot compose the output video.")
            }
            try picture.insertTimeRange(span, of: frames, at: span.start)
            picture.preferredTransform = try await frames.load(.preferredTransform)
            let start = CMTime(seconds: origin, preferredTimescale: 60000)
            let window = CMTimeRange(start: start + span.start, duration: span.duration)
            var kept = 0
            for sound in sounds {
                let overlap = try await sound.load(.timeRange).intersection(window)
                guard overlap.duration > .zero, let track = composition.addMutableTrack(withMediaType: .audio, preferredTrackID: kCMPersistentTrackID_Invalid) else {
                    continue
                }
                try track.insertTimeRange(overlap, of: sound, at: overlap.start - start)
                kept += 1
            }
            guard kept > 0 else { return false }
            // Passthrough keeps encoded video and audio; an audio codec MP4 cannot carry falls back to re-encoding.
            // When both fail, the error reports both, since the passthrough failure is usually the informative one.
            var failures: [String] = []
            for (name, preset) in [("passthrough", AVAssetExportPresetPassthrough), ("re-encode", AVAssetExportPresetHighestQuality)] {
                try? FileManager.default.removeItem(at: muxed)
                guard let session = AVAssetExportSession(asset: composition, presetName: preset) else {
                    failures.append(name + ": export preset unavailable"); continue
                }
                do {
                    try await session.export(to: muxed, as: .mp4)
                    return true
                } catch {
                    failures.append(name + ": " + error.localizedDescription)
                }
            }
            throw WorkerFailure(
                code: "invalid_output", message: "Cannot keep the source audio in the MP4 output (" + failures.joined(separator: "; ") + ").")
        }
        try cancellation.check()
        guard composed else { return false }
        try FileManager.default.removeItem(at: video)
        try FileManager.default.moveItem(at: muxed, to: video)
        return true
    }
}
