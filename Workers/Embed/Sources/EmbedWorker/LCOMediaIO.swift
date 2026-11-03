// SPDX-License-Identifier: MIT
// LCO-Embedding Omni's media front end, matching the makers' readers rather than CoreImage defaults:
// - images as PIL reads them: raw stored RGB values (no colour management), RGBA composited onto white (qwen_omni_utils
//   to_rgb), resized with PIL's antialiased bicubic (a = -0.5, support widened when shrinking, 8-bit between passes);
// - video as FFmpeg decodes it: Y'CbCr converted with the track's tagged matrix and range, BT.601 when untagged
//   (FFmpeg's default), chroma replicated per 2×2 block; each sampled frame is resized as it is decoded, so only the
//   small frames are kept;
// - audio mono as librosa's mean of the channels.
import AVFoundation
import CoreGraphics
import Foundation
import ArsenalWire
import IntegrationKit

/// An 8-bit RGB image, interleaved, row-major.
struct LCORGB: Equatable {
    let width, height: Int
    var bytes: [UInt8]
}

enum LCOImageIO {
    /// The stored values of an image file, alpha composited onto white.
    static func load(_ url: URL, maxPixels: Int = 64_000_000) throws -> LCORGB { try rgb(NativeMedia.image(url, maxPixels: maxPixels)) }

    static func rgb(_ image: CGImage) throws -> LCORGB {
        let width = image.width, height = image.height
        // An RGB source renders in its own colour space, so values pass through unconverted (PIL ignores ICC
        // profiles); other models (grey, CMYK, indexed) are converted to sRGB.
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgba = [UInt8](repeating: 255, count: width * height * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { return false }
            context.setFillColor(red: 1, green: 1, blue: 1, alpha: 1)
            context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard drawn else { throw WorkerFailure(code: "invalid_request", message: "Image could not be decoded.") }
        var out = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) { out[i * 3] = rgba[i * 4]; out[i * 3 + 1] = rgba[i * 4 + 1]; out[i * 3 + 2] = rgba[i * 4 + 2] }
        return LCORGB(width: width, height: height, bytes: out)
    }
}

enum LCOResample {
    private static func cubic(_ x: Double) -> Double {
        let a = -0.5, x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }
    /// PIL precompute_coeffs for one axis: first input index and normalized weights per output position.
    static func coefficients(input: Int, output: Int) -> [(start: Int, weights: [Double])] {
        let scale = Double(input) / Double(output), filterScale = max(scale, 1), support = 2 * filterScale
        return (0..<output).map { x in
            let center = (Double(x) + 0.5) * scale
            let low = max(Int(center - support + 0.5), 0), high = min(Int(center + support + 0.5), input)
            var weights = (low..<high).map { cubic((Double($0) - center + 0.5) / filterScale) }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            return (low, weights)
        }
    }
    private static func clamp8(_ v: Double) -> UInt8 { UInt8(max(0, min(255, v.rounded(.toNearestOrAwayFromZero)))) }

    /// PIL Image.resize(BICUBIC): horizontal pass, 8-bit, then vertical pass; a same-size axis is copied.
    static func bicubic(_ image: LCORGB, width: Int, height: Int) -> LCORGB {
        var current = image
        if width != image.width {
            let coeffs = coefficients(input: image.width, output: width)
            var out = [UInt8](repeating: 0, count: width * image.height * 3)
            image.bytes.withUnsafeBufferPointer { src in
                for y in 0..<image.height {
                    let row = y * image.width * 3
                    for x in 0..<width {
                        let (start, weights) = coeffs[x]
                        var r = 0.0, g = 0.0, b = 0.0
                        for (k, w) in weights.enumerated() {
                            let p = row + (start + k) * 3
                            r += w * Double(src[p]); g += w * Double(src[p + 1]); b += w * Double(src[p + 2])
                        }
                        let o = (y * width + x) * 3
                        out[o] = clamp8(r); out[o + 1] = clamp8(g); out[o + 2] = clamp8(b)
                    }
                }
            }
            current = LCORGB(width: width, height: image.height, bytes: out)
        }
        if height != current.height {
            let coeffs = coefficients(input: current.height, output: height)
            var out = [UInt8](repeating: 0, count: width * height * 3)
            let stride = width * 3
            current.bytes.withUnsafeBufferPointer { src in
                for y in 0..<height {
                    let (start, weights) = coeffs[y]
                    for i in 0..<stride {
                        var v = 0.0
                        for (k, w) in weights.enumerated() { v += w * Double(src[(start + k) * stride + i]) }
                        out[y * stride + i] = clamp8(v)
                    }
                }
            }
            current = LCORGB(width: width, height: height, bytes: out)
        }
        return current
    }

    /// Clockwise quarter turns, for a video track's display transform.
    static func rotate(_ image: LCORGB, quarterTurns: Int) -> LCORGB {
        let turns = ((quarterTurns % 4) + 4) % 4
        guard turns != 0 else { return image }
        let (w, h) = (image.width, image.height)
        let (ow, oh) = turns == 2 ? (w, h) : (h, w)
        var out = [UInt8](repeating: 0, count: image.bytes.count)
        for y in 0..<h {
            for x in 0..<w {
                let (nx, ny): (Int, Int) = turns == 1 ? (h - 1 - y, x) : turns == 2 ? (w - 1 - x, h - 1 - y) : (y, w - 1 - x)
                for c in 0..<3 { out[(ny * ow + nx) * 3 + c] = image.bytes[(y * w + x) * 3 + c] }
            }
        }
        return LCORGB(width: ow, height: oh, bytes: out)
    }
}

enum LCOPatches {
    static let mean: [Float] = [0.48145466, 0.4578275, 0.40821073], std: [Float] = [0.26862954, 0.26130258, 0.27577711]
    /// Rescale, CLIP-normalize and patchify same-size frames (an odd count repeats the last frame) into the
    /// processor's [patches, 3·2·14·14] layout: rows (time, block row, block column, merge row, merge column),
    /// columns (channel, frame in pair, pixel row, pixel column).
    static func make(_ frames: [LCORGB], patch: Int = 14, merge: Int = 2, temporal: Int = 2) -> ([Float], LCOGrid) {
        precondition(!frames.isEmpty)
        var frames = frames
        while frames.count % temporal != 0 { frames.append(frames[frames.count - 1]) }
        let w = frames[0].width, h = frames[0].height
        let gt = frames.count / temporal, gh = h / patch, gw = w / patch
        let columns = 3 * temporal * patch * patch
        var out = [Float](repeating: 0, count: gt * gh * gw * columns)
        var row = 0
        for t in 0..<gt {
            for bh in 0..<(gh / merge) {
                for bw in 0..<(gw / merge) {
                    for mh in 0..<merge {
                        for mw in 0..<merge {
                            let py = (bh * merge + mh) * patch, px = (bw * merge + mw) * patch
                            var column = row * columns
                            for c in 0..<3 {
                                for f in 0..<temporal {
                                    let bytes = frames[t * temporal + f].bytes
                                    for y in 0..<patch {
                                        for x in 0..<patch {
                                            out[column] = (Float(bytes[((py + y) * w + px + x) * 3 + c]) / 255 - mean[c]) / std[c]
                                            column += 1
                                        }
                                    }
                                }
                            }
                            row += 1
                        }
                    }
                }
            }
        }
        return (out, LCOGrid(t: gt, h: gh, w: gw))
    }
}

enum LCOYCbCr {
    /// (Kr, Kb) of the track's tagged matrix; BT.601 when untagged, as FFmpeg's conversion assumes.
    static func coefficients(_ matrix: String?) -> (Double, Double) {
        switch matrix {
        case String(kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2): return (0.2126, 0.0722)
        case String(kCMFormatDescriptionYCbCrMatrix_ITU_R_2020): return (0.2627, 0.0593)
        case String(kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995): return (0.212, 0.087)
        default: return (0.299, 0.114)
        }
    }
    /// One bi-planar 4:2:0 frame to RGB, chroma replicated over each 2×2 block.
    static func rgb(_ buffer: CVPixelBuffer, matrix: String?, fullRange: Bool) -> LCORGB? {
        guard CVPixelBufferGetPlaneCount(buffer) == 2 else { return nil }
        CVPixelBufferLockBaseAddress(buffer, .readOnly)
        defer { CVPixelBufferUnlockBaseAddress(buffer, .readOnly) }
        let width = CVPixelBufferGetWidthOfPlane(buffer, 0), height = CVPixelBufferGetHeightOfPlane(buffer, 0)
        guard let yBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 0), let cBase = CVPixelBufferGetBaseAddressOfPlane(buffer, 1) else { return nil }
        let yStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 0), cStride = CVPixelBufferGetBytesPerRowOfPlane(buffer, 1)
        let yp = yBase.assumingMemoryBound(to: UInt8.self), cp = cBase.assumingMemoryBound(to: UInt8.self)
        let (kr, kb) = coefficients(matrix), kg = 1 - kr - kb
        let (yOffset, yScale, cScale) = fullRange ? (0.0, 1 / 255.0, 1 / 255.0) : (16.0, 1 / 219.0, 1 / 224.0)
        var out = [UInt8](repeating: 0, count: width * height * 3)
        // FFmpeg's table conversion truncates (fixed-point shifts); rounding sat 1 to 2 levels high on its output.
        func clamp(_ v: Double) -> UInt8 { UInt8(max(0, min(255, (v * 255).rounded(.down)))) }
        for y in 0..<height {
            for x in 0..<width {
                let luma = (Double(yp[y * yStride + x]) - yOffset) * yScale
                let c = (y / 2) * cStride + (x / 2) * 2
                let pb = (Double(cp[c]) - 128) * cScale, pr = (Double(cp[c + 1]) - 128) * cScale
                let r = luma + 2 * (1 - kr) * pr, b = luma + 2 * (1 - kb) * pb
                let g = (luma - kr * r - kb * b) / kg
                let o = (y * width + x) * 3
                out[o] = clamp(r); out[o + 1] = clamp(g); out[o + 2] = clamp(b)
            }
        }
        return LCORGB(width: width, height: height, bytes: out)
    }
}

enum LCOVideoIO {
    struct Track {
        let frames: Int
        let fps: Double
        let width, height: Int // after the display rotation
        let quarterTurns: Int
        let matrix: String?
        let fullRange: Bool
    }
    static func track(_ url: URL, cancellation: NativeCancellation) throws -> Track {
        let (frames, fps) = try lcoAwait { try await OvisVideoMetadata.read(url: url, cancellation: cancellation) }
        return try lcoAwait {
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else {
                throw WorkerFailure(code: "invalid_request", message: "Video has no video track.")
            }
            let size = try await track.load(.naturalSize), transform = try await track.load(.preferredTransform)
            let description = try await track.load(.formatDescriptions).first
            let matrix = description.flatMap { CMFormatDescriptionGetExtension($0, extensionKey: kCMFormatDescriptionExtension_YCbCrMatrix) as? String }
            let full = description.flatMap { CMFormatDescriptionGetExtension($0, extensionKey: kCMFormatDescriptionExtension_FullRangeVideo) as? Bool } ?? false
            let turns = Int((atan2(Double(transform.b), Double(transform.a)) / (Double.pi / 2)).rounded())
            let (w, h) = (Int(size.width.rounded()), Int(size.height.rounded()))
            return Track(frames: frames, fps: fps, width: turns % 2 == 0 ? w : h, height: turns % 2 == 0 ? h : w, quarterTurns: turns, matrix: matrix, fullRange: full)
        }
    }
    /// Decode the frames at `indices` (sorted), resizing each to `width`×`height` as it arrives.
    static func frames(_ url: URL, track: Track, indices: [Int], width: Int, height: Int, cancellation: NativeCancellation) throws -> [LCORGB] {
        let asset = AVURLAsset(url: url)
        let videoTrack = try lcoAwait { try await asset.loadTracks(withMediaType: .video).first }
        guard let videoTrack else { throw WorkerFailure(code: "invalid_request", message: "Video has no video track.") }
        let reader = try AVAssetReader(asset: asset)
        let format = track.fullRange ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: format])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw WorkerFailure(code: "invalid_request", message: "Cannot decode video frames.") }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? WorkerFailure(code: "invalid_request", message: "Cannot start video reader.") }
        defer { reader.cancelReading() }
        var result: [LCORGB] = [], index = 0, cursor = 0
        while cursor < indices.count, let sample = output.copyNextSampleBuffer() {
            try cancellation.check()
            guard CMSampleBufferGetNumSamples(sample) > 0, let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            if indices[cursor] == index {
                guard let decoded = LCOYCbCr.rgb(pixels, matrix: track.matrix, fullRange: track.fullRange) else {
                    throw WorkerFailure(code: "invalid_request", message: "Unsupported decoded video layout.")
                }
                let frame = LCOResample.bicubic(LCOResample.rotate(decoded, quarterTurns: track.quarterTurns), width: width, height: height)
                while cursor < indices.count, indices[cursor] == index { result.append(frame); cursor += 1 }
            }
            index += 1
        }
        guard reader.status != .failed, result.count == indices.count else { throw WorkerFailure(code: "invalid_request", message: "Video contains no readable sampled frames.") }
        return result
    }
}

enum LCOAudioIO {
    /// 16 kHz mono as librosa.load(sr=16000, mono=True): every channel resampled, then their mean.
    static func mono(_ url: URL, maximumSamples: Int) throws -> [Float] {
        let channels = Int(try NativeMedia.openAudio(url).processingFormat.channelCount)
        guard (1...8).contains(channels) else { throw WorkerFailure(code: "invalid_request", message: "Audio has \(channels) channels; at most 8 are read.") }
        if channels > 2 { return try SurroundAudio.mono(url, sampleRate: LCOMel.sampleRate, maximumSamples: maximumSamples) }
        let audio = try NativeMedia.audio(url, sampleRate: LCOMel.sampleRate, channels: channels, maximumFrames: maximumSamples)
        guard channels > 1 else { return audio.samples }
        return stride(from: 0, to: audio.samples.count, by: channels).map { start in
            audio.samples[start..<(start + channels)].reduce(0, +) / Float(channels)
        }
    }
}

func lcoAwait<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<T, Error>?
    Task {
        do { result = .success(try await operation()) } catch { result = .failure(error) }
        semaphore.signal()
    }
    semaphore.wait()
    return try result!.get()
}
