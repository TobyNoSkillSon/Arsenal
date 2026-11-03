// SPDX-License-Identifier: MIT
// EmbeddingGemma 2's media rules as its processor (Transformers 5.22.1) applies them: pictures decoded as PIL reads
// them (EXIF orientation applied, convert("RGB"), no colour management), Gemma4ImageProcessor / EmbeddingGemma2-
// VideoProcessor aspect-preserving resize with torchvision's antialiased uint8 bicubic, rescale to [0, 1] and 16-pixel
// patches with (x, y) positions; video sampled at 1 frame a second, at most 32 frames.
// The picture decoder and the resampler are copied from Workers/D1/Sources/D1Engine/Pictures.swift (Arsenal
// contributors, MIT), where both are checked bit-exact against Pillow and torch. The video reader converts Y'CbCr as
// FFmpeg's default rgb24 conversion does (tagged matrix and range, BT.601 when untagged, chroma per 2×2 block), the
// method of the LCO-Embedding worker's media reader (Arsenal contributors, MIT).
import ArsenalWire
import AVFoundation
import CoreGraphics
import Foundation
import ImageIO
import IntegrationKit
import MLX

/// An 8-bit RGB picture, rows top to bottom, channels interleaved.
struct EG2Picture: Equatable {
    let width: Int
    let height: Int
    var pixels: [UInt8]

    static func failure(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_request", message: message) }

    /// Pillow's `Image.open` + `ImageOps.exif_transpose` + `convert("RGB")`: decoder bytes without colour conversion.
    static func load(_ url: URL, maxPixels: Int = 64_000_000) throws -> EG2Picture {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
            CGImageSourceGetCount(source) > 0,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let w = properties[kCGImagePropertyPixelWidth] as? Int, let h = properties[kCGImagePropertyPixelHeight] as? Int
        else { throw failure("Cannot read the image at \(url.lastPathComponent).") }
        guard w > 0, h > 0, w <= maxPixels / h else { throw failure("The image is empty or larger than \(maxPixels / 1_000_000) megapixels.") }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw failure("Cannot decode the image at \(url.lastPathComponent).")
        }
        let decoded = try raw16(image) ?? raw(image) ?? drawn(image)
        return decoded.oriented((properties[kCGImagePropertyOrientation] as? NSNumber)?.intValue ?? 1)
    }

    /// EXIF orientation 1–8 applied as Pillow's `exif_transpose` does.
    func oriented(_ orientation: Int) -> EG2Picture {
        guard (2...8).contains(orientation) else { return self }
        let swap = orientation >= 5
        let ow = swap ? height : width, oh = swap ? width : height
        var out = [UInt8](repeating: 0, count: pixels.count)
        for y in 0..<oh {
            for x in 0..<ow {
                // Source coordinate of output (x, y).
                let (sx, sy): (Int, Int)
                switch orientation {
                case 2: (sx, sy) = (width - 1 - x, y)
                case 3: (sx, sy) = (width - 1 - x, height - 1 - y)
                case 4: (sx, sy) = (x, height - 1 - y)
                case 5: (sx, sy) = (y, x)
                case 6: (sx, sy) = (y, height - 1 - x)
                case 7: (sx, sy) = (width - 1 - y, height - 1 - x)
                default: (sx, sy) = (width - 1 - y, x)
                }
                let s = 3 * (sy * width + sx), d = 3 * (y * ow + x)
                out[d] = pixels[s]; out[d + 1] = pixels[s + 1]; out[d + 2] = pixels[s + 2]
            }
        }
        return EG2Picture(width: ow, height: oh, pixels: out)
    }

    private static func drawn(_ image: CGImage) throws -> EG2Picture {
        let w = image.width, h = image.height
        let space: CGColorSpace
        if let own = image.colorSpace, own.model == .rgb {
            space = own
        } else if let own = image.colorSpace, own.model == .indexed, let base = own.baseColorSpace, base.model == .rgb {
            space = base
        } else if let own = image.colorSpace, own.model == .monochrome {
            var g = [UInt8](repeating: 0, count: w * h)
            let ok = g.withUnsafeMutableBytes { buffer -> Bool in
                guard
                    let context = CGContext(
                        data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: own,
                        bitmapInfo: CGImageAlphaInfo.none.rawValue)
                else { return false }
                context.interpolationQuality = .none
                context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
                return true
            }
            guard ok else { throw failure("The image could not be converted to RGB.") }
            return EG2Picture(width: w, height: h, pixels: g.flatMap { [$0, $0, $0] })
        } else {
            space = CGColorSpace(name: CGColorSpace.sRGB)!
        }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let ok = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard ok else { throw failure("The image could not be converted to RGB.") }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) { rgb[3 * i] = rgba[4 * i]; rgb[3 * i + 1] = rgba[4 * i + 1]; rgb[3 * i + 2] = rgba[4 * i + 2] }
        return EG2Picture(width: w, height: h, pixels: rgb)
    }

    /// 8-bit RGB(A)/gray(A) bytes straight from the decoder, alpha dropped (Pillow's convert('RGB')).
    private static func raw(_ image: CGImage) -> EG2Picture? {
        guard image.bitsPerComponent == 8, let space = image.colorSpace, let data = image.dataProvider?.data else { return nil }
        let info = image.bitmapInfo
        guard info.intersection(.byteOrderMask) == [] || info.intersection(.byteOrderMask) == .byteOrder32Big, !info.contains(.floatComponents) else { return nil }
        let alpha = image.alphaInfo
        let channels: Int
        switch space.model {
        case .rgb: channels = 3
        case .monochrome: channels = 1
        default: return nil
        }
        let stride = channels + (alpha != .none ? 1 : 0)
        guard image.bitsPerPixel == stride * 8,
            alpha == .none || alpha == .last || alpha == .first || alpha == .noneSkipLast || alpha == .noneSkipFirst
        else { return nil }
        let offset = (alpha == .first || alpha == .noneSkipFirst) ? 1 : 0
        let w = image.width, h = image.height, row = image.bytesPerRow
        guard CFDataGetLength(data) >= row * (h - 1) + w * stride else { return nil }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        let bytes = CFDataGetBytePtr(data)!
        for y in 0..<h {
            for x in 0..<w {
                let p = y * row + x * stride + offset, o = 3 * (y * w + x)
                if channels == 3 {
                    rgb[o] = bytes[p]; rgb[o + 1] = bytes[p + 1]; rgb[o + 2] = bytes[p + 2]
                } else {
                    rgb[o] = bytes[p]; rgb[o + 1] = bytes[p]; rgb[o + 2] = bytes[p]
                }
            }
        }
        return EG2Picture(width: w, height: h, pixels: rgb)
    }

    /// PNG grayscale16 opens in Pillow's integer mode (clamped at 255); RGB16/RGBA16 take the high byte.
    private static func raw16(_ image: CGImage) -> EG2Picture? {
        guard image.bitsPerComponent == 16, !image.bitmapInfo.contains(.floatComponents), let space = image.colorSpace,
            let data = image.dataProvider?.data
        else { return nil }
        let channels: Int
        switch space.model {
        case .rgb: channels = 3
        case .monochrome: channels = 1
        default: return nil
        }
        let alpha = image.alphaInfo
        guard alpha == .none || alpha == .last || alpha == .first || alpha == .noneSkipLast || alpha == .noneSkipFirst else { return nil }
        let stride = (channels + (alpha == .none ? 0 : 1)) * 2
        guard image.bitsPerPixel == stride * 8 else { return nil }
        let offset = (alpha == .first || alpha == .noneSkipFirst) ? 2 : 0
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let little = order == .byteOrder16Little || order == .byteOrder32Little
        let w = image.width, h = image.height, row = image.bytesPerRow
        guard CFDataGetLength(data) >= row * (h - 1) + w * stride else { return nil }
        let bytes = CFDataGetBytePtr(data)!
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for y in 0..<h {
            for x in 0..<w {
                let p = y * row + x * stride + offset
                for c in 0..<3 {
                    let i = p + (channels == 1 ? 0 : c * 2)
                    let value = little ? Int(bytes[i]) | Int(bytes[i + 1]) << 8 : Int(bytes[i]) << 8 | Int(bytes[i + 1])
                    rgb[3 * (y * w + x) + c] = channels == 1 && alpha == .none ? UInt8(clamping: value) : UInt8(value >> 8)
                }
            }
        }
        return EG2Picture(width: w, height: h, pixels: rgb)
    }
}

/// torch's antialiased uint8 bicubic (`tvF.resize(..., antialias=True)` on uint8 tensors): a = -0.5, support scaled
/// when shrinking, int16 weights at the precision the pass's largest weight allows; width first, then height, each pass
/// rounded to 8 bits and skipped when its size is unchanged.
enum EG2Resample {
    static func cubic(_ x: Double) -> Double {
        let a = -0.5, x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    static func weights(_ input: Int, _ output: Int) -> [(Int, [Double])] {
        let scale = Double(input) / Double(output), filterScale = max(scale, 1), reach = 2 * filterScale
        return (0..<output).map { i in
            let center = scale * (Double(i) + 0.5)
            let lo = max(Int(center - reach + 0.5), 0), hi = min(Int(center + reach + 0.5), input)
            var w = (0..<(hi - lo)).map { cubic((Double($0 + lo) - center + 0.5) / filterScale) }
            let total = w.reduce(0, +)
            if total != 0 { w = w.map { $0 / total } }
            return (lo, w)
        }
    }

    static func resize(_ p: EG2Picture, width: Int, height: Int) -> EG2Picture {
        var out = p
        if width != p.width { out = pass(out, horizontal: true, size: width) }
        if height != p.height { out = pass(out, horizontal: false, size: height) }
        return out
    }

    private static func pass(_ p: EG2Picture, horizontal: Bool, size: Int) -> EG2Picture {
        let table = weights(horizontal ? p.width : p.height, size)
        let largest = table.flatMap(\.1).max() ?? 1
        var bits = 0
        while bits < 21 && Int(0.5 + largest * Double(1 << (bits + 1))) < (1 << 15) { bits += 1 }
        let fixed = table.map { lo, w in (lo, w.map { Int64(($0 * Double(1 << bits) + ($0 < 0 ? -0.5 : 0.5)).rounded(.towardZero)) }) }
        let w = horizontal ? size : p.width, h = horizontal ? p.height : size
        var out = [UInt8](repeating: 0, count: w * h * 3)
        let half = Int64(1) << (bits - 1)
        p.pixels.withUnsafeBufferPointer { src in
            out.withUnsafeMutableBufferPointer { dst in
                for y in 0..<h {
                    for x in 0..<w {
                        let (lo, k) = fixed[horizontal ? x : y]
                        for c in 0..<3 {
                            var acc = half
                            for (j, weight) in k.enumerated() {
                                let s = horizontal ? (y * p.width + lo + j) * 3 + c : ((lo + j) * p.width + x) * 3 + c
                                acc += weight * Int64(src[s])
                            }
                            dst[(y * w + x) * 3 + c] = UInt8(clamping: acc >> Int64(bits))
                        }
                    }
                }
            }
        }
        return EG2Picture(width: w, height: h, pixels: out)
    }
}

/// One picture as the vision tower reads it: its real patches only (padding patches never change real outputs, so they
/// are not built) and the number of soft tokens it becomes.
struct EG2Patches {
    /// `[patches, 16·16·3]` in the processor's (row, column, channel) order, values in [0, 1].
    let pixels: MLXArray
    /// Patch grid: `columns` × `rows`; positions are (x, y) = (column, row), row-major.
    let columns: Int
    let rows: Int
    var softTokens: Int { columns * rows / 9 }
}

enum EG2Vision {
    static let patch = 16, pooling = 3
    static let imageSoftTokens = 280, videoSoftTokens = 140
    static let supportedSoftTokens = [70, 140, 280, 560, 1120]

    /// `get_aspect_ratio_preserving_size`: the largest size within `maxSoftTokens · 9` patches whose sides are multiples
    /// of 48 px.
    static func targetSize(height: Int, width: Int, maxSoftTokens: Int) throws -> (height: Int, width: Int) {
        let maxPatches = maxSoftTokens * pooling * pooling
        let totalPx = Double(height * width), targetPx = Double(maxPatches * patch * patch)
        let factor = (targetPx / totalPx).squareRoot()
        let side = pooling * patch
        var th = Int(floor(factor * Double(height) / Double(side))) * side
        var tw = Int(floor(factor * Double(width) / Double(side))) * side
        guard th > 0 || tw > 0 else { throw EG2Picture.failure("The image is too small or too thin to resize.") }
        let maxSide = (maxPatches / (pooling * pooling)) * side
        if th == 0 {
            th = side; tw = min(Int(floor(Double(width) / Double(height))) * side, maxSide)
        } else if tw == 0 {
            tw = side; th = min(Int(floor(Double(height) / Double(width))) * side, maxSide)
        }
        guard Double(th * tw) <= targetPx else { throw EG2Picture.failure("The image's aspect ratio is too extreme for the patch budget.") }
        return (th, tw)
    }

    /// Resize, rescale by 1/255 and cut into 16-pixel patches as `convert_image_to_patches` does.
    static func patches(_ picture: EG2Picture, maxSoftTokens: Int) throws -> EG2Patches {
        let (h, w) = try targetSize(height: picture.height, width: picture.width, maxSoftTokens: maxSoftTokens)
        let resized = (h == picture.height && w == picture.width) ? picture : EG2Resample.resize(picture, width: w, height: h)
        let rows = h / patch, columns = w / patch
        let image = MLXArray(resized.pixels, [rows, patch, columns, patch, 3]).asType(.float32) * Float(1.0 / 255.0)
        // (rows, py, columns, px, c) → (rows, columns, py, px, c), row-major patches.
        let laid = image.transposed(0, 2, 1, 3, 4).reshaped(rows * columns, patch * patch * 3)
        return EG2Patches(pixels: laid, columns: columns, rows: rows)
    }

    /// EmbeddingGemma2VideoProcessor.sample_frames at fps 1 with the uniform 32-frame cap.
    /// Each candidate frame is a function of its position, so the cap picks positions before any list is built: a
    /// crafted near-zero rate would otherwise ask for billions of positions, or trap converting an infinite count.
    static func sampleFrames(total: Int, fps: Double, targetFPS: Double = 1, maxFrames: Int = 32) -> [Int] {
        guard total > 0 else { return [] }
        let n: Int, frame: (Int) -> Int
        if fps > 0, fps.isFinite {
            let step = fps / targetFPS, duration = Double(total) / fps
            n = Int(min(max(1, duration * targetFPS), Double(1 << 40)))
            frame = { Int(min(Double(total - 1), Double($0) * step)) }
        } else {
            n = total; frame = { $0 }
        }
        guard n > maxFrames else { return (0..<n).map(frame) }
        // np.linspace(0, n - 1, maxFrames, dtype=int): arange · step, the last point set to n - 1, truncated.
        let step = Double(n - 1) / Double(max(maxFrames - 1, 1))
        return (0..<maxFrames).map { i in frame(i == maxFrames - 1 ? n - 1 : Int(Double(i) * step)) }
    }
}

/// Video as the maker's PyAV reader sees it: frame count and average rate, sampled frames as RGB.
enum EG2VideoIO {
    struct Track {
        let frames: Int
        let fps: Double
        let quarterTurns: Int
        let matrix: String?
        let fullRange: Bool
    }

    static func track(_ url: URL, cancellation: NativeCancellation) throws -> Track {
        let (frames, fps) = try eg2Await { try await OvisVideoMetadata.read(url: url, cancellation: cancellation) }
        return try eg2Await {
            let asset = AVURLAsset(url: url)
            guard let track = try await asset.loadTracks(withMediaType: .video).first else { throw EG2Picture.failure("The video has no video track.") }
            let transform = try await track.load(.preferredTransform)
            let description = try await track.load(.formatDescriptions).first
            let matrix = description.flatMap { CMFormatDescriptionGetExtension($0, extensionKey: kCMFormatDescriptionExtension_YCbCrMatrix) as? String }
            let full = description.flatMap { CMFormatDescriptionGetExtension($0, extensionKey: kCMFormatDescriptionExtension_FullRangeVideo) as? Bool } ?? false
            let turns = Int((atan2(Double(transform.b), Double(transform.a)) / (Double.pi / 2)).rounded())
            return Track(frames: frames, fps: fps, quarterTurns: (turns % 4 + 4) % 4, matrix: matrix, fullRange: full)
        }
    }

    /// (Kr, Kb) of the tagged matrix; BT.601 when untagged, as FFmpeg assumes.
    static func coefficients(_ matrix: String?) -> (Double, Double) {
        switch matrix {
        case String(kCMFormatDescriptionYCbCrMatrix_ITU_R_709_2): return (0.2126, 0.0722)
        case String(kCMFormatDescriptionYCbCrMatrix_ITU_R_2020): return (0.2627, 0.0593)
        case String(kCMFormatDescriptionYCbCrMatrix_SMPTE_240M_1995): return (0.212, 0.087)
        default: return (0.299, 0.114)
        }
    }

    static func rgb(_ buffer: CVPixelBuffer, matrix: String?, fullRange: Bool) -> EG2Picture? {
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
        func clamp(_ v: Double) -> UInt8 { UInt8(max(0, min(255, (v * 255).rounded()))) }
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
        return EG2Picture(width: width, height: height, pixels: out)
    }

    /// Decode the frames at the sorted `indices`, display rotation applied, each passed through `map` as it arrives.
    static func frames<T>(_ url: URL, track: Track, indices: [Int], cancellation: NativeCancellation, map: (EG2Picture) throws -> T) throws -> [T] {
        let asset = AVURLAsset(url: url)
        guard let videoTrack = try eg2Await({ try await asset.loadTracks(withMediaType: .video).first }) else { throw EG2Picture.failure("The video has no video track.") }
        let reader = try AVAssetReader(asset: asset)
        let format = track.fullRange ? kCVPixelFormatType_420YpCbCr8BiPlanarFullRange : kCVPixelFormatType_420YpCbCr8BiPlanarVideoRange
        let output = AVAssetReaderTrackOutput(track: videoTrack, outputSettings: [kCVPixelBufferPixelFormatTypeKey as String: format])
        output.alwaysCopiesSampleData = false
        guard reader.canAdd(output) else { throw EG2Picture.failure("Cannot decode the video's frames.") }
        reader.add(output)
        guard reader.startReading() else { throw reader.error ?? EG2Picture.failure("Cannot start reading the video.") }
        defer { reader.cancelReading() }
        let orientation = [1, 6, 3, 8][track.quarterTurns]
        var result: [T] = [], index = 0, cursor = 0
        while cursor < indices.count, let sample = output.copyNextSampleBuffer() {
            try cancellation.check()
            guard CMSampleBufferGetNumSamples(sample) > 0, let pixels = CMSampleBufferGetImageBuffer(sample) else { continue }
            if indices[cursor] == index {
                guard let decoded = rgb(pixels, matrix: track.matrix, fullRange: track.fullRange) else { throw EG2Picture.failure("Unsupported decoded video layout.") }
                let frame = try map(decoded.oriented(orientation))
                while cursor < indices.count, indices[cursor] == index { result.append(frame); cursor += 1 }
            }
            index += 1
        }
        guard reader.status != .failed, result.count == indices.count else { throw EG2Picture.failure("The video has no readable sampled frames.") }
        return result
    }
}

func eg2Await<T>(_ operation: @escaping () async throws -> T) throws -> T {
    let semaphore = DispatchSemaphore(value: 0)
    var result: Result<T, Error>?
    Task {
        do { result = .success(try await operation()) } catch { result = .failure(error) }
        semaphore.signal()
    }
    semaphore.wait()
    return try result!.get()
}
