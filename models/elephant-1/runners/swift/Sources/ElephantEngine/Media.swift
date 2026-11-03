// SPDX-License-Identifier: MIT
// elephant-1's image front end as the maker builds it (OneJev qev/media.py and train/mm.py with transformers 5.19.0's
// torchvision-backed Qwen2VLImageProcessor):
// - decode as PIL's Image.open(path).convert("RGB"): stored colour values, no colour management, alpha dropped (not
//   composited), grey replicated, 16-bit RGB(A) and grey+alpha to their high byte, 16-bit grey clipped to 255. EXIF
//   orientation is not applied (PIL does not apply it either). JPEG is decoded by ImageIO, not libjpeg-turbo, so a
//   JPEG's pixels can differ from PIL's by a level or two (measured on the fixtures; see the model README).
// - smart_resize to a multiple of patch x merge within the processor's pixel bounds;
// - torchvision's antialiased bicubic on 8-bit images (PIL's filter, a = -0.5, support widened when shrinking, with
//   torchvision's int16 fixed-point weights; horizontal pass, then vertical);
// - rescale and normalise fused as torchvision does, (x - 127.5) / 127.5 in float32, and patchify in merge-block order.
import CoreGraphics
import Foundation
import ImageIO

/// An 8-bit RGB image, interleaved, row-major.
public struct RGBImage: Equatable, Sendable {
    public let width: Int, height: Int
    public var bytes: [UInt8]
    public init(width: Int, height: Int, bytes: [UInt8]) { self.width = width; self.height = height; self.bytes = bytes }
}

public struct ImageProcessorConfig: Sendable {
    public let patch: Int, merge: Int, temporal: Int, minPixels: Int, maxPixels: Int
    public let mean: [Float], std: [Float], rescale: Float
    public var factor: Int { patch * merge }

    public init(json: [String: Any]) throws {
        guard let p = json["image_processor"] as? [String: Any], let size = p["size"] as? [String: Int],
            let minPixels = size["shortest_edge"], let maxPixels = size["longest_edge"]
        else { throw PromptError("processor_config.json lacks image_processor size.") }
        patch = p["patch_size"] as? Int ?? 16; merge = p["merge_size"] as? Int ?? 2; temporal = p["temporal_patch_size"] as? Int ?? 2
        self.minPixels = minPixels; self.maxPixels = maxPixels
        mean = (p["image_mean"] as? [Double] ?? [0.5, 0.5, 0.5]).map(Float.init)
        std = (p["image_std"] as? [Double] ?? [0.5, 0.5, 0.5]).map(Float.init)
        rescale = Float(p["rescale_factor"] as? Double ?? 1.0 / 255)
        guard p["resample"] as? Int ?? 3 == 3, p["do_resize"] as? Bool ?? true, p["do_normalize"] as? Bool ?? true else {
            throw PromptError("Only the bicubic, normalising image processor is implemented.")
        }
    }
}

/// A picture's file and stored size, read without decoding its pixels.
public struct ImageSource: Sendable {
    public let url: URL
    public let width: Int, height: Int
}

public enum ImageInput {
    /// Files above this many pixels are refused before decoding (a 64-megapixel decode is about 190 MB of RGB).
    public static let maxSourcePixels = 64_000_000

    /// The file types the maker's PIL path is matched for and model.toml promises (ImageIO also opens HEIC, TIFF,
    /// WebP, GIF, BMP, …; those are refused, not decoded differently from the maker). A multi-picture JPEG (MPO, some
    /// camera and portrait photos) is a JPEG whose first picture PIL reads, as ImageIO's index 0 does.
    public static let acceptedTypes: Set<String> = ["public.png", "public.jpeg", "public.mpo-image"]

    /// The picture's size from its header (no pixel decode).
    public static func probe(_ url: URL) throws -> ImageSource {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, nil) else { throw PromptError("The image could not be read.") }
        guard let type = CGImageSourceGetType(source) as String?, acceptedTypes.contains(type) else {
            throw PromptError("The image must be a PNG or JPEG file.")
        }
        guard let props = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let width = props[kCGImagePropertyPixelWidth] as? Int, let height = props[kCGImagePropertyPixelHeight] as? Int,
            width > 0, height > 0
        else { throw PromptError("The image could not be read.") }
        guard width <= maxSourcePixels / height else { throw PromptError("The image is larger than \(maxSourcePixels / 1_000_000) megapixels.") }
        return ImageSource(url: url, width: width, height: height)
    }

    /// The file's pixels as PIL's `Image.open(path).convert("RGB")` gives them. A file whose data ends early is refused,
    /// as PIL refuses it ("image file is truncated"): ImageIO decodes a cut JPEG with the missing part grey and reports
    /// it only through the source's status after the decode; a cut PNG yields no pixel data at all.
    public static func load(_ url: URL) throws -> RGBImage {
        let source = try probe(url)
        guard let file = CGImageSourceCreateWithURL(url as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(file, 0, [kCGImageSourceShouldCache: false] as CFDictionary),
            image.width == source.width, image.height == source.height
        else { throw PromptError("The image could not be decoded.") }
        // The decoded copy lives only inside this closure, so a redrawn layout never holds it beside its own buffer.
        let stored: RGBImage? = try {
            guard let data = image.dataProvider?.data, CGImageSourceGetStatusAtIndex(file, 0) == .statusComplete else {
                throw PromptError("The image file is truncated or damaged.")
            }
            return raw(image, data)
        }()
        return try stored ?? drawn(image)
    }

    /// Grey or RGB, 8 or 16 bits per component, with or without straight alpha: the stored values, as PIL converts them.
    static func raw(_ image: CGImage, _ data: CFData) -> RGBImage? {
        guard let model = image.colorSpace?.model, model == .rgb || model == .monochrome, [8, 16].contains(image.bitsPerComponent),
            let ptr = CFDataGetBytePtr(data)
        else { return nil }
        let colors = model == .rgb ? 3 : 1, bpc = image.bitsPerComponent, bytes = bpc / 8
        let info = image.alphaInfo
        let hasAlphaSlot: Bool, alphaFirst: Bool
        switch info {
        case .none: hasAlphaSlot = false; alphaFirst = false
        case .last, .noneSkipLast: hasAlphaSlot = true; alphaFirst = false
        case .first, .noneSkipFirst: hasAlphaSlot = true; alphaFirst = true
        default: return nil // premultiplied or alpha-only: drawn instead
        }
        let components = colors + (hasAlphaSlot ? 1 : 0)
        guard image.bitsPerPixel == components * bpc, image.bitmapInfo.contains(.floatComponents) == false else { return nil }
        let order = image.bitmapInfo.intersection(.byteOrderMask)
        let little: Bool
        switch (order, bytes) {
        case ([], _), (.byteOrder16Big, 2), (.byteOrder32Big, 1): little = false
        case (.byteOrder16Little, 2): little = true
        default: return nil
        }
        let hasAlpha = info == .last || info == .first
        let width = image.width, height = image.height, stride = image.bytesPerRow
        let first = alphaFirst ? 1 : 0
        var out = [UInt8](repeating: 0, count: width * height * 3)
        for y in 0..<height {
            for x in 0..<width {
                let base = y * stride + x * components * bytes
                func value(_ c: Int) -> Int {
                    let p = base + (first + c) * bytes
                    if bytes == 1 { return Int(ptr[p]) }
                    return little ? Int(ptr[p]) | Int(ptr[p + 1]) << 8 : Int(ptr[p]) << 8 | Int(ptr[p + 1])
                }
                let o = (y * width + x) * 3
                for c in 0..<3 {
                    let v = value(colors == 3 ? c : 0)
                    let byte: Int
                    if bytes == 1 {
                        byte = v
                    } else if colors == 1 && !hasAlpha {
                        byte = Swift.min(v, 255) // PIL I;16 -> RGB clips
                    } else {
                        byte = v >> 8 // PIL reads 16-bit RGB(A) and LA as their high byte
                    }
                    out[o + c] = UInt8(byte)
                }
            }
        }
        return RGBImage(width: width, height: height, bytes: out)
    }

    /// Other layouts (palette without straight data, CMYK, premultiplied, float): drawn in their own RGB space, or
    /// sRGB, without filtering. CMYK and float images can differ from PIL's own conversions.
    static func drawn(_ image: CGImage) throws -> RGBImage {
        let width = image.width, height = image.height
        let space = image.colorSpace.flatMap { $0.model == .rgb ? $0 : nil } ?? CGColorSpace(name: CGColorSpace.sRGB)!
        var rgbx = [UInt8](repeating: 0, count: width * height * 4)
        let ok = rgbx.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: width, height: height))
            return true
        }
        guard ok else { throw PromptError("The image could not be decoded.") }
        var out = [UInt8](repeating: 0, count: width * height * 3)
        for i in 0..<(width * height) { out[i * 3] = rgbx[i * 4]; out[i * 3 + 1] = rgbx[i * 4 + 1]; out[i * 3 + 2] = rgbx[i * 4 + 2] }
        return RGBImage(width: width, height: height, bytes: out)
    }

    /// transformers' qwen2_vl smart_resize (Python round = half to even).
    public static func smartResize(height: Int, width: Int, factor: Int, minPixels: Int, maxPixels: Int) throws -> (height: Int, width: Int) {
        guard Double(Swift.max(height, width)) / Double(Swift.min(height, width)) <= 200 else {
            throw PromptError("Image aspect ratio must be smaller than 200.")
        }
        let f = Double(factor), h = Double(height), w = Double(width)
        var hBar = (h / f).rounded(.toNearestOrEven) * f, wBar = (w / f).rounded(.toNearestOrEven) * f
        if hBar * wBar > Double(maxPixels) {
            let beta = (h * w / Double(maxPixels)).squareRoot()
            hBar = Swift.max(f, (h / beta / f).rounded(.down) * f); wBar = Swift.max(f, (w / beta / f).rounded(.down) * f)
        } else if hBar * wBar < Double(minPixels) {
            let beta = (Double(minPixels) / (h * w)).squareRoot()
            hBar = (h * beta / f).rounded(.up) * f; wBar = (w * beta / f).rounded(.up) * f
        }
        return (Int(hBar), Int(wBar))
    }

    /// Image tokens for an image of this size: (H/patch)(W/patch)/merge^2 after smart_resize.
    public static func tokens(height: Int, width: Int, config c: ImageProcessorConfig) throws -> (count: Int, grid: THW) {
        let (h, w) = try smartResize(height: height, width: width, factor: c.factor, minPixels: c.minPixels, maxPixels: c.maxPixels)
        let grid = THW(1, h / c.patch, w / c.patch)
        return (grid.h * grid.w / (c.merge * c.merge), grid)
    }

    /// Resize, rescale, normalise and patchify one image: values [patches * 3 * T * P * P] and its grid.
    public static func pixels(_ image: RGBImage, config c: ImageProcessorConfig) throws -> ([Float], THW) {
        let (h, w) = try smartResize(height: image.height, width: image.width, factor: c.factor, minPixels: c.minPixels, maxPixels: c.maxPixels)
        let resized = Resample.bicubic(image, width: w, height: h)
        let gh = h / c.patch, gw = w / c.patch, p = c.patch, m = c.merge, frames = c.temporal
        let columns = 3 * frames * p * p
        // torchvision fuses rescale into normalisation: (x - mean / rescale) / (std / rescale), float32.
        let mean = c.mean.map { $0 / c.rescale }, std = c.std.map { $0 / c.rescale }
        var out = [Float](repeating: 0, count: gh * gw * columns)
        var row = 0
        let bytes = resized.bytes
        for bh in 0..<(gh / m) {
            for bw in 0..<(gw / m) {
                for mh in 0..<m {
                    for mw in 0..<m {
                        let py = (bh * m + mh) * p, px = (bw * m + mw) * p
                        var column = row * columns
                        for ch in 0..<3 {
                            for _ in 0..<frames {
                                for y in 0..<p {
                                    for x in 0..<p {
                                        out[column] = (Float(bytes[((py + y) * w + px + x) * 3 + ch]) - mean[ch]) / std[ch]
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
        return (out, THW(1, gh, gw))
    }
}

enum Resample {
    private static func cubic(_ x: Double) -> Double {
        let a = -0.5, x = Swift.abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }
    /// Antialiased filter taps for one axis (PIL precompute_coeffs = torch _compute_indices_min_size_weights_aa):
    /// first input index and normalised weights per output position.
    static func coefficients(input: Int, output: Int) -> [(start: Int, weights: [Double])] {
        let scale = Double(input) / Double(output), filterScale = Swift.max(scale, 1), support = 2 * filterScale
        return (0..<output).map { x in
            let center = (Double(x) + 0.5) * scale
            let low = Swift.max(Int(center - support + 0.5), 0), high = Swift.min(Int(center + support + 0.5), input)
            var weights = (low..<high).map { cubic((Double($0) - center + 0.5) / filterScale) }
            let total = weights.reduce(0, +)
            if total != 0 { weights = weights.map { $0 / total } }
            return (low, weights)
        }
    }

    /// torchvision's uint8 path (PIL-SIMD): weights as int16 with the largest precision that keeps the largest weight
    /// below 2^15, rounded half away from zero; sums start at 2^(precision-1), shift right, clip to 0...255.
    static func fixedPoint(_ coeffs: [(start: Int, weights: [Double])]) -> (precision: Int, taps: [(start: Int, weights: [Int32])]) {
        let maxWeight = coeffs.flatMap(\.weights).max() ?? 1
        var precision = 0
        while precision < 22 {
            let next = Int(0.5 + maxWeight * Double(1 << (precision + 1)))
            if next >= (1 << 15) { break }
            precision += 1
        }
        let taps = coeffs.map { c in
            (
                c.start,
                c.weights.map { w -> Int32 in
                    let v = w * Double(1 << precision)
                    return Int32(Int16(v < 0 ? Int(-0.5 + v) : Int(0.5 + v)))
                }
            )
        }
        return (precision, taps)
    }

    private static func clip8(_ v: Int32, _ precision: Int) -> UInt8 { UInt8(Swift.max(0, Swift.min(255, v >> Int32(precision)))) }

    /// Horizontal pass to 8 bits, then vertical; a same-size axis is copied.
    static func bicubic(_ image: RGBImage, width: Int, height: Int) -> RGBImage {
        var current = image
        if width != image.width {
            let (precision, taps) = fixedPoint(coefficients(input: image.width, output: width))
            let initial = Int32(1 << (precision - 1))
            var out = [UInt8](repeating: 0, count: width * image.height * 3)
            for y in 0..<image.height {
                let row = y * image.width * 3
                for x in 0..<width {
                    let (start, weights) = taps[x]
                    var r = initial, g = initial, b = initial
                    for (k, w) in weights.enumerated() {
                        let p = row + (start + k) * 3
                        r += w * Int32(image.bytes[p]); g += w * Int32(image.bytes[p + 1]); b += w * Int32(image.bytes[p + 2])
                    }
                    let o = (y * width + x) * 3
                    out[o] = clip8(r, precision); out[o + 1] = clip8(g, precision); out[o + 2] = clip8(b, precision)
                }
            }
            current = RGBImage(width: width, height: image.height, bytes: out)
        }
        if height != current.height {
            let (precision, taps) = fixedPoint(coefficients(input: current.height, output: height))
            let initial = Int32(1 << (precision - 1))
            var out = [UInt8](repeating: 0, count: width * height * 3)
            let stride = width * 3
            for y in 0..<height {
                let (start, weights) = taps[y]
                for i in 0..<stride {
                    var v = initial
                    for (k, w) in weights.enumerated() { v += w * Int32(current.bytes[(start + k) * stride + i]) }
                    out[y * stride + i] = clip8(v, precision)
                }
            }
            current = RGBImage(width: width, height: height, bytes: out)
        }
        return current
    }
}

// MARK: - Frame lists (videos)

/// The checkpoint's video processor settings (processor_config.json "video_processor", Qwen3VLVideoProcessor).
public struct VideoProcessorConfig: Sendable {
    public let patch: Int, merge: Int, temporal: Int, minPixels: Int, maxPixels: Int, maxVideoTokens: Int
    public let mean: [Float], std: [Float], rescale: Float
    public var factor: Int { patch * merge }

    public init(json: [String: Any]) throws {
        guard let p = json["video_processor"] as? [String: Any], let size = p["size"] as? [String: Int],
            let minPixels = size["shortest_edge"], let maxPixels = size["longest_edge"]
        else { throw PromptError("processor_config.json lacks video_processor size.") }
        patch = p["patch_size"] as? Int ?? 16; merge = p["merge_size"] as? Int ?? 2; temporal = p["temporal_patch_size"] as? Int ?? 2
        self.minPixels = minPixels; self.maxPixels = maxPixels
        maxVideoTokens = p["max_video_tokens"] as? Int ?? 768
        mean = (p["image_mean"] as? [Double] ?? [0.5, 0.5, 0.5]).map(Float.init)
        std = (p["image_std"] as? [Double] ?? [0.5, 0.5, 0.5]).map(Float.init)
        rescale = Float(p["rescale_factor"] as? Double ?? 1.0 / 255)
        guard p["resample"] as? Int ?? 3 == 3, p["do_resize"] as? Bool ?? true, p["do_normalize"] as? Bool ?? true else {
            throw PromptError("Only the bicubic, normalising video processor is implemented.")
        }
    }
}

/// A frame list as the makers send it ({"type": "video", "frames": [...], "fps": f}): stills of one size, in order.
public struct VideoSource: Sendable {
    public let frames: [URL]
    public let width: Int, height: Int
    public let fps: Double
}

public enum VideoInput {
    /// Arsenal's bound on frames per video (the video processor's max_frames).
    public static let maxFrames = 768

    /// Every frame's size from its header; all frames must share one size, as the makers' processor stacks them.
    public static func probe(_ frames: [URL], fps: Double, sizes known: [ImageSource]? = nil) throws -> VideoSource {
        guard frames.count >= 2, frames.count <= maxFrames else {
            throw PromptError("A video needs 2 to \(maxFrames) frames (the processor pairs frames in time).")
        }
        guard fps > 0, fps.isFinite else { throw PromptError("fps must be a positive number.") }
        let sizes = try known ?? frames.map { try ImageInput.probe($0) }
        guard sizes.allSatisfy({ $0.width == sizes[0].width && $0.height == sizes[0].height }) else {
            throw PromptError("All frames of a video must have the same size.")
        }
        return VideoSource(frames: frames, width: sizes[0].width, height: sizes[0].height, fps: fps)
    }

    /// Qwen3VLVideoProcessor.resize with cap_pixels_per_frame (the makers' videos_kwargs): per-frame pixels at most
    /// max_video_tokens x factor^2 and longest_edge / frames, at least 1.05 x shortest_edge; then the video smart_resize.
    public static func size(frames n: Int, height: Int, width: Int, config c: VideoProcessorConfig) throws -> (height: Int, width: Int) {
        let f = c.factor
        let frameCap = c.maxVideoTokens * f * f
        let perFrame = Swift.max(Swift.min(frameCap, c.maxPixels / n), Int(Double(c.minPixels) * 1.05))
        return try smartResize(frames: n, height: height, width: width, factor: f, temporal: c.temporal, minPixels: c.minPixels, maxPixels: perFrame * n)
    }

    /// transformers 5.19 qwen3_vl video smart_resize (Python round = half to even, int() truncates).
    public static func smartResize(
        frames n: Int, height: Int, width: Int, factor: Int, temporal: Int, minPixels: Int, maxPixels: Int
    ) throws -> (height: Int, width: Int) {
        guard n >= temporal else { throw PromptError("t:\(n) must be larger than temporal_factor:\(temporal)") }
        var height = height, width = width
        if height < factor || width < factor {
            let scale = Swift.max(Double(factor) / Double(height), Double(factor) / Double(width))
            height = Int(Double(height) * scale); width = Int(Double(width) * scale)
        }
        guard Double(Swift.max(height, width)) / Double(Swift.min(height, width)) <= 200 else {
            throw PromptError("Video aspect ratio must be smaller than 200.")
        }
        let f = Double(factor), h = Double(height), w = Double(width)
        var hBar = (h / f).rounded(.toNearestOrEven) * f, wBar = (w / f).rounded(.toNearestOrEven) * f
        let tBar = (Double(n) / Double(temporal)).rounded(.toNearestOrEven) * Double(temporal)
        if tBar * hBar * wBar > Double(maxPixels) {
            let beta = (Double(n) * h * w / Double(maxPixels)).squareRoot()
            hBar = Swift.max(f, (h / beta / f).rounded(.down) * f); wBar = Swift.max(f, (w / beta / f).rounded(.down) * f)
        } else if tBar * hBar * wBar < Double(minPixels) {
            let beta = (Double(minPixels) / (Double(n) * h * w)).squareRoot()
            hBar = (h * beta / f).rounded(.up) * f; wBar = (w * beta / f).rounded(.up) * f
        }
        return (Int(hBar), Int(wBar))
    }

    /// The video's grid: frame pairs x patch rows x patch columns.
    public static func grid(_ v: VideoSource, config c: VideoProcessorConfig) throws -> THW {
        let (h, w) = try size(frames: v.frames.count, height: v.height, width: v.width, config: c)
        return THW((v.frames.count + c.temporal - 1) / c.temporal, h / c.patch, w / c.patch)
    }

    /// Qwen3VLProcessor._calculate_timestamps: frame index / fps, the last index repeated to fill the last pair, each
    /// pair's mean; shown as "<%.1f seconds>".
    public static func timestamps(frames n: Int, fps: Double, temporal: Int) -> [Double] {
        var indices = Array(0..<n)
        while indices.count % temporal != 0 { indices.append(indices.last!) }
        let t = indices.map { Double($0) / fps }
        return stride(from: 0, to: t.count, by: temporal).map { (t[$0] + t[$0 + temporal - 1]) / 2 }
    }

    public static func label(_ seconds: Double) -> String { "<" + String(format: "%.1f", seconds) + " seconds>" }

    /// Decode, resize and patchify a frame list one frame at a time: rows (frame pair, block row, block column, merge
    /// row, merge column), columns (channel, frame in pair, pixel row, pixel column); an odd count repeats the last frame.
    public static func pixels(_ v: VideoSource, config c: VideoProcessorConfig, cancelled: () -> Bool = { false }) throws -> ([Float], THW) {
        let (h, w) = try size(frames: v.frames.count, height: v.height, width: v.width, config: c)
        var frames: [[UInt8]] = []
        for url in v.frames {
            if cancelled() { throw Cancelled() }
            let image = try ImageInput.load(url)
            guard image.width == v.width, image.height == v.height else { throw PromptError("A frame changed size while it was read.") }
            frames.append(Resample.bicubic(image, width: w, height: h).bytes)
        }
        while frames.count % c.temporal != 0 { frames.append(frames[frames.count - 1]) }
        let gt = frames.count / c.temporal, gh = h / c.patch, gw = w / c.patch, p = c.patch, m = c.merge, tp = c.temporal
        let columns = 3 * tp * p * p
        let mean = c.mean.map { $0 / c.rescale }, std = c.std.map { $0 / c.rescale }
        var out = [Float](repeating: 0, count: gt * gh * gw * columns)
        var row = 0
        for t in 0..<gt {
            for bh in 0..<(gh / m) {
                for bw in 0..<(gw / m) {
                    for mh in 0..<m {
                        for mw in 0..<m {
                            let py = (bh * m + mh) * p, px = (bw * m + mw) * p
                            var column = row * columns
                            for ch in 0..<3 {
                                for k in 0..<tp {
                                    let bytes = frames[t * tp + k]
                                    for y in 0..<p {
                                        for x in 0..<p {
                                            out[column] = (Float(bytes[((py + y) * w + px + x) * 3 + ch]) - mean[ch]) / std[ch]
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
        return (out, THW(gt, gh, gw))
    }
}
