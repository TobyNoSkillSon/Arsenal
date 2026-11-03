// SPDX-License-Identifier: MIT
// Pictures as the vendor stack sees them: Pillow's decode + convert('RGB'), the vendor's bicubic cap (Pillow,
// 22-bit fixed point) and LFM2-VL's processor (torchvision's antialiased uint8 bicubic: int16 weights, precision
// chosen per pass). Both resamplers are bit-exact with their originals (log/007).
import CoreGraphics
import Foundation
import ImageIO

/// An 8-bit RGB picture, rows top to bottom, channels interleaved.
public struct Picture: Sendable {
    public let width: Int
    public let height: Int
    public var pixels: [UInt8]

    public init(width: Int, height: Int, pixels: [UInt8]) {
        self.width = width
        self.height = height
        self.pixels = pixels
    }

    /// Decodes a local file. Pixel values are taken as stored (no colour management, EXIF orientation not applied),
    /// as Pillow's `Image.open(path).convert('RGB')`; an alpha channel is dropped, not composited.
    public static func load(_ url: URL, maximumWorkingBytes: Int? = nil) throws -> Picture {
        guard let source = CGImageSourceCreateWithURL(url as CFURL, [kCGImageSourceShouldCache: false] as CFDictionary),
            CGImageSourceGetCount(source) > 0,
            let properties = CGImageSourceCopyPropertiesAtIndex(source, 0, nil) as? [CFString: Any],
            let w = properties[kCGImagePropertyPixelWidth] as? Int,
            let h = properties[kCGImagePropertyPixelHeight] as? Int
        else { throw JSONError("cannot identify image file '\(url.path)'") }
        guard w > 0, h > 0, w <= 178_956_970 / h else { throw JSONError("The picture is empty or too large to decode.") }
        // Inspect the header before asking ImageIO for decoded pixels. Reserve for the decoder (up to RGBA32),
        // RGB conversion and both resize passes, plus the two coefficient tables and capped output.
        let workingBytes = w * h * 32 + (w + h) * 128 + PictureProcessor.maxPixels * 9
        if let maximumWorkingBytes, workingBytes > maximumWorkingBytes {
            throw JSONError(PictureBudget.message)
        }
        guard let image = CGImageSourceCreateImageAtIndex(source, 0, [kCGImageSourceShouldCache: false] as CFDictionary) else {
            throw JSONError("cannot identify image file '\(url.path)'")
        }
        if let raw = raw16(image) { return raw }
        if let raw = raw(image) { return raw }
        // Other layouts (16-bit, indexed, CMYK, ...): draw into an 8-bit RGB context in the picture's own RGB space
        // when it has one, so no colour conversion is applied; otherwise sRGB.
        let space: CGColorSpace
        if let own = image.colorSpace, own.model == .rgb {
            space = own
        } else if let own = image.colorSpace, own.model == .indexed, let base = own.baseColorSpace, base.model == .rgb {
            space = base
        } else if let own = image.colorSpace, own.model == .monochrome {
            return try gray(image)
        } else {
            space = CGColorSpace(name: CGColorSpace.sRGB)!
        }
        var rgba = [UInt8](repeating: 0, count: w * h * 4)
        let drawn = rgba.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space,
                    bitmapInfo: CGImageAlphaInfo.noneSkipLast.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw JSONError("The picture could not be converted to RGB.") }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[3 * i] = rgba[4 * i]
            rgb[3 * i + 1] = rgba[4 * i + 1]
            rgb[3 * i + 2] = rgba[4 * i + 2]
        }
        return Picture(width: w, height: h, pixels: rgb)
    }

    /// 8-bit RGB(A)/gray(A) bytes straight from the decoder, alpha dropped (Pillow's convert('RGB')).
    private static func raw(_ image: CGImage) -> Picture? {
        guard image.bitsPerComponent == 8, let space = image.colorSpace, let data = image.dataProvider?.data else { return nil }
        let info = image.bitmapInfo
        guard info.intersection(.byteOrderMask) == [] || info.intersection(.byteOrderMask) == .byteOrder32Big else { return nil }
        guard !info.contains(.floatComponents) else { return nil }
        let alpha = image.alphaInfo
        let channels: Int
        switch space.model {
        case .rgb: channels = 3
        case .monochrome: channels = 1
        default: return nil
        }
        let hasAlpha = alpha != .none
        let stride = channels + (hasAlpha ? 1 : 0)
        guard image.bitsPerPixel == stride * 8 else { return nil }
        // Premultiplied colour cannot be recovered exactly; let the drawing path handle it.
        guard alpha == .none || alpha == .last || alpha == .first || alpha == .noneSkipLast || alpha == .noneSkipFirst else { return nil }
        let offset = (alpha == .first || alpha == .noneSkipFirst) ? 1 : 0
        let w = image.width
        let h = image.height
        let row = image.bytesPerRow
        guard CFDataGetLength(data) >= row * (h - 1) + w * stride else { return nil }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        let bytes = CFDataGetBytePtr(data)!
        for y in 0..<h {
            for x in 0..<w {
                let p = y * row + x * stride + offset
                let o = 3 * (y * w + x)
                if channels == 3 {
                    rgb[o] = bytes[p]
                    rgb[o + 1] = bytes[p + 1]
                    rgb[o + 2] = bytes[p + 2]
                } else {
                    rgb[o] = bytes[p]
                    rgb[o + 1] = bytes[p]
                    rgb[o + 2] = bytes[p]
                }
            }
        }
        return Picture(width: w, height: h, pixels: rgb)
    }

    /// PNG grayscale16 opens in Pillow's integer mode: convert('RGB') clamps at 255, it does not
    /// normalise the 16-bit range. RGB16/RGBA16/gray-alpha16 open in an 8-bit mode and take the high byte.
    private static func raw16(_ image: CGImage) -> Picture? {
        guard image.bitsPerComponent == 16, !image.bitmapInfo.contains(.floatComponents),
            let space = image.colorSpace, let data = image.dataProvider?.data
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
        return Picture(width: w, height: h, pixels: rgb)
    }

    private static func gray(_ image: CGImage) throws -> Picture {
        let w = image.width
        let h = image.height
        var g = [UInt8](repeating: 0, count: w * h)
        let drawn = g.withUnsafeMutableBytes { buffer -> Bool in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: image.colorSpace ?? CGColorSpaceCreateDeviceGray(),
                    bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { return false }
            context.interpolationQuality = .none
            context.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
            return true
        }
        guard drawn else { throw JSONError("The picture could not be converted to RGB.") }
        var rgb = [UInt8](repeating: 0, count: w * h * 3)
        for i in 0..<(w * h) {
            rgb[3 * i] = g[i]
            rgb[3 * i + 1] = g[i]
            rgb[3 * i + 2] = g[i]
        }
        return Picture(width: w, height: h, pixels: rgb)
    }
}

/// Per wire request: charge only capped pictures, including earlier groups. Failed items are transactional.
/// The working limit includes retained pictures plus one image's decode/conversion/resize reservation.
public struct PictureBudget: Sendable {
    public static let retainedLimit = 128 * 1024 * 1024
    public static let workingLimit = 512 * 1024 * 1024
    public static let message = "Images exceed this request's decoded-picture memory limit (128 MiB retained, 512 MiB working)."
    public fileprivate(set) var retainedBytes = 0
    public init() {}
}

/// Separable antialiased bicubic resampling of 8-bit pictures (a = -0.5, support 2 scaled when shrinking): width
/// first, then height, each pass rounded to 8 bits, a pass skipped when its size is unchanged.
public enum Resample {
    public enum Precision: Sendable {
        /// Pillow's `ImagingResample`: 22-bit fixed-point weights.
        case pillow
        /// torch's uint8 antialiased kernel: int16 weights, the precision the pass's largest weight allows.
        case torchUInt8
    }

    static func cubic(_ x: Double) -> Double {
        let a = -0.5
        let x = abs(x)
        if x < 1 { return ((a + 2) * x - (a + 3)) * x * x + 1 }
        if x < 2 { return (((x - 5) * x + 8) * x - 4) * a }
        return 0
    }

    /// Per output index: first input index and normalised weights.
    static func weights(_ input: Int, _ output: Int, filter: (Double) -> Double = cubic, support: Double = 2) -> [(Int, [Double])] {
        let scale = Double(input) / Double(output)
        let filterScale = max(scale, 1)
        let reach = support * filterScale
        return (0..<output).map { i in
            let center = scale * (Double(i) + 0.5)
            let lo = max(Int(center - reach + 0.5), 0)
            let hi = min(Int(center + reach + 0.5), input)
            var w = (0..<(hi - lo)).map { filter((Double($0 + lo) - center + 0.5) / filterScale) }
            let total = w.reduce(0, +)
            if total != 0 { w = w.map { $0 / total } }
            return (lo, w)
        }
    }

    public static func resize(_ p: Picture, width: Int, height: Int, _ precision: Precision) -> Picture {
        var out = p
        if width != p.width { out = pass(out, horizontal: true, size: width, precision) }
        if height != p.height { out = pass(out, horizontal: false, size: height, precision) }
        return out
    }

    private static func pass(_ p: Picture, horizontal: Bool, size: Int, _ precision: Precision) -> Picture {
        let table = weights(horizontal ? p.width : p.height, size)
        let bits: Int
        switch precision {
        case .pillow: bits = 22
        case .torchUInt8:
            let largest = table.flatMap(\.1).max() ?? 1
            var b = 0
            while b < 21 && Int(0.5 + largest * Double(1 << (b + 1))) < (1 << 15) { b += 1 }
            bits = b
        }
        let fixed = table.map { lo, w in (lo, w.map { Int64(($0 * Double(1 << bits) + ($0 < 0 ? -0.5 : 0.5)).rounded(.towardZero)) }) }
        let w = horizontal ? size : p.width
        let h = horizontal ? p.height : size
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
        return Picture(width: w, height: h, pixels: out)
    }
}

/// LFM2-VL's image processor (transformers 5.14 `Lfm2VlImageProcessor`/`Lfm2VlProcessor`) at the checkpoint's settings.
public struct PictureProcessor: Sendable {
    let patch = 16
    let downsample = 2
    let tile = 512
    let minTokens = 64
    let maxTokens = 256
    let tolerance = 2.0
    /// `_target_ratios(2, 10)` in the processor's (set-ordered, then area-sorted) order.
    static let ratios: [(Int, Int)] = [
        (1, 2), (2, 1), (3, 1), (1, 3), (2, 2), (4, 1), (1, 4), (5, 1), (1, 5), (1, 6), (6, 1), (3, 2), (2, 3), (7, 1), (1, 7), (4, 2), (2, 4),
        (1, 8), (8, 1), (1, 9), (3, 3), (9, 1), (2, 5), (5, 2), (10, 1), (1, 10)
    ]
    public static let maxPixels = 1024 * 1024 // the vendor's VISION_MAX_PIXELS

    public init(config url: URL) throws {
        let root = try JSONSerialization.jsonObject(with: Data(contentsOf: url)) as? [String: Any]
        guard let p = root?["image_processor"] as? [String: Any], p["image_processor_type"] as? String == "Lfm2VlImageProcessor",
            p["downsample_factor"] as? Int == 2, p["encoder_patch_size"] as? Int == 16, p["tile_size"] as? Int == 512,
            p["min_image_tokens"] as? Int == 64, p["max_image_tokens"] as? Int == 256, p["min_tiles"] as? Int == 2,
            p["max_tiles"] as? Int == 10, p["use_thumbnail"] as? Bool == true, p["do_image_splitting"] as? Bool == true,
            p["max_pixels_tolerance"] as? Double == 2.0, p["resample"] as? Int == 3,
            p["image_mean"] as? [Double] == [0.5, 0.5, 0.5], p["image_std"] as? [Double] == [0.5, 0.5, 0.5]
        else { throw JSONError("processor_config.json is not the LFM2-VL setting this runner implements") }
    }

    /// Decode and cap each source inside its own autorelease pool; no full-size source survives to the next.
    /// Commit the budget only after the complete item succeeds, so one bad path does not consume later items' quota.
    public func loadCapped(_ urls: [URL], budget: inout PictureBudget) throws -> [Picture] {
        var charged = budget.retainedBytes
        var out: [Picture] = []
        for url in urls {
            guard charged + Self.maxPixels * 3 <= PictureBudget.retainedLimit else { throw JSONError(PictureBudget.message) }
            let picture = try autoreleasepool {
                cap(try Picture.load(url, maximumWorkingBytes: PictureBudget.workingLimit - charged))
            }
            guard picture.pixels.count <= PictureBudget.retainedLimit - charged else { throw JSONError(PictureBudget.message) }
            charged += picture.pixels.count
            out.append(picture)
        }
        budget.retainedBytes = charged
        return out
    }

    /// The vendor's `cap_pixels`: at most 1,048,576 pixels, Pillow bicubic.
    public func cap(_ p: Picture) -> Picture {
        guard p.width * p.height > Self.maxPixels else { return p }
        let scale = (Double(Self.maxPixels) / Double(p.width * p.height)).squareRoot()
        return Resample.resize(p, width: max(1, Int(Double(p.width) * scale)), height: max(1, Int(Double(p.height) * scale)), .pillow)
    }

    /// Python's round(): half to even.
    private func roundBy(_ n: Int, _ factor: Int) -> Int { Int((Double(n) / Double(factor)).rounded(.toNearestOrEven)) * factor }

    func smartResize(height: Int, width: Int) -> (height: Int, width: Int) {
        let total = patch * downsample
        let minPixels = Double(minTokens * patch * patch * downsample * downsample)
        let maxPixels = Double(maxTokens * patch * patch * downsample * downsample)
        var h = max(total, roundBy(height, total))
        var w = max(total, roundBy(width, total))
        if Double(h * w) > maxPixels {
            let beta = (Double(height * width) / maxPixels).squareRoot()
            h = max(total, Int((Double(height) / beta / Double(total)).rounded(.down)) * total)
            w = max(total, Int((Double(width) / beta / Double(total)).rounded(.down)) * total)
        } else if Double(h * w) < minPixels {
            let beta = (minPixels / Double(height * width)).squareRoot()
            h = Int((Double(height) * beta / Double(total)).rounded(.up)) * total
            w = Int((Double(width) * beta / Double(total)).rounded(.up)) * total
        }
        return (h, w)
    }

    func tooLarge(height: Int, width: Int) -> Bool {
        let total = patch * downsample
        let h = max(patch, roundBy(height, total))
        let w = max(patch, roundBy(width, total))
        return Double(h * w) > Double(maxTokens * patch * patch * downsample * downsample) * tolerance
    }

    func grid(height: Int, width: Int) -> (cols: Int, rows: Int) {
        let aspect = Double(width) / Double(height)
        var bestDiff = Double.infinity
        var best = (1, 1)
        let area = Double(width * height)
        for r in Self.ratios {
            let diff = abs(aspect - Double(r.0) / Double(r.1))
            if diff < bestDiff {
                bestDiff = diff
                best = r
            } else if diff == bestDiff && area > 0.5 * Double(tile * tile * r.0 * r.1) {
                best = r
            }
        }
        return best
    }

    /// One picture's layout: its pieces in feature order (tiles row by row, then the thumbnail; or one resized image).
    public struct Layout: Sendable {
        public let rows: Int
        public let cols: Int
        public let size: (height: Int, width: Int) // the single image's or the thumbnail's
    }

    public func layout(_ p: Picture) -> Layout {
        let size = smartResize(height: p.height, width: p.width)
        guard tooLarge(height: p.height, width: p.width) else { return Layout(rows: 1, cols: 1, size: size) }
        let (cols, rows) = grid(height: p.height, width: p.width) // at least two tiles (min_tiles 2)
        return Layout(rows: rows, cols: cols, size: size)
    }

    /// The pieces' pixels in feature order: tiles of the picture resized to the grid, row by row, then the thumbnail;
    /// or the one resized picture.
    public func pieces(_ p: Picture, _ layout: Layout) -> [Picture] {
        guard layout.rows * layout.cols > 1 else { return [Resample.resize(p, width: layout.size.width, height: layout.size.height, .torchUInt8)] }
        let full = Resample.resize(p, width: tile * layout.cols, height: tile * layout.rows, .torchUInt8)
        var out: [Picture] = []
        for r in 0..<layout.rows {
            for c in 0..<layout.cols {
                var px = [UInt8](repeating: 0, count: tile * tile * 3)
                for y in 0..<tile {
                    let from = ((r * tile + y) * full.width + c * tile) * 3
                    px.replaceSubrange((y * tile * 3)..<((y + 1) * tile * 3), with: full.pixels[from..<(from + tile * 3)])
                }
                out.append(Picture(width: tile, height: tile, pixels: px))
            }
        }
        out.append(Resample.resize(p, width: layout.size.width, height: layout.size.height, .torchUInt8))
        return out
    }

    /// Feature rows one piece yields after the 2x2 pixel unshuffle.
    func tokens(_ piece: (height: Int, width: Int)) -> Int {
        let h = Int((Double(piece.height / patch) / Double(downsample)).rounded(.up))
        let w = Int((Double(piece.width / patch) / Double(downsample)).rounded(.up))
        return h * w
    }

    /// `_build_image_tokens` with image special tokens.
    func expansion(_ layout: Layout) -> String {
        var s = "<|image_start|>"
        if layout.rows > 1 || layout.cols > 1 {
            let perTile = tokens((tile, tile))
            for r in 0..<layout.rows {
                for c in 0..<layout.cols {
                    s += "<|img_row_\(r + 1)_col_\(c + 1)|>" + String(repeating: "<image>", count: perTile)
                }
            }
            s += "<|img_thumbnail|>" + String(repeating: "<image>", count: tokens(layout.size))
        } else {
            s += String(repeating: "<image>", count: tokens(layout.size))
        }
        return s + "<|image_end|>"
    }

    /// `Lfm2VlProcessor.__call__`'s text side: every `<image>` in `text` replaced by its picture's expansion. The
    /// processor refuses a text whose `<image>` count differs from the number of pictures.
    public func expand(_ text: String, _ layouts: [Layout]) throws -> String {
        let parts = text.components(separatedBy: "<image>")
        guard parts.count - 1 == layouts.count else {
            throw JSONError("The number of images in the text [\(parts.count - 1)] and images [\(layouts.count)] should be the same.")
        }
        var out = ""
        for (i, layout) in layouts.enumerated() { out += parts[i] + expansion(layout) }
        return out + parts[layouts.count]
    }

    /// What the chat template writes for `n` pictures at the head of a user turn.
    public static func markup(_ n: Int) -> String { String(repeating: "<image>", count: n) }

    /// Normalised patches of one piece, (h/16 * w/16, 16*16*3) fp32: (x - 127.5) / 127.5, patch rows then columns
    /// then channels (`convert_image_to_patches`).
    func patches(_ p: Picture) -> (values: [Float], gridH: Int, gridW: Int) {
        let gh = p.height / patch
        let gw = p.width / patch
        var out = [Float](repeating: 0, count: gh * gw * patch * patch * 3)
        var o = 0
        for py in 0..<gh {
            for px in 0..<gw {
                for y in 0..<patch {
                    for x in 0..<patch {
                        let s = ((py * patch + y) * p.width + px * patch + x) * 3
                        for c in 0..<3 {
                            out[o] = (Float(p.pixels[s + c]) - 127.5) / 127.5
                            o += 1
                        }
                    }
                }
            }
        }
        return (out, gh, gw)
    }
}
