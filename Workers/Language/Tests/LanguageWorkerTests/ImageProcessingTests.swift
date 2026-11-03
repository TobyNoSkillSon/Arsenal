// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import CoreImage
import ImageIO
import UniformTypeIdentifiers
import Testing
import MLX
import ArsenalWire
import IntegrationKit
@testable import LanguageWorker

/// Pixel-path parity with the makers' Pillow pipelines. References were captured once with Pillow 12.3.0
/// (`Image.frombytes("RGB", (7, 5), bytes).resize(size, filter).tobytes()`), Chandra's own `scale_to_fit`
/// (chandra/model/util.py) and Transformers' `smart_resize`; they are fixed numbers, not recomputed here.
struct ImageProcessingTests {
    init() { MLXReady.ensure() }
    /// ((x·37 + y·91 + c·53)·7) mod 256 over a 7×5 RGB image.
    private var source: RGBImage {
        var bytes: [UInt8] = []
        for y in 0..<5 { for x in 0..<7 { for c in 0..<3 { bytes.append(UInt8(((x * 37 + y * 91 + c * 53) * 7) % 256)) } } }
        return RGBImage(width: 7, height: 5, bytes: bytes)
    }

    @Test func pillowResampleMatchesCapturedReference() {
        let bicubicDown: [UInt8] = [99, 171, 172, 59, 181, 179, 70, 145, 186, 192, 165, 166, 121, 172, 173, 62, 176, 180]
        let lanczosDown: [UInt8] = [
            55, 166, 180, 60, 169, 185, 63, 178, 191, 66, 115, 196, 201, 181, 154, 77, 185, 159, 68, 193, 165, 83, 158, 170, 189, 155, 169, 199, 161, 174,
            93, 166, 180, 54, 182, 185
        ]
        let bilinearMixed: [UInt8] = [
            3, 118, 233, 9, 124, 239, 15, 130, 245, 45, 160, 189, 51, 166, 195, 57, 141, 201, 114, 229, 117, 120, 235, 123, 126, 160, 129, 159, 185, 158, 82,
            191, 164, 82, 146, 170, 198, 112, 227, 18, 118, 233, 9, 124, 239, 156, 181, 154, 79, 187, 160, 78, 193, 166, 136, 222, 110, 134, 228, 116, 119,
            234, 122, 205, 150, 179, 161, 156, 185, 47, 162, 191, 247, 106, 221, 178, 112, 227, 3, 118, 233
        ]
        // First two output rows of the 11×8 bicubic upscale.
        let bicubicUpRows: [UInt8] = [
            0, 106, 239, 0, 107, 240, 0, 109, 242, 0, 111, 244, 0, 113, 246, 0, 115, 248, 2, 117, 250, 4, 119, 252, 6, 121, 254, 8, 134, 255, 9, 143, 255,
            43, 171, 172, 44, 172, 173, 47, 174, 175, 58, 176, 177, 63, 178, 179, 65, 180, 181, 67, 182, 183, 69, 185, 185, 71, 186, 187, 73, 121, 189, 74,
            74, 190
        ]
        #expect(PILResample.resize(source, width: 3, height: 2, filter: .bicubic).bytes == bicubicDown)
        #expect(PILResample.resize(source, width: 4, height: 3, filter: .lanczos).bytes == lanczosDown)
        #expect(PILResample.resize(source, width: 3, height: 9, filter: .bilinear).bytes == bilinearMixed)
        #expect(Array(PILResample.resize(source, width: 11, height: 8, filter: .bicubic).bytes.prefix(66)) == bicubicUpRows)
    }

    @Test func documentResizeIsPillowBicubic() {
        let resized = ImagePixels.rgb(DocumentRegions.resize(ImagePixels.ciImage(source), width: 3, height: 2))
        let reference: [UInt8] = [99, 171, 172, 59, 181, 179, 70, 145, 186, 192, 165, 166, 121, 172, 173, 62, 176, 180]
        #expect(resized.width == 3 && resized.height == 2)
        // The sRGB→working space→sRGB round trip through Core Image may move a byte by one.
        #expect(zip(resized.bytes, reference).allSatisfy { abs(Int($0) - Int($1)) <= 1 })
    }

    /// A Display P3 screenshot stores grey 128; PIL ignores the ICC profile and reads 128 (0.502).
    @Test(.heavy) func displayP3GreyReadsLikePillow() throws {
        NativeRuntime.prepare()
        let p3 = try #require(CGColorSpace(name: CGColorSpace.displayP3))
        let provider = try #require(CGDataProvider(data: Data(repeating: 128, count: 4 * 4 * 4) as CFData))
        let cg = try #require(
            CGImage(
                width: 4, height: 4, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: 16, space: p3,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.noneSkipLast.rawValue), provider: provider, decode: nil, shouldInterpolate: false,
                intent: .defaultIntent))
        let file = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-p3-grey-\(UUID().uuidString).png")
        defer { try? FileManager.default.removeItem(at: file) }
        let destination = try #require(CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, cg, nil)
        #expect(CGImageDestinationFinalize(destination))
        // The worker's image() helper decodes files this way.
        let image = try #require(CIImage(contentsOf: file, options: [.applyOrientationProperty: true]))
        let rgb = ImagePixels.rgb(image)
        #expect(rgb.bytes.allSatisfy { abs(Int($0) - 128) <= 1 })
        let normalized = ImagePixels.normalized(rgb, mean: [0.5, 0.5, 0.5], std: [0.5, 0.5, 0.5]).asArray(Float.self)
        #expect(normalized.allSatisfy { abs($0 - (128.0 / 255 - 0.5) / 0.5) < 0.01 })
        // The pinned Qwen3-VL image path (MediaProcessing.asMLXArray) rendered RGBAf without a colour space, which
        // returns Core Image's linear working values (≈0.216 here), not the 0.502 PIL reads.
        var linear = [Float](repeating: 0, count: 4 * 4 * 4)
        linear.withUnsafeMutableBytes { buffer in
            CIContext().render(image, toBitmap: buffer.baseAddress!, rowBytes: 4 * 16, bounds: image.extent, format: .RGBAf, colorSpace: nil)
        }
        #expect(stride(from: 0, to: linear.count, by: 4).allSatisfy { linear[$0] < 0.3 })
    }

    @Test func qwenSmartResizeUsesPythonRounding() throws {
        let cases: [([Int], [Int])] = [
            ([200, 300, 32, 200704, 8294400], [384, 576]), ([80, 112, 32, 3136, 16777216], [64, 128]), ([3000, 4000, 32, 65536, 1048576], [864, 1152]),
            ([2112, 1632, 32, 65536, 16777216], [2112, 1632]), ([48, 80, 32, 1024, 4096], [64, 64])
        ]
        for (input, expected) in cases {
            let size = try QwenVisionProcessor.smartResize(height: input[0], width: input[1], factor: input[2], minPixels: input[3], maxPixels: input[4])
            #expect([size.height, size.width] == expected, "\(input)")
        }
    }

    @Test func chandraScaleToFitMatchesMaker() {
        let fit = QwenVisionConfiguration.ScaleToFit(maxPixels: 3072 * 2048, minPixels: 1792 * 28, grid: 28)
        let cases: [([Int], [Int])] = [
            ([1000, 800], [1008, 812]), ([4000, 3000], [2884, 2156]), ([2480, 3508], [2100, 2968]), ([1700, 28], [1736, 28]), ([640, 480], [644, 476]),
            ([3072, 2048], [3052, 2044])
        ]
        for (input, expected) in cases {
            let size = QwenVisionProcessor.scaleToFit(width: input[0], height: input[1], fit: fit)
            #expect([size?.width ?? input[0], size?.height ?? input[1]] == expected, "\(input)")
        }
    }

    /// Folder limits reach the decoded processor configuration and the patch grid (Ovis 448²–2880²).
    @Test func folderPixelLimitsReachTheProcessor() throws {
        var value: JSONObject = [
            "size": ["shortest_edge": 65536, "longest_edge": 16777216], "min_pixels": 65536, "max_pixels": 16777216, "patch_size": 16, "merge_size": 2,
            "temporal_patch_size": 2, "image_mean": [0.5, 0.5, 0.5], "image_std": [0.5, 0.5, 0.5]
        ]
        try LanguageNative.applyPixelLimits(&value, spec: ["min_pixels": 200704, "max_pixels": 8294400], architecture: "qwen3_5")
        let config = try JSONDecoder().decode(QwenVisionConfiguration.self, from: WireJSON.encode(value))
        #expect(config.minPixels == 200704 && config.maxPixels == 8294400)
        // Infinity's 2,048 floor does not lower the processor's own 65,536.
        var infinity: JSONObject = ["min_pixels": 65536, "max_pixels": 16777216]
        try LanguageNative.applyPixelLimits(&infinity, spec: ["min_pixels": 2048, "max_pixels": 16777216], architecture: "qwen3_5")
        #expect(infinity["min_pixels"]?.number == 65536)
        let processor = QwenVisionProcessor(config, tokenizer: RandomTokenizer(imageID: 260, startID: 261, endID: 262, imageToken: "<|image_pad|>"))
        let small = CIImage(color: CIColor(red: 0.2, green: 0.4, blue: 0.6)).cropped(to: CGRect(x: 0, y: 0, width: 300, height: 200))
        let (patches, grid) = try processor.pixels(small)
        #expect(grid.t == 1 && grid.h == 24 && grid.w == 36)
        #expect(patches.shape == [24 * 36, 3 * 2 * 16 * 16])
        // A per-call cap (describe/tag) only lowers the budget: 262,144 px on a 2000×1500 source gives 576×416.
        let large = CIImage(color: CIColor(red: 0.2, green: 0.4, blue: 0.6)).cropped(to: CGRect(x: 0, y: 0, width: 2000, height: 1500))
        let full: JSONObject = [
            "min_pixels": 3136, "max_pixels": 16777216, "patch_size": 16, "merge_size": 2, "temporal_patch_size": 2, "image_mean": [0.5, 0.5, 0.5],
            "image_std": [0.5, 0.5, 0.5]
        ]
        let uncapped = QwenVisionProcessor(
            try JSONDecoder().decode(QwenVisionConfiguration.self, from: WireJSON.encode(full)),
            tokenizer: RandomTokenizer(imageID: 260, startID: 261, endID: 262, imageToken: "<|image_pad|>"))
        let (_, capped) = try uncapped.pixels(large, maxPixels: 262_144)
        #expect(capped.h * 16 == 416 && capped.w * 16 == 576)
    }

    @Test func pdfRenderScalePerMaker() {
        let letter = CGSize(width: 612, height: 792)
        #expect(abs(LanguageNative.pdfScale(page: letter, spec: [:]) - 2) < 1e-9)
        // Chandra: 192 dpi, raised so the short side reaches 1,024 px.
        let chandra: JSONObject = ["pdf_dpi": 192, "pdf_min_short_side": 1024]
        #expect(abs(LanguageNative.pdfScale(page: letter, spec: chandra) - 192.0 / 72) < 1e-9)
        #expect(abs(LanguageNative.pdfScale(page: CGSize(width: 300, height: 400), spec: chandra) - 1024.0 / 300) < 1e-9)
        // Infinity-Parser2: 300 dpi (2550×3300 for Letter).
        #expect(abs(LanguageNative.pdfScale(page: letter, spec: ["pdf_dpi": 300]) - 300.0 / 72) < 1e-9)
        // Oversized pages stay within 16,777,216 px.
        let poster = CGSize(width: 2384, height: 3370)
        let scale = LanguageNative.pdfScale(page: poster, spec: ["pdf_dpi": 300])
        #expect(poster.width * scale * poster.height * scale <= 16_777_216 + 1)
    }
}
