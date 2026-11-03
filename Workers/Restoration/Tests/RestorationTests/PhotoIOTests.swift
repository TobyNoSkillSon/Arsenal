// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import Testing
import UniformTypeIdentifiers
import ArsenalWire
@testable import RestorationWorker

/// Photo decoding as the providers read files: raw samples, alpha dropped rather than blended, EXIF orientation.
struct PhotoIOTests {
    private func write(_ rgba: [UInt8], width: Int, height: Int, orientation: Int? = 1) throws -> URL {
        let url = FileManager.default.temporaryDirectory.appendingPathComponent(UUID().uuidString + ".png")
        let image = try #require(
            CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 32, bytesPerRow: width * 4, space: CGColorSpace(name: CGColorSpace.sRGB)!,
                bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.last.rawValue), provider: CGDataProvider(data: Data(rgba) as CFData)!, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent))
        let dest = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(dest, image, orientation.map { [kCGImagePropertyOrientation: $0] as CFDictionary })
        #expect(CGImageDestinationFinalize(dest))
        return url
    }

    @Test func transparentPixelsKeepTheirColour() throws {
        // A fully transparent white pixel and a half-transparent red one: both read as their stored colour.
        let url = try write([255, 255, 255, 0, 200, 10, 20, 128], width: 2, height: 1)
        defer { try? FileManager.default.removeItem(at: url) }
        let pixels = try PhotoIO.rgb(url)
        #expect(pixels.bytes == [255, 255, 255, 200, 10, 20])
        // A mask: luma of the stored colour, alpha ignored (PIL convert("L")).
        #expect(try PhotoIO.mask(url).bytes == [255, 68])
    }

    @Test func exifOrientationIsApplied() throws {
        // 2x1: red, green; orientation 6 (rotate 90 degrees clockwise) gives 1x2 with red on top.
        let url = try write([255, 0, 0, 255, 0, 255, 0, 255], width: 2, height: 1, orientation: 6)
        defer { try? FileManager.default.removeItem(at: url) }
        let pixels = try PhotoIO.rgb(url)
        #expect(pixels.width == 1 && pixels.height == 2)
        #expect(pixels.bytes == [255, 0, 0, 0, 255, 0])
        for orientation in 1...8 {
            let o = PhotoIO.orient([1, 2, 3, 4, 5, 6], channels: 1, width: 3, height: 2, orientation: orientation)
            #expect(o.bytes.sorted() == [1, 2, 3, 4, 5, 6])
            #expect(orientation >= 5 ? (o.width, o.height) == (2, 3) : (o.width, o.height) == (3, 2))
        }
        #expect(PhotoIO.orient([1, 2, 3, 4, 5, 6], channels: 1, width: 3, height: 2, orientation: 3).bytes == [6, 5, 4, 3, 2, 1])
    }

    @Test func untaggedMaskTakesThePhotosOrientation() throws {
        // A photo stored 2x1 (red, green) tagged 6: upright it is 1x2, red on top. A mask drawn on the stored pixels
        // (white over red, black over green) with no orientation of its own must cover the upright red pixel.
        let photoURL = try write([255, 0, 0, 255, 0, 255, 0, 255], width: 2, height: 1, orientation: 6)
        let rawMask = try write([255, 255, 255, 255, 0, 0, 0, 255], width: 2, height: 1, orientation: nil)
        let uprightMask = try write([255, 255, 255, 255, 0, 0, 0, 255], width: 1, height: 2, orientation: nil)
        let taggedMask = try write([255, 255, 255, 255, 0, 0, 0, 255], width: 2, height: 1, orientation: 8)
        defer { for url in [photoURL, rawMask, uprightMask, taggedMask] { try? FileManager.default.removeItem(at: url) } }
        let photo = try PhotoIO.rgb(photoURL)
        #expect(photo.orientation == 6 && photo.width == 1 && photo.height == 2)
        let raw = try PhotoIO.mask(rawMask, photo: photo)
        #expect(raw.width == 1 && raw.height == 2 && raw.bytes == [255, 0])
        // Without a photo the untagged mask stays as stored.
        #expect(try PhotoIO.mask(rawMask).width == 2)
        // Already upright: left as is.
        let upright = try PhotoIO.mask(uprightMask, photo: photo)
        #expect(upright.width == 1 && upright.height == 2 && upright.bytes == [255, 0])
        // Its own tag wins over the photo's (8 rotates the other way: the white pixel ends at the bottom).
        let tagged = try PhotoIO.mask(taggedMask, photo: photo)
        #expect(tagged.width == 1 && tagged.height == 2 && tagged.bytes == [0, 255])
    }

    @Test func lamaRefusesPhotosAbove4K() throws {
        try LaMaAdapter.checkSize(width: 3840, height: 2160)
        try LaMaAdapter.checkSize(width: 2160, height: 3840)
        #expect(throws: WorkerFailure.self) { try LaMaAdapter.checkSize(width: 4000, height: 3000) }
        #expect(throws: WorkerFailure.self) { try LaMaAdapter.checkSize(width: 15, height: 1000) }
        do { try LaMaAdapter.checkSize(width: 4000, height: 3000) } catch let failure as WorkerFailure {
            #expect(failure.code == "invalid_request" && failure.message.contains("8.3 megapixels") && failure.message.contains("4000×3000 (12.0 MP)"))
        }
    }
}
