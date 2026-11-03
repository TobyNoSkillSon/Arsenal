// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import UniformTypeIdentifiers

public enum PNG {
    public struct WriteError: LocalizedError { public let message: String; public var errorDescription: String? { message } }

    /// RGB8 row-major pixels → PNG file (sRGB, no alpha).
    public static func write(_ pixels: [UInt8], width: Int, height: Int, to url: URL) throws {
        guard pixels.count == width * height * 3 else { throw WriteError(message: "Pixel buffer does not match \(width)×\(height).") }
        let data = CFDataCreate(nil, pixels, pixels.count)!
        guard let provider = CGDataProvider(data: data),
            let image = CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 24, bytesPerRow: width * 3,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue),
                provider: provider, decode: nil, shouldInterpolate: false, intent: .defaultIntent),
            let destination = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil)
        else { throw WriteError(message: "Could not create the PNG encoder.") }
        CGImageDestinationAddImage(destination, image, nil)
        guard CGImageDestinationFinalize(destination) else { throw WriteError(message: "Could not write \(url.lastPathComponent).") }
    }
}
