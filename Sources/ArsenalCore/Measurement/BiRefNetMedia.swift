// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ImageIO

extension ModelIntakeCommand {
    static func biRefNetImages(in directory: URL) throws -> [URL] {
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        return try (0..<3).map { index in
            let w = 512, h = 384
            guard
                let context = CGContext(
                    data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4,
                    space: CGColorSpaceCreateDeviceRGB(), bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { throw ManifestError.invalid("Cannot create fixed image") }
            context.setFillColor(CGColor(red: 0.82, green: 0.86, blue: 0.9, alpha: 1))
            context.fill(CGRect(x: 0, y: 0, width: w, height: h))
            if index == 2 { context.clear(CGRect(x: 0, y: 0, width: w, height: h)) }
            context.setFillColor(CGColor(red: 0.12, green: 0.3, blue: 0.18, alpha: index == 2 ? 0.5 : 1))
            context.fillEllipse(in: CGRect(x: 140, y: 55, width: 220, height: 280))
            if index == 1 {
                context.setStrokeColor(CGColor(red: 0.05, green: 0.1, blue: 0.04, alpha: 1)); context.setLineWidth(2)
                for i in 0..<16 {
                    context.move(to: CGPoint(x: 250, y: 180)); context.addLine(to: CGPoint(x: 20 + i * 30, y: 350)); context.strokePath()
                }
            }
            guard let image = context.makeImage() else { throw ManifestError.invalid("Cannot render fixed image") }
            let file = directory.appendingPathComponent("birefnet-fixed-\(index).png")
            guard let destination = CGImageDestinationCreateWithURL(file as CFURL, "public.png" as CFString, 1, nil) else {
                throw ManifestError.invalid("Cannot save fixed image")
            }
            CGImageDestinationAddImage(destination, image, nil)
            guard CGImageDestinationFinalize(destination) else { throw ManifestError.invalid("Cannot finalize fixed image") }
            return file
        }
    }
    static func biRefNetMaskValues(_ reply: JSONObject) throws -> (shape: [Int], values: [UInt8]) {
        guard let path = reply["files"]?.array?.first(where: { $0["id"] == "mask" })?["path"]?.string,
            let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw ManifestError.invalid("Missing mask PNG") }
        var pixels = [UInt8](repeating: 0, count: image.width * image.height)
        try pixels.withUnsafeMutableBytes { buffer in
            guard
                let context = CGContext(
                    data: buffer.baseAddress, width: image.width, height: image.height, bitsPerComponent: 8, bytesPerRow: image.width,
                    space: CGColorSpaceCreateDeviceGray(), bitmapInfo: CGImageAlphaInfo.none.rawValue)
            else { throw ManifestError.invalid("Cannot decode mask") }
            context.draw(image, in: CGRect(x: 0, y: 0, width: image.width, height: image.height))
        }
        return ([image.height, image.width], pixels)
    }
    static func biRefNetMaskMAE(reference: [UInt8], candidate: [UInt8]) throws -> Double {
        guard !reference.isEmpty, reference.count == candidate.count else { throw ManifestError.invalid("Mask count differs") }
        return zip(reference, candidate).reduce(0) { $0 + abs(Double($1.0) - Double($1.1)) } / (255 * Double(reference.count))
    }
    static func biRefNetMaskDiagnostic(reference: JSONObject, candidate: JSONObject) throws -> JSONObject {
        let a = try biRefNetMaskValues(reference), b = try biRefNetMaskValues(candidate)
        guard a.shape == b.shape else { throw ManifestError.invalid("Mask shape differs") }
        let value = try biRefNetMaskMAE(reference: a.values, candidate: b.values)
        return [
            "diagnostic_only": true, "affects_precision_gate": false, "mean_absolute_error": .number(value),
            "bound": .number(biRefNetMaskMAEBound), "passed": .bool(value <= biRefNetMaskMAEBound), "shape": .array(a.shape.map { .number(Double($0)) })
        ]
    }
    static func biRefNetSaveSanity(reply: JSONObject, to directory: URL, label: String) throws {
        for file in reply["files"]?.array ?? [] {
            guard let path = file["path"]?.string, let name = file["id"]?.string else { throw ManifestError.invalid("Invalid sanity output") }
            try FileManager.default.copyItem(at: URL(fileURLWithPath: path), to: directory.appendingPathComponent(label + "-" + name + ".png"))
        }
    }
}
