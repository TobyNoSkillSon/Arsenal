// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import CoreImage.CIFilterBuiltins
import CoreGraphics
import ImageIO
import UniformTypeIdentifiers
import IntegrationKit
import ArsenalWire

/// Coordinates are top-left pixels at the API boundary; Core Image crops use bottom-left pixels.
enum DocumentRegions {
    static func crop(_ image: CIImage, box: CGRect, polygon: [CGPoint] = [], angle: Int = 0) throws -> CIImage {
        let bounds = CGRect(origin: .zero, size: image.extent.size)
        guard box.width >= 1, box.height >= 1, bounds.contains(box) else { throw invalid("Invalid document crop.") }
        let rect = CGRect(x: image.extent.minX + box.minX, y: image.extent.maxY - box.maxY, width: box.width, height: box.height)
        var crop = image.cropped(to: rect).transformed(by: CGAffineTransform(translationX: -rect.minX, y: -rect.minY))
        if !polygon.isEmpty {
            let width = Int(box.width), height = Int(box.height)
            guard
                let context = CGContext(
                    data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: width * 4, space: CGColorSpaceCreateDeviceRGB(),
                    bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)
            else { throw failed("Cannot allocate polygon crop.") }
            context.setFillColor(CGColor(gray: 0, alpha: 1)); context.fill(CGRect(x: 0, y: 0, width: width, height: height))
            context.beginPath()
            for (index, point) in polygon.enumerated() {
                let mapped = CGPoint(x: point.x - box.minX, y: box.maxY - point.y)
                if index == 0 { context.move(to: mapped) } else { context.addLine(to: mapped) }
            }
            context.closePath(); context.clip()
            guard let cg = CIContext().createCGImage(crop, from: crop.extent) else { throw failed("Cannot render polygon crop.") }
            context.draw(cg, in: CGRect(x: 0, y: 0, width: width, height: height))
            guard let masked = context.makeImage() else { throw failed("Cannot finish polygon crop.") }; crop = CIImage(cgImage: masked)
        }
        if angle != 0 {
            crop = crop.transformed(by: CGAffineTransform(rotationAngle: CGFloat(angle) * .pi / 180))
            crop = crop.transformed(by: CGAffineTransform(translationX: -crop.extent.minX, y: -crop.extent.minY))
        }
        return crop
    }
    /// PIL `Image.resize((width, height), BICUBIC)` on the sRGB bytes: Keys a = -0.5, support widened on downscales
    /// (antialiased) and window weights renormalized at the edges. Returns an opaque sRGB image at the origin.
    static func resize(_ image: CIImage, width: Int, height: Int) -> CIImage {
        ImagePixels.ciImage(PILResample.resize(ImagePixels.rgb(image), width: width, height: height, filter: .bicubic))
    }
    /// NaviDC-OCR resize_by_need: pad strips beyond 50:1 with white (centered), then upscale a short edge below 28 px.
    static func recognitionCrop(_ image: CIImage) -> CIImage {
        var crop = image.transformed(by: CGAffineTransform(translationX: -image.extent.minX, y: -image.extent.minY))
        let width = crop.extent.width.rounded(), height = crop.extent.height.rounded()
        guard width >= 1, height >= 1 else { return crop }
        if max(width, height) / min(width, height) > 50 {
            let paddedWidth = width > height ? width : (height / 50).rounded(.up), paddedHeight = width > height ? (width / 50).rounded(.up) : height
            // PIL pastes at the top-left offset; Core Image places the same pixels from the bottom edge.
            let left = ((paddedWidth - width) / 2).rounded(.towardZero), top = ((paddedHeight - height) / 2).rounded(.towardZero)
            let canvas = CGRect(x: 0, y: 0, width: paddedWidth, height: paddedHeight)
            crop = crop.cropped(to: CGRect(x: 0, y: 0, width: width, height: height))
                .transformed(by: CGAffineTransform(translationX: left, y: paddedHeight - height - top))
                .composited(over: CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: canvas)).cropped(to: canvas)
        }
        let short = min(crop.extent.width, crop.extent.height)
        if short < 28 {
            let scale = 28 / short
            crop = resize(crop, width: Int((crop.extent.width * scale).rounded(.up)), height: Int((crop.extent.height * scale).rounded(.up)))
        }
        return crop
    }
    static func figures(page: inout JSONObject, image: CIImage, root: URL, cancellation: NativeCancellation) throws -> [JSONValue] {
        var files: [JSONValue] = [], figures: [JSONValue] = []
        let markdown = page["markdown"]?.string ?? ""
        let regex = try NSRegularExpression(pattern: #"<img src="images/bbox_\d+_\d+_\d+_\d+\.jpg"\s*/>"#)
        let sourceTags = regex.matches(in: markdown, range: NSRange(markdown.startIndex..., in: markdown)).map { (markdown as NSString).substring(with: $0.range) }
        var linkedMarkdown = markdown
        for (index, item) in (page["images"]?.array ?? []).enumerated() {
            try cancellation.check()
            guard var figure = item.object, let x = figure["top_left_x"]?.number, let y = figure["top_left_y"]?.number,
                let right = figure["bottom_right_x"]?.number, let bottom = figure["bottom_right_y"]?.number
            else { throw invalid("Malformed figure coordinates.") }
            // The card's save_renderable_markdown_with_visual_regions skips empty or inverted regions.
            guard right > x, bottom > y else { continue }
            let crop = try crop(image, box: CGRect(x: x, y: y, width: right - x, height: bottom - y))
            let id = "page-\(Int(page["index"]?.number ?? 0))-figure-\(index)"
            let file = root.appendingPathComponent(id + ".png")
            guard !FileManager.default.fileExists(atPath: file.path), let cg = CIContext().createCGImage(crop, from: crop.extent),
                let destination = CGImageDestinationCreateWithURL(file as CFURL, UTType.png.identifier as CFString, 1, nil)
            else { throw failed("Cannot create figure file.") }
            CGImageDestinationAddImage(destination, cg, nil)
            guard CGImageDestinationFinalize(destination) else { throw failed("Cannot write figure file.") }
            files.append(.object(try NativeMedia.file(file, root: root, mime: "image/png", id: id)))
            figure["id"] = .string(id); figures.append(.object(figure))
            if index < sourceTags.count { linkedMarkdown = linkedMarkdown.replacingOccurrences(of: sourceTags[index], with: "<img src=\"" + file.lastPathComponent + "\" />") }
        }
        page["images"] = .array(figures)
        page["markdown"] = .string(linkedMarkdown)
        return files
    }
    /// Generated coordinates outside the page are the parser's rejection; rendering and file failures are not.
    private static func invalid(_ message: String) -> WorkerFailure { WorkerFailure(code: "invalid_model_output", message: message) }
    private static func failed(_ message: String) -> WorkerFailure { WorkerFailure(code: "output_failed", message: message) }
}
