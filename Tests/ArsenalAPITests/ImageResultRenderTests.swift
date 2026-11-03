// SPDX-License-Identifier: MIT
import CoreGraphics
import Foundation
import ImageIO
import MCP
import Testing
import UniformTypeIdentifiers
@testable import ArsenalAPI
@testable import ArsenalCore

/// An image model's schema-3 `files` result as the agent receives it (Qwen-Image's generate/edit shape): every file as a path
/// line with its size, a resource link and a thumbnail, and transparency kept in the thumbnail of an RGBA PNG.
@Suite struct ImageResultRenderTests {
    /// A PNG whose left half is transparent (RGBA) or fully opaque (RGB).
    func png(_ url: URL, width: Int, height: Int, transparent: Bool) throws {
        let channels = transparent ? 4 : 3
        var bytes = [UInt8](repeating: 0, count: width * height * channels)
        for y in 0..<height {
            for x in 0..<width {
                let i = (y * width + x) * channels
                bytes[i] = 200; bytes[i + 1] = UInt8(x % 256); bytes[i + 2] = UInt8(y % 256)
                if transparent { bytes[i + 3] = x < width / 2 ? 0 : 255 }
            }
        }
        let info = CGBitmapInfo(rawValue: (transparent ? CGImageAlphaInfo.last : CGImageAlphaInfo.none).rawValue)
        let image = try #require(
            CGImage(
                width: width, height: height, bitsPerComponent: 8, bitsPerPixel: 8 * channels, bytesPerRow: width * channels,
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: info, provider: CGDataProvider(data: Data(bytes) as CFData)!, decode: nil,
                shouldInterpolate: false, intent: .defaultIntent))
        let destination = try #require(CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil))
        CGImageDestinationAddImage(destination, image, nil)
        #expect(CGImageDestinationFinalize(destination))
    }

    @Test func imagesResultRendersPathsLinksAndAlphaThumbnails() async throws {
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-image-render-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let job = support.appendingPathComponent("jobs/job-1")
        try FileManager.default.createDirectory(at: job, withIntermediateDirectories: true)
        let sticker = job.appendingPathComponent("image-1-00000007.png"), photo = job.appendingPathComponent("image-2-00000008.png")
        try png(sticker, width: 512, height: 256, transparent: true)
        try png(photo, width: 512, height: 256, transparent: false)
        func file(_ url: URL, alpha: Bool) throws -> JSONValue {
            [
                "id": .string(url.lastPathComponent), "path": .string(url.path), "mime": "image/png", "width": 512, "height": 256, "alpha": .bool(alpha),
                "bytes": .number(Double(try FileDigest.bytes(url))), "sha256": .string(try FileDigest.sha256(url))
            ]
        }
        let result: JSONObject = ["files": [try file(sticker, alpha: true), try file(photo, alpha: false)], "seed": 7, "seeds": [7, 8], "seconds": 12.5]
        let router = APIRouter(service: StubModelsService(), support: support)
        let entry = CatalogEntry(row: ModelRow(id: "qwen-image-2.1", name: "Qwen-Image-2.1", kind: "images"))
        let operation = ModelOperation(name: "edit", title: "Edit", inputSchema: ["type": "object"], output: "image", stream: false)
        let rendered = try await router.mcpRenderFiles(
            result, entry: entry, operation: operation, seconds: 12.5, directory: job, requester: "Test", voice: nil, play: false)
        let lines = rendered.text.components(separatedBy: "\n")
        #expect(lines.count == 3 && lines[0].hasPrefix("Created 2 images with qwen-image-2.1 (") && lines[0].hasSuffix(", seed 7:"), "\(rendered.text)")
        #expect(lines[1].hasPrefix(sticker.path + " (512×256, PNG, ") && lines[2].hasPrefix(photo.path + " (512×256, PNG, "))
        // One resource link and one thumbnail per image; the RGBA image's thumbnail keeps its transparent half.
        let thumbnails = rendered.extra.compactMap { content -> Data? in
            if case .image(let data, _, _, _) = content { return Data(base64Encoded: data) }
            return nil
        }
        let links = rendered.extra.filter {
            if case .resourceLink = $0 { return true }; return false
        }
        #expect(thumbnails.count == 2 && links.count == 2)
        let thumbnail = try #require(CGImageSourceCreateImageAtIndex(CGImageSourceCreateWithData(thumbnails[0] as CFData, nil)!, 0, nil))
        #expect(thumbnail.width == 256 && thumbnail.height == 128)
        #expect([.first, .last, .premultipliedFirst, .premultipliedLast].contains(thumbnail.alphaInfo))
        // The JSON form keeps the model's own fields (alpha, seeds) next to the service's file metadata.
        let files = rendered.json["files"]?.array ?? []
        #expect(files.count == 2 && files[0]["alpha"]?.bool == true && files[1]["alpha"]?.bool == false && rendered.json["seeds"]?.array?.count == 2)
        #expect(try String(contentsOf: job.appendingPathComponent("output.txt"), encoding: .utf8) == rendered.text)
    }
}
