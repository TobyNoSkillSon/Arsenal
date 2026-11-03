// SPDX-License-Identifier: MIT
import AppKit
import CryptoKit
import Foundation
import ImageIO
import ArsenalCore
import UniformTypeIdentifiers

/// A stand-in background remover for the stub service: BiRefNet's `remove_background` contract (input
/// models/birefnet/ops/remove_background.input.json, reply files `foreground` RGBA and `mask` grey at the input's size).
/// For tests, renders and the stub app; no model, no GPU.
enum CutoutFixture {
    static let id = "birefnet"

    static var row: ModelRow {
        var row = ModelRow(
            id: id, name: "BiRefNet", kind: "images", params: "220M", licence: "MIT", precisions: ["fp16"], states: ["fp16": PrecisionState(state: .ready)],
            selectedPrecision: "fp16", fast: false, runner: "native", summary: "Remove an image's background.", runnerLanguage: "swift", fastExact: false)
        row.tools = [ToolKind.background.rawValue]
        return row
    }
    /// The folder fields the stub service reads (its tools).
    static let metadata: [String: JSONObject] = [id: ["tools": [.string(ToolKind.background.rawValue)]]]

    /// models/birefnet/ops/remove_background.input.json, as the folder declares it (BackgroundToolTests compares them).
    static let input: JSONObject = [
        "$schema": "https://json-schema.org/draft/2020-12/schema", "type": "object", "additionalProperties": false, "required": ["image"],
        "properties": [
            "image": [
                "type": "object", "additionalProperties": false, "required": ["path"], "x-arsenal-file": ["accept": ["image/png", "image/jpeg"]],
                "properties": ["path": ["type": "string", "minLength": 1, "maxLength": 4096]]
            ]
        ]
    ]
    static let operation = ModelOperation(
        name: "remove_background", title: "Remove background", inputSchema: input, output: "file", stream: true,
        policy: OperationPolicy(timeoutS: 180, firstResponseTimeoutS: 180), schemaVersion: 3)
    static var descriptor: OperationDescriptor {
        OperationDescriptor(name: operation.name, title: operation.title, inputSchema: input, output: "file", stream: true, timeout: 180)
    }

    /// The runner's reply: the input with its background transparent and the alpha as a grey mask. The stand-in for
    /// segmentation: a pixel is subject where it differs from the background at the left edge of its row (a soft edge
    /// over 30 levels), which follows a figure's silhouette on a plain or vertically graded backdrop.
    static func run(params: JSONObject, directory: URL) throws -> JSONObject {
        guard let path = params["image"]?["path"]?.string, let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil),
            let image = CGImageSourceCreateImageAtIndex(source, 0, nil)
        else { throw WorkerFailure(code: "invalid_request", message: "image.path is not a readable PNG or JPEG.") }
        let w = image.width, h = image.height
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        let space = CGColorSpace(name: CGColorSpace.sRGB)!
        guard
            let rgbaContext = CGContext(
                data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w * 4, space: space, bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue),
            let grey8 = CGContext(data: nil, width: w, height: h, bitsPerComponent: 8, bytesPerRow: w, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: 0),
            let rgbaData = rgbaContext.data, let greyData = grey8.data
        else { throw WorkerFailure(code: "protocol", message: "Could not draw the fixture cut-out.") }
        rgbaContext.draw(image, in: CGRect(x: 0, y: 0, width: w, height: h))
        let rgba = rgbaData.bindMemory(to: UInt8.self, capacity: w * h * 4), grey = greyData.bindMemory(to: UInt8.self, capacity: w * h)
        let rgbaRow = rgbaContext.bytesPerRow, greyRow = grey8.bytesPerRow
        for y in 0..<h {
            let row = y * rgbaRow, n = min(3, w)
            let edge = (0..<3).map { c in (0..<n).map { Int(rgba[row + $0 * 4 + c]) }.reduce(0, +) / n }
            for x in 0..<w {
                let i = row + x * 4
                let diff = (0..<3).map { abs(Int(rgba[i + $0]) - edge[$0]) }.max() ?? 0
                let a = max(0, min(255, (diff - 18) * 255 / 30))
                grey[y * greyRow + x] = UInt8(a)
                for c in 0..<3 { rgba[i + c] = UInt8(Int(rgba[i + c]) * a / 255) }
                rgba[i + 3] = UInt8(a)
            }
        }
        var files: [JSONValue] = []
        for (order, (name, context)) in [("foreground", rgbaContext), ("mask", grey8)].enumerated() {
            let url = directory.appendingPathComponent("\(name).png")
            guard let cg = context.makeImage(), let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else {
                throw WorkerFailure(code: "protocol", message: "Could not encode the fixture cut-out.")
            }
            CGImageDestinationAddImage(dest, cg, nil)
            guard CGImageDestinationFinalize(dest) else { throw WorkerFailure(code: "protocol", message: "Could not encode the fixture cut-out.") }
            let data = try Data(contentsOf: url)
            files.append(
                [
                    "id": .string(name), "path": .string(url.path), "mime": "image/png", "width": .number(Double(w)), "height": .number(Double(h)),
                    "order": .number(Double(order)), "label": .string(name == "foreground" ? "Foreground" : "Mask"), "bytes": .number(Double(data.count)),
                    "sha256": .string(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
                ])
        }
        return ["files": .array(files), "metadata": ["model": .string(id), "engine": "fixture"]]
    }
}
