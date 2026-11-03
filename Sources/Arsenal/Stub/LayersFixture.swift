// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore

// A deterministic new-kind fixture for the generic tool: `local.layers-fixture` with one operation, `separate`, that
// takes an image and a layer count and returns that many ordered RGBA PNGs with labels and depth ranges. It is protocol
// and UI evidence for a layers-shaped result, not a depth model: the layers are drawn shapes, whatever the input.

enum LayersFixture {
    static let id = "layers-fixture"
    static let kind = KindSpec(
        id: "local.layers-fixture", title: "Layers", order: 9, columns: [KindColumn(id: "seconds", title: "Seconds", unit: "s")], icon: "square.3.layers.3d",
        titleSingular: "Layers",
        operations: [OpDeclaration(name: "separate", title: "Separate Layers", input: "", output: "file", stream: false, defaultOp: true)])

    static var row: ModelRow {
        var row = ModelRow(
            id: id, name: "Layer Fixture", kind: kind.id, params: "", licence: "MIT", precisions: ["fp16"], states: ["fp16": PrecisionState(state: .loaded)],
            selectedPrecision: "fp16", fast: false, runner: "native", summary: "Deterministic layer separator for protocol and UI tests.", runnerLanguage: "swift",
            fastExact: false)
        row.tools = ["operation"]; row.local = true; row.activity = "default"
        return row
    }

    /// The folder fields the stub service reads (its tools).
    static let metadata: [String: JSONObject] = [id: ["tools": ["operation"]]]

    static let input: JSONObject = [
        "type": "object", "required": ["image"], "additionalProperties": false,
        "properties": [
            "image": [
                "type": "object", "title": "Image", "required": ["path"], "additionalProperties": false, "properties": ["path": ["type": "string"]],
                "x-arsenal-file": ["accept": ["image/png", "image/jpeg"]]
            ],
            "layers": ["type": "integer", "title": "Layers", "minimum": 2, "maximum": 8, "default": 4],
            "fill_hidden": ["type": "boolean", "title": "Fill hidden areas", "default": false]
        ]
    ]
    static let output: JSONObject = [
        "type": "object", "required": ["files", "metadata"],
        "properties": [
            "files": [
                "type": "array", "minItems": 1, "maxItems": 8,
                "items": [
                    "type": "object", "required": ["id", "path", "mime", "label", "order"],
                    "properties": [
                        "id": ["type": "string"], "path": ["type": "string"], "mime": ["const": "image/png"], "label": ["type": "string"],
                        "order": ["type": "integer"], "width": ["type": "integer"], "height": ["type": "integer"],
                        "depth_min": ["type": "number"], "depth_max": ["type": "number"]
                    ]
                ]
            ],
            "metadata": ["type": "object"]
        ]
    ]
    static let operation = ModelOperation(
        name: "separate", title: "Separate Layers", inputSchema: input, output: "file", stream: false, outputSchema: output,
        policy: OperationPolicy(timeoutS: 60, firstResponseTimeoutS: 30), schemaVersion: 3)
    static var descriptor: OperationDescriptor {
        OperationDescriptor(name: "separate", title: "Separate Layers", inputSchema: input, outputSchema: output, output: "file", stream: false, timeout: 60)
    }

    /// Layers nearest first, Background always last: "Near, Middle, Far, Background" up to four; beyond the named slots
    /// by order ("Near, 2, 3, 4, 5, Background"), never Background in the middle or an unnamed "Layer 5".
    static func names(_ count: Int) -> [String] {
        guard count > 1 else { return ["Background"].suffix(count) }
        if count <= 4 { return Array(["Near", "Middle", "Far"].prefix(count - 1)) + ["Background"] }
        return ["Near"] + (2..<count).map(String.init) + ["Background"]
    }

    /// The runner's reply: `count` PNGs in `directory`, nearest first.
    static func run(params: JSONObject, directory: URL, size: Int = 256) throws -> JSONObject {
        let count = Int(params["layers"]?.number ?? 4)
        guard (2...8).contains(count) else { throw WorkerFailure(code: "invalid_request", message: "layers must be 2 to 8.") }
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        var files: [JSONValue] = []
        let names = Self.names(count)
        for i in 0..<count {
            let url = directory.appendingPathComponent("layer-\(i).png")
            try png(layer: i, of: count, size: size).write(to: url, options: .atomic)
            let lo = Double(i) / Double(count), hi = Double(i + 1) / Double(count)
            files.append(
                [
                    "id": .string("layer-\(i)"), "path": .string(url.path), "mime": "image/png", "label": .string(names[i]), "order": .number(Double(i)),
                    "width": .number(Double(size)), "height": .number(Double(size)), "depth_min": .number(lo), "depth_max": .number(hi)
                ])
        }
        return ["files": .array(files), "metadata": ["depth_units": "relative", "order": "near-to-far", "dimensions": ["width": 256, "height": 256]]]
    }

    /// One layer: a shape of its own colour on transparency, larger and further back as the index grows.
    static func png(layer i: Int, of count: Int, size: Int) throws -> Data {
        guard
            let rep = NSBitmapImageRep(
                bitmapDataPlanes: nil, pixelsWide: size, pixelsHigh: size, bitsPerSample: 8, samplesPerPixel: 4, hasAlpha: true, isPlanar: false,
                colorSpaceName: .deviceRGB, bytesPerRow: 0, bitsPerPixel: 0), let context = NSGraphicsContext(bitmapImageRep: rep)
        else { throw WorkerFailure(code: "protocol", message: "Could not draw a fixture layer.") }
        NSGraphicsContext.saveGraphicsState(); defer { NSGraphicsContext.restoreGraphicsState() }
        NSGraphicsContext.current = context
        let s = CGFloat(size)
        let hues: [CGFloat] = [0.02, 0.12, 0.55, 0.62, 0.3, 0.8, 0.45, 0.9]
        NSColor(calibratedHue: hues[i % hues.count], saturation: 0.55, brightness: 0.9, alpha: 1).setFill()
        if i == count - 1 {
            NSBezierPath(rect: NSRect(x: 0, y: 0, width: s, height: s * 0.45)).fill()
        } else {
            let r = s * min(0.4, 0.14 + 0.05 * CGFloat(i))
            let x = s * (0.3 + 0.2 * CGFloat(i % 3)), y = s * (0.65 - 0.06 * CGFloat(i))
            let centre = NSPoint(x: min(max(x, r), s - r), y: min(max(y, r), s - r))
            NSBezierPath(ovalIn: NSRect(x: centre.x - r, y: centre.y - r, width: 2 * r, height: 2 * r)).fill()
        }
        guard let data = rep.representation(using: .png, properties: [:]) else { throw WorkerFailure(code: "protocol", message: "Could not encode a fixture layer.") }
        return data
    }
}
