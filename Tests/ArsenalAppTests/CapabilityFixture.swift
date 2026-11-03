// SPDX-License-Identifier: MIT
import AVFoundation
import AppKit
import CryptoKit
import Foundation
import ImageIO
@testable import Arsenal
import ArsenalCore
import UniformTypeIdentifiers

/// The capability windows' fixtures: a model per window declaring that window's operations with the input and output
/// schemas read from the repository's kinds (Resources/kinds/<kind>/ops), stand-in runners whose replies follow those
/// output schemas (files written into the job's folder with ids, as a runner writes them), and media to feed them
/// (photos, clips, videos, PDFs). No model, no weights, no GPU.
enum CapabilityFixture {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    /// Each operation's kind, title, output and stream, as the kinds' TOML declares them.
    static let declared: [String: (kind: String, title: String, output: String, stream: Bool)] = [
        "transcribe": ("audio", "Transcribe", "json", true), "diarize": ("audio", "Who spoke when", "json", false),
        "end_of_turn": ("audio", "End of turn", "json", false), "separate_audio": ("audio", "Separate audio", "file", true),
        "translate": ("chat", "Translate", "json", false), "read_document": ("chat", "Read document", "json", false), "extract": ("chat", "Extract", "json", false),
        "embed": ("embeddings", "Embed", "json", false), "rerank": ("embeddings", "Rerank", "json", false),
        "detect": ("perception", "Find objects", "json", false), "ground_screen": ("perception", "Ground screen", "json", false),
        "segment": ("perception", "Select and segment", "file", false), "segment_session": ("perception", "Image session", "json", false),
        "track_video": ("perception", "Track selection", "file", true), "video_session": ("perception", "Create or close a video session", "json", false),
        "estimate_depth": ("perception", "Depth", "file", false),
        "upscale_image": ("restoration", "Upscale image", "file", true), "upscale_video": ("restoration", "Upscale video", "file", true),
        "interpolate_video": ("restoration", "Smooth video", "file", true), "image_to_3d": ("geometry", "Make 3D", "file", true)
    ]

    /// A schema from Resources/kinds/<kind>/ops.
    static func schema(_ op: String, _ part: String) -> JSONObject {
        guard let kind = declared[op]?.kind else { return [:] }
        let url = repo.appendingPathComponent("Resources/kinds/\(kind)/ops/\(op).\(part).json")
        guard let data = try? Data(contentsOf: url), case .object(let object)? = try? JSONDecoder().decode(JSONValue.self, from: data) else { return [:] }
        return object
    }
    static func operation(_ op: String, input: JSONObject? = nil) -> ModelOperation {
        let d = declared[op]!
        return ModelOperation(
            name: op, title: d.title, inputSchema: input ?? schema(op, "input"), output: d.output, stream: d.stream, outputSchema: schema(op, "output"),
            policy: OperationPolicy(timeoutS: 120, firstResponseTimeoutS: 120), schemaVersion: 3)
    }
    static func descriptor(_ op: String, input: JSONObject? = nil) -> OperationDescriptor {
        let o = operation(op, input: input)
        return OperationDescriptor(name: o.name, title: o.title, inputSchema: o.inputSchema, outputSchema: o.outputSchema, output: o.output, stream: o.stream, timeout: 120)
    }

    /// One fixture model per window, declaring the window's operations (as a schema-3 folder would, `tools` absent).
    /// Make 3D's model answers with a splat (PLY and preview) instead of a GLB mesh.
    nonisolated(unsafe) static var splatResult = false
    static let models: [(id: String, name: String, kind: String, ops: [String])] = [
        ("whisper-fixture", "Whisper Fixture", "audio", ["transcribe", "diarize"]),
        ("smart-turn-fixture", "Smart Turn Fixture", "audio", ["end_of_turn"]),
        ("demucs-fixture", "Demucs Fixture", "audio", ["separate_audio"]),
        ("translate-fixture", "Translate Fixture", "chat", ["translate"]),
        ("ocr-fixture", "OCR Fixture", "chat", ["read_document"]),
        ("extract-fixture", "Extract Fixture", "chat", ["extract"]),
        ("rerank-fixture", "Rerank Fixture", "embeddings", ["rerank"]),
        ("embed-fixture", "Embed Fixture", "embeddings", ["embed"]),
        ("sam-fixture", "SAM Fixture", "perception", ["segment", "segment_session", "track_video", "video_session"]),
        ("dino-fixture", "Grounding Fixture", "perception", ["detect", "ground_screen"]),
        ("depth-fixture", "Depth Fixture", "perception", ["estimate_depth"]),
        ("upscale-fixture", "Upscale Fixture", "restoration", ["upscale_image", "upscale_video"]),
        ("rife-fixture", "RIFE Fixture", "restoration", ["interpolate_video"]),
        ("trellis-fixture", "TRELLIS Fixture", "geometry", ["image_to_3d"])
    ]
    static func row(_ id: String, state: ModelState = .ready, loaded: Bool = false) -> ModelRow {
        let m = models.first { $0.id == id }!
        var row = ModelRow(
            id: m.id, name: m.name, kind: m.kind, params: "", licence: "MIT", precisions: ["fp16"], states: ["fp16": PrecisionState(state: loaded ? .loaded : state)],
            selectedPrecision: "fp16", fast: false, runner: "native", summary: "", runnerLanguage: "swift", fastExact: true)
        row.tools = [ToolKind.operation.rawValue]
        return row
    }
    static var rows: [ModelRow] { models.map { row($0.id) } }
    static var metadata: [String: JSONObject] { Dictionary(uniqueKeysWithValues: models.map { ($0.id, ["tools": ["operation"]]) }) }
    static var operations: [String: [ModelOperation]] { Dictionary(uniqueKeysWithValues: models.map { m in (m.id, m.ops.map { operation($0) }) }) }
    static func choice(_ id: String, resident: Bool = false, unavailable: OperationProblem? = nil) -> ModelChoice {
        ModelChoice(model: ToolModel(row(id)), resident: resident, unavailable: unavailable)
    }

    /// A service with the fixture models and every runner installed.
    static func service(support: URL, rows extra: [ModelRow] = []) async -> FixtureService {
        let service = FixtureService(rows: rows + extra, metadata: metadata, supportDirectory: support, operations: operations)
        await install(service)
        return service
    }
    static func install(_ service: FixtureService) async {
        for op in declared.keys { await service.setRunner(op, { try run(op, params: $0) }) }
    }

    // MARK: Runners

    /// A schema-valid reply for each operation.
    @Sendable static func run(_ op: String, params: JSONObject) throws -> JSONObject {
        let dir = params["out_dir"]?.string.map { URL(fileURLWithPath: $0) }
        switch op {
        case "transcribe":
            let speakers = params["diarize"]?.bool == true
            func segment(_ start: Double, _ end: Double, _ text: String, _ speaker: String) -> JSONValue {
                var s: JSONObject = ["start": .number(start), "end": .number(end), "text": .string(text)]
                if speakers { s["speaker"] = .string(speaker) }
                return .object(s)
            }
            var reply: JSONObject = [
                "text": "Thanks for coming in. Let's look at the numbers. Revenue is up eight percent.", "language": "en", "duration": 9.5,
                "segments": [
                    segment(0, 2.4, "Thanks for coming in.", "SPEAKER_00"), segment(2.6, 5.1, "Let's look at the numbers.", "SPEAKER_01"),
                    segment(5.4, 9.5, "Revenue is up eight percent.", "SPEAKER_00")
                ]
            ]
            if params["word_timestamps"]?.bool == true {
                reply["words"] = [["word": "Thanks", "start": 0, "end": 0.4], ["word": "for", "start": 0.4, "end": 0.6], ["word": "coming", "start": 0.6, "end": 1.0]]
            }
            return reply
        case "diarize":
            return [
                "segments": [
                    ["start": 0, "end": 2.4, "speaker": "SPEAKER_00"], ["start": 2.6, "end": 5.1, "speaker": "SPEAKER_01"], ["start": 5.4, "end": 9.5, "speaker": "SPEAKER_00"]
                ],
                "speakers": ["SPEAKER_00", "SPEAKER_01"]
            ]
        case "end_of_turn": return ["prediction": true, "probability": 0.91]
        case "separate_audio":
            let stems = params["stems"]?.array?.compactMap(\.string) ?? (params["prompt"]?.string.map { _ in ["target", "residual"] } ?? ["vocals", "drums", "bass", "other"])
            let files = try stems.enumerated().map { i, stem in
                try file(dir!, name: stem, ext: "wav", mime: "audio/wav", id: stem) { try tone($0, frequency: 220 * Double(i + 1), seconds: 1.2) }
            }
            return ["files": .array(files)]
        case "translate":
            let texts = params["text"]?.array?.compactMap(\.string) ?? []
            let target = params["target_lang"]?.string ?? "de"
            return ["translations": .array(texts.map { ["text": .string("[\(target)] " + $0), "detected_source_language": "en"] })]
        case "read_document":
            return [
                "pages": [
                    [
                        "index": 0,
                        "markdown":
                            "# Quarterly report\n\nRevenue grew **8%** on the year.\n\n| Region | Q1 | Q2 |\n|---|---|---|\n| North | 12 | 14 |\n| South | 9 | 11 |\n\n- Costs flat\n- Margin up",
                        "dimensions": ["width": 1240, "height": 1754, "dpi": 150]
                    ],
                    ["index": 1, "markdown": "## Outlook\n\nThe second half depends on the new line."]
                ],
                "model": "ocr-fixture"
            ]
        case "extract":
            if let schema = params["json_schema"]?.object, let properties = schema["properties"]?.object {
                var data: JSONObject = [:]
                for (key, p) in properties {
                    switch p["type"]?.string {
                    case "number", "integer": data[key] = 42
                    case "boolean": data[key] = true
                    case "array": data[key] = ["first", "second"]
                    default: data[key] = .string("\(key) value")
                    }
                }
                return ["data": .object(data)]
            }
            let labels = params["labels"]?.array?.compactMap(\.string) ?? []
            return [
                "entities": .array(
                    labels.enumerated().map {
                        ["label": .string($0.element), "text": .string("Ada Lovelace"), "start": 0, "end": 12, "score": .number(0.9 - Double($0.offset) * 0.1)]
                    })
            ]
        case "embed":
            let inputs = params["input"]?.array ?? []
            return ["embeddings": .array(inputs.map { .array(vector($0).map(JSONValue.number)) }), "input_tokens": .number(Double(inputs.count * 8))]
        case "rerank":
            let query = params["query"]?.string ?? ""
            let docs = params["documents"]?.array?.compactMap(\.string) ?? []
            let scores = docs.map { overlap(query, $0) }
            let order = scores.indices.sorted { scores[$0] > scores[$1] }
            return ["results": .array(order.map { ["index": .number(Double($0)), "relevance_score": .number(scores[$0])] })]
        case "detect":
            let labels = params["labels"]?.array?.compactMap(\.string) ?? [params["phrase"]?.string ?? "object"]
            return [
                "coordinate_system": "normalized",
                "detections": .array(
                    labels.enumerated().map { i, label in
                        let x = 0.12 + Double(i) * 0.28
                        return [
                            "label": .string(label), "score": .number(0.92 - Double(i) * 0.15), "box": ["xmin": .number(x), "ymin": 0.25, "xmax": .number(x + 0.24), "ymax": 0.8]
                        ]
                    })
            ]
        case "ground_screen":
            let size = imageSize(params["image"]?["path"]?.string) ?? PixelSize(width: 1440, height: 900)
            return [
                "coordinate_system": "pixels", "image_size": ["width": .number(Double(size.width)), "height": .number(Double(size.height))],
                "detections": [
                    [
                        "label": .string(params["instruction"]?.string ?? "target"), "score": 0.88,
                        "point": ["x": .number(Double(size.width) * 0.7), "y": .number(Double(size.height) * 0.2)]
                    ]
                ]
            ]
        case "segment_session", "video_session":
            if params["action"]?.string == "close" { return ["session_id": params["session_id"] ?? "closed", "closed": true] }
            return ["session_id": .string("\(op)-\(UUID().uuidString.prefix(8))")]
        case "segment":
            let path = params["image"]?["path"]?.string
            let size = imageSize(path) ?? PixelSize(width: 600, height: 400)
            let point = SpatialValue.points(params["points"].flatMap(OperationForm.pretty) ?? "").first
            let box = SpatialValue.boxes(params["boxes"].flatMap(OperationForm.pretty) ?? "").first
            let centre = point.map { CGPoint(x: $0.x, y: $0.y) } ?? box.map { CGPoint(x: ($0.xmin + $0.xmax) / 2, y: ($0.ymin + $0.ymax) / 2) } ?? CGPoint(x: 0.5, y: 0.5)
            let count = params["multimask"]?.bool == true ? 3 : 1
            let files = try (0..<count).map { i in
                try file(dir!, name: "mask-\(i)", ext: "png", mime: "image/png", id: "mask-\(i)", size: size) {
                    try mask($0, size: size, centre: centre, radius: 0.18 + Double(i) * 0.07)
                }
            }
            var reply: JSONObject = ["files": .array(files), "scores": .array((0..<count).map { .number(0.95 - Double($0) * 0.1) })]
            if let session = params["session_id"] { reply["session_id"] = session }
            return reply
        case "estimate_depth":
            let paths = params["images"]?.array?.compactMap { $0["path"]?.string } ?? [params["image"]?["path"]?.string].compactMap { $0 }
            let files = try paths.enumerated().map { i, path in
                let size = imageSize(path) ?? PixelSize(width: 640, height: 480)
                return try file(dir!, name: "depth-\(i)", ext: "png", mime: "image/png", id: "depth-\(i)", size: size) { try depth($0, size: size) }
            }
            let metric = params["mode"]?.string == "metric"
            var reply: JSONObject = ["files": .array(files), "mode": metric ? "metric" : "relative", "depth_unit": metric ? "meters" : "relative"]
            if params["intrinsics"]?.bool == true { reply["intrinsics"] = [[812, 0, 320], [0, 812, 240], [0, 0, 1]] }
            return reply
        case "upscale_image":
            let path = params["image"]?["path"]?.string ?? ""
            let scale = params["scale"]?.number.flatMap(\.roundedInt) ?? 2
            let size = imageSize(path).map { PixelSize(width: $0.width * scale, height: $0.height * scale) } ?? PixelSize(width: 800, height: 600)
            return ["files": [try file(dir!, name: "upscaled", ext: "png", mime: "image/png", id: "image", size: size) { try scaled(path, to: $0, size: size) }]]
        case "upscale_video", "track_video", "interpolate_video":
            if op == "interpolate_video", params["frames"] != nil {
                let frames = try (0..<3).map { i in
                    try file(dir!, name: "frame-\(i)", ext: "png", mime: "image/png", id: "frame-\(i)", size: PixelSize(width: 320, height: 180)) {
                        try picture($0, width: 320, height: 180)
                    }
                }
                return ["files": .array(frames)]
            }
            let video = try file(dir!, name: op, ext: "mp4", mime: "video/mp4", id: op == "track_video" ? "masks" : "video") {
                try VideoMaker.write($0, frames: 12, width: 320, height: 180, mask: op == "track_video")
            }
            return ["files": [video]]
        case "image_to_3d":
            if splatResult { return try splat(dir!) }
            return ["files": [try file(dir!, name: "model", ext: "glb", mime: "model/gltf-binary", id: "mesh") { try GLBMaker.cube($0) }]]
        default:
            throw WorkerFailure(code: "unsupported_op", message: "The fixture does not run \(op).")
        }
    }

    /// A splat model's answer: the PLY and its rendered preview.
    static func splat(_ dir: URL) throws -> JSONObject {
        let ply = try file(dir, name: "splat", ext: "ply", mime: "application/x-ply", id: "splat") { url in
            var data = Data("ply\nformat binary_little_endian 1.0\nelement vertex 262144\nproperty float x\nproperty float f_dc_0\nend_header\n".utf8)
            data.append(Data(count: 8))
            try data.write(to: url)
        }
        let size = PixelSize(width: 768, height: 768)
        let preview = try file(dir, name: "preview", ext: "png", mime: "image/png", id: "preview", size: size) { url in
            // ARSENAL_QA_SPLAT_PREVIEW: a real model's preview, for a render of what a person sees
            if let real = ProcessInfo.processInfo.environment["ARSENAL_QA_SPLAT_PREVIEW"] {
                try FileManager.default.copyItem(at: URL(fileURLWithPath: real), to: url)
            } else {
                try picture(url, width: 768, height: 768)
            }
        }
        return ["files": [ply, preview]]
    }

    // MARK: Files

    /// A file the runner made: written by `write`, described as a runner describes it.
    static func file(_ dir: URL, name: String, ext: String, mime: String, id: String, size: PixelSize? = nil, write: (URL) throws -> Void) throws -> JSONValue {
        try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
        let url = dir.appendingPathComponent("\(name).\(ext)")
        try write(url)
        let data = try Data(contentsOf: url)
        var entry: JSONObject = [
            "id": .string(id), "path": .string(url.path), "mime": .string(mime), "bytes": .number(Double(data.count)),
            "sha256": .string(SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined())
        ]
        if let size { entry["width"] = .number(Double(size.width)); entry["height"] = .number(Double(size.height)) }
        return .object(entry)
    }
    static func imageSize(_ path: String?) -> PixelSize? { path.flatMap { PixelSize.read(URL(fileURLWithPath: $0)) } }

    /// A 44.1 kHz mono sine.
    static func tone(_ url: URL, frequency: Double, seconds: Double) throws {
        let format = AVAudioFormat(standardFormatWithSampleRate: 44_100, channels: 1)!
        let frames = AVAudioFrameCount(44_100 * seconds)
        let buffer = AVAudioPCMBuffer(pcmFormat: format, frameCapacity: frames)!
        buffer.frameLength = frames
        for i in 0..<Int(frames) {
            let envelope = min(1, Double(i) / 2000) * min(1, Double(Int(frames) - i) / 2000)
            buffer.floatChannelData![0][i] = Float(sin(2 * .pi * frequency * Double(i) / 44_100) * 0.4 * envelope)
        }
        let file = try AVAudioFile(forWriting: url, settings: format.settings)
        try file.write(from: buffer)
    }

    /// A white ellipse around `centre` on black, the picture's size.
    static func mask(_ url: URL, size: PixelSize, centre: CGPoint, radius: Double) throws {
        let c = CGContext(data: nil, width: size.width, height: size.height, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpaceCreateDeviceGray(), bitmapInfo: 0)!
        c.setFillColor(gray: 0, alpha: 1); c.fill(CGRect(x: 0, y: 0, width: size.width, height: size.height))
        c.setFillColor(gray: 1, alpha: 1)
        let w = Double(size.width), h = Double(size.height)
        c.fillEllipse(in: CGRect(x: centre.x * w - radius * w, y: (1 - centre.y) * h - radius * h * 1.3, width: radius * 2 * w, height: radius * 2.6 * h))
        try write(c.makeImage()!, to: url)
    }

    /// A 16-bit depth map: far at the top, near at the bottom, a near disc in the middle.
    static func depth(_ url: URL, size: PixelSize) throws {
        var pixels = [UInt16](repeating: 0, count: size.width * size.height)
        for y in 0..<size.height {
            for x in 0..<size.width {
                let dx = Double(x - size.width / 2) / Double(size.width), dy = Double(y - size.height / 2) / Double(size.height)
                let disc = dx * dx + dy * dy < 0.03 ? 0.35 : 0
                pixels[y * size.width + x] = UInt16(min(1, Double(y) / Double(size.height) * 0.65 + disc) * 65535).littleEndian
            }
        }
        let provider = CGDataProvider(data: Data(bytes: pixels, count: pixels.count * 2) as CFData)!
        let image = CGImage(
            width: size.width, height: size.height, bitsPerComponent: 16, bitsPerPixel: 16, bytesPerRow: size.width * 2, space: CGColorSpaceCreateDeviceGray(),
            bitmapInfo: CGBitmapInfo(rawValue: CGImageAlphaInfo.none.rawValue | CGBitmapInfo.byteOrder16Little.rawValue), provider: provider, decode: nil,
            shouldInterpolate: false, intent: .defaultIntent)!
        try write(image, to: url)
    }

    /// The picture at `size`.
    static func scaled(_ path: String, to url: URL, size: PixelSize) throws {
        guard let source = CGImageSourceCreateWithURL(URL(fileURLWithPath: path) as CFURL, nil), let image = CGImageSourceCreateImageAtIndex(source, 0, nil) else {
            throw WorkerFailure(code: "invalid_request", message: "image.path is not readable.")
        }
        let c = CGContext(
            data: nil, width: size.width, height: size.height, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        c.interpolationQuality = .high
        c.draw(image, in: CGRect(x: 0, y: 0, width: size.width, height: size.height))
        try write(c.makeImage()!, to: url)
    }

    static func write(_ image: CGImage, to url: URL) throws {
        guard let dest = CGImageDestinationCreateWithURL(url as CFURL, UTType.png.identifier as CFString, 1, nil) else { throw WorkerFailure(code: "protocol", message: "png") }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw WorkerFailure(code: "protocol", message: "png") }
    }

    /// A deterministic unit-ish vector for an input: letters' counts (texts sharing words point the same way).
    static func vector(_ input: JSONValue) -> [Double] {
        let text = input.string ?? input["text"]?.string ?? input["image"]?["path"]?.string ?? ""
        var v = [Double](repeating: 0.01, count: 16)
        for word in text.lowercased().split(whereSeparator: { !$0.isLetter }) { v[abs(word.hashValueStable) % 16] += 1 }
        return v
    }
    /// Shared words over the query's words.
    static func overlap(_ query: String, _ doc: String) -> Double {
        let q = Set(query.lowercased().split(whereSeparator: { !$0.isLetter })), d = Set(doc.lowercased().split(whereSeparator: { !$0.isLetter }))
        return q.isEmpty ? 0 : (Double(q.intersection(d).count) / Double(q.count)) * 0.9 + 0.05
    }

    // MARK: Inputs

    static func photo(_ url: URL, width: Int = 900, height: Int = 600) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try picture(url, width: width, height: height, type: url.pathExtension.lowercased() == "png" ? .png : .jpeg)
        return url
    }
    /// A photo-like picture: sky over sand with a figure in the middle.
    static func picture(_ url: URL, width: Int, height: Int, type: UTType = .png) throws {
        let c = CGContext(
            data: nil, width: width, height: height, bitsPerComponent: 8, bytesPerRow: 0, space: CGColorSpace(name: CGColorSpace.sRGB)!,
            bitmapInfo: CGImageAlphaInfo.premultipliedLast.rawValue)!
        let w = CGFloat(width), h = CGFloat(height)
        c.setFillColor(CGColor(red: 0.45, green: 0.66, blue: 0.9, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: w, height: h))
        c.setFillColor(CGColor(red: 0.93, green: 0.82, blue: 0.62, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: w, height: h * 0.35))
        c.setFillColor(CGColor(red: 0.85, green: 0.3, blue: 0.25, alpha: 1)); c.fillEllipse(in: CGRect(x: w * 0.3, y: -h * 0.2, width: w * 0.4, height: h * 0.75))
        c.setFillColor(CGColor(red: 0.96, green: 0.78, blue: 0.64, alpha: 1)); c.fillEllipse(in: CGRect(x: w * 0.41, y: h * 0.45, width: w * 0.18, height: h * 0.3))
        guard let image = c.makeImage(), let dest = CGImageDestinationCreateWithURL(url as CFURL, type.identifier as CFString, 1, nil) else {
            throw WorkerFailure(code: "protocol", message: "picture")
        }
        CGImageDestinationAddImage(dest, image, nil)
        guard CGImageDestinationFinalize(dest) else { throw WorkerFailure(code: "protocol", message: "picture") }
    }
    static func clip(_ url: URL, seconds: Double = 9.5) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try tone(url, frequency: 180, seconds: seconds)
        return url
    }
    static func video(_ url: URL) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        try VideoMaker.write(url, frames: 24, width: 480, height: 270, mask: false)
        return url
    }
    /// A two-page PDF with a heading and a table's worth of text.
    static func pdf(_ url: URL) throws -> URL {
        try FileManager.default.createDirectory(at: url.deletingLastPathComponent(), withIntermediateDirectories: true)
        var box = CGRect(x: 0, y: 0, width: 612, height: 792)
        let c = CGContext(url as CFURL, mediaBox: &box, nil)!
        for page in 0..<2 {
            c.beginPDFPage(nil)
            c.setFillColor(CGColor(gray: 1, alpha: 1)); c.fill(box)
            c.setFillColor(CGColor(gray: 0.15, alpha: 1))
            c.fill(CGRect(x: 60, y: 700, width: 300, height: 24))
            for line in 0..<14 { c.fill(CGRect(x: 60, y: 640 - line * 22, width: 480 - (line % 3) * 60, height: 8)) }
            if page == 0 { c.stroke(CGRect(x: 60, y: 200, width: 480, height: 90)) }
            c.endPDFPage()
        }
        c.closePDF()
        return url
    }
}

extension Substring {
    /// The same hash every run (Swift's own hash is seeded per process).
    var hashValueStable: Int { unicodeScalars.reduce(5381) { ($0 &* 33) &+ Int($1.value) } }
}

/// Short H.264 videos for the fixtures: a moving disc on a gradient, or a white disc on black (a mask).
enum VideoMaker {
    static func write(_ url: URL, frames: Int, width: Int, height: Int, mask: Bool) throws {
        try? FileManager.default.removeItem(at: url)
        let writer = try AVAssetWriter(outputURL: url, fileType: .mp4)
        let input = AVAssetWriterInput(mediaType: .video, outputSettings: [AVVideoCodecKey: AVVideoCodecType.h264, AVVideoWidthKey: width, AVVideoHeightKey: height])
        let adaptor = AVAssetWriterInputPixelBufferAdaptor(
            assetWriterInput: input,
            sourcePixelBufferAttributes: [
                kCVPixelBufferPixelFormatTypeKey as String: kCVPixelFormatType_32ARGB, kCVPixelBufferWidthKey as String: width, kCVPixelBufferHeightKey as String: height
            ])
        writer.add(input)
        guard writer.startWriting() else { throw writer.error ?? WorkerFailure(code: "protocol", message: "video") }
        writer.startSession(atSourceTime: .zero)
        for i in 0..<frames {
            while !input.isReadyForMoreMediaData { Thread.sleep(forTimeInterval: 0.002) }
            var buffer: CVPixelBuffer?
            CVPixelBufferCreate(nil, width, height, kCVPixelFormatType_32ARGB, nil, &buffer)
            guard let buffer else { continue }
            CVPixelBufferLockBaseAddress(buffer, [])
            let c = CGContext(
                data: CVPixelBufferGetBaseAddress(buffer), width: width, height: height, bitsPerComponent: 8, bytesPerRow: CVPixelBufferGetBytesPerRow(buffer),
                space: CGColorSpace(name: CGColorSpace.sRGB)!, bitmapInfo: CGImageAlphaInfo.noneSkipFirst.rawValue)!
            let t = Double(i) / Double(max(1, frames - 1))
            if mask {
                c.setFillColor(CGColor(gray: 0, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: width, height: height))
                c.setFillColor(CGColor(gray: 1, alpha: 1))
            } else {
                c.setFillColor(CGColor(red: 0.45, green: 0.66, blue: 0.9, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: width, height: height))
                c.setFillColor(CGColor(red: 0.93, green: 0.82, blue: 0.62, alpha: 1)); c.fill(CGRect(x: 0, y: 0, width: width, height: height / 3))
                c.setFillColor(CGColor(red: 0.85, green: 0.3, blue: 0.25, alpha: 1))
            }
            c.fillEllipse(in: CGRect(x: Double(width) * (0.15 + 0.5 * t), y: Double(height) * 0.3, width: Double(height) * 0.4, height: Double(height) * 0.4))
            CVPixelBufferUnlockBaseAddress(buffer, [])
            adaptor.append(buffer, withPresentationTime: CMTime(value: CMTimeValue(i), timescale: 12))
        }
        input.markAsFinished()
        let done = DispatchSemaphore(value: 0)
        writer.finishWriting { done.signal() }
        done.wait()
        if writer.status != .completed { throw writer.error ?? WorkerFailure(code: "protocol", message: "video") }
    }
}

/// A coloured cube as a binary glTF: positions, normals and indices, one material.
enum GLBMaker {
    static func cube(_ url: URL) throws { try data().write(to: url) }
    static func data() -> Data {
        var positions: [Float] = [], normals: [Float] = [], indices: [UInt16] = []
        let faces: [([Float], [[Float]])] = [
            ([0, 0, 1], [[-1, -1, 1], [1, -1, 1], [1, 1, 1], [-1, 1, 1]]), ([0, 0, -1], [[1, -1, -1], [-1, -1, -1], [-1, 1, -1], [1, 1, -1]]),
            ([1, 0, 0], [[1, -1, 1], [1, -1, -1], [1, 1, -1], [1, 1, 1]]), ([-1, 0, 0], [[-1, -1, -1], [-1, -1, 1], [-1, 1, 1], [-1, 1, -1]]),
            ([0, 1, 0], [[-1, 1, 1], [1, 1, 1], [1, 1, -1], [-1, 1, -1]]), ([0, -1, 0], [[-1, -1, -1], [1, -1, -1], [1, -1, 1], [-1, -1, 1]])
        ]
        for (normal, corners) in faces {
            let base = UInt16(positions.count / 3)
            for corner in corners { positions += corner.map { $0 * 0.5 }; normals += normal }
            indices += [base, base + 1, base + 2, base, base + 2, base + 3]
        }
        var bin = Data()
        bin.append(contentsOf: positions.withUnsafeBufferPointer { Data(buffer: $0) })
        let normalOffset = bin.count
        bin.append(contentsOf: normals.withUnsafeBufferPointer { Data(buffer: $0) })
        let indexOffset = bin.count
        bin.append(contentsOf: indices.withUnsafeBufferPointer { Data(buffer: $0) })
        while bin.count % 4 != 0 { bin.append(0) }
        let json: [String: Any] = [
            "asset": ["version": "2.0"], "scene": 0, "scenes": [["nodes": [0]]], "nodes": [["mesh": 0, "rotation": [0.2, 0.35, 0, 0.92]]],
            "meshes": [["primitives": [["attributes": ["POSITION": 0, "NORMAL": 1], "indices": 2, "material": 0]]]],
            "materials": [["pbrMetallicRoughness": ["baseColorFactor": [0.85, 0.42, 0.2, 1], "metallicFactor": 0, "roughnessFactor": 0.6]]],
            "buffers": [["byteLength": bin.count]],
            "bufferViews": [
                ["buffer": 0, "byteOffset": 0, "byteLength": positions.count * 4], ["buffer": 0, "byteOffset": normalOffset, "byteLength": normals.count * 4],
                ["buffer": 0, "byteOffset": indexOffset, "byteLength": indices.count * 2]
            ],
            "accessors": [
                ["bufferView": 0, "componentType": 5126, "count": positions.count / 3, "type": "VEC3", "min": [-0.5, -0.5, -0.5], "max": [0.5, 0.5, 0.5]],
                ["bufferView": 1, "componentType": 5126, "count": normals.count / 3, "type": "VEC3"],
                ["bufferView": 2, "componentType": 5123, "count": indices.count, "type": "SCALAR"]
            ]
        ]
        var jsonData = (try? JSONSerialization.data(withJSONObject: json)) ?? Data()
        while jsonData.count % 4 != 0 { jsonData.append(0x20) }
        var out = Data()
        func u32(_ v: UInt32) { withUnsafeBytes(of: v.littleEndian) { out.append(contentsOf: $0) } }
        u32(0x4654_6C67); u32(2); u32(UInt32(12 + 8 + jsonData.count + 8 + bin.count))
        u32(UInt32(jsonData.count)); u32(0x4E4F_534A); out.append(jsonData)
        u32(UInt32(bin.count)); u32(0x004E_4942); out.append(bin)
        return out
    }
}
