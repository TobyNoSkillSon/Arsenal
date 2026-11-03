// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalWire
import CoreImage
import IntegrationKit
@testable import LanguageWorker

/// Maker examples and format variants, passed through the serving parsers and the production validator.
/// Sources: pinned model cards; INF-MLLM 011866e; UI-Venus-2 1ffbaf6;
/// UI-Venus-1.5 192a924; GroundGUI a1a02a9; NaviDC-OCR 1e71f4f.
struct ParserFixtures {
    init() { MLXReady.ensure() }
    private var root: URL {
        URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    }
    private func validate(_ result: JSONObject, model: String, op: String, output: String = "json", directory: URL? = nil) throws {
        let temporary = root.appendingPathComponent("lab/g1/parser-tests/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: temporary, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: temporary) }
        // Retired models keep their folders under retired-models/; the parser code they exercise still ships.
        let shipping = root.appendingPathComponent("models/" + model)
        let folder = FileManager.default.fileExists(atPath: shipping.path) ? shipping : root.appendingPathComponent("retired-models/" + model)
        let schema = try WireJSON.decode(Data(contentsOf: folder.appendingPathComponent("ops/\(op).output.json")))
        let request: JSONObject = [
            "operation": .string(op), "output": .string(output), "schema": .object(schema), "result": .object(result), "directory": .string((directory ?? temporary).path)
        ]
        let file = temporary.appendingPathComponent("output.json"); try WireJSON.encode(request).write(to: file)
        let process = Process(); process.executableURL = root.appendingPathComponent(".build/debug/Arsenal")
        process.arguments = ["runner-check", "--validate-output", file.path]
        let pipe = Pipe(); process.standardOutput = pipe; process.standardError = pipe
        try process.run(); process.waitUntilExit()
        let diagnostic = String(decoding: pipe.fileHandleForReading.readDataToEndOfFile(), as: UTF8.self)
        #expect(process.terminationStatus == 0, "\(diagnostic)")
    }
    @Test func infinityDocumentedLayout() throws {
        // Synthetic page in the reference shape: a list of category/text/bbox blocks on a 0–1000 grid.
        let raw =
            #"[{"category":"header","text":"Running head","bbox":[0,0,1000,50]},{"category":"title","text":"Example document","bbox":[100,100,900,200]},{"category":"table","text":"<table><tr><td>42</td></tr></table>","bbox":[100,300,900,800]}]"#
        let page = try LanguageOutput.document(id: "infinity-parser2-flash", index: 0, text: raw, size: CGSize(width: 2400, height: 1800))
        let blocks = page.object!["blocks"]!.array!
        #expect(blocks.count == 3); #expect(page.object!["tables"]!.array!.count == 1)
        #expect(blocks[2].object!["box"]!.object!["ymax"]?.number == 1440)
        #expect(page.object!["markdown"]?.string == "Example document\n\n<table><tr><td>42</td></tr></table>")
        try validate(["pages": [page]], model: "infinity-parser2-flash", op: "read_document")
        #expect(throws: (any Error).self) { try LanguageOutput.document(id: "infinity-parser2-flash", index: 0, text: "{}", size: CGSize(width: 10, height: 10)) }
        // Lenient like restore_abs_bbox_coordinates: elements stay; boxes are ordered and clamped, unreadable boxes dropped.
        let lenient = try LanguageOutput.document(
            id: "infinity-parser2-flash", index: 0,
            text: #"[{"category":"text","text":"x","bbox":[900,0,100,1200]},{"category":"text","text":"y","bbox":[1,2]},{"category":"text","text":"z"}]"#,
            size: CGSize(width: 10, height: 10))
        let blocks2 = try #require(lenient.object?["blocks"]?.array)
        #expect(blocks2.count == 3); #expect(lenient.object?["markdown"]?.string == "x\n\ny\n\nz")
        #expect(blocks2[0].object?["box"]?.object?["xmin"]?.number == 1); #expect(blocks2[0].object?["box"]?.object?["ymax"]?.number == 10)
        #expect(blocks2[1].object?["box"] == nil); #expect(blocks2[2].object?["box"] == nil)
        try validate(["pages": [lenient]], model: "infinity-parser2-flash", op: "read_document")
    }
    @Test(.heavy) func infinityMakerUtilsExamples() throws {
        // INF-MLLM 011866e tests/test_utils.py: restore_abs_bbox_coordinates on a 3000x2000 image and the full
        // doc2json pipeline on a 1000x1000 image.
        let restored = try LanguageOutput.document(
            id: "infinity-parser2-flash", index: 0, text: #"[{"bbox": [0, 0, 500, 500], "text": "hello"}]"#, size: CGSize(width: 3000, height: 2000))
        let box = try #require(restored.object?["blocks"]?.array?.first?.object?["box"]?.object)
        #expect(box["xmax"]?.number == 1500); #expect(box["ymax"]?.number == 1000)
        let pipeline = try LanguageOutput.document(
            id: "infinity-parser2-flash", index: 0, text: "```json\n[{\"bbox\": [0, 0, 500, 500], \"category\": \"text\", \"text\": \"Test\"}]\n```",
            size: CGSize(width: 1000, height: 1000))
        #expect(pipeline.object?["blocks"]?.array?.count == 1); #expect(pipeline.object?["blocks"]?.array?.first?.object?["category"]?.string == "text")
        // convert_json_to_markdown strips header and footer by default.
        let markdown = try LanguageOutput.document(
            id: "infinity-parser2-flash", index: 0,
            text:
                #"[{"category": "header", "text": "Header content", "bbox": [0, 0, 1, 1]}, {"category": "text", "text": "Main content", "bbox": [0, 0, 1, 1]}, {"category": "footer", "text": "Footer content", "bbox": [0, 0, 1, 1]}]"#,
            size: CGSize(width: 1000, height: 1000))
        #expect(markdown.object?["markdown"]?.string == "Main content")
    }
    @Test func nuExtractCardExamples() throws {
        // NuExtract3 README: receipt and ICL examples (including nulls, arrays and numeric values).
        for raw in [
            #"{"store":"Trader Joe's","date":null,"total":12.40,"currency":"USD","items":[{"name":"apples","price":null},{"name":"coffee","price":null}]}"#,
            #"{"names":["-JOHN-","-MARY-","-JAMES-"]}"#
        ] {
            let data = try LanguageOutput.json(raw)
            #expect(data.object != nil)
            try validate(["data": data], model: "nuextract3", op: "extract")
        }
        let example = try NuExamples.prepare([["text": "Stephen is the manager at Susan's store.", "output": ["names": ["-STEPHEN-", "-SUSAN-"]]]]) { _ in
            throw WorkerFailure(code: "test", message: "Unexpected image")
        }
        #expect(example.messages.count == 1); #expect(example.images.isEmpty)
        #expect(example.messages[0]["role"] as? String == "developer")
        let parts = try #require(example.messages[0]["content"] as? [[String: String]])
        #expect(parts.count == 2); #expect(try LanguageOutput.json(parts[1]["text"]!).object?["names"]?.array?.count == 2)
    }
    @Test func venusCoordinateConventions() throws {
        // Both maker grounding implementations use normalized 0–1000, including UI-Venus-1.5.
        // A resize from original 2400x1800 to 1200x900 must not turn [250,750] into resized pixels.
        let original = CGSize(width: 2400, height: 1800)
        for model in ["ui-venus-2-9b", "ui-venus-1.5-2b"] {
            for raw in ["[250,750]", "[100,500,400,1000]", "[100,500], [400,1000]"] {
                let result = try LanguageOutput.ground(id: model, text: raw, label: "Save", size: original)
                #expect(result["detections"]?.array?.count == 1)
                let point = result["detections"]!.array![0].object!["point"]!.object!
                #expect(point["x"]?.number == 600); #expect(point["y"]?.number == 1350)
                // Equivalent resized-space conversion: [300,675] * original/processed.
                #expect(point["x"]?.number == 300.0 * 2400.0 / 1200.0)
                #expect(point["y"]?.number == 675.0 * 1800.0 / 900.0)
                try validate(result, model: model, op: "ground_screen")
            }
            let refusal = try LanguageOutput.ground(id: model, text: "[-1,-1]", label: "Missing", size: original)
            #expect(refusal["detections"]!.array!.isEmpty); try validate(refusal, model: model, op: "ground_screen")
        }
    }
    @Test func groundGUIReferenceToolCall() throws {
        let result = try LanguageOutput.ground(
            id: "groundgui-8b", text: #"<tool_call>{"name":"computer_use","arguments":{"action":"left_click","coordinate":[250,750]}}</tool_call>"#, label: "Save",
            size: CGSize(width: 2560, height: 1440))
        let point = result["detections"]!.array![0].object!["point"]!.object!
        #expect(point["x"]?.number == 640); #expect(point["y"]?.number == 1080)
        try validate(result, model: "groundgui-8b", op: "ground_screen")
    }
    /// Synthetic layout and recognition replies in the NaviDC-OCR client formats.
    @Test func teleLayoutCropRecognitionReassembly() throws {
        let image = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: CGRect(x: 0, y: 0, width: 200, height: 100))
        var prompts: [String] = [], sizes: [CGSize] = []
        let result = try TeleDocument.page(index: 0, image: image, distorted: false, cancellation: NativeCancellation()) { prompt, crop, _ in
            prompts.append(prompt); sizes.append(crop.extent.size)
            if prompts.count == 1 { return "<box:0 0 500 500><label:text><up>\n<box:500 0 1000 500><label:table><right>\n<box:0 500 500 500 500 1000 0 1000><label:equation><up>" }
            if prompts.count == 2 { return "Hello" }
            if prompts.count == 3 { return "<fcel>42<nl>" }
            return "\\[x=1\\]"
        }
        #expect(prompts.count == 4); #expect(sizes[0] == CGSize(width: 1036, height: 1036))
        #expect(sizes[1] == CGSize(width: 100, height: 50)); #expect(abs(sizes[2].width - 50) < 0.001)
        #expect(result.object!["blocks"]!.array!.count == 3)
        #expect(result.object!["markdown"]?.string == "Hello\n\n<table><tr><td>42</td></tr></table>\n\n$$x=1$$")
        try validate(["pages": [result]], model: "teleocr", op: "read_document")
    }
    @Test func ovisCroppedFiguresAndFiles() throws {
        let directory = root.appendingPathComponent("lab/g1/figure-tests/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: 200, height: 100))
        var page = try #require(LanguageOutput.document(id: "ovisocr2", index: 2, text: #"Figure: <img src="images/bbox_100_200_600_800.jpg" />"#, size: image.extent.size).object)
        let files = try DocumentRegions.figures(page: &page, image: image, root: directory, cancellation: NativeCancellation())
        #expect(files.count == 1)
        #expect(page["markdown"]?.string == "Figure: <img src=\"page-2-figure-0.png\" />")
        #expect(files[0].object?["width"]?.number == 100); #expect(files[0].object?["height"]?.number == 60)
        #expect(files[0].object?["id"] == page["images"]?.array?[0].object?["id"])
        try validate(["pages": [.object(page)], "files": .array(files)], model: "ovisocr2", op: "read_document", output: "file", directory: directory)
    }
    @Test func topLeftCropAndPolygonMask() throws {
        let bounds = CGRect(x: 0, y: 0, width: 20, height: 20)
        let red = CIImage(color: CIColor(red: 1, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 10, width: 20, height: 10))
        let blue = CIImage(color: CIColor(red: 0, green: 0, blue: 1)).cropped(to: bounds)
        let image = red.composited(over: blue)
        let top = try DocumentRegions.crop(image, box: CGRect(x: 0, y: 0, width: 10, height: 10))
        let bottom = try DocumentRegions.crop(image, box: CGRect(x: 0, y: 10, width: 10, height: 10))
        func pixel(_ image: CIImage, x: Int, y: Int) -> [UInt8] {
            var bytes = [UInt8](repeating: 0, count: 4)
            CIContext().render(image, toBitmap: &bytes, rowBytes: 4, bounds: CGRect(x: x, y: y, width: 1, height: 1), format: .RGBA8, colorSpace: CGColorSpaceCreateDeviceRGB())
            return bytes
        }
        #expect(pixel(top, x: 5, y: 5).prefix(3) == [255, 0, 0])
        #expect(pixel(bottom, x: 5, y: 5).prefix(3) == [0, 0, 255])
        let polygon = try DocumentRegions.crop(image, box: CGRect(x: 0, y: 0, width: 10, height: 10), polygon: [CGPoint(x: 0, y: 0), CGPoint(x: 10, y: 0), CGPoint(x: 0, y: 10)])
        #expect(pixel(polygon, x: 1, y: 8).prefix(3) == [255, 0, 0])
        #expect(pixel(polygon, x: 8, y: 1).prefix(3) == [0, 0, 0])
    }
    @Test func paddleSpatialPositions() {
        let (positions, next) = PaddlePositions.imagePrompt(count: 10, imageStart: 2, height: 2, width: 3)
        #expect(positions == [[0, 1, 2, 2, 2, 2, 2, 2, 5, 6], [0, 1, 2, 2, 2, 3, 3, 3, 5, 6], [0, 1, 2, 3, 4, 2, 3, 4, 5, 6]])
        #expect(next == 7)
    }
}

/// Reference post-processing paths: PaddleX c852371, GroundGUI a1a02a9, UI-Venus 1ffbaf6, INF-MLLM 011866e,
/// NuExtract3 c99dc8f card, OvisOCR2 card, NaviDC-OCR 1e71f4f.
extension ParserFixtures {
    @Test func paddleSpottingTablesAndFormulas() throws {
        let size = CGSize(width: 2000, height: 1000)
        let raw =
            "Hello<|LOC_100|><|LOC_200|><|LOC_300|><|LOC_200|><|LOC_300|><|LOC_400|><|LOC_100|><|LOC_400|>"
            + "World<|LOC_500|><|LOC_500|><|LOC_600|><|LOC_500|><|LOC_600|><|LOC_600|><|LOC_500|><|LOC_600|><|LOC_7|>"
        let page = try #require(LanguageOutput.document(id: "paddleocr-vl-1.6", index: 0, text: raw, size: size, mode: "text_spotting").object)
        let blocks = try #require(page["blocks"]?.array)
        #expect(blocks.count == 2); #expect(page["markdown"]?.string == "Hello\n\nWorld")
        let first = try #require(blocks[0].object)
        #expect(first["text"]?.string == "Hello")
        #expect(first["polygon"]?.array?.first?.object?["x"]?.number == 200); #expect(first["polygon"]?.array?.first?.object?["y"]?.number == 200)
        #expect(first["box"]?.object?["xmax"]?.number == 600); #expect(first["box"]?.object?["ymax"]?.number == 400)
        try validate(["pages": [.object(page)]], model: "paddleocr-vl-1.6", op: "read_document")
        #expect(throws: (any Error).self) { try LanguageOutput.document(id: "paddleocr-vl-1.6", index: 0, text: "plain text", size: size, mode: "text_spotting") }
        let blank = try LanguageOutput.document(id: "paddleocr-vl-1.6", index: 0, text: "", size: size, mode: "text_spotting")
        #expect(blank.object?["blocks"]?.array?.isEmpty == true)
        let table = try LanguageOutput.document(id: "paddleocr-vl-1.6", index: 0, text: "<fcel>1<fcel>2<nl>", size: size, mode: "table")
        #expect(table.object?["markdown"]?.string == "<table><tr><td>1</td><td>2</td></tr></table>")
        #expect(table.object?["tables"]?.array?.count == 1)
        let formula = try LanguageOutput.document(id: "paddleocr-vl-1.6", index: 0, text: "\\[x^2\\]", size: size, mode: "formula")
        #expect(formula.object?["markdown"]?.string == " $$ x^2 $$ ")
        try validate(["pages": [table, formula]], model: "paddleocr-vl-1.6", op: "read_document")
    }
    @Test func groundGUIPlatformsAndFallbacks() throws {
        let size = CGSize(width: 1000, height: 2000)
        let mobile = try LanguageOutput.ground(
            id: "groundgui-8b", text: #"<tool_call>{"name": "mobile_use", "arguments": {"action": "click", "coordinate": [500, 250]}}</tool_call>"#, label: "Send", size: size)
        #expect(mobile["detections"]?.array?.first?.object?["point"]?.object?["y"]?.number == 500)
        try validate(mobile, model: "groundgui-8b", op: "ground_screen")
        let abstain = try LanguageOutput.ground(
            id: "groundgui-8b", text: #"<tool_call>{"name": "computer_use", "arguments": {"action": "abstain"}}</tool_call>"#, label: "x", size: size)
        #expect(abstain["detections"]?.array?.isEmpty == true)
        // eval/prompts.py falls back to exactly two integers when the call is not valid JSON; an unclosed tag keeps the tail.
        let loose = try LanguageOutput.ground(id: "groundgui-8b", text: "<tool_call>left_click(250, 750)", label: "x", size: size)
        #expect(loose["detections"]?.array?.first?.object?["point"]?.object?["x"]?.number == 250)
        // Qwen3VLPrompt falls back to GUIOwlPrompt's regexes, which also read a bare [x, y].
        let bare = try LanguageOutput.ground(id: "groundgui-8b", text: "[250, 750]", label: "x", size: size)
        #expect(bare["detections"]?.array?.first?.object?["point"]?.object?["y"]?.number == 1500)
        #expect(throws: (any Error).self) { try LanguageOutput.ground(id: "groundgui-8b", text: "no point", label: "x", size: size) }
        // The last valid tool call wins; a coordinate past the grid is clamped to the image.
        let last = try LanguageOutput.ground(
            id: "groundgui-8b",
            text:
                #"<tool_call>{"name":"computer_use","arguments":{"action":"left_click","coordinate":[100,100]}}</tool_call><tool_call>{"name":"computer_use","arguments":{"action":"left_click","coordinate":[1200,500]}}</tool_call><tool_call>{bad}</tool_call>"#,
            label: "x", size: size)
        #expect(last["detections"]?.array?.first?.object?["point"]?.object?["x"]?.number == 1000)
        #expect(last["detections"]?.array?.first?.object?["point"]?.object?["y"]?.number == 1000)
        // A box answer keeps its box beside the center point.
        let boxed = try LanguageOutput.ground(id: "ui-venus-2-9b", text: "[100,300,300,500]", label: "x", size: size)
        #expect(boxed["detections"]?.array?.first?.object?["box"]?.object?["ymax"]?.number == 1000)
        try validate(boxed, model: "ui-venus-2-9b", op: "ground_screen")
        let venus = try LanguageOutput.ground(id: "ui-venus-1.5-2b", text: "[250,750]\n[1,1]", label: "x", size: size)
        #expect(venus["detections"]?.array?.first?.object?["point"]?.object?["x"]?.number == 250)
        let numbers = try LanguageOutput.ground(id: "ui-venus-2-9b", text: "(100, 300, 300, 500)", label: "x", size: size)
        #expect(numbers["detections"]?.array?.first?.object?["point"]?.object?["y"]?.number == 800)
    }
    @Test func infinityFencesTruncationAndMarkdown() throws {
        let size = CGSize(width: 1000, height: 1000)
        let fenced = "Here:\n```json\n[{\"bbox\": [100, 100, 900, 200], \"category\": \"title\", \"text\": \"A\"}]\n```"
        let page = try LanguageOutput.document(id: "infinity-parser2-flash", index: 0, text: fenced, size: size)
        #expect(page.object?["markdown"]?.string == "A")
        let truncated = "[{\"bbox\": [100, 100, 900, 200], \"category\": \"title\", \"text\": \"A\"}, {\"bbox\": [100, 300"
        #expect(LanguageOutput.infinityJSON(truncated) == "[{\"bbox\": [100, 100, 900, 200], \"category\": \"title\", \"text\": \"A\"}]")
        let recovered = try LanguageOutput.document(id: "infinity-parser2-flash", index: 0, text: truncated, size: size)
        #expect(recovered.object?["blocks"]?.array?.count == 1)
        try validate(["pages": [page, recovered]], model: "infinity-parser2-flash", op: "read_document")
        let figure = try LanguageOutput.document(id: "infinity-parser2-flash", index: 0, text: "[{\"bbox\": [0, 0, 10, 10], \"category\": \"figure\"}]", size: size)
        #expect(figure.object?["blocks"]?.array?.first?.object?["text"]?.string == "")
        let markdown = try LanguageOutput.document(id: "infinity-parser2-flash", index: 0, text: "```markdown\n# Title\n\nBody\n```", size: size, mode: "markdown")
        #expect(markdown.object?["markdown"]?.string == "# Title\n\nBody")
    }
    @Test func nuExtractReasoningAndExampleSpelling() throws {
        let data = try LanguageOutput.json("<think>\nThe date is absent.\n</think>\n\n{\"date\": null, \"url\": \"https://a.b/c\"}")
        #expect(data.object?["url"]?.string == "https://a.b/c")
        let example = try NuExamples.prepare([["text": "x", "output": ["b": "1/2", "a": 1]]]) { _ in throw WorkerFailure(code: "test", message: "Unexpected image") }
        let parts = try #require(example.messages[0]["content"] as? [[String: String]])
        #expect(parts.last?["text"] == #"{"a": 1, "b": "1/2"}"#)
    }
    @Test func nuExtractExamplesFollowTemplateOrder() throws {
        // The card's receipt template (json.dumps spelling); example outputs are written in its key order.
        let template = #"{"store": "verbatim-string", "date": "date-time", "total": "number", "items": [{"name": "verbatim-string", "price": "number"}]}"#
        let output: JSONValue = ["items": [["price": .null, "name": "apples"]], "total": 12.4, "store": "Trader Joe's", "date": .null, "note": "x"]
        let example = try NuExamples.prepare([["text": "receipt", "output": output]], template: template) { _ in
            throw WorkerFailure(code: "test", message: "Unexpected image")
        }
        let parts = try #require(example.messages[0]["content"] as? [[String: String]])
        #expect(parts.last?["text"] == #"{"store": "Trader Joe's", "date": null, "total": 12.4, "items": [{"name": "apples", "price": null}], "note": "x"}"#)
        // The card's ICL example output spelling.
        let card: JSONValue = ["text": "Stephen is the manager at Susan's store.", "output": ["names": ["-STEPHEN-", "-SUSAN-"]]]
        let names = try NuExamples.prepare([card], template: #"{"names": ["string"]}"#) { _ in throw WorkerFailure(code: "test", message: "Unexpected image") }
        #expect((names.messages[0]["content"] as? [[String: String]])?.last?["text"] == #"{"names": ["-STEPHEN-", "-SUSAN-"]}"#)
        let spec = try WireJSON.decode(Data(contentsOf: root.appendingPathComponent("models/nuextract3/runners/native/language.json")))
        #expect(LanguageDecoding.make(spec: spec, input: [:], thinking: false).temperature == 0.2)
        #expect(LanguageDecoding.make(spec: spec, input: [:], thinking: true).temperature == 0.6)
        let schema = try WireJSON.decode(Data(contentsOf: root.appendingPathComponent("models/nuextract3/ops/extract.input.json")))
        #expect(schema["properties"]?.object?["template"]?.object?["x-arsenal-ordered"]?.bool == true)
    }
    @Test func ovisFigureClampAndSkip() throws {
        let directory = root.appendingPathComponent("lab/g1/figure-tests/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let image = CIImage(color: CIColor(red: 0, green: 1, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: 200, height: 100))
        let text = #"<img src="images/bbox_900_0_1200_500.jpg" /> <img src="images/bbox_500_500_500_600.jpg" />"#
        var page = try #require(LanguageOutput.document(id: "ovisocr2", index: 0, text: text, size: image.extent.size).object)
        #expect(page["images"]?.array?.first?.object?["bottom_right_x"]?.number == 200)
        let files = try DocumentRegions.figures(page: &page, image: image, root: directory, cancellation: NativeCancellation())
        #expect(files.count == 1); #expect(page["images"]?.array?.count == 1)
        #expect(files[0].object?["width"]?.number == 20)
        #expect(LanguageOutput.cleanTruncatedRepeats("short") == "short")
        let looped = String(repeating: "a", count: 7990) + String(repeating: "xyz", count: 40)
        #expect(LanguageOutput.cleanTruncatedRepeats(looped) == String(repeating: "a", count: 7990) + "xyz")
    }
    @Test func teleDirectLayoutAndEquations() throws {
        // Direct layout modes: parsed blocks, no recognition; garbage is the parser's rejection.
        let page = try #require(
            LanguageOutput.document(id: "teleocr", index: 1, text: "<box:0 0 500 500><label:title><up>\nnoise", size: CGSize(width: 200, height: 100), mode: "layout").object)
        #expect(page["blocks"]?.array?.first?.object?["category"]?.string == "title")
        #expect(page["blocks"]?.array?.first?.object?["box"]?.object?["xmax"]?.number == 100)
        try validate(["pages": [.object(page)]], model: "teleocr", op: "read_document")
        #expect(throws: (any Error).self) {
            try LanguageOutput.document(id: "teleocr", index: 0, text: "not a layout", size: CGSize(width: 10, height: 10), mode: "distorted_layout")
        }
        #expect(TeleDocument.layoutPrompt(distorted: true) == "\nMulti-point Layout Segmentation Analysis.")
        // Card post_process: strip \[ and \] independently; wrap unless the content starts and ends with $.
        #expect(try TeleDocument.content("\\[x=1", type: "equation") == "$$x=1$$")
        #expect(try TeleDocument.content("$x=1$", type: "equation") == "$x=1$")
        #expect(try TeleDocument.content("$x=1", type: "equation") == "$$$x=1$$")
        #expect(try TeleDocument.content("plain", type: "table") == "")
    }
    @Test func teleTolerantLayoutAndCropPreparation() throws {
        let blocks = try TeleDocument.parse(
            "noise\n<box:0 0 500 500><label:text><up>\n<box:0 0 0 0><label:text><up>\n<box:1 2 3 4><label:bogus><up>", size: CGSize(width: 200, height: 100))
        #expect(blocks.count == 1)
        let image = CIImage(color: CIColor(red: 1, green: 1, blue: 1)).cropped(to: CGRect(x: 0, y: 0, width: 200, height: 100))
        #expect(throws: (any Error).self) {
            try TeleDocument.page(index: 0, image: image, distorted: false, cancellation: NativeCancellation()) { _, _, _ in "not a layout" }
        }
        let strip = DocumentRegions.recognitionCrop(CIImage(color: CIColor(red: 0, green: 0, blue: 0)).cropped(to: CGRect(x: 0, y: 0, width: 10, height: 600)))
        // 60:1 is padded to 12x600, then the short edge is scaled by 28/12 with ceil, exactly as resize_by_need computes it.
        let scale: CGFloat = 28 / 12
        #expect(strip.extent.size == CGSize(width: (12 * scale).rounded(.up), height: (600 * scale).rounded(.up)))
        #expect(TeleDocument.markdown("Intro", type: "title") == "# Intro")
        #expect(TeleDocument.markdown("12", type: "page_number") == nil)
        #expect(TeleDocument.markdown("abc", type: "char") == nil)
    }
}

/// Truncation reaches the result: pages flag it, parsed operations carry finish_reason.
extension ParserFixtures {
    @Test func truncationFieldsValidate() throws {
        let page: JSONValue = ["index": 0, "markdown": "Cut off", "truncated": true]
        for model in ["chandra-ocr-2", "teleocr", "paddleocr-vl-1.6", "infinity-parser2-flash", "nuextract3"] {
            try validate(["pages": [page]], model: model, op: "read_document")
        }
        // OvisOCR2 always returns the joined Markdown file beside its pages (LanguageNative.documentResult).
        let directory = root.appendingPathComponent("lab/g1/truncation-tests/" + UUID().uuidString)
        try FileManager.default.createDirectory(at: directory, withIntermediateDirectories: true)
        defer { try? FileManager.default.removeItem(at: directory) }
        let document = directory.appendingPathComponent("document.md")
        try Data("Cut off".utf8).write(to: document)
        let markdown = try NativeMedia.file(document, root: directory, mime: "text/markdown", id: "document")
        try validate(["pages": [page], "files": [.object(markdown)]], model: "ovisocr2", op: "read_document", output: "file", directory: directory)
        for model in ["hy-mt2-7b", "index-translate-2b", "index-translate-9b"] {
            try validate(["translations": [["text": "Cześć", "detected_source_language": "und", "finish_reason": "length"]]], model: model, op: "translate")
        }
        try validate(["data": ["a": 1], "finish_reason": "stop"], model: "nuextract3", op: "extract")
        for model in ["qwen3.5-2b", "lfm2.5-vl-450m", "gemma-4-e4b-it"] {
            try validate(["text": "A cat", "finish_reason": "length"], model: model, op: "describe")
            try validate(["tags": ["cat"], "finish_reason": "stop"], model: model, op: "tag")
        }
        for model in ["groundgui-8b", "ui-venus-2-9b", "ui-venus-1.5-2b"] {
            try validate(["detections": [], "coordinate_system": "pixels", "finish_reason": "stop"], model: model, op: "ground_screen")
        }
    }
}

/// Chandra OCR 2 (datalab-to/chandra output.py parse_markdown and parse_layout). Synthetic page in the maker prompt's
/// documented format (layout divs with data-bbox/data-label, <math>, HTML tables); the maker repository has no
/// example output.
extension ParserFixtures {
    @Test(.heavy) func chandraMarkdownAndLayout() throws {
        let raw =
            #"<div data-bbox="0 0 1000 50" data-label="Page-Header"><p>Running head</p></div>"#
            + #"<div data-bbox="100 100 900 200" data-label="Section-Header"><h1>Results &amp; methods</h1></div>"#
            + #"<div data-bbox="100 220 900 400" data-label="Text">Energy <b>is</b> conserved for snake_case at 5$.</div>"#
            + #"<div data-bbox="100 420 900 600" data-label="Equation-Block"><p><math display="block">E=mc^2</math></p></div>"#
            + #"<div data-bbox="100 620 900 800" data-label="Table"><table><tr><th>A</th><td data-bbox="1 2 3 4">1</td></tr></table></div>"#
            + #"<div data-bbox="100 820 500 900" data-label="List-Group"><ol><li>First</li><li>Second <i>item</i></li></ol></div>"#
            + #"<div data-bbox="0 0 1000 1000" data-label="Blank-Page"></div>"#
            + #"<div data-bbox="bad" data-label="Figure"><img alt="A chart"/>Bars rise.</div>"#
        let page = try #require(LanguageOutput.document(id: "chandra-ocr-2", index: 0, text: raw, size: CGSize(width: 2000, height: 1000)).object)
        #expect(
            page["markdown"]?.string
                == "# Results & methods\n\nEnergy **is** conserved for snake\\_case at 5\\$.\n\n$$E=mc^2$$\n\n"
                + "<table><tr><th>A</th><td data-bbox=\"1 2 3 4\">1</td></tr></table>\n\n1. First\n2. Second *item*\n\nBars rise.")
        let blocks = try #require(page["blocks"]?.array)
        #expect(blocks.map { $0.object?["category"]?.string } == ["Page-Header", "Section-Header", "Text", "Equation-Block", "Table", "List-Group", "Figure"])
        let header = try #require(blocks[1].object?["box"]?.object)
        #expect(header["xmin"]?.number == 200); #expect(header["ymin"]?.number == 100); #expect(header["xmax"]?.number == 1800)
        // parse_layout strips nested data-bbox and falls back to [0, 0, 1, 1] for an unreadable box.
        #expect(blocks[4].object?["text"]?.string == "<table><tr><th>A</th><td>1</td></tr></table>")
        #expect(blocks[6].object?["box"]?.object?["xmax"]?.number == 2)
        #expect(page["tables"]?.array?.count == 1)
        try validate(["pages": [.object(page)]], model: "chandra-ocr-2", op: "read_document")
    }
}

/// PaddleX c852371 uilts.py otsl_pad_to_sqr_v2 and truncate_repetitive_content, as the pipeline applies them.
extension ParserFixtures {
    @Test func paddleRaggedTableAndRepetition() throws {
        let size = CGSize(width: 32, height: 32)
        // Rows of 2, 3 and 1 cells pad to the width of the longest row that holds an <fcel>.
        let ragged = try LanguageOutput.document(
            id: "paddleocr-vl-1.6", index: 0, text: "<fcel>a<fcel>b<nl><fcel>1<fcel>2<fcel>3<nl><fcel>x<nl>", size: size, mode: "table")
        #expect(
            ragged.object?["markdown"]?.string
                == "<table><tr><td>a</td><td>b</td><td></td></tr><tr><td>1</td><td>2</td><td>3</td></tr><tr><td>x</td><td></td><td></td></tr></table>")
        // Trailing empty cells beyond the cheapest width are trimmed.
        #expect(OTSLOutput.padToSquare("<fcel>a<ecel><ecel><ecel><nl><fcel>b<nl><fcel>c<nl>") == "<fcel>a<nl><fcel>b<nl><fcel>c<nl>")
        #expect(OTSLOutput.padToSquare("<fcel>a<fcel>b") == "<fcel>a<fcel>b<nl>")
        // No cells: the model text is kept (PaddleX) or emptied (TeleOCR card).
        #expect(try OTSLOutput.paddle("plain") == "plain"); #expect(try OTSLOutput.html("plain") == "")
        try validate(["pages": [ragged]], model: "paddleocr-vl-1.6", op: "read_document")
        #expect(PaddleOutput.truncateRepetitive("Intro " + String(repeating: "abcdefghij", count: 20), minCount: 50) == "Intro ")
        #expect(PaddleOutput.truncateRepetitive(String(repeating: "ab", count: 40), minCount: 50) == "ab")
        #expect(PaddleOutput.truncateRepetitive(String(repeating: "same line\n", count: 12), minCount: 50) == "same line")
        #expect(PaddleOutput.truncateRepetitive(String(repeating: "ab", count: 40), minCount: 5000) == String(repeating: "ab", count: 40))
        let looped = try LanguageOutput.document(id: "paddleocr-vl-1.6", index: 0, text: String(repeating: "ab", count: 40), size: size, mode: "ocr")
        #expect(looped.object?["markdown"]?.string == "ab")
    }
}
