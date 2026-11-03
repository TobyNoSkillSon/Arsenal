// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ArsenalWire

/// Parse only the maker's output format. Invalid trained output is a visible error.
enum LanguageOutput {
    /// Reasoning answers end their visible thinking with the non-special `</think>`; the answer follows it.
    static func answer(_ text: String) -> String {
        guard let close = text.range(of: "</think>", options: .backwards) else { return text }
        return String(text[close.upperBound...])
    }
    static func json(_ text: String) throws -> JSONValue {
        var value = answer(text).trimmingCharacters(in: .whitespacesAndNewlines)
        if value.hasPrefix("```"), let first = value.firstIndex(of: "\n"), let last = value.range(of: "```", options: .backwards), first < last.lowerBound {
            value = String(value[value.index(after: first)..<last.lowerBound])
        }
        let decoder = JSONDecoder()
        do { return try decoder.decode(JSONValue.self, from: Data(value.utf8)) } catch {
            throw WorkerFailure(code: "invalid_model_output", message: "The model did not return valid JSON.")
        }
    }

    // MARK: Screen grounding

    static func ground(id: String, text: String, label: String, size: CGSize) throws -> JSONObject {
        let empty: JSONObject = ["detections": [], "coordinate_system": "pixels", "image_size": imageSize(size)]
        var coordinates: [Double]
        var box: [Double]?
        if id == "groundgui-8b" {
            guard let point = try groundGUIPoint(text) else { throw invalidCoordinates() }
            coordinates = point
        } else {
            // UI-Venus ui_venus2_gd.py extract_coordinates_qwen35 (also covers the 1.5 parser's formats).
            let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if let corners = first(in: value, pattern: #"\[\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)\s*\]"#)
                ?? first(in: value, pattern: #"\[\s*(\d+)\s*,\s*(\d+)\s*\]\s*,\s*\[\s*(\d+)\s*,\s*(\d+)\s*\]"#)
            {
                coordinates = [(corners[0] + corners[2]) / 2, (corners[1] + corners[3]) / 2]; box = corners
            } else if let point = first(in: value, pattern: #"\[\s*(-?\d+)\s*,\s*(-?\d+)\s*\]"#) {
                coordinates = point
            } else {
                let all = numbers(in: value, pattern: #"-?\d+\.?\d*"#)
                if all.count == 2 {
                    coordinates = all
                } else if all.count >= 4 {
                    if all[0] == -1 && all[1] == -1 { return empty }
                    coordinates = [(all[0] + all[2]) / 2, (all[1] + all[3]) / 2]; box = Array(all.prefix(4))
                } else {
                    throw invalidCoordinates()
                }
            }
        }
        if coordinates == [-1, -1] { return empty }
        guard coordinates.count == 2, coordinates.allSatisfy(\.isFinite) else { throw invalidCoordinates() }
        // Both makers emit a 0–1000 grid over the whole screenshot, so the processor's resize cancels out. The makers do
        // not clamp; values past the grid are clamped to the image here.
        let width = Double(size.width), height = Double(size.height)
        func x(_ value: Double) -> JSONValue { .number(min(max(value * width / 1000, 0), width)) }
        func y(_ value: Double) -> JSONValue { .number(min(max(value * height / 1000, 0), height)) }
        var detection: JSONObject = ["label": .string(label), "point": ["x": x(coordinates[0]), "y": y(coordinates[1])]]
        if let box, box.allSatisfy(\.isFinite) {
            detection["box"] = ["xmin": x(min(box[0], box[2])), "ymin": y(min(box[1], box[3])), "xmax": x(max(box[0], box[2])), "ymax": y(max(box[1], box[3]))]
        }
        return ["detections": [.object(detection)], "coordinate_system": "pixels", "image_size": imageSize(size)]
    }
    /// GroundGUI eval/prompts.py Qwen3VLPrompt.extract_coordinates: the last <tool_call> JSON whose arguments hold a
    /// coordinate (terminate/abstain is a refusal, [-1, -1]); otherwise GUIOwlPrompt's fallback: refusal markers, then the
    /// last "(x, y)", then the last "[x, y]". Nil when nothing parses.
    static func groundGUIPoint(_ text: String) throws -> [Double]? {
        let source = text as NSString
        let calls = try NSRegularExpression(pattern: #"<tool_call>\s*(\{.*?\})\s*</tool_call>"#, options: [.dotMatchesLineSeparators])
            .matches(in: text, range: NSRange(location: 0, length: source.length))
        for match in calls.reversed() {
            guard let action = try? JSONDecoder().decode(JSONValue.self, from: Data(source.substring(with: match.range(at: 1)).utf8)).object else { continue }
            let arguments = action["arguments"]?.object ?? [:]
            if let name = arguments["action"]?.string, ["terminate", "abstain"].contains(name) { return [-1, -1] }
            guard let coordinate = arguments["coordinate"]?.array, coordinate.count >= 2 else { continue }
            let values = coordinate.prefix(2).compactMap { $0.number ?? $0.string.flatMap { Double($0.trimmingCharacters(in: .whitespaces)) } }
            if values.count == 2 { return values }
        }
        if text.contains("\"action\": \"terminate\"") || text.contains("\"status\": \"failure\"") || text.contains("\"action\": \"abstain\"") { return [-1, -1] }
        for pattern in [#"\((\d+),\s*(\d+)\)"#, #"\[(\d+),\s*(\d+)\]"#] {
            let regex = try NSRegularExpression(pattern: pattern)
            if let last = regex.matches(in: text, range: NSRange(location: 0, length: source.length)).last {
                return [1, 2].compactMap { Double(source.substring(with: last.range(at: $0))) }
            }
        }
        return nil
    }
    private static func imageSize(_ size: CGSize) -> JSONValue { ["width": .number(size.width), "height": .number(size.height)] }
    private static func first(in text: String, pattern: String) -> [Double]? {
        guard let regex = try? NSRegularExpression(pattern: pattern),
            let match = regex.firstMatch(in: text, range: NSRange(text.startIndex..., in: text))
        else { return nil }
        let values = (1..<match.numberOfRanges).compactMap { Range(match.range(at: $0), in: text).flatMap { Double(text[$0]) } }
        return values.count == match.numberOfRanges - 1 ? values : nil
    }
    private static func numbers(in text: String, pattern: String) -> [Double] {
        guard let regex = try? NSRegularExpression(pattern: pattern) else { return [] }
        return regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).compactMap { match in
            Range(match.range, in: text).flatMap { range in
                let token = String(text[range])
                return Double(token.hasSuffix(".") ? String(token.dropLast()) : token)
            }
        }
    }
    private static func invalidCoordinates() -> WorkerFailure {
        WorkerFailure(code: "invalid_model_output", message: "The model did not return a single valid grounding point or box on its 0–1000 grid.")
    }

    // MARK: Documents

    static func document(id: String, index: Int, text: String, size: CGSize, mode: String = "ocr") throws -> JSONValue {
        if id == "teleocr", ["layout", "distorted_layout"].contains(mode) { return try TeleDocument.layout(text, index: index, size: size) }
        var text = text
        if id == "ovisocr2" { text = cleanTruncatedRepeats(text.trimmingCharacters(in: .whitespacesAndNewlines)) }
        if id == "infinity-parser2-flash", mode == "markdown" { text = infinityMarkdown(text) }
        var spotting: [JSONValue]?
        if id == "chandra-ocr-2" {
            let parsed = ChandraOutput.parse(text, size: size)
            text = parsed.markdown; spotting = parsed.blocks
        }
        if id == "paddleocr-vl-1.6" {
            // pipeline.py order: repetition truncation, LaTeX delimiters, OTSL table, spotting.
            text = PaddleOutput.truncateRepetitive(text, minCount: mode == "table" ? 5000 : 50)
            text = PaddleOutput.latex(text)
            if mode == "table" { text = try OTSLOutput.paddle(text) }
            if mode == "text_spotting" {
                let parsed = try PaddleOutput.spotting(text, size: size)
                text = parsed.markdown; spotting = parsed.blocks
            }
        }
        var page: JSONObject = ["index": .number(Double(index)), "markdown": .string(text), "dimensions": ["width": .number(size.width), "height": .number(size.height)]]
        if let spotting { page["blocks"] = .array(spotting) }
        if id == "infinity-parser2-flash", mode == "ocr" {
            let value = try json(infinityJSON(answer(text)))
            guard let list = value.object?["elements"]?.array ?? value.array else {
                throw WorkerFailure(code: "invalid_model_output", message: "Infinity layout must be a block array.")
            }
            // utils.convert_json_to_markdown flattens multi-page lists one level.
            let elements = list.flatMap { $0.array ?? [$0] }
            page["markdown"] = .string(
                elements.compactMap { item in
                    guard let object = item.object, !["header", "footer", "page_footnote"].contains(object["category"]?.string ?? "") else { return nil }
                    guard let value = object["text"]?.string, !value.isEmpty else { return nil }
                    return value
                }.joined(separator: "\n\n"))
            // restore_abs_bbox_coordinates keeps every element; a box that is not four numbers stays unconverted there,
            // so the block keeps its text without a box. Boxes are clamped to the 0–1000 grid and ordered.
            page["blocks"] = .array(
                elements.compactMap { item in
                    guard let object = item.object else { return nil }
                    var block: JSONObject = [
                        "category": .string(object["category"]?.string ?? "unknown"), "text": .string(object["text"]?.string ?? ""), "coordinate_system": "pixels"
                    ]
                    let numbers = object["bbox"]?.array?.compactMap(\.number) ?? []
                    if numbers.count == 4, object["bbox"]?.array?.count == 4, numbers.allSatisfy(\.isFinite) {
                        let v = numbers.map { min(max($0, 0), 1000) }
                        block["box"] = [
                            "xmin": .number(min(v[0], v[2]) * size.width / 1000), "ymin": .number(min(v[1], v[3]) * size.height / 1000),
                            "xmax": .number(max(v[0], v[2]) * size.width / 1000), "ymax": .number(max(v[1], v[3]) * size.height / 1000)
                        ]
                    }
                    return .object(block)
                })
            page["tables"] = .array(
                elements.enumerated().compactMap { index, item in
                    guard item.object?["category"]?.string == "table", let content = item.object?["text"]?.string else { return nil }
                    return ["id": .string("table-\(index)"), "content": .string(content)]
                })
        } else {
            let regex = try NSRegularExpression(pattern: #"(?s)<table\b.*?</table>"#)
            page["tables"] = .array(
                regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).enumerated().compactMap { index, match in
                    guard let range = Range(match.range, in: text) else { return nil }
                    return ["id": .string("table-\(index)"), "content": .string(String(text[range]))]
                })
        }
        if id == "ovisocr2" {
            // Card: <img src="images/bbox_l_t_r_b.jpg" /> with [0, 1000) coordinates, rounded and clamped to the page.
            let regex = try NSRegularExpression(pattern: #"<img src="images/bbox_(\d+)_(\d+)_(\d+)_(\d+)\.jpg"\s*/>"#)
            page["images"] = .array(
                regex.matches(in: text, range: NSRange(text.startIndex..., in: text)).enumerated().map { index, match in
                    let numbers = (1...4).map { Double((text as NSString).substring(with: match.range(at: $0))) ?? 0 }
                    func scaled(_ value: Double, _ extent: CGFloat) -> JSONValue {
                        let limit = Double(extent)
                        return .number(max(0, min(limit, (value * limit / 1000).rounded(.toNearestOrEven))))
                    }
                    return [
                        "id": .string("image-\(index)"), "top_left_x": scaled(numbers[0], size.width), "top_left_y": scaled(numbers[1], size.height),
                        "bottom_right_x": scaled(numbers[2], size.width), "bottom_right_y": scaled(numbers[3], size.height)
                    ]
                })
        }
        return .object(page)
    }

    /// INF-MLLM utils.extract_json_content followed by truncate_last_incomplete_element.
    static func infinityJSON(_ text: String) -> String {
        var value = text
        if let open = value.range(of: "```json\n") {
            let rest = value[open.upperBound...]
            value = (rest.range(of: "\n```").map { String(rest[..<$0.lowerBound]) } ?? String(rest)).trimmingCharacters(in: .whitespacesAndNewlines)
        }
        let marker = "{\"bbox\":"
        guard value.count > 65536 || !value.trimmingCharacters(in: .whitespacesAndNewlines).hasSuffix("]"),
            value.components(separatedBy: marker).count > 2, let last = value.range(of: marker, options: .backwards)
        else { return value }
        var truncated = String(value[..<last.lowerBound])
        while let tail = truncated.last, tail.isWhitespace { truncated.removeLast() }
        if truncated.hasSuffix(",") { truncated.removeLast(); truncated += "]" }
        return truncated
    }
    /// INF-MLLM utils.postprocess_doc2md_result: drop one surrounding Markdown code fence.
    static func infinityMarkdown(_ text: String) -> String {
        var value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        for pattern in [#"^```markdown\s*\n?"#, #"^```\s*\n?"#, #"\n?```$"#] {
            value = value.replacingOccurrences(of: pattern, with: "", options: .regularExpression)
        }
        return value.trimmingCharacters(in: .whitespacesAndNewlines)
    }
    /// OvisOCR2 card `_clean_truncated_repeats`, on Unicode scalars as Python indexes strings.
    static func cleanTruncatedRepeats(_ text: String) -> String {
        let scalars = Array(text.unicodeScalars), count = scalars.count
        guard count >= 8000 else { return text }
        for unit in 1...min(200, count - 1) where scalars[count - 1] == scalars[count - 1 - unit] {
            var matched = 1, index = count - 2
            while index >= unit && scalars[index] == scalars[index - unit] {
                matched += 1; index -= 1
            }
            let total = matched + unit
            if total / unit >= 5 && total >= 100 {
                var view = String.UnicodeScalarView()
                view.append(contentsOf: scalars[..<(count - total + unit)])
                view.append(contentsOf: scalars[(count - total % unit)...])
                return String(view)
            }
        }
        return text
    }
}
