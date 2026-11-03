// SPDX-License-Identifier: MIT
import Foundation
import CoreImage
import ArsenalWire
import IntegrationKit

/// NaviDC-OCR TeleOCR_client.py: layout lines retain reading order, normalized boxes/polygons and orientation.
/// Markdown assembly follows the same repository's vlm_middle_json_mkcontent.py block rules.
enum TeleDocument {
    struct Block {
        let category: String
        let points: [CGPoint]
        let angle: Int
        var box: CGRect {
            let xs = points.map(\.x), ys = points.map(\.y)
            return CGRect(x: xs.min()!, y: ys.min()!, width: xs.max()! - xs.min()!, height: ys.max()! - ys.min()!)
        }
    }
    /// Not sent to recognition (prepare_for_extract skip_list).
    static let skip: Set<String> = ["image", "list", "equation_block"]
    /// Page furniture kept as blocks but left out of the Markdown body (vlm_magic_model discarded_blocks).
    static let discarded: Set<String> = ["header", "footer", "page_number", "aside_text", "page_footnote"]
    static let categories: Set<String> = [
        "text", "title", "table", "image", "code", "algorithm", "header", "footer", "page_number", "page_footnote", "aside_text", "equation", "equation_block", "ref_text", "list",
        "phonetic", "table_caption", "image_caption", "code_caption", "table_footnote", "image_footnote", "unknown", "seal", "char"
    ]
    /// parse_layout_output: malformed lines, unknown labels and invalid boxes are skipped, as the reference does.
    static func parse(_ text: String, size: CGSize) throws -> [Block] {
        let regex = try NSRegularExpression(pattern: #"^<box:([\d\s]+)><label:(\w+)><([^>]+)>$"#)
        return text.components(separatedBy: "\n").compactMap { line -> Block? in
            let line = line.trimmingCharacters(in: .whitespacesAndNewlines)
            guard let match = regex.firstMatch(in: line, range: NSRange(line.startIndex..., in: line)) else { return nil }
            let fields = (1...3).map { (line as NSString).substring(with: match.range(at: $0)) }
            let values = fields[0].split(whereSeparator: \.isWhitespace).compactMap { Double($0) }
            let category = fields[1].lowercased()
            guard values.count >= 4, values.count % 2 == 0, values.count <= 256, values.allSatisfy({ $0 >= 0 && $0 <= 1000 }), categories.contains(category) else {
                return nil
            }
            let points = stride(from: 0, to: values.count, by: 2).map { index in
                CGPoint(
                    x: (CGFloat(min(values[index], 999)) * size.width / 1000).rounded(.towardZero),
                    y: (CGFloat(min(values[index + 1], 999)) * size.height / 1000).rounded(.towardZero))
            }
            let angle = [("up", 0), ("right", 90), ("down", 180), ("left", 270)].first { fields[2].contains($0.0) }?.1 ?? 0
            let block = Block(category: category, points: points, angle: angle)
            return block.box.width >= 1 && block.box.height >= 1 ? block : nil
        }
    }
    /// The TeleOCR_client sampling kind of a direct read_document mode (`penalties` keys in language.json).
    static func kind(mode: String) -> String? {
        ["table": "table", "scientific_figure": "char", "seal": "seal", "layout": "layout", "distorted_layout": "layout", "formula": "equation", "code": "code"][mode]
    }
    static func prompt(_ type: String) -> String {
        switch type {
        case "table": return "\nThis is the image of a table. Please output the table in OTSL format."
        case "char": return "\nThis is a scientific figure. Please extract the table implied by this figure."
        case "equation": return "\nPlease write out the expression of the formula in the image using LaTeX format."
        case "code": return "\nThe image contains a code snippet, please output the parsing result."
        case "seal": return "\nSeal Recognition:"
        default: return "\nPlease output the text content from the image."
        }
    }
    static func content(_ text: String, type: String) throws -> String {
        if ["table", "char"].contains(type) { return try OTSLOutput.html(text) }
        var value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        if type == "equation", !value.isEmpty {
            // Card post_process: removeprefix("\\["), removesuffix("\\]"), strip; wrap unless it starts and ends with $.
            if value.hasPrefix("\\[") { value.removeFirst(2) }
            if value.hasSuffix("\\]") { value.removeLast(2) }
            value = value.trimmingCharacters(in: .whitespacesAndNewlines)
            if !(value.hasPrefix("$") && value.hasSuffix("$")) { value = "$$" + value + "$$" }
        }
        return value
    }
    /// mk_blocks_to_markdown: titles become headings, code is fenced, short scientific-figure text and blocks without
    /// a Markdown rule (seal, unknown, image) stay in `blocks` only.
    static func markdown(_ text: String, type: String) -> String? {
        let value = text.trimmingCharacters(in: .whitespacesAndNewlines)
        guard !value.isEmpty, !discarded.contains(type), !["seal", "unknown", "image", "list", "equation_block"].contains(type) else { return nil }
        switch type {
        case "title": return "# " + value
        case "code": return value.hasPrefix("```") ? value : "```\n" + value + "\n```"
        case "char": return value.count > 5 ? value : nil
        default: return value
        }
    }
    /// `infer(prompt, image, kind)`: kind is `layout` or the block category, which selects the maker's sampling penalties.
    /// TeleOCR_client LAYOUT_PROMPTS (Detection, Segmentation).
    static func layoutPrompt(distorted: Bool) -> String { distorted ? "\nMulti-point Layout Segmentation Analysis." : "\nAnalyze the image layout." }
    /// The client sends the layout call a 1036×1036 bicubic resize of the page (37×37 merged patches).
    static func layoutImage(_ image: CIImage) -> CIImage { DocumentRegions.resize(image, width: 1036, height: 1036) }
    /// Direct layout and distorted_layout modes: the parsed layout blocks without recognition.
    static func layout(_ raw: String, index: Int, size: CGSize) throws -> JSONValue {
        let blocks = try parse(raw, size: size)
        guard !blocks.isEmpty || raw.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { throw invalid() }
        return [
            "index": .number(Double(index)), "markdown": "", "tables": [],
            "blocks": .array(
                blocks.map { block in
                    [
                        "category": .string(block.category), "text": "", "coordinate_system": "pixels",
                        "box": ["xmin": .number(block.box.minX), "ymin": .number(block.box.minY), "xmax": .number(block.box.maxX), "ymax": .number(block.box.maxY)]
                    ]
                }),
            "dimensions": ["width": .number(size.width), "height": .number(size.height)]
        ]
    }
    static func page(
        index: Int, image: CIImage, distorted: Bool, cancellation: NativeCancellation, infer: (String, CIImage, String) throws -> String
    ) throws -> JSONValue {
        let raw = try infer(layoutPrompt(distorted: distorted), layoutImage(image), "layout")
        let blocks = try parse(raw, size: image.extent.size)
        guard !blocks.isEmpty || raw.isEmpty else { throw invalid() }
        var output: [JSONValue] = [], body: [String] = [], tables: [JSONValue] = []
        for (number, block) in blocks.enumerated() where block.category != "equation_block" {
            try cancellation.check()
            let text: String
            if skip.contains(block.category) {
                text = ""
            } else {
                let crop = try DocumentRegions.crop(image, box: block.box, polygon: block.points.count > 2 ? block.points : [], angle: block.angle)
                text = try content(infer(prompt(block.category), DocumentRegions.recognitionCrop(crop), block.category), type: block.category)
            }
            output.append([
                "category": .string(block.category), "text": .string(text), "coordinate_system": "pixels",
                "box": ["xmin": .number(block.box.minX), "ymin": .number(block.box.minY), "xmax": .number(block.box.maxX), "ymax": .number(block.box.maxY)]
            ])
            if ["table", "char"].contains(block.category), !text.isEmpty { tables.append(["id": .string("table-\(number)"), "content": .string(text)]) }
            if let entry = markdown(text, type: block.category) { body.append(entry) }
        }
        return [
            "index": .number(Double(index)), "markdown": .string(body.joined(separator: "\n\n")), "blocks": .array(output), "tables": .array(tables),
            "dimensions": ["width": .number(image.extent.width), "height": .number(image.extent.height)]
        ]
    }
    static func invalid() -> WorkerFailure { WorkerFailure(code: "invalid_model_output", message: "TeleOCR returned no valid layout lines.") }
}
