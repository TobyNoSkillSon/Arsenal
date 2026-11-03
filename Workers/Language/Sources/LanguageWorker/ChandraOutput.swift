// SPDX-License-Identifier: MIT
import Foundation
import CoreGraphics
import ArsenalWire

/// Chandra OCR 2 output, ported from datalab-to/chandra `chandra/output.py`: the model writes one top-level
/// `<div data-bbox="x0 y0 x1 y1" data-label="…">` per layout block on a 0–1000 grid.
/// - `parse_html` + `parse_markdown`: Blank-Page, Page-Header and Page-Footer blocks are dropped, text-only Text
///   blocks are wrapped in `<p>`, src-less images outside Image/Figure blocks are removed, then markdownify runs with
///   ATX headings, `-` bullets, `$…$`/`$$…$$` math, escaped `_`, `*` and `$`, and tables kept as HTML.
/// - `parse_layout`: every block except Blank-Page with its box scaled by width/1000 and height/1000 and its inner
///   HTML without nested data-bbox attributes.
/// Image and Figure blocks link no extracted image file here; their description text stays in the Markdown and the
/// block keeps the region.
enum ChandraOutput {
    static func parse(_ raw: String, size: CGSize) -> (markdown: String, blocks: [JSONValue]) {
        let divs = HTML.parse(raw).compactMap { node -> HTML.Element? in
            if case .element(let element) = node, element.name == "div" { return element }
            return nil
        }
        var html: [HTML.Node] = [], blocks: [JSONValue] = []
        for div in divs {
            let label = div.attribute("data-label")
            if label == "Blank-Page" { continue }
            blocks.append(block(div, label: label, size: size))
            if label == "Page-Header" || label == "Page-Footer" { continue }
            html += content(div, label: label)
        }
        return (Markdownify().convert(html).trimmingCharacters(in: .whitespacesAndNewlines), blocks)
    }
    /// parse_html's per-block rewrite of the div's children.
    private static func content(_ div: HTML.Element, label: String?) -> [HTML.Node] {
        if label == "Image" || label == "Figure" { return div.children.removingElements(named: "img") }
        let children = div.children.removingElements(named: "img", where: { ($0.attribute("src") ?? "").isEmpty })
        let inner = HTML.serialize(children).trimmingCharacters(in: .whitespacesAndNewlines)
        // Text blocks without inner markup get their paragraph tags (parse_html).
        if label == "Text", !inner.contains("<") { return HTML.parse("<p>" + inner + "</p>") }
        return children
    }
    private static func block(_ div: HTML.Element, label: String?, size: CGSize) -> JSONValue {
        let numbers = (div.attribute("data-bbox") ?? "").split(separator: " ", omittingEmptySubsequences: false).map { Int($0) }
        let bbox = numbers.count == 4 && numbers.allSatisfy({ $0 != nil }) ? numbers.map { $0! } : [0, 0, 1, 1]
        let width = Double(size.width), height = Double(size.height)
        func scaled(_ value: Int, _ extent: Double) -> JSONValue { .number(min(max(0, Double(Int(Double(value) * extent / 1000))), extent)) }
        var copy = div.children
        copy.removeAttribute("data-bbox")
        return [
            "category": .string(label ?? "block"), "text": .string(HTML.serialize(copy)), "coordinate_system": "pixels",
            "box": ["xmin": scaled(bbox[0], width), "ymin": scaled(bbox[1], height), "xmax": scaled(bbox[2], width), "ymax": scaled(bbox[3], height)]
        ]
    }
}

/// A small HTML tree in the shape BeautifulSoup's html.parser builds: no implied end tags, void elements, entities.
enum HTML {
    final class Element {
        let name: String
        var attributes: [(String, String)]
        var children: [Node] = []
        init(name: String, attributes: [(String, String)]) { self.name = name; self.attributes = attributes }
        func attribute(_ key: String) -> String? { attributes.first { $0.0 == key }?.1 }
        func copy() -> Element {
            let element = Element(name: name, attributes: attributes)
            element.children = children.map { $0.copy() }
            return element
        }
    }
    enum Node {
        case text(String)
        case element(Element)
        func copy() -> Node {
            if case .element(let element) = self { return .element(element.copy()) }
            return self
        }
    }
    static let void: Set<String> = ["br", "img", "hr", "input", "meta", "link", "col", "area", "base", "wbr", "source", "embed", "param", "track"]

    static func parse(_ html: String) -> [Node] {
        let root = Element(name: "", attributes: [])
        var stack = [root]
        var text = ""
        let chars = Array(html)
        var index = 0
        func flush() {
            if !text.isEmpty { stack[stack.count - 1].children.append(.text(entities(text))); text = "" }
        }
        while index < chars.count {
            let char = chars[index]
            guard char == "<", index + 1 < chars.count else { text.append(char); index += 1; continue }
            let next = chars[index + 1]
            if next == "!" {
                // Comments and doctypes carry no content.
                let close = String(chars[index...]).hasPrefix("<!--") ? "-->" : ">"
                flush()
                if let end = find(close, in: chars, from: index + 2) { index = end + close.count } else { index = chars.count }
                continue
            }
            if next == "/" {
                guard let end = find(">", in: chars, from: index + 2) else { text.append(char); index += 1; continue }
                let name = String(chars[(index + 2)..<end]).trimmingCharacters(in: .whitespaces).lowercased()
                flush()
                if let position = stack.lastIndex(where: { $0.name == name }), position > 0 { stack.removeSubrange(position...) }
                index = end + 1
                continue
            }
            guard next.isLetter, let end = tagEnd(chars, from: index + 1) else { text.append(char); index += 1; continue }
            flush()
            var body = String(chars[(index + 1)..<end])
            var selfClosing = false
            if body.hasSuffix("/") { selfClosing = true; body.removeLast() }
            let (name, attributes) = tag(body)
            let element = Element(name: name, attributes: attributes)
            stack[stack.count - 1].children.append(.element(element))
            if !selfClosing && !void.contains(name) { stack.append(element) }
            index = end + 1
        }
        flush()
        return root.children
    }
    /// The `>` closing a start tag, skipping quoted attribute values.
    private static func tagEnd(_ chars: [Character], from start: Int) -> Int? {
        var quote: Character?
        var index = start
        while index < chars.count {
            let char = chars[index]
            if let open = quote { if char == open { quote = nil } } else if char == "\"" || char == "'" { quote = char } else if char == ">" { return index }
            index += 1
        }
        return nil
    }
    private static func find(_ needle: String, in chars: [Character], from start: Int) -> Int? {
        let target = Array(needle)
        guard start <= chars.count - target.count else { return nil }
        for index in start...(chars.count - target.count) where Array(chars[index..<(index + target.count)]) == target { return index }
        return nil
    }
    private static func tag(_ body: String) -> (String, [(String, String)]) {
        let chars = Array(body)
        var index = 0
        var name = ""
        while index < chars.count, !chars[index].isWhitespace { name.append(chars[index]); index += 1 }
        var attributes: [(String, String)] = []
        while index < chars.count {
            while index < chars.count, chars[index].isWhitespace { index += 1 }
            var key = ""
            while index < chars.count, !chars[index].isWhitespace, chars[index] != "=" { key.append(chars[index]); index += 1 }
            while index < chars.count, chars[index].isWhitespace { index += 1 }
            var value = ""
            if index < chars.count, chars[index] == "=" {
                index += 1
                while index < chars.count, chars[index].isWhitespace { index += 1 }
                if index < chars.count, chars[index] == "\"" || chars[index] == "'" {
                    let quote = chars[index]; index += 1
                    while index < chars.count, chars[index] != quote { value.append(chars[index]); index += 1 }
                    index += 1
                } else {
                    while index < chars.count, !chars[index].isWhitespace { value.append(chars[index]); index += 1 }
                }
            }
            if !key.isEmpty, !attributes.contains(where: { $0.0 == key.lowercased() }) { attributes.append((key.lowercased(), entities(value))) }
        }
        return (name.lowercased(), attributes)
    }
    static func entities(_ text: String) -> String {
        guard text.contains("&") else { return text }
        let named = ["amp": "&", "lt": "<", "gt": ">", "quot": "\"", "apos": "'", "nbsp": "\u{00A0}"]
        guard let regex = try? NSRegularExpression(pattern: "&(#[0-9]+|#[xX][0-9a-fA-F]+|[a-zA-Z]+);") else { return text }
        var result = "", last = text.startIndex
        for match in regex.matches(in: text, range: NSRange(text.startIndex..., in: text)) {
            guard let range = Range(match.range, in: text), let body = Range(match.range(at: 1), in: text) else { continue }
            result += text[last..<range.lowerBound]
            let entity = String(text[body])
            var value: String?
            if entity.hasPrefix("#x") || entity.hasPrefix("#X") {
                value = UInt32(entity.dropFirst(2), radix: 16).flatMap { Unicode.Scalar($0) }.map { String(Character($0)) }
            } else if entity.hasPrefix("#") {
                value = UInt32(entity.dropFirst()).flatMap { Unicode.Scalar($0) }.map { String(Character($0)) }
            } else {
                value = named[entity]
            }
            result += value ?? String(text[range])
            last = range.upperBound
        }
        return result + text[last...]
    }
    /// BeautifulSoup's default (minimal) serialization: `&`, `<`, `>` escaped in text, attribute values double-quoted.
    static func serialize(_ nodes: [Node]) -> String { nodes.map(serialize).joined() }
    static func serialize(_ node: Node) -> String {
        switch node {
        case .text(let text): return escape(text)
        case .element(let element):
            let attributes = element.attributes.map { " \($0.0)=\"\(escape($0.1).replacingOccurrences(of: "\"", with: "&quot;"))\"" }.joined()
            if void.contains(element.name) { return "<\(element.name)\(attributes)/>" }
            return "<\(element.name)\(attributes)>\(serialize(element.children))</\(element.name)>"
        }
    }
    static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;").replacingOccurrences(of: "<", with: "&lt;").replacingOccurrences(of: ">", with: "&gt;")
    }
    /// Concatenated text of a subtree.
    static func text(_ nodes: [Node]) -> String {
        nodes.map { node in
            switch node {
            case .text(let text): return text
            case .element(let element): return text(element.children)
            }
        }.joined()
    }
}

extension Array where Element == HTML.Node {
    /// A copy without elements of this name (at any depth) that match the condition.
    func removingElements(named name: String, where condition: (HTML.Element) -> Bool = { _ in true }) -> [HTML.Node] {
        compactMap { node in
            guard case .element(let element) = node else { return node }
            if element.name == name && condition(element) { return nil }
            let copy = HTML.Element(name: element.name, attributes: element.attributes)
            copy.children = element.children.removingElements(named: name, where: condition)
            return .element(copy)
        }
    }
    /// Removes an attribute from every element (copies the elements first).
    mutating func removeAttribute(_ key: String) {
        self = map { node in
            guard case .element(let element) = node else { return node }
            let copy = HTML.Element(name: element.name, attributes: element.attributes.filter { $0.0 != key })
            copy.children = element.children
            copy.children.removeAttribute(key)
            return .element(copy)
        }
    }
}

/// The markdownify conversions Chandra's output uses (MarkdownConverter with Chandra's Markdownify overrides).
struct Markdownify {
    func convert(_ nodes: [HTML.Node]) -> String {
        let text = children(nodes, parents: [])
        return text.replacingOccurrences(of: #"\n{3,}"#, with: "\n\n", options: .regularExpression)
    }
    private func children(_ nodes: [HTML.Node], parents: [String]) -> String {
        nodes.indices.map { index in
            switch nodes[index] {
            case .text(let text):
                // A list item's last text, or text before a nested list, loses its trailing whitespace.
                var listEnd = false
                if parents.last == "li" {
                    if index == nodes.count - 1 { listEnd = true } else if case .element(let next) = nodes[index + 1], ["ul", "ol"].contains(next.name) { listEnd = true }
                }
                return process(text, parents: parents, listEnd: listEnd)
            case .element(let element): return convert(element, parents: parents)
            }
        }.joined()
    }
    /// process_text: collapse spaces and tabs outside `pre`; escape outside pre/code/kbd/samp/math.
    private func process(_ text: String, parents: [String], listEnd: Bool) -> String {
        var value = text
        if !parents.contains("pre") { value = value.replacingOccurrences(of: #"[\t ]+"#, with: " ", options: .regularExpression) }
        if !parents.contains(where: { ["pre", "code", "kbd", "samp", "math"].contains($0) }) {
            value = value.replacingOccurrences(of: "*", with: "\\*").replacingOccurrences(of: "_", with: "\\_").replacingOccurrences(of: "$", with: "\\$")
        }
        if listEnd { value = value.trimmingTrailingWhitespace() }
        return value
    }
    private func convert(_ element: HTML.Element, parents: [String]) -> String {
        let name = element.name
        if name == "table" { return "\n\n" + HTML.serialize(.element(element)) + "\n\n" }
        let inner = parents + [name]
        if name == "ul" || name == "ol" {
            var number = Int(element.attribute("start") ?? "") ?? 1
            var items = ""
            for child in element.children {
                if case .element(let item) = child, item.name == "li" {
                    items += listItem(children(item.children, parents: inner + ["li"]), bullet: name == "ol" ? "\(number)." : "-")
                    number += 1
                } else {
                    items += children([child], parents: inner)
                }
            }
            if parents.last == "li" { return "\n" + items.trimmingTrailingWhitespace() }
            return "\n\n" + items + "\n\n"
        }
        let text = children(element.children, parents: inner)
        let heading = parents.contains { ["h1", "h2", "h3", "h4", "h5", "h6"].contains($0) }
        switch name {
        case "p", "div", "section", "article":
            if heading { return " " + text.trimmingCharacters(in: .whitespacesAndNewlines) + " " }
            let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return body.isEmpty ? "" : "\n\n" + body + "\n\n"
        case "br": return heading ? "" : "  \n"
        case "b", "strong": return inline(text, "**", "**")
        case "i", "em": return inline(text, "*", "*")
        case "del", "s": return inline(text, "~~", "~~")
        case "sub", "sup": return inline(text, "<\(name)>", "</\(name)>")
        case "code", "kbd", "samp": return parents.contains("pre") ? text : inline(text, "`", "`")
        case "pre": return "\n\n```\n" + text + "\n```\n\n"
        case "h1", "h2", "h3", "h4", "h5", "h6":
            let level = Int(String(name.last!)) ?? 1
            let body = text.replacingOccurrences(of: "\n", with: " ").trimmingCharacters(in: .whitespaces)
            return "\n\n" + String(repeating: "#", count: level) + " " + body + "\n\n"
        case "hr": return "\n\n---\n\n"
        case "math":
            let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return element.attribute("display") == "block" ? "\n$$" + body + "$$\n" : " $" + body + "$ "
        case "a":
            let label = text.replacingOccurrences(of: #"([\[\]()])"#, with: #"\\$1"#, options: .regularExpression)
            guard let href = element.attribute("href"), !href.isEmpty else { return label }
            if label == href { return "<" + href + ">" }
            return inline(label, "[", "](\(href))")
        case "img":
            let alt = element.attribute("alt") ?? "", src = element.attribute("src") ?? ""
            return "![\(alt)](\(src))"
        case "li": return listItem(text, bullet: "-")
        case "blockquote":
            let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
            return body.isEmpty ? "" : "\n\n" + body.components(separatedBy: "\n").map { "> " + $0 }.joined(separator: "\n") + "\n\n"
        default: return text
        }
    }
    /// markdownify chomp: surrounding spaces move outside the markup; empty content drops it.
    private func inline(_ text: String, _ prefix: String, _ suffix: String) -> String {
        let leading = text.first == " " ? " " : "", trailing = text.last == " " ? " " : ""
        let body = text.trimmingCharacters(in: .whitespaces)
        return body.isEmpty ? "" : leading + prefix + body + suffix + trailing
    }
    /// convert_li: the bullet, then the item's text with continuation lines indented under it.
    private func listItem(_ text: String, bullet: String) -> String {
        let body = text.trimmingCharacters(in: .whitespacesAndNewlines)
        let indent = String(repeating: " ", count: bullet.count + 1)
        let lines = body.components(separatedBy: "\n")
        let indented = ([lines.first ?? ""] + lines.dropFirst().map { $0.isEmpty ? "" : indent + $0 }).joined(separator: "\n")
        return bullet + " " + indented + "\n"
    }
}

private extension String {
    func trimmingTrailingWhitespace() -> String {
        var value = self
        while let last = value.last, last.isWhitespace { value.removeLast() }
        return value
    }
}
