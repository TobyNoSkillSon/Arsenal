// SPDX-License-Identifier: MIT
import Foundation
import ArsenalCore
import MCP

/// The closed list of categories an MCP error line starts with (`[memory] …`). Every other failure maps onto one of them.
enum MCPCategory: String, Sendable, CaseIterable {
    case invalidInput = "invalid_input", unknownModel = "unknown_model", notFound = "not_found", memory, disk, download, busy, cancelled, unsupported, approval
    case failed, limit
    /// Nothing an agent can do: a person must act (accept a gated model's terms on Hugging Face).
    case waitingOnPerson = "waiting_on_person"
}

/// A refusal or failure the agent reads as one `[category] text` line, with `isError` true.
struct MCPFailure: Error, Sendable, LocalizedError, Equatable {
    let category: MCPCategory
    let text: String
    init(_ category: MCPCategory, _ text: String) { self.category = category; self.text = text }
    var line: String { "[\(category.rawValue)] \(text)" }
    var errorDescription: String? { line }
    var result: CallTool.Result { mcpText(line, isError: true) }
}

/// A text result: the agent's whole context is `text`; `extra` carries non-text blocks (resource links, thumbnails).
func mcpText(_ text: String, isError: Bool = false, extra: [Tool.Content] = []) -> CallTool.Result {
    .init(content: [.text(text: text, annotations: nil, _meta: nil)] + extra, structuredContent: nil, isError: isError)
}

/// `format:"json"`: the data as `structuredContent`, with the identical serialized JSON as the text block.
func mcpJSON(_ object: JSONObject, extra: [Tool.Content] = []) throws -> CallTool.Result {
    let text = String(decoding: try WireJSON.encode(object), as: UTF8.self)
    let value: MCP.Value? = try mcpValue(object)
    return CallTool.Result(content: [.text(text: text, annotations: nil, _meta: nil)] + extra, structuredContent: value, isError: false)
}

/// The first text block of a result.
func mcpResultText(_ result: CallTool.Result) -> String {
    for block in result.content { if case .text(let text, _, _) = block { return text } }
    return ""
}

/// Units in text: memory in the app's and Activity Monitor's GB (2^30 bytes); download and disk sizes in decimal units.
enum MCPFormat {
    static let locale = Locale(identifier: "en_US_POSIX")
    static func trimmed(_ value: Double, decimals: Int) -> String {
        var text = String(format: "%.\(decimals)f", locale: locale, value)
        if text.contains(".") {
            while text.hasSuffix("0") { text.removeLast() }
            if text.hasSuffix(".") { text.removeLast() }
        }
        return text
    }
    /// A detected language as its English name ("pl" → "Polish"); nil when nothing was detected ("und", "unknown",
    /// empty), so results never show a raw placeholder code. A name the model already wrote ("English") stays as is.
    static func language(_ value: String?) -> String? {
        guard let raw = value?.trimmingCharacters(in: .whitespacesAndNewlines), !raw.isEmpty,
            !["und", "unknown", "auto", "none", "null", "zxx", "mis", "mul", "n/a"].contains(raw.lowercased())
        else { return nil }
        let code = raw.split { $0 == "-" || $0 == "_" }.first.map(String.init) ?? raw
        guard code.count <= 3, code.allSatisfy(\.isLetter) else { return raw }
        return Locale(identifier: "en_US").localizedString(forLanguageCode: code.lowercased()) ?? raw
    }
    /// Decimal bytes: "180 MB", "2.5 GB", "640 KB".
    static func size(_ bytes: Int64) -> String {
        let value = Double(max(0, bytes))
        if value == 0 { return "0 MB" }
        if value < 1000 { return value == 1 ? "1 byte" : "\(Int(value)) bytes" }
        if value < 1e6 { return "\(Int((value / 1e3).rounded())) KB" }
        if value < 1e9 { return trimmed(value / 1e6, decimals: value < 1e7 ? 1 : 0) + " MB" }
        return trimmed(value / 1e9, decimals: value < 1e11 ? 1 : 0) + " GB"
    }
    /// "1.2 of 3.4 GB": both figures in the total's unit.
    static func progress(_ done: Int64, of total: Int64) -> String {
        let (scale, unit, decimals): (Double, String, Int) = total >= 1_000_000_000 ? (1e9, "GB", 1) : total >= 1_000_000 ? (1e6, "MB", total < 10_000_000 ? 1 : 0) : (1e3, "KB", 0)
        return trimmed(Double(done) / scale, decimals: decimals) + " of " + trimmed(Double(total) / scale, decimals: decimals) + " " + unit
    }
    /// Memory for structured answers, in the GB the text says (the app's, as Activity Monitor counts it), two decimals:
    /// the 100 GB limit is 100, never its 107.37 decimal gigabytes.
    static func memoryValue(decimalGB: Double) -> JSONValue {
        .number((MemoryDisplay.gib(decimalGB: max(0, decimalGB)) * 100).rounded() / 100)
    }
    /// Memory from decimal GB to the app's GB: "4.1 GB" under 10, "61 GB" above.
    static func memory(decimalGB: Double) -> String {
        let value = MemoryDisplay.gib(decimalGB: max(0, decimalGB))
        return trimmed(value, decimals: value < 9.95 ? 1 : 0) + " GB"
    }
    /// A remaining time: "under a minute", "about 4 min", "about 1.6 h".
    static func remaining(_ seconds: Double) -> String {
        guard seconds.isFinite else { return "time unknown" }
        if seconds < 60 { return "under a minute" }
        if seconds < 90 * 60 { return "about \(max(1, Int((seconds / 60).rounded()))) min" }
        return "about " + trimmed(seconds / 3600, decimals: 1) + " h"
    }
    /// An elapsed time: "4 s", "3 min 40 s", "1 h 5 min".
    static func elapsed(_ seconds: Double) -> String {
        let total = max(0, Int(seconds.rounded()))
        if total < 60 { return "\(total) s" }
        if total < 3600 { return total % 60 == 0 ? "\(total / 60) min" : "\(total / 60) min \(total % 60) s" }
        return (total % 3600) / 60 == 0 ? "\(total / 3600) h" : "\(total / 3600) h \((total % 3600) / 60) min"
    }
    /// "11 MB/s", "2.5 MB/s", "800 KB/s".
    static func rate(_ bytesPerSecond: Double) -> String {
        if bytesPerSecond < 1e6 { return "\(max(1, Int((bytesPerSecond / 1e3).rounded()))) KB/s" }
        return trimmed(bytesPerSecond / 1e6, decimals: bytesPerSecond < 1e7 ? 1 : 0) + " MB/s"
    }
    /// Seconds of media: "3 min 12 s of audio".
    static func clock(_ seconds: Double) -> String {
        let tenths = max(0, Int((seconds * 10).rounded()))
        return String(format: "%02d:%02d.%d", locale: locale, tenths / 600, (tenths / 10) % 60, tenths % 10)
    }
    static func number(_ value: Double, decimals: Int = 2) -> String { trimmed(value, decimals: decimals) }
    /// Declared phrases joined into one line; a phrase's own final period is not doubled.
    static func phrases(_ items: [String]) -> String {
        items.map { $0.trimmingCharacters(in: .whitespaces).replacingOccurrences(of: #"\.$"#, with: "", options: .regularExpression) }.joined(separator: "; ")
    }
    static func plural(_ count: Int, _ word: String, _ plural: String? = nil) -> String { "\(count) " + (count == 1 ? word : plural ?? word + "s") }
}
