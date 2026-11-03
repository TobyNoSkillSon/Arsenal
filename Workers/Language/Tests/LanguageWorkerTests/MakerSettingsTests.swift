// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalWire
@testable import LanguageWorker

/// The language folders' budgets and bounds follow each maker's own client or card, and every generating operation can
/// report a stopped loop.
struct MakerSettingsTests {
    private static let models = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        .deletingLastPathComponent().deletingLastPathComponent().appendingPathComponent("models")
    private func json(_ path: String) throws -> JSONObject {
        let shipped = Self.models.appendingPathComponent(path)
        let url =
            FileManager.default.fileExists(atPath: shipped.path)
            ? shipped : Self.models.deletingLastPathComponent().appendingPathComponent("retired-models").appendingPathComponent(path)
        return try WireJSON.decode(Data(contentsOf: url))
    }

    /// chandra settings.py MAX_OUTPUT_TOKENS 12384; TeleOCR card example max_new_tokens 4096; Index-Translate card client
    /// max_tokens 1024; NuExtract3 card example max_new_tokens 4096; UI-Venus cards max_new_tokens 256.
    @Test(arguments: [
        ("chandra-ocr-2", "read_document", 12384), ("teleocr", "read_document", 4096), ("index-translate-2b", "translate", 1024),
        ("index-translate-9b", "translate", 1024), ("nuextract3", "extract", 4096), ("nuextract3", "read_document", 4096),
        ("ui-venus-1.5-2b", "ground_screen", 256), ("ui-venus-2-9b", "ground_screen", 256)
    ])
    func outputBudgetsFollowTheMakers(id: String, operation: String, tokens: Int) throws {
        let schema = try json("\(id)/ops/\(operation).input.json")
        #expect(schema["properties"]?.object?["max_tokens"]?.object?["default"]?.number == Double(tokens))
    }

    @Test(arguments: [
        ("chandra-ocr-2", "read_document"), ("teleocr", "read_document"), ("nuextract3", "read_document"), ("nuextract3", "extract"),
        ("index-translate-2b", "translate"), ("index-translate-9b", "translate"), ("ui-venus-1.5-2b", "ground_screen"), ("ui-venus-2-9b", "ground_screen")
    ])
    func outputsCanSayALoopWasStopped(id: String, operation: String) throws {
        let properties = try #require(try json("\(id)/ops/\(operation).output.json")["properties"]?.object)
        let reason: JSONObject?
        switch operation {
        case "read_document": reason = properties["pages"]?.object?["items"]?.object?["properties"]?.object?["finish_reason"]?.object
        case "translate": reason = properties["translations"]?.object?["items"]?.object?["properties"]?.object?["finish_reason"]?.object
        default: reason = properties["finish_reason"]?.object
        }
        #expect(reason?["enum"]?.array?.contains("repetition") == true)
        #expect(reason?["enum"]?.array?.contains("length") == true)
    }

    @Test func documentPagesSayWhyTheyStopped() {
        var page: JSONObject = ["index": 0, "markdown": "x"]
        LanguageNative.mark(&page, reason: "stop")
        #expect(page["truncated"] == nil && page["finish_reason"] == nil)
        LanguageNative.mark(&page, reason: "repetition")
        #expect(page["truncated"] == true && page["finish_reason"]?.string == "repetition")
    }

    /// ui_venus2_gd.py sends max_pixels 12,845,056 (and the 1.5 client's qwen_vl_utils resize defaults to the same
    /// 16384·28·28); the checkpoint processors allow 16,777,216. The floor stays the processor's 65,536 px.
    @Test(arguments: ["ui-venus-1.5-2b", "ui-venus-2-9b"]) func venusPixelBoundsFollowTheGroundingClient(id: String) throws {
        let spec = try json("\(id)/runners/native/language.json")
        var processor: JSONObject = ["min_pixels": 65536, "max_pixels": 16_777_216]
        try LanguageNative.applyPixelLimits(&processor, spec: spec, architecture: id == "ui-venus-1.5-2b" ? "qwen3_vl" : "qwen3_5")
        #expect(processor["max_pixels"]?.number == 12_845_056)
        #expect(processor["min_pixels"]?.number == 65536)
    }

    /// Index-Translate cards: "--max-model-len 32768, matching the GitHub 2B/9B Translate serving preset".
    @Test(arguments: ["index-translate-2b", "index-translate-9b"]) func indexServesTheMakersContext(id: String) throws {
        #expect(try json("\(id)/runners/native/language.json")["max_context"]?.number == 32768)
    }
}
