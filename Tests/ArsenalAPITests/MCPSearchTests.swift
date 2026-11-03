// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import ArsenalCore
import MCP
@testable import ArsenalAPI

/// `arsenal_find` relevance over the shipped catalog: the queries an agent walk-through got wrong (21 Oct) and a few
/// more, each asserting which models lead and which unrelated ones never appear.
@Suite struct MCPSearchTests {
    static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
    /// Every shipped catalog model as `arsenal_find` sees it: its manifest's words, tasks, languages, media and operations.
    static let shipped: (entries: [CatalogEntry], operations: [String: [String]]) = {
        let catalog = (try? WireJSON.decode(Data(contentsOf: repo.appendingPathComponent("Resources/catalog/catalog.json")))) ?? [:]
        var entries: [CatalogEntry] = [], operations: [String: [String]] = [:]
        for model in catalog["models"]?.array ?? [] {
            guard let manifest = model["manifest"]?.object, let id = manifest["id"]?.string else { continue }
            var row = ModelRow(id: id, name: manifest["name"]?.string ?? id, kind: manifest["kind"]?.string ?? "", summary: manifest["summary"]?.string ?? "")
            row.purpose = manifest["purpose"]?.string ?? ""
            entries.append(CatalogEntry(row: row, metadata: manifest))
            operations[id] = manifest["op"]?.array?.compactMap { $0["name"]?.string } ?? []
        }
        return (entries, operations)
    }()
    /// Ids of the models a query finds, best first (ties by id).
    func search(_ query: String) -> [String] {
        let scores = MCPSearch.relevant(Self.shipped.entries, operations: Self.shipped.operations, query: query)
        return scores.keys.sorted { scores[$0]! != scores[$1]! ? scores[$0]! > scores[$1]! : $0 < $1 }
    }

    @Test func theShippedCatalogIsLoaded() {
        #expect(Self.shipped.entries.count >= 37 && Self.shipped.operations["index-translate-2b"] == ["translate"])
    }

    @Test(arguments: [
        // (query, models that must lead in any order, models that must not appear), against the 101-model catalog the release
        // task signs from this branch; with the 77-model signed catalog the queries that name this branch's models fail.
        // Withdrawn models (retired-models/) are not in the catalog, so they never lead.
        ("read a scanned invoice", ["chandra-ocr-2", "teleocr"], ["kokoro-82m", "laya-typed-decisions", "birefnet"]),
        ("translate Polish", ["index-translate-2b", "index-translate-9b"], ["birefnet", "parakeet-v3-ultra", "qwen3-asr-0.6b", "qwen3-asr-1.7b", "d1-3b"]),
        (
            "transcribe speech", ["parakeet-v3-ultra", "qwen3-asr-0.6b", "qwen3-asr-1.7b", "moss-transcribe-diarize"],
            ["birefnet", "kokoro-82m", "index-translate-2b", "pyannote-community-1"]
        ),
        ("depth", ["da3metric-large", "depth-anything-v2-small", "da3-small", "moge-3-vitl"], ["birefnet", "trellis2-4b"]),
        ("remove background", ["birefnet"], ["trellis2-4b", "sam-audio-small", "re-use"]),
        ("find a button on screen", ["ui-venus-1.5-2b"], ["rf-detr-nano", "birefnet", "kokoro-82m"]),
        ("separate vocals", ["htdemucs-ft"], ["ace-step-1.5", "yue2-3b", "kokoro-82m", "birefnet"]),
        ("transcribe Japanese", ["qwen3-asr-0.6b", "qwen3-asr-1.7b"], ["parakeet-v3-ultra", "index-translate-2b"]),
        ("clone a voice", ["voxcpm2", "pocket-tts", "qwen3-tts-base"], ["qwen3-tts-voicedesign", "kokoro-82m"]),
        ("restore the treble of an old recording", ["audiosr-basic"], ["birefnet", "kokoro-82m"]),
        ("read this text aloud", ["voxcpm2", "fish-drama-3", "kokoro-82m", "pocket-tts", "qwen3-tts-base", "qwen3-tts-voicedesign"], ["teleocr"]),
        ("search photos", ["tipsv2-l14", "embeddinggemma-2", "lco-embedding-omni-7b", "ovis-vl-embedding-2b"], ["jina-reranker-v3.5", "qwen3-reranker-8b", "harrier-oss-v1-0.6b"]),
        (
            "who spoke when in a meeting", ["diarizen-large-v2", "sortformer-4spk-v2", "moss-transcribe-diarize", "pyannote-community-1", "nemotron-3-diarization"],
            ["birefnet", "kokoro-82m", "smart-turn-v3"]
        ),
        (
            "identify speakers", ["moss-transcribe-diarize", "sortformer-4spk-v2", "diarizen-large-v2", "nemotron-3-diarization", "pyannote-community-1"],
            ["birefnet", "kokoro-82m", "htdemucs-ft"]
        ),
        ("speaker turns in a meeting", ["diarizen-large-v2", "moss-transcribe-diarize", "sortformer-4spk-v2", "pyannote-community-1"], ["birefnet", "kokoro-82m", "smart-turn-v3"]),
        ("turn taking", ["smart-turn-v3"], ["pyannote-community-1", "kokoro-82m", "birefnet"]),
        ("finished speaking", ["smart-turn-v3"], ["pyannote-community-1", "kokoro-82m", "birefnet"]),
        ("isolate vocals", ["melband-roformer-kim", "sam-audio-large", "tiger-dnr", "htdemucs-ft"], ["kokoro-82m", "birefnet", "pyannote-community-1"]),
        ("stems", ["htdemucs-ft"], ["kokoro-82m", "birefnet", "pyannote-community-1"]),
        ("remove a sound", ["htdemucs-ft"], ["re-use", "birefnet", "kokoro-82m"]),
        ("remove the sound from a video", ["tiger-dnr", "htdemucs-ft"], ["re-use", "birefnet", "kokoro-82m"]),
        ("how far", ["da3metric-large", "da3-small", "depth-anything-v2-small", "moge-3-vitl"], ["birefnet", "trellis2-4b", "kokoro-82m"]),
        ("click on a button", ["ui-venus-1.5-2b"], ["rf-detr-nano", "birefnet", "kokoro-82m"]),
        ("hero background image", ["flux3-image-klein", "flux2-klein-4b"], ["kokoro-82m", "parakeet-v3-ultra", "index-translate-2b"]),
        ("edit a photo", [], ["kokoro-82m", "birefnet"])
    ])
    func goldenQueriesLeadWithWhatTheModelDoes(query: String, lead: [String], absent: [String]) {
        let found = search(query)
        #expect(Set(found.prefix(lead.count)) == Set(lead), "\(query): \(found)")
        for id in absent { #expect(!found.contains(id), "\(query) listed \(id): \(found)") }
    }

    @Test func speakerWordsInAMixedQueryKeepTheDiarizationModels() {
        let found = search("speech emotion speaker identify voice")
        #expect(found.contains("pyannote-community-1") && found.contains("nemotron-3-diarization"), "\(found)")
    }

    @Test func nothingFitsMeansNoModelsNotPadding() {
        // VoxCPM2 steers the emotion of the speech it makes, so it is the one honest fit; nothing recognises emotion.
        #expect(search("speech emotion") == ["voxcpm2"])
        #expect(search("play chess").isEmpty)
        #expect(search("").isEmpty && search("the a of").isEmpty)
    }

    @Test func wordsMeetOnlyAsTheSameWord() {
        #expect(MCPSearch.similar("translation", "translate") && MCPSearch.similar("transcribe", "transcription") && MCPSearch.similar("removal", "remove"))
        #expect(!MCPSearch.similar("translate", "transparent") && !MCPSearch.similar("transcribe", "translate") && !MCPSearch.similar("speech", "speed"))
        #expect(MCPSearch.parse("Speech to text in Polish") == .init(specific: ["transcribe", "polish"], generic: []))
        #expect(MCPSearch.parse("read the text aloud") == .init(specific: ["speak"], generic: ["text"]))
    }

    @Test func taskAndLanguageFiltersAcceptNamesAndCodes() {
        #expect(MCPSearch.languageCode("Polish") == "pl" && MCPSearch.languageCode("pl-PL") == "pl" && MCPSearch.languageCode(" PL ") == "pl")
        #expect(MCPSearch.languageFit(["mul"], code: "pl") == .multilingual && MCPSearch.languageFit(["en", "pl"], code: "pl") == .exact)
        #expect(MCPSearch.languageFit(["und"], code: "pl") == .none && MCPSearch.languageFit(["en-US"], code: "en") == .exact)
        let byID = Dictionary(uniqueKeysWithValues: Self.shipped.entries.map { ($0.id, $0) })
        func fits(_ id: String, _ task: String) -> Bool { MCPSearch.matches(byID[id]!, operations: Self.shipped.operations[id] ?? [], task: task) }
        for (id, task) in [
            ("index-translate-2b", "translate"), ("index-translate-2b", "translation"), ("parakeet-v3-ultra", "speech-to-text"), ("parakeet-v3-ultra", "transcription"),
            ("teleocr", "ocr"), ("teleocr", "read_document"), ("moge-3-vitl", "depth"), ("birefnet", "background removal"), ("birefnet", "remove_background"),
            ("flux2-klein-4b", "image"), ("flux2-klein-4b", "images"), ("kokoro-82m", "text-to-speech"), ("kokoro-82m", "speak"), ("seedvr2-3b", "upscale"),
            ("htdemucs-ft", "separate vocals"), ("ddcolor-modelscope", "colorize"), ("ddcolor-modelscope", "colourise"),
            ("big-lama", "inpaint"), ("big-lama", "erase")
        ] {
            #expect(fits(id, task), "\(id) should match task \(task)")
        }
        for (id, task) in [
            ("parakeet-v3-ultra", "translate"), ("birefnet", "translate"), ("kokoro-82m", "ocr"), ("index-translate-2b", "transcribe"),
            ("big-lama", "remove background"), ("ddcolor-modelscope", "upscale")
        ] {
            #expect(!fits(id, task), "\(id) should not match task \(task)")
        }
    }

    @Test func findFiltersByTaskAndLanguageAndSaysWhenNothingFits() async throws {
        func op(_ name: String) -> ModelOperation { ModelOperation(name: name, title: name, inputSchema: ["type": "object"], output: "json", stream: false) }
        var translator = ModelRow(id: "translator", name: "Translator", kind: "chat", precisions: ["fp16"], selectedPrecision: "fp16", downloadBytes: ["fp16": 1_000_000_000])
        translator.purpose = "Translate with glossary and style"
        var listener = ModelRow(id: "listener", name: "Listener", kind: "audio", precisions: ["fp16"], selectedPrecision: "fp16", downloadBytes: ["fp16": 1_000_000_000])
        listener.purpose = "Speech transcription"
        let h = try await MCPHarness(
            rows: [translator, listener],
            metadata: ["translator": ["tasks": ["translate"], "languages": ["mul"]], "listener": ["tasks": ["transcribe"], "languages": ["en", "pl"]]],
            operations: ["translator": [op("translate")], "listener": [op("transcribe")]])
        defer { Task { await h.close() } }
        for language in ["pl", "Polish", "pl-PL"] {
            let text = try await h.text("find", ["task": "translate", "language": .string(language)])
            #expect(text.hasPrefix("1 model for task \"translate\" in language \"\(language)\"") && text.contains("• translator — "), "\(text)")
        }
        let listed = try await h.text("find", ["language": "pl"])
        let first = try #require(listed.range(of: "• listener")), second = try #require(listed.range(of: "• translator"))
        #expect(first.lowerBound < second.lowerBound, "a model declaring Polish comes before a multilingual one: \(listed)")
        #expect(!(try await h.text("find", ["task": "translate", "language": "ja"])).contains("listener"))
        #expect(try await h.text("find", ["task": "speech to text"]).contains("• listener"))
        let none = try await h.text("find", ["query": "speech emotion"])
        #expect(none == "No model for \"speech emotion\". Try other words, or \(MCPNames.prefix)find() with no arguments for every model.", "\(none)")
        let polish = try await h.text("find", ["query": "translate Polish"])
        #expect(polish.hasPrefix("1 model for \"translate Polish\"") && !polish.contains("listener"), "\(polish)")
    }
}
