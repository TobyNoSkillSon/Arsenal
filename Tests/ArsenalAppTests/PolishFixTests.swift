// SPDX-License-Identifier: MIT
import AppKit
import CryptoKit
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalAPI
@testable import ArsenalCore

/// The polish fixes (lab/polish-audit's list): each finding with the check that keeps it fixed. Windows are laid out
/// offscreen (far off any screen, never activated); no model runs.
@MainActor @Suite(.menuState, .serialized) final class PolishFixTests {
    init() {
        _ = NSApplication.shared
        ToolWindow.keepOffscreenFrames = true
        ControlFrames.recording = true
        SoundPlayer.silent = true
    }
    private let scratch = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-polish-fix-\(UUID().uuidString)", isDirectory: true)
    deinit { try? FileManager.default.removeItem(at: scratch) }
    private static let repo = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()

    // MARK: Windows

    private struct Lab {
        let api: APIHost
        let session: CapabilitySession
        let window: ToolWindowController
        @MainActor func stop() { session.discard(); window.close(); api.stop() }
    }
    /// A capability window over the fixture models behind the app's API, open offscreen at `size`.
    private func lab(_ tool: ToolKind, _ model: String, size: NSSize = ToolLayout.size) async throws -> Lab {
        let support = scratch.appendingPathComponent("\(tool.rawValue)-\(UUID().uuidString.prefix(6))")
        let service = await CapabilityFixture.service(support: support)
        let api = APIHost(service: service, support: support, port: 0)
        await api.start()
        let port = try #require(api.port)
        let models = CapabilityFixture.models.filter { ToolCatalog.windows(declared: Set($0.ops)).contains(tool) }.map { CapabilityFixture.choice($0.id, resident: true) }
        let (session, view) = CapabilityWindows.make(
            tool, models: models, selected: model, client: { ArsenalClient(port: port) }, inbox: support.appendingPathComponent("inbox"),
            output: support.appendingPathComponent("Results"), scratch: support.appendingPathComponent("window"))
        session.prepare()
        try await waitUntil(timeout: 10) { !session.loading }
        let window = ToolWindowController(title: tool.windowTitle, autosave: "", size: size, minSize: ToolLayout.minSize, content: view)
        let w = try #require(window.window)
        w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: size), display: false)
        w.orderFrontRegardless()
        w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
        return Lab(api: api, session: session, window: window)
    }
    private func settle(_ lab: Lab) async throws {
        lab.window.window?.contentView?.layoutSubtreeIfNeeded(); lab.window.window?.displayIfNeeded()
        try await Task.sleep(for: .milliseconds(200))
        lab.window.window?.contentView?.layoutSubtreeIfNeeded(); lab.window.window?.displayIfNeeded()
    }
    private func frame(_ lab: Lab, _ id: String) -> CGRect? { ControlFrames.frames[lab.window.scope]?[id] }
    private func run(_ lab: Lab) async throws {
        lab.session.start()
        try await waitUntil(timeout: 15) { !lab.session.running }
        #expect(lab.session.problem == nil, "\(lab.session.problem?.message ?? "")")
    }

    // 1. Copy never sits on a result's first figure.
    @Test func copyLeavesTheResultsFiguresClear() async throws {
        let search = try await lab(.search, "rerank-fixture")
        defer { search.stop() }
        search.session.set("query", .text("red apples"))
        search.session.set("documents", .text("red apples are sweet\nbananas are yellow"))
        try await run(search)
        try await settle(search)
        let copy = try #require(frame(search, "copy")), score = try #require(frame(search, "score-0"))
        #expect(!copy.intersects(score), "Copy \(copy) overlaps the first score \(score)")

        let photo = try CapabilityFixture.photo(scratch.appendingPathComponent("photo.jpg"), width: 640, height: 480)
        let find = try await lab(.find, "dino-fixture")
        defer { find.stop() }
        find.session.setSubject([photo]); find.session.set("labels", .text("person, umbrella"))
        try await run(find)
        try await settle(find)
        let findCopy = try #require(frame(find, "copy"))
        let rows = (ControlFrames.frames[find.window.scope] ?? [:]).filter { $0.key.hasPrefix("detection-") }.map(\.value)
        #expect(!rows.isEmpty)
        for row in rows { #expect(!findCopy.intersects(row), "Copy \(findCopy) overlaps a detection row \(row)") }
    }

    // 8. Smooth Video's run button sits at the column's foot, as every window's does, at every height.
    @Test(arguments: [ToolLayout.size, ToolLayout.minSize]) func smoothRunsFromTheColumnsFoot(_ size: NSSize) async throws {
        let video = try CapabilityFixture.video(scratch.appendingPathComponent("walk.mp4"))
        let smooth = try await lab(.smooth, "rife-fixture", size: size)
        defer { smooth.stop() }
        try await settle(smooth)
        let empty = try #require(frame(smooth, "run"))
        smooth.session.setSubject([video]); smooth.session.set("factor", .text("4"))
        try await settle(smooth)
        let ready = try #require(frame(smooth, "run"))
        let upscale = try await lab(.upscale, "upscale-fixture", size: size)
        defer { upscale.stop() }
        try await settle(upscale)
        let reference = try #require(frame(upscale, "run"))
        // The same place as Upscale's, empty and with a video, so nothing jumps when the video arrives.
        #expect(abs(empty.minY - reference.minY) < 1 && abs(ready.minY - reference.minY) < 1, "Smooth \(empty) / \(ready) vs Upscale \(reference)")
        #expect(size.height - ready.maxY < 40)
    }

    // 2. Chat-kind models that read documents, translate or extract are shown by what they do.
    @Test func modelsThatDoNotChatAreShownByWhatTheyDo() throws {
        #expect(KindPresentation.id(kind: "chat", tasks: ["read_document"]) == "documents")
        #expect(KindPresentation.id(kind: "chat", tasks: ["translate"]) == "translation")
        #expect(KindPresentation.id(kind: "chat", tasks: ["extract", "read_document"]) == "extraction")
        #expect(KindPresentation.id(kind: "chat", tasks: ["chat", "translate"]) == "chat")
        #expect(KindPresentation.id(kind: "audio", tasks: ["transcribe"]) == "audio")
        // Every shipped folder: none shows as Chat (no chat model ships).
        let kinds = Self.repo.appendingPathComponent("Resources/kinds")
        var shown: [String: String] = [:]
        for id in try FileManager.default.contentsOfDirectory(atPath: Self.repo.appendingPathComponent("models").path) where !id.hasPrefix(".") {
            guard let m = try? ModelManifest.load(Self.repo.appendingPathComponent("models/\(id)"), requirePins: false, kindsDirectory: kinds), m.kind == "chat" else { continue }
            shown[id] = KindPresentation.id(kind: m.kind, tasks: m.tasks)
        }
        for id in ["teleocr", "index-translate-2b", "index-translate-9b", "nuextract3"] {
            #expect(shown[id] != nil && shown[id] != "chat", "\(id): \(shown[id] ?? "-")")
        }

        // Get more models: the kind, its title and icon, and the sidebar.
        var ocr = BrowserTests.entry("ocr", "OCR 2", kind: "chat", params: "4.5B", bytes: 1_000_000)
        ocr.tasks = ["read_document"]
        var translate = BrowserTests.entry("tr", "Translate 2B", kind: "chat", params: "2.2B", bytes: 1_000_000)
        translate.tasks = ["translate"]
        let browser = BrowserController(service: FixtureService(rows: []), physicalBytes: 128 * 1_073_741_824, chip: "M5 Max")
        browser.apply(entries: [BrowserTests.entries[2], ocr, translate], kinds: BrowserTests.kinds)
        let item = try #require(browser.items.first { $0.id == "ocr" })
        #expect(item.kind == "documents" && item.kindTitle == "Documents" && item.icon == "doc.text.viewfinder")
        #expect(browser.items.first { $0.id == "tr" }?.kindTitle == "Translation")
        #expect(browser.kindRows.map(\.kind.id) == ["voices", "documents", "translation"])
        browser.state.choose(kind: "documents")
        #expect(browser.shown.map(\.id) == ["ocr"])

        // The Models table: groups by what they do; moving one moves its kind in the saved order.
        let table = TableController(service: FixtureService(rows: []))
        table.apply(rows: [BrowserTests.entries[2].row, ocr.row, translate.row], kinds: BrowserTests.kinds, savedOrder: ["voices", "decisions", "chat"])
        table.setShownKinds(["ocr": "documents", "tr": "translation"])
        #expect(table.groups.map(\.kind.title) == ["Speech", "Documents", "Translation"])
        #expect(table.groups.map { $0.rows.map(\.id) } == [["kokoro"], ["ocr"], ["tr"]])
        table.moveGroup("translation", toSlot: 0)
        #expect(table.groupOrder.first == "chat")
        #expect(TableStats.help(kind: table.groups[0].kind, column: "speed") == TableStats.help(kind: BrowserTests.kinds[2], column: "speed"))
    }

    // Groups named by what their models do or the tool they open; audio that only separates is its own group. The
    // service's ids and titles are untouched (API, MCP, saved order).
    @Test func groupsAreNamedByWhatTheirModelsDo() {
        let expected = [
            "voices": "Speech", "sound": "Music & Sounds", "enhancement": "Audio Cleanup", "audio": "Transcription", "perception": "Objects & Depth",
            "restoration": "Upscale & Smooth", "embeddings": "Search"
        ]
        for (id, title) in expected { #expect(KindPresentation.title(id) == title, "\(id)") }
        #expect(KindPresentation.id(kind: "audio", tasks: ["separate_audio"]) == "separation")
        #expect(KindPresentation.id(kind: "audio", tasks: ["transcribe"]) == "audio" && KindPresentation.id(kind: "audio", tasks: ["diarize"]) == "audio")
        // Short enough for the sidebar's row at its narrowest, chosen (semibold) and with a two-digit count: no title cut.
        let font = NSFont.systemFont(ofSize: 13, weight: .semibold)
        let count = ("88" as NSString).size(withAttributes: [.font: NSFont.monospacedDigitSystemFont(ofSize: 11.5, weight: .regular)]).width
        // The sidebar's padding and the row's, the icon and the two 9 pt gaps.
        let room = BrowserSidebar.minWidth - 2 * 10 - 2 * 10 - 18 - 2 * 9 - count
        for title in Array(expected.values) + KindPresentation.splits.map(\.title) {
            let width = (title as NSString).size(withAttributes: [.font: font]).width
            #expect(width <= room, "\(title): \(width) of \(room) pt")
        }
        let spec = KindSpec(id: "voices", title: "Voices", order: 0, columns: [])
        #expect(KindPresentation.named(spec).title == "Speech" && spec.title == "Voices")
        #expect(ModelPicker.heading("sound") == "Music & Sounds")
    }

    // 3. One runner is no choice: no card, no radio, no history; several say how they differ in words.
    @Test func oneRunnerShowsItsFiguresWithoutACardOrARadio() throws {
        var entry = BrowserTests.entries[2]
        var native = CatalogRunner(name: "native", language: "swift", engine: "voices", acceptedDefault: true, selected: true)
        native.features = [.sequentialOnly]
        native.history = RunnerHistory(path: "log/005.md", title: "Qualified intake recipes", verdict: "kept")
        entry.runners = [native]
        let one = ModelDetail(BrowserItem(entry, kind: BrowserTests.kinds[0]))
        #expect(BrowserRunnerList.layout(one) == .figures)
        #expect(Self.radios(BrowserRunnerList(detail: one, palette: .dark, choose: { _ in })) == 0)

        var second = CatalogRunner(name: "mlx-lm", language: "swift", engine: "mlx-lm")
        second.features = [.batching]
        entry.runners = [native, second]
        let two = ModelDetail(BrowserItem(entry, kind: BrowserTests.kinds[0]))
        #expect(BrowserRunnerList.layout(two) == .cards(radios: true))
        #expect(Self.radios(BrowserRunnerList(detail: two, palette: .dark, choose: { _ in })) == 2)
        #expect(two.runners.map { $0.features.map(\.title) } == [["One request at a time"], ["Batches requests"]])
        // Only one of two can run at this precision: cards say why, with no radio.
        var limited = entry
        limited.formats = [ModelFormat(id: "bf16", bitsPerWeight: 16, source: .vendor, runners: ["native"])]
        let three = ModelDetail(BrowserItem(limited, kind: BrowserTests.kinds[0]))
        #expect(BrowserRunnerList.layout(three) == .cards(radios: false))
        #expect(Self.radios(BrowserRunnerList(detail: three, palette: .dark, choose: { _ in })) == 0)
    }
    /// The radios a runner list draws, laid out in an offscreen window.
    private static func radios(_ list: BrowserRunnerList) -> Int {
        let host = NSHostingView(rootView: list.frame(width: 320))
        let window = NSWindow(contentRect: NSRect(x: -9000, y: -9000, width: 320, height: 400), styleMask: .borderless, backing: .buffered, defer: false)
        window.contentView = host
        host.layoutSubtreeIfNeeded()
        defer { window.contentView = nil }
        func count(_ v: NSView) -> Int { (v is RunnerChoiceButton ? 1 : 0) + v.subviews.map(count).reduce(0, +) }
        return count(host)
    }

    // 4. Parameters in one format, wherever the folder's spelling: one decimal in B, whole numbers in M (round 2).
    @Test func paramsReadInOneFormat() {
        let cases = [
            "2038M": "2B", "3.70B": "3.7B", "3151M": "3.2B", "8.1M": "8M", "782M": "782M", "30.5M": "31M", "1.7B": "1.7B", "27B": "27B",
            "3.12B": "3.1B", "5.7M": "6M", "99M": "99M", "999.8M": "1B", "1.46B": "1.5B", "8.19B": "8.2B", "5.01B": "5B"
        ]
        for (raw, shown) in cases { #expect(ParamsText.display(raw) == shown, "\(raw)") }
        #expect(BrowserItem(BrowserTests.entry("a", "A", kind: "voices", params: "2038M", bytes: 1), kind: BrowserTests.kinds[0]).params == "2B")
    }

    // 5. The tool window's top line names the fast path only when it runs.
    @Test func theTopLineNamesOnlyTheFastPath() {
        #expect(ToolModel(id: "a", name: "A", kind: "audio", dtype: "fp16", mode: "exact").detail == "16-bit")
        #expect(ToolModel(id: "a", name: "A", kind: "audio", dtype: "int4", mode: "fast").detail == "4-bit \u{00b7} Fast")
    }

    // 6. Licences by name, never cut mid-word, the orange explained.
    @Test func licencesReadByNameAndTheirOrangeIsExplained() throws {
        #expect(LicenceName.short("Apache-2.0", name: "Apache License 2.0") == "Apache 2.0")
        #expect(LicenceName.short("LicenseRef-SAM", name: "LicenseRef-SAM") == "SAM License")
        #expect(LicenceName.short("CC-BY-NC-4.0", name: nil) == "CC BY-NC 4.0")
        #expect(LicenceName.short("LicenseRef-Something-New", name: nil) == "Something New")
        #expect(LicenceName.short("LicenseRef-Mine", name: "Mine Licence") == "Mine Licence")
        let breeze = BrowserItem(BrowserTests.entries[3], kind: BrowserTests.kinds[0])
        #expect(breeze.licenceHelp.hasPrefix("Orange: commercial use is limited."))
        #expect(!BrowserItem(BrowserTests.entries[0], kind: BrowserTests.kinds[1]).licenceHelp.contains("Orange"))
        // Every catalog licence fits its cell at the column's narrowest, in two lines at most, without truncation.
        let catalog = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: Self.repo.appendingPathComponent("Resources/catalog/catalog.json")))
        let cell = WrapCell()
        let width = BrowserColumns.specs[.licence]!.min
        let oneLine = cell.field.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: 10_000, height: 1000)).height
        for model in catalog["models"]?.array ?? [] {
            guard let l = model["manifest"]?["licence"]?.object, let spdx = l["spdx"]?.string else { continue }
            let name = LicenceName.short(spdx, name: l["name"]?.string)
            #expect(!name.hasPrefix("LicenseRef") && !name.contains("-Clause") || name.contains(" "), "\(spdx) → \(name)")
            cell.field.stringValue = name
            let needed = cell.field.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: width, height: 1000)).height
            #expect(needed <= oneLine * 2 + 1, "\(name) needs \(needed) pt at \(width) pt")
            // No word is wider than the cell (it would be cut mid-word).
            for word in name.split(separator: " ") {
                cell.field.stringValue = String(word)
                #expect(cell.field.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: 10_000, height: 1000)).width <= width, "\(word)")
            }
        }
    }

    // Kind and Languages wrap at a word too: no word of theirs is wider than its column at its narrowest.
    @Test func kindsAndLanguagesNeverCutMidWord() {
        let cell = WrapCell()
        func widest(_ text: String) -> CGFloat {
            text.split(separator: " ").map { word -> CGFloat in
                cell.field.stringValue = String(word)
                return cell.field.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: 10_000, height: 1000)).width
            }.max() ?? 0
        }
        let kinds = ["Speech", "Decisions", "Images", "Chat", "3D"] + KindPresentation.names.values.map(\.title) + KindPresentation.splits.map(\.title)
        for title in kinds { #expect(widest(title) <= BrowserColumns.specs[.kind]!.min, "\(title)") }
        for text in ["100+ languages", "Any language", "12 languages", "English"] { #expect(widest(text) <= BrowserColumns.specs[.languages]!.min, "\(text)") }
    }

    // 7. A greyed 4-bit cell says the rule that greyed it; there is no 8-bit column to grey.
    @Test func greyedTiersSayWhy() {
        #expect(PrecisionColumn.all == ["16", "4"] && ModelHelp.unavailable(ModelRow(id: "x", name: "X", kind: "audio", precisions: ["fp16"]))["8"] == nil)
        var small = ModelRow(id: "s", name: "S", kind: "audio", params: "782M", licence: "MIT", precisions: ["fp16"], selectedPrecision: "fp16")
        small.formats = [ModelFormat(id: "fp16", bitsPerWeight: 16, source: .vendor, runners: ["native"])]
        #expect(ModelHelp.unavailable(small)["8"] == nil)
        #expect(ModelHelp.unavailable(small)["4"]?.contains("4-bit would save less than 1 GB") == true)
        var large = small
        large.params = "2038M"
        #expect(ModelHelp.unavailable(large)["8"] == nil && ModelHelp.unavailable(large)["4"]?.hasPrefix("Not offered.") == true)
        var local = small
        local.local = true
        #expect(ModelHelp.unavailable(local)["8"] == nil && ModelHelp.unavailable(local)["4"] == "Not offered. This folder declares no 4-bit format.")
    }

    /// Core review 10: the raw labels shown to people get plain titles (TeleOCR's "ocr" mode, Index Translate's hard and
    /// soft constraints), and Translate puts its text box first, the image drop zone after it. Real schemas, no model.
    @Test func rawSchemaLabelsReadAsWordsAndTranslateStartsWithItsText() throws {
        func descriptor(_ model: String, _ op: String) throws -> OperationDescriptor {
            let url = Self.repo.appendingPathComponent("models/\(model)/ops/\(op).input.json")
            let schema = try #require(try WireJSON.decode(Data(contentsOf: url)) as JSONObject?)
            return OperationDescriptor(name: op, title: op, inputSchema: schema, outputSchema: nil, output: "json", stream: false, timeout: 120)
        }
        func session(_ tool: ToolKind, _ model: String, _ op: OperationDescriptor) -> CapabilitySession {
            let (s, _) = CapabilityWindows.make(
                tool, models: [CapabilityFixture.choice(model, resident: true)], selected: model, client: { nil }, inbox: scratch, output: scratch, scratch: scratch)
            s.preview(operations: [op], selected: op.name)
            return s
        }
        let translate = session(.translate, "translate-fixture", try descriptor("index-translate-2b", "translate"))
        let fields = try #require(translate.form?.fields)
        #expect(fields.first?.key == "text", "\(fields.map(\.key))")
        let image = try #require(fields.firstIndex { $0.key == "image" })
        #expect(image > 0)
        #expect(fields.first { $0.key == "hard_constraints" }?.title == "Rules" && fields.first { $0.key == "soft_constraints" }?.title == "Preferences")
        #expect(FieldControl.choiceTitle("ocr") == "Full page" && FieldControl.choiceTitle("distorted_layout") == "Distorted layout")
        #expect(FieldControl.choiceTitle("en") == "English" && FieldControl.choiceTitle("table") == "Table" && FieldControl.choiceTitle("512x512") == "512x512")
        let document = session(.document, "ocr-fixture", try descriptor("teleocr", "read_document"))
        #expect(document.form?.fields.first { $0.key == "mode" }?.title == "Mode")
    }

    // 9. Options named for people, JSON-only inputs left to the API, enumerable values as choices.
    @Test func optionsReadAsWordsWithTheirOwnControls() throws {
        func session(_ tool: ToolKind, _ model: String, _ op: String) -> CapabilitySession {
            let (s, _) = CapabilityWindows.make(
                tool, models: [CapabilityFixture.choice(model, resident: true)], selected: model, client: { nil }, inbox: scratch, output: scratch, scratch: scratch)
            s.preview(operations: [CapabilityFixture.descriptor(op)], selected: op)
            return s
        }
        func field(_ s: CapabilitySession, _ key: String) -> ToolField? { s.form?.fields.first { $0.key == key } }
        let search = session(.search, "rerank-fixture", "rerank")
        let titles = search.optionFields.map(\.title)
        #expect(!titles.contains("Top") && !titles.contains("Tokens per document"), "\(titles)")
        #expect(field(search, "top_n")?.title == "Results" && field(search, "max_tokens_per_doc")?.unit == "tokens")
        let select = session(.select, "sam-fixture", "segment")
        #expect(!select.optionFields.contains { $0.key == "visual_exemplars" })
        let document = session(.document, "ocr-fixture", "read_document")
        #expect(field(document, "document_annotation_format")?.title == "Fields to extract" && field(document, "document_annotation_format")?.startsEmpty == true)
        let transcribe = session(.transcribe, "whisper-fixture", "transcribe")
        let task = try #require(field(transcribe, "task"))
        #expect(task.title == "Output" && task.choiceTitles["translate"] == "English translation" && task.defaultValue == "transcribe")
        let separate = session(.separate, "demucs-fixture", "separate_audio")
        #expect(field(separate, "mode")?.title == "Keep" && field(separate, "mode")?.choiceTitles["remove"] == "Everything else")
        let smooth = session(.smooth, "rife-fixture", "interpolate_video")
        #expect(field(smooth, "factor")?.unit == "\u{00d7}")
        let make = session(.make3d, "trellis-fixture", "image_to_3d")
        #expect(field(make, "resolution")?.control == .generic(.choice([512, 1024, 1536])) && field(make, "resolution")?.title == "Detail")
        #expect(field(make, "texture_size")?.control == .generic(.choice([512, 1024, 2048, 4096])) && field(make, "texture_size")?.choiceTitles["2048"] == "2048 px")
        // A chosen value goes as the number the schema takes.
        make.set("texture_size", .choice(2))
        make.set("image", .files([try CapabilityFixture.photo(scratch.appendingPathComponent("3d.jpg"), width: 64, height: 64)]))
        let input = try #require(make.form).input(make.values, seed: nil) { $0.path }
        #expect(input["texture_size"] == .number(2048))
        for s in [search, select, document, transcribe, separate, smooth, make] { s.discard() }
    }

    // 10. Decoded media held to a byte budget, and let go when the last tool window closes.
    @Test func mediaCachesKeepToTheirBudgetAndEmptyWithTheLastWindow() throws {
        let cache = MediaCache<String, Int>(limit: 100) { $0 }
        cache.set(40, for: "a"); cache.set(40, for: "b")
        _ = cache["a"]
        cache.set(40, for: "c")
        #expect(cache["b"] == nil && cache["a"] == 40 && cache["c"] == 40 && cache.bytes == 80)
        cache.set(500, for: "huge")
        #expect(cache["huge"] == nil && cache.count == 2)

        let photo = try CapabilityFixture.photo(scratch.appendingPathComponent("cached.jpg"), width: 320, height: 240)
        func open() throws -> ToolWindowController {
            let wc = ToolWindowController(title: "t", autosave: "", size: ToolLayout.minSize, minSize: ToolLayout.minSize, content: Color.clear)
            let w = try #require(wc.window)
            w.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: ToolLayout.minSize), display: false)
            w.orderFrontRegardless(); w.setFrameOrigin(NSPoint(x: -9000, y: -9000))
            return wc
        }
        let first = try open(), second = try open()
        #expect(Thumbnail.image(photo) != nil && Thumbnail.cache.count >= 1)
        first.close()
        #expect(Thumbnail.cache.count >= 1)
        second.close()
        #expect(Thumbnail.cache.count == 0 && Thumbnail.cache.bytes == 0)
    }

    // Lower priority: result files are read off the main thread.
    @Test func resultsAreReadOffTheMainThread() async throws {
        let folder = scratch.appendingPathComponent("results")
        let client = ArsenalClient(port: 1)
        // Callable from a detached task: `result` is not tied to the main actor.
        let result = try await Task.detached {
            try await ToolFiles.result(op: "x", data: Data("{\"text\": \"hi\"}".utf8), mime: "application/json", client: client, into: folder)
        }.value
        #expect(result.fields == [Pair(key: "text", value: "hi")])
    }

    // Lower priority: hovers in plain words.
    @Test func hoversSpeakPlainly() {
        let (session, _) = CapabilityWindows.make(
            .search, models: [CapabilityFixture.choice("rerank-fixture")], selected: "rerank-fixture", client: { nil }, inbox: scratch, output: scratch, scratch: scratch)
        defer { session.discard() }
        session.preview(operations: [CapabilityFixture.descriptor("rerank")], run: ToolRun(model: "m", operation: "rerank", peakMB: 1200))
        #expect(!session.memoryFigure().help.contains("MLX"))
        session.preview(operations: [CapabilityFixture.descriptor("rerank")], run: ToolRun(model: "m", operation: "rerank", memoryMB: 1200))
        let help = session.memoryFigure().help
        #expect(!help.contains("footprint") && !help.contains("worker") && !help.contains("sampled"))
        #expect(!TierControl.headerHelp.contains("checkpoint"))
        #expect(!ModelHelp.origin("int8", native: "bf16", format: ModelFormat(id: "int8", bitsPerWeight: 8.5, source: .derived, runners: [], method: "affine-g64")).contains("g64"))
    }

    // Lower priority: languages in words.
    @Test func languagesReadInWords() {
        func text(_ languages: [String]) -> String {
            BrowserItem(BrowserTests.entry("a", "A", kind: "voices", params: "1B", bytes: 1, languages: languages), kind: nil).languagesText
        }
        #expect(text(["multi"]) == "100+ languages" && text(["und"]) == "Any language" && text(["en"]) == "English" && text(["zh", "en", "ja"]) == "3 languages")
    }

    // Lower priority: Label speakers shows what it holds and sends it.
    @Test func labelSpeakersShowsAndSendsItsState() async throws {
        let clip = try CapabilityFixture.clip(scratch.appendingPathComponent("meeting.wav"))
        let lab = try await lab(.transcribe, "whisper-fixture")
        defer { lab.stop() }
        lab.session.setSubject([clip]); lab.session.set("diarize", .toggle(true))
        try await settle(lab)
        let toggle = try #require(Self.find(NSSwitch.self, in: lab.window.window?.contentView) { $0.accessibilityLabel() == "Label speakers" || $0.state == .on })
        #expect(toggle.state == .on)
        let input = try #require(lab.session.form).input(lab.session.requestValues(), seed: nil) { $0.path }
        #expect(input["diarize"] == .bool(true))
    }
    /// Review 2 (switches drawn Off in the result renders while the run used On): through a real run and its result,
    /// the session holds On, the run sent On, and the window's NSSwitch is On. The renders' Off is the offscreen capture
    /// drawing the switch's first state, not the binding.
    @Test func switchesStayOnThroughARunAndItsResult() async throws {
        let clip = try CapabilityFixture.clip(scratch.appendingPathComponent("meeting2.wav"))
        let transcribe = try await lab(.transcribe, "whisper-fixture")
        defer { transcribe.stop() }
        transcribe.session.setSubject([clip]); transcribe.session.set("diarize", .toggle(true))
        try await run(transcribe)
        try await settle(transcribe)
        #expect(transcribe.session.value("diarize") == .toggle(true) && transcribe.session.run?.input["diarize"] == .bool(true))
        let speakers = try #require(Self.find(NSSwitch.self, in: transcribe.window.window?.contentView) { _ in true })
        #expect(speakers.state == .on, "the first switch (Label speakers) after the result")

        let photo = try CapabilityFixture.photo(scratch.appendingPathComponent("room3.jpg"), width: 320, height: 240)
        let depth = try await lab(.depth, "depth-fixture")
        defer { depth.stop() }
        depth.session.setSubject([photo]); depth.session.set("intrinsics", .toggle(true))
        try await run(depth)
        try await settle(depth)
        #expect(depth.session.value("intrinsics") == .toggle(true) && depth.session.run?.input["intrinsics"] == .bool(true))
        let on = Self.all(NSSwitch.self, in: depth.window.window?.contentView).map(\.state)
        #expect(on.contains(.on), "Camera parameters after the result: \(on.map(\.rawValue))")
    }
    static func all<T: NSView>(_ type: T.Type, in view: NSView?) -> [T] {
        guard let view else { return [] }
        return ((view as? T).map { [$0] } ?? []) + view.subviews.flatMap { all(type, in: $0) }
    }
    static func find<T: NSView>(_ type: T.Type, in view: NSView?, where match: (T) -> Bool) -> T? {
        guard let view else { return nil }
        if let hit = view as? T, match(hit) { return hit }
        for sub in view.subviews { if let hit = find(type, in: sub, where: match) { return hit } }
        return nil
    }

    // Lower priority: a voice's menu actions come only with the loaded model (never for one only downloaded).
    @Test func speechActionsNeedTheModelLoaded() async throws {
        let service = FixtureService(rows: MenuTests.downloaded)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: scratch.appendingPathComponent("menu")))
        app.tools.playback = FixturePlayback()
        await app.table.refresh(); await app.refreshMenuState()
        #expect(app.table.rows.first { $0.id == "kokoro-82m" }?.state == .ready)
        #expect(!app.menu.items.contains { $0.title == "Speak Clipboard" })
        app.table.stop()
    }

    // MARK: Round 2 (lab/design-review's findings)

    // R2-1. Get more models at its default size: Kind and Languages fit their words on one line (taken from Status, empty
    // for nearly every model), Download fits "On this Mac", Status's words wrap without a cut.
    @Test func getMoreModelsCellsFitOneLineAtTheDefaultWindow() throws {
        let defaults = UserDefaults.standard
        let oldOpen = defaults.object(forKey: BrowserRootView.openKey), oldWidth = defaults.object(forKey: BrowserRootView.widthKey)
        let oldWeights = defaults.object(forKey: BrowserColumns.weightsKey)
        defer {
            defaults.set(oldOpen, forKey: BrowserRootView.openKey); defaults.set(oldWidth, forKey: BrowserRootView.widthKey)
            defaults.set(oldWeights, forKey: BrowserColumns.weightsKey)
        }
        defaults.set(true, forKey: BrowserRootView.openKey); defaults.removeObject(forKey: BrowserRootView.widthKey)
        defaults.removeObject(forKey: BrowserColumns.weightsKey)
        let controller = BrowserController(service: FixtureService(rows: []), chip: "Fixture Mac")
        controller.apply(entries: BrowserTests.entries, kinds: BrowserTests.kinds)
        let windowController = BrowserWindowController(controller: controller)
        let window = try #require(windowController.window)
        defer { window.close() }
        window.setFrame(NSRect(origin: NSPoint(x: -9000, y: -9000), size: BrowserWindowController.defaultSize), display: false)
        window.contentView?.layoutSubtreeIfNeeded()
        windowController.list.applyLayout()
        let layout = windowController.list.layout
        #expect(layout.hidden.isEmpty, "\(layout.hidden)")
        let cell = WrapCell()
        func oneLine(_ text: String) -> CGFloat {
            cell.field.stringValue = text
            return cell.field.cell!.cellSize(forBounds: NSRect(x: 0, y: 0, width: 10_000, height: 1000)).width
        }
        let kinds = ["Speech", "Decisions", "Images", "3D"] + KindPresentation.names.values.map(\.title) + KindPresentation.splits.map(\.title)
        for title in kinds { #expect(oneLine(title) <= layout.widths[.kind]!, "Kind \(title): \(oneLine(title)) > \(layout.widths[.kind]!)") }
        for text in ["100+ languages", "Any language", "10 languages", "16 languages"] {
            #expect(oneLine(text) <= layout.widths[.languages]!, "Languages \(text): \(oneLine(text)) > \(layout.widths[.languages]!)")
        }
        let label = NSTextField(labelWithString: "On this Mac")
        label.font = .monospacedDigitSystemFont(ofSize: 13, weight: .regular)
        #expect(label.cell!.cellSize.width <= layout.widths[.download]!)
        for status in BrowserItem.Status.allCases where !status.rawValue.isEmpty {
            for word in status.rawValue.split(separator: " ") { #expect(oneLine(String(word)) <= layout.widths[.status]!, "\(word)") }
        }
        // At the narrowest widths the minimums still let every word stand whole (the wrapping fallback).
        for title in kinds { for word in title.split(separator: " ") { #expect(oneLine(String(word)) <= BrowserColumns.specs[.kind]!.min, "\(word)") } }
        for word in ["Leaving", "catalog", "Checks", "failed"] { #expect(oneLine(word) <= BrowserColumns.specs[.status]!.min, "\(word)") }
    }

    // R2-2. Params: the size the name carries (as Vella and the makers say it), else one decimal in B, whole M.
    @Test func paramsFollowTheNameOrOneFormat() throws {
        #expect(ParamsText.display("2038M", name: "Qwen3 ASR 1.7B") == "1.7B")
        #expect(ParamsText.display("782M", name: "Qwen3 ASR 0.6B") == "0.6B")
        #expect(ParamsText.display("596M", name: "Harrier OSS v1 0.6B") == "0.6B")
        #expect(ParamsText.display("3.12B", name: "d1 \u{00b7} 3B") == "3B")
        #expect(ParamsText.display("2.21B", name: "UI Venus 1.5 2B") == "2B")
        #expect(ParamsText.display("82M", name: "Kokoro 82M") == "82M")
        #expect(ParamsText.display("8.1M", name: "Pyannote Community-1") == "8M")
        #expect(ParamsText.display("8M", name: "Smart Turn v3") == "8M")
        #expect(ParamsText.display("870M", name: "SAM 3.1") == "870M")
        #expect(ParamsText.display("8.19B", name: "Qwen3-Reranker-8B") == "8.2B", "a hyphenated name carries no word of its own")
        #expect(ParamsText.display("300M", name: "Fixture 7B") == "300M", "a size far from the count never stands in for it")
        // Every shipped model reads in one format.
        let catalog = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: Self.repo.appendingPathComponent("Resources/catalog/catalog.json")))
        for model in catalog["models"]?.array ?? [] {
            guard let m = model["manifest"], let params = m["params"]?.string, let name = m["name"]?.string else { continue }
            let shown = ParamsText.display(params, name: name)
            #expect(shown.range(of: #"^[0-9]+(\.[0-9])?B$|^[0-9]+M$|^[0-9]+K$"#, options: .regularExpression) != nil, "\(name): \(params) → \(shown)")
        }
    }

    // R2-3. The expanded row's figures say which precision they are for (the row's own figure can be another one's).
    @Test func detailFiguresNameTheirPrecision() throws {
        var entry = BrowserTests.entries[2]
        entry.runners = [CatalogRunner(name: "native", language: "swift", engine: "voices", acceptedDefault: true, selected: true)]
        let detail = ModelDetail(BrowserItem(entry, kind: BrowserTests.kinds[0]))
        #expect(BrowserRunnerList.layout(detail) == .figures)
        #expect(detail.runners.first?.precision == entry.row.selectedPrecision)
        // Round 1 (dropin-detail): the chip said "fp16" beside "Sizes 16-bit". It names the precision as the table does.
        #expect(detail.runners.first?.precisionLabel == WeightsPresentation.bitLabel(entry.row.selectedPrecision, formats: entry.row.formats))
        #expect(detail.runners.first?.precisionLabel.hasSuffix("-bit") == true)
        // Drawn: the table's gold chip above the figures (the model is in the table, so its precision is the chosen one).
        var added = entry
        added.added = true
        let chosen = ModelDetail(BrowserItem(added, kind: BrowserTests.kinds[0]))
        #expect(chosen.runners.first?.precisionChosen == true)
        var unlabelled = chosen
        unlabelled.runners[0].precision = ""
        // The missing figures' yellow triangles are drawn either way: the chip adds its own gold.
        let with = Self.goldPixels(chosen), without = Self.goldPixels(unlabelled)
        #expect(with - without > 40, "gold pixels with the chip \(with), without \(without)")
    }
    /// Pixels of the precision chip's gold in the runner list, drawn offscreen.
    private static func goldPixels(_ detail: ModelDetail) -> Int {
        let host = NSHostingView(rootView: BrowserRunnerList(detail: detail, palette: .dark, choose: { _ in }).frame(width: 320).background(Color.black))
        let window = NSWindow(contentRect: NSRect(x: -9000, y: -9000, width: 320, height: 200), styleMask: .borderless, backing: .buffered, defer: false)
        window.contentView = host
        defer { window.contentView = nil }
        host.frame = NSRect(x: 0, y: 0, width: 320, height: 200)
        host.layoutSubtreeIfNeeded()
        guard let rep = host.bitmapImageRepForCachingDisplay(in: host.bounds) else { return 0 }
        host.cacheDisplay(in: host.bounds, to: rep)
        let gold = ArsenalPalette.gold.usingColorSpace(.sRGB)!
        var count = 0
        for x in 0..<rep.pixelsWide {
            for y in 0..<rep.pixelsHigh {
                // The bitmap's own (calibrated) values, within 0.1 of the sRGB gold; the yellow triangles have no blue.
                guard let c = rep.colorAt(x: x, y: y) else { continue }
                if abs(c.redComponent - gold.redComponent) < 0.1, abs(c.greenComponent - gold.greenComponent) < 0.1, abs(c.blueComponent - gold.blueComponent) < 0.1 {
                    count += 1
                }
            }
        }
        return count
    }

    // R2-4. Speak Clipboard and the folder's own actions, through the real service: a model not on this Mac offers none.
    @Test func voiceActionsNeedTheModelOnThisMacThroughTheRealService() async throws {
        let (app, service, support) = try await realApp(["kokoro-82m"])
        defer { app.table.stop(); app.tools.closeAll(); try? FileManager.default.removeItem(at: support) }
        // Not on this Mac, Kokoro is not on the table (it holds only models with weights or a download).
        #expect(!app.table.rows.contains { $0.id == "kokoro-82m" })
        #expect(try await service.onDiskModels().allSatisfy { $0.id != "kokoro-82m" })
        #expect(await service.menuActions().isEmpty)
        #expect(await service.menuSections().isEmpty)
        let titles = app.menu.items.map(\.title)
        #expect(!titles.contains { $0.hasPrefix("Speak Clipboard") || $0.hasPrefix("Read Selection Aloud") }, "\(titles)")
        // No table model speaks, so there is no Speak entry either, and no "not installed — Get" row.
        #expect(!app.menu.items.contains { $0.title == "Speak\u{2026}" && !$0.isHidden }, "\(titles)")
        #expect(!titles.contains { $0.contains("not installed") }, "\(titles)")
        await service.shutdown()
    }

    // R2-10, by task: the main menu lists tasks in their fixed order (Speak, …, the capability windows), each serving the
    // table's models that do it; models that work in the background (rankings, embeddings) appear nowhere in it.
    @Test func menuToolsFollowTheTablesGroups() async throws {
        let ids = ["jina-reranker-v3.5", "index-translate-2b", "htdemucs-ft", "harrier-oss-v1-0.6b", "kokoro-82m", "index-translate-9b"]
        let (app, service, support) = try await realApp(ids)
        defer { app.table.stop(); app.tools.closeAll(); try? FileManager.default.removeItem(at: support) }
        // The table holds models with weights or a Get under way: each model's Get is under way, so all six are on it.
        let real = try #require(service as? RealModelsService)
        for id in ids { await real.holdGetUnderWay(id, true) }
        await app.table.refresh(); await app.refreshMenuState(); app.rebuildMenu()
        #expect(Set(app.table.rows.map(\.id)) == Set(ids))
        let entries = app.tools.taskEntries(app.table.menuRows, includeHidden: true)
        let keys = entries.map { AppDelegate.taskKey($0.task) }
        let menuKeys = app.menu.items.compactMap { $0.representedObject as? String }.filter(keys.contains)
        #expect(menuKeys == keys, "menu \(menuKeys) vs tasks \(keys)")
        let order = ToolCenter.taskOrder.map(AppDelegate.taskKey)
        let positions = keys.compactMap { order.firstIndex(of: $0) }
        #expect(positions == positions.sorted(), "\(keys) in task order")
        func serving(_ title: String) -> Set<String> { Set(entries.first { $0.title == title }?.all.map(\.id) ?? []) }
        #expect(serving("Speak\u{2026}") == ["kokoro-82m"])
        #expect(serving("Translate\u{2026}") == ["index-translate-2b", "index-translate-9b"])
        #expect(serving("Separate Audio\u{2026}") == ["htdemucs-ft"])
        for id in ["jina-reranker-v3.5", "harrier-oss-v1-0.6b"] {
            let row = try #require(app.table.rows.first { $0.id == id })
            #expect(app.tools.backgroundOnly(row) && app.tools.tasks(row).isEmpty, "\(id)")
            #expect(!app.menu.everyItem.contains { $0.title.contains(row.name) || ($0.representedObject as? [String])?.contains(id) == true }, "\(id)")
        }
        #expect(!app.table.rows.contains { row in app.menu.items.contains { $0.title == row.name } }, "no model's name in the main menu")
        for id in ids { await real.holdGetUnderWay(id, false) }
        await service.shutdown()
    }

    /// The app over the real service: this checkout's model folders and kinds under a dev-signed catalog, the given
    /// models added to the table (none installed), the menu built.
    private func realApp(_ ids: [String]) async throws -> (AppDelegate, any ModelsService, URL) {
        let fm = FileManager.default
        let root = scratch.appendingPathComponent("real-\(UUID().uuidString.prefix(6))")
        let resources = root.appendingPathComponent("bundle/Contents/Resources")
        let kinds = resources.appendingPathComponent("kinds"), models = resources.appendingPathComponent("models")
        try fm.createDirectory(at: models, withIntermediateDirectories: true)
        try fm.copyItem(at: Self.repo.appendingPathComponent("Resources/kinds"), to: kinds)
        func inventory(_ dir: URL) throws -> [CatalogFile] {
            try SafePath.files(in: dir).map {
                CatalogFile(path: $0, sha256: try FileDigest.sha256(dir.appendingPathComponent($0)), bytes: try FileDigest.bytes(dir.appendingPathComponent($0)))
            }
        }
        var entries: [CatalogModel] = []
        for id in ids {
            let folder = models.appendingPathComponent(id)
            try fm.copyItem(at: shippedOrRetiredFolder(Self.repo, id), to: folder)
            entries.append(CatalogModel(manifest: try ModelManifest.load(folder, kindsDirectory: kinds), files: try inventory(folder)))
        }
        let data = try CatalogDocument(sequence: 1, sourceCommit: String(repeating: "a", count: 40), models: entries, kindFiles: try inventory(kinds)).encoded()
        let key = Curve25519.Signing.PrivateKey()
        try fm.createDirectory(at: resources.appendingPathComponent("catalog"), withIntermediateDirectories: true)
        try data.write(to: resources.appendingPathComponent("catalog/catalog.json"))
        try key.signature(for: data).write(to: resources.appendingPathComponent("catalog/catalog.json.sig"))
        try key.publicKey.rawRepresentation.write(to: resources.appendingPathComponent("catalog-public-key"))
        let paths = ArsenalPaths(support: root.appendingPathComponent("support"), cache: root.appendingPathComponent("cache"))
        let (service, error) = await ServiceFactory.make(paths: paths, environment: [:], bundle: resources)
        #expect(error == nil, "\(error ?? "")")
        #expect(service is RealModelsService)
        for id in ids { try await service.add(id: id) }
        let app = AppDelegate(service: service, paths: paths)
        app.tools.playback = FixturePlayback()
        await app.table.refresh(); await app.refreshMenuState()
        app.rebuildMenu()
        return (app, service, root)
    }

    // R2-5. Separate Audio at its narrowest: the envelope shrinks; the stem's duration keeps its own width.
    @Test func stemDurationsStayClearOfTheirEnvelopes() async throws {
        let song = try CapabilityFixture.clip(scratch.appendingPathComponent("song.wav"), seconds: 3)
        let separate = try await lab(.separate, "demucs-fixture", size: ToolLayout.minSize)
        defer { separate.stop() }
        separate.session.setSubject([song]); separate.session.set("stems", .text("vocals, drums, bass, other"))
        try await run(separate)
        try await settle(separate)
        let frames = ControlFrames.frames[separate.window.scope] ?? [:]
        let waves = frames.filter { $0.key.hasPrefix("wave-") }
        #expect(waves.count >= 2)
        for (key, wave) in waves {
            let duration = try #require(frames["duration-" + key.dropFirst("wave-".count)])
            #expect(wave.maxX <= duration.minX + 0.5, "\(key): envelope \(wave) runs into its duration \(duration)")
            #expect(duration.width >= 37.5)
        }
    }

    // R2-6. Find Objects: a box's label reads whole, past its box when it must, and never covers a neighbour's.
    @Test func boxLabelsReadWholeWithoutCoveringEachOther() {
        let picture = CGRect(x: 50, y: 40, width: 284, height: 189)
        func detection(_ id: Int, _ label: String, _ score: Double) -> Detection { Detection(id: id, label: label, score: score, box: nil, point: nil) }
        // The design review's minimum-width picture: three boxes 68 pt wide, 12 pt apart.
        let ds = [detection(0, "person", 0.94), detection(1, "umbrella", 0.88), detection(2, "towel", 0.62)]
        let boxes = ds.enumerated().map { i, d in (id: d.id, box: CGRect(x: 84 + CGFloat(i) * 79, y: 90, width: 68, height: 100), width: DetectionOverlay.natural(d)) }
        let tags = DetectionOverlay.tags(boxes, picture: picture)
        for b in boxes {
            let t = tags[b.id]!
            #expect(t.width == b.width, "label \(b.id) cut to \(t.width) of \(b.width)")
            #expect(picture.contains(t), "label \(b.id) \(t) leaves the picture")
            for o in boxes where o.id != b.id { #expect(!t.intersects(tags[o.id]!), "labels \(b.id) and \(o.id) overlap") }
        }
        #expect(boxes.contains { $0.width > $0.box.width }, "the fixture needs a label wider than its box")
        // A box at the picture's top: its label sits inside the box's top edge.
        let top = DetectionOverlay.tags([(id: 0, box: CGRect(x: 60, y: 40, width: 120, height: 80), width: 70)], picture: picture)
        #expect(top[0]!.minY == 42)
    }

    // R2-7. The Models table: every shipped model's purpose reads whole under its name (two lines at most, no word cut),
    // inside the row's height.
    @Test func purposesReadWholeInTheModelsTable() throws {
        let width = ModelTable.W.model - ModelTable.W.icon - ModelTable.W.iconGap
        let font = NSFont.systemFont(ofSize: ModelTable.deltaSize, weight: .medium), name = NSFont.systemFont(ofSize: ModelTable.valueSize)
        func line(_ f: NSFont) -> CGFloat { ceil(f.ascender - f.descender + f.leading) }
        #expect(line(name) + ModelTable.W.lineGap + 2 * line(font) <= ModelTable.rowHeight)
        #expect(ModelTableRow.purposeLines(StatusLine(text: "x", tone: .plain, help: "")) == 2)
        #expect(ModelTableRow.purposeLines(StatusLine(text: "x", tone: .busy, help: "")) == 1)
        let catalog = try JSONDecoder().decode(JSONValue.self, from: Data(contentsOf: Self.repo.appendingPathComponent("Resources/catalog/catalog.json")))
        var checked = 0
        for model in catalog["models"]?.array ?? [] {
            guard let purpose = model["manifest"]?["purpose"]?.string else { continue }
            let attributes: [NSAttributedString.Key: Any] = [.font: font]
            let height = (purpose as NSString).boundingRect(with: NSSize(width: width, height: 1000), options: [.usesLineFragmentOrigin], attributes: attributes).height
            #expect(height <= 2 * line(font) + 0.5, "\(purpose) needs \(height) pt at \(width) pt")
            for word in purpose.split(separator: " ") { #expect((String(word) as NSString).size(withAttributes: attributes).width <= width, "\(word)") }
            checked += 1
        }
        #expect(checked >= 37)
    }

    // R2-8. Smart Turn (the end of a spoken turn) is with Speech, in the table and in Get more models.
    @Test func endOfTurnSitsWithSpeech() {
        #expect(KindPresentation.id(kind: "audio", tasks: ["end_of_turn"]) == "voices")
        #expect(KindPresentation.id(kind: "audio", tasks: ["transcribe"]) == "audio")
        #expect(KindPresentation.id(kind: "audio", tasks: ["diarize"]) == "audio")
        let audio = KindSpec(id: "audio", title: "Audio", order: 6, columns: [])
        let kinds = KindSpec.defaults + [audio]
        let rows = [
            ModelRow(id: "kokoro", name: "Kokoro", kind: "voices"), ModelRow(id: "turn", name: "Smart Turn", kind: "audio"),
            ModelRow(id: "asr", name: "Parakeet", kind: "audio")
        ]
        let controller = TableController(service: FixtureService(rows: rows, kinds: kinds), chip: "M5 Max")
        controller.apply(rows: rows, kinds: kinds)
        controller.setShownKinds(["turn": "voices"])
        let groups = controller.groups.map { ($0.kind.title, $0.rows.map(\.id)) }
        #expect(groups.first { $0.0 == "Speech" }?.1 == ["kokoro", "turn"], "\(groups)")
        #expect(groups.first { $0.0 == "Transcription" }?.1 == ["asr"], "\(groups)")
        // Alone in the table, it still shows under Speech.
        controller.apply(rows: [rows[1]], kinds: kinds)
        #expect(controller.groups.map(\.kind.title) == ["Speech"])
        // Get more models counts it under Speech.
        var entry = BrowserTests.entry("turn", "Smart Turn", kind: "audio", params: "8M", bytes: 1)
        entry.tasks = ["end_of_turn"]
        let item = BrowserItem(entry, kind: KindPresentation.expand(kinds).first { $0.id == KindPresentation.id(kind: entry.kind, tasks: entry.tasks) })
        #expect(item.kind == "voices" && item.kindTitle == "Speech")
    }

    // R2-9. A copy already on this Mac reads as such in the Download column and the Sizes list.
    @Test func aLocalCopyReadsOnThisMac() {
        var entry = BrowserTests.entry("a", "A", kind: "voices", params: "1B", bytes: 0)
        entry.row.localCopyBytes = ["bf16": 4_000_000]
        let item = BrowserItem(entry, kind: BrowserTests.kinds[0])
        #expect(item.downloadText == "On this Mac")
        #expect(ModelDetail.sizes(entry).first { $0.dtype == "bf16" }?.note == "on this Mac")
    }
}

extension RealModelsService {
    /// Holds a model's Get as under way without a download (the table holds models with weights or a Get under way).
    fileprivate func holdGetUnderWay(_ id: String, _ held: Bool) { gets[id] = held ? Task {} : nil }
}
