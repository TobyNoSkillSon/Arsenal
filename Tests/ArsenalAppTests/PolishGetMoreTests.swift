// SPDX-License-Identifier: MIT
import AppKit
import SwiftUI
import Testing
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) struct PolishGetMoreTests {
    static let id = "kokoro"
    static var metadata: [String: JSONObject] {
        [
            id: [
                "runner": .array([
                    ["name": "native", "language": "swift", "worker": "voices"],
                    ["name": "sequential-fixture", "language": "swift", "worker": "voices"],
                    ["name": "unavailable-fixture", "language": "swift", "worker": "voices", "chips": ["unsupported-test-chip"]]
                ])
            ]
        ]
    }
    static var row: ModelRow {
        var row = BrowserTests.entries[2].row
        row.runner = "native"; row.runnerLanguage = "swift"; row.defaultRunner = "native"
        return row
    }

    @Test func runnerHitTestedClickPersistsAndTheTableReadsTheSameChoice() async throws {
        _ = NSApplication.shared
        let defaults = UserDefaults.standard
        let oldSidebar = defaults.object(forKey: BrowserRootView.openKey)
        let oldFrame = defaults.object(forKey: "NSWindow Frame GetMoreModels")
        defer {
            defaults.set(oldSidebar, forKey: BrowserRootView.openKey)
            defaults.set(oldFrame, forKey: "NSWindow Frame GetMoreModels")
        }
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-getmore-\(UUID().uuidString)")
        defer { try? FileManager.default.removeItem(at: support) }
        let service = StubModelsService(rows: [Self.row], metadata: Self.metadata, supportDirectory: support)
        let controller = BrowserController(service: service)
        await controller.refresh()
        controller.expanded = [Self.id]
        let windowController = BrowserWindowController(controller: controller)
        let window = try #require(windowController.window)
        defer { window.close() }
        window.setFrame(NSRect(x: -9000, y: -9000, width: 1440, height: 820), display: false)
        windowController.list.expansionChanged()
        window.contentView?.layoutSubtreeIfNeeded()
        for index in 0..<windowController.list.lines.count { _ = windowController.list.table.rowView(atRow: index, makeIfNecessary: true) }
        window.contentView?.layoutSubtreeIfNeeded()
        let action = try #require(Self.accessible(window.contentView!, id: "browser-runner-sequential-fixture"))
        let button = try #require(action as? NSButton)
        let point = button.convert(NSPoint(x: button.bounds.midX, y: button.bounds.midY), to: nil)
        let frame = try #require(window.contentView?.superview)
        #expect(frame.hitTest(frame.convert(point, from: nil)) === button)
        button.performClick(nil)
        try await waitUntil { controller.items.first?.entry.row.runner == "sequential-fixture" }
        let config = try ArsenalConfiguration.read(support.appendingPathComponent("config.json"))
        #expect(config.runnerOverrides[Self.id] == "sequential-fixture")
        // The table holds the model once its weights are on this Mac; it reads the choice the browser saved.
        var held = Self.row
        held.states[held.selectedPrecision] = PrecisionState(state: .ready)
        let reopened = StubModelsService(rows: [held], metadata: Self.metadata, supportDirectory: support)
        let tableRows = await reopened.rows()
        #expect(tableRows.first?.runner == "sequential-fixture")
        #expect(tableRows.first?.runnerOverride == "sequential-fixture")
        let restored = await reopened.catalog(query: .init())
        #expect(restored.first?.runners.first(where: \.selected)?.name == "sequential-fixture")
        let settingsRow = try #require(tableRows.first)
        let setting = try #require(TableController.runnerSetting(settingsRow, restored.first?.runners ?? []))
        #expect(setting.value == .string("sequential-fixture"))
        let disabled = try #require(Self.accessible(window.contentView!, id: "browser-runner-unavailable-fixture") as? NSButton)
        #expect(!disabled.isEnabled)
        disabled.performClick(nil)
        #expect(controller.items.first?.entry.row.runner == "sequential-fixture")
        let native = try #require(Self.accessible(window.contentView!, id: "browser-runner-native") as? NSButton)
        #expect(native.accessibilityLabel()?.contains("default; restores automatic selection") == true)
        native.performClick(nil)
        try await waitUntil { controller.items.first?.entry.row.runner == "native" }
        #expect(try ArsenalConfiguration.read(support.appendingPathComponent("config.json")).runnerOverrides[Self.id] == nil)
    }

    @Test func removedFormatSupportDisablesOldPassingRunnerAndRefusesPersistence() async throws {
        _ = NSApplication.shared
        let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-getmore-support-\(UUID().uuidString)")
        let defaults = UserDefaults.standard
        let oldFrame = defaults.object(forKey: "NSWindow Frame GetMoreModels")
        let oldSidebar = defaults.object(forKey: BrowserRootView.openKey)
        defer {
            try? FileManager.default.removeItem(at: support)
            defaults.set(oldFrame, forKey: "NSWindow Frame GetMoreModels")
            defaults.set(oldSidebar, forKey: BrowserRootView.openKey)
        }
        var row = Self.row
        row.selectedPrecision = "int8"
        row.formats = [
            ModelFormat(id: "bf16", bitsPerWeight: 16, source: .vendor, runners: ["native", "sequential-fixture"]),
            ModelFormat(id: "int8", bitsPerWeight: 8.5, source: .derived, runners: ["native"])
        ]
        var oldPass = PrecisionStatSet(kind: .voices, dtype: "int8", runner: "sequential-fixture")
        oldPass.availability = StatValue(state: .measured)
        row.precisionStats = ["int8": oldPass]
        let service = StubModelsService(rows: [row], metadata: Self.metadata, supportDirectory: support)
        let controller = BrowserController(service: service)
        await controller.refresh()
        let item = try #require(controller.items.first)
        let detail = ModelDetail(item)
        let removed = try #require(detail.runners.first { $0.name == "sequential-fixture" })
        #expect(!removed.eligible && removed.reason == "int8: This runner does not support this precision.")
        #expect(!detail.choosable)
        #expect(detail.runners.first { $0.name == "native" }?.eligible == true)
        var restored = item.entry
        restored.formats[1].runners.append("sequential-fixture")
        #expect(ModelDetail(BrowserItem(restored, kind: BrowserTests.kinds[0])).runners.first { $0.name == "sequential-fixture" }?.eligible == true)
        // Missing historical stats must not make unsupported precision eligible either.
        restored.formats[1].runners = ["native"]
        restored.runners[1].stats = [:]
        #expect(ModelDetail(BrowserItem(restored, kind: BrowserTests.kinds[0])).runners[1].eligible == false)
        controller.expanded = [Self.id]
        let windowController = BrowserWindowController(controller: controller)
        let window = try #require(windowController.window)
        defer { window.close() }
        window.setFrame(NSRect(x: -9000, y: -9000, width: 1440, height: 820), display: false)
        windowController.list.expansionChanged()
        for index in 0..<windowController.list.lines.count { _ = windowController.list.table.rowView(atRow: index, makeIfNecessary: true) }
        window.contentView?.layoutSubtreeIfNeeded()
        // One runner can run at int8: no radio to choose with (the user's polish list: no radio with one choice), the other
        // card says why it cannot.
        #expect(BrowserRunnerList.layout(detail) == .cards(radios: false))
        #expect(Self.accessible(window.contentView!, id: "browser-runner-sequential-fixture") as? NSButton == nil)
        #expect(Self.accessible(window.contentView!, id: "browser-runner-native") as? NSButton == nil)
        controller.chooseRunner(Self.id, "sequential-fixture")
        #expect(controller.lastError == "Kokoro 82M: " + (removed.reason ?? ""))
        #expect(controller.footer.error)
        #expect(await service.catalog(query: .init()).first?.row.runner == "native")
        #expect(try ArsenalConfiguration.read(support.appendingPathComponent("config.json")).runnerOverrides[Self.id] == nil)
    }

    @Test func derivedSizesUseTheRealCatalogForecastRatherThanDownloadBytes() throws {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let model = try ModelManifest.load(root.appendingPathComponent("models/kokoro-82m"), requirePins: false)
        let sourceBytes = model.weights.bytes!.values.reduce(Int64(0), +)
        let row = ModelRow(
            id: model.id, name: model.name, kind: model.kind, precisions: model.precisions,
            selectedPrecision: "fp16", downloadBytes: ["fp16": sourceBytes, "int8": sourceBytes, "int4": sourceBytes])
        // RealModelsService.catalog passes the manifest's sizes and formats separately from the Get plan.
        var entry = CatalogEntry(
            row: row, metadata: ["weights": ["dtype": "fp32"], "derive": ["fp16": "cast:kokoro", "int8": "mlx-affine-g64:kokoro-82m", "int4": "mlx-affine-g64:kokoro-82m"]],
            sizesByDtype: model.sizesByDtype)
        entry.formats = model.formats
        let sizes = ModelDetail.sizes(entry)
        // Kokoro ships FP16 only (int8 withdrawn on 22 Oct, int4 on 26 Oct: under 1 GB at 16-bit); FP16 is cast from the FP32 download.
        #expect(sizes == [.init(dtype: "fp16", size: "~171 MB", note: "from a 342 MB download", label: "16-bit")])
        #expect(entry.sizesByDtype["fp16"] != entry.row.downloadBytes["fp16"])
        // The fixture's 8-bit forecast (~182 MB) is not listed: 8-bit is not offered.
        #expect(ModelDetail.sizes(BrowserTests.entries[2]).map(\.dtype) == ["bf16"])
    }

    static func accessible(_ object: Any, id: String, depth: Int = 0) -> (any NSAccessibilityProtocol)? {
        guard depth < 30, let element = object as? any NSAccessibilityProtocol else { return nil }
        if element.accessibilityIdentifier() == id { return element }
        for child in element.accessibilityChildren() ?? [] {
            if let found = accessible(child, id: id, depth: depth + 1) { return found }
        }
        if let view = object as? NSView {
            for child in view.subviews {
                if let found = accessible(child, id: id, depth: depth + 1) { return found }
            }
        }
        return nil
    }

    @Test func runnersStayOnTheRightAtEveryBrowserWidth() {
        #expect(ModelDetailView.runnerWidth(550) == 260)
        #expect(ModelDetailView.runnerWidth(1100) == 320)
        #expect(ModelDetailView.stacked(550))
        #expect(!ModelDetailView.stacked(1100))
        let detail = ModelDetail(BrowserItem(BrowserTests.entries[2], kind: BrowserTests.kinds[0]))
        #expect(detail.runners[0].stats.allSatisfy { $0.missing })
        #expect(detail.runners[0].stats.map(\.title) == ["Speed", "Peak memory"])
    }
    @Test func perRunnerStatsNeverBorrowAnotherPrecisionOrRunner() {
        let provenance = StatProvenance(chip: "m5-max", date: "2026-10-09", method: "CPU fixture", baseline: "own 16-bit", source: "test fixture")
        var runner = CatalogRunner(name: "native", language: "swift", engine: "chat")
        var stats = PrecisionStatSet(kind: .chat, dtype: "int4", runner: "native")
        stats.set(.tokensPerSecond, StatValue(state: .measured, value: 12.34, provenance: provenance))
        stats.set(.batteryPercentPer100000Tokens, StatValue(state: .quickCheck, value: 1.27, provenance: provenance))
        runner.stats = ["int4": stats]
        // The table's two figures, formatted as the table formats them; battery and context are not shown.
        let rows = BrowserRunnerStats.rows(kind: "chat", dtype: "int4", runner: runner, host: "M5 Max")
        #expect(rows.map(\.title) == ["Speed", "Peak memory"])
        #expect(rows[0].value == "12 tok/s" && !rows[0].missing && rows[0].reference == nil)
        #expect(rows[0].help.contains("Measured on M5 Max"))
        #expect(rows[1].missing && rows[1].help.contains(TableFigures.notMeasured))
        // Another Mac: the figure is the M5 Max reference.
        #expect(BrowserRunnerStats.rows(kind: "chat", dtype: "int4", runner: runner, host: "M1").first?.reference == "M5 Max")
        #expect(BrowserRunnerStats.rows(kind: "chat", dtype: "bf16", runner: runner).allSatisfy { $0.missing })
        stats.runner = "another-runner"; runner.stats["int4"] = stats
        #expect(BrowserRunnerStats.rows(kind: "chat", dtype: "int4", runner: runner).allSatisfy { $0.missing })
    }

    @Test func declaredFeaturesAndUnavailablePrecisionReachTheRunnerPanel() {
        var entry = BrowserTests.entries[2]
        var runner = CatalogRunner(name: "native", language: "swift", engine: "voices", selected: true)
        runner.features = [.sequentialOnly]
        runner.stats = ["bf16": PrecisionStatSet(kind: .voices, dtype: "bf16", runner: "native", unavailableReason: "Quality gate failed")]
        entry.runners = [runner]
        let detail = ModelDetail(BrowserItem(entry, kind: BrowserTests.kinds[0]))
        #expect(!detail.runners[0].eligible)
        #expect(detail.runners[0].reason == "bf16: Quality gate failed")
        #expect(detail.runners[0].features.map(\.help) == ["Processes one request at a time."])
        #expect(detail.runners[0].stats.allSatisfy { $0.missing && $0.help == "bf16 is unavailable. Quality gate failed." })
        runner.features = []; entry.runners = [runner]
        #expect(ModelDetail(BrowserItem(entry, kind: BrowserTests.kinds[0])).runners[0].features.isEmpty)
    }
}
