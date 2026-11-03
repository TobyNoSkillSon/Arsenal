// SPDX-License-Identifier: MIT
import AppKit
import ArsenalCore
import SwiftUI
import Testing
@testable import Arsenal

@MainActor @Suite(.menuState, .serialized) final class PolishTableTests {
    static let kinds = KindSpec.defaults
    static var rows: [ModelRow] {
        let provenance = StatProvenance(chip: "M5 Max", date: "2026-10-09", method: "CPU render fixture", baseline: "Own 16-bit fixture")
        return ModelStatKind.measurable.map { kind in
            var row = ModelRow(
                id: kind.rawValue, name: kind == .chat ? "Long model name with a deliberately extended edition" : kind.rawValue.capitalized + " model",
                kind: kind.rawValue, params: "4B", precisions: ["bf16", "int8", "int4"], states: ["int4": .init(state: kind == .chat ? .loaded : .ready)],
                selectedPrecision: "int4", fast: false, keepHot: kind == .chat, fastExact: false)
            for dtype in row.precisions {
                var set = PrecisionStatSet(kind: kind, dtype: dtype, runner: "native")
                if kind == .images {
                    set.availability = .unavailable("Quality gate failed")
                } else if kind != .sound {
                    set.availability = StatValue(state: .measured, provenance: provenance)
                    for metric in kind.metrics {
                        let value: Double
                        switch metric {
                        case .agreementPercent, .nextTokenAgreementPercent: value = 100
                        case .maximumContextTokens: value = 262_144
                        case .tokensPerSecond: value = dtype == "int4" ? 42.5 : 21.0
                        case .realTimeFactor: value = 0.025
                        case .peakMemoryGB, .loadPeakMemoryGB: value = 8.4
                        case .spectralDistance: value = dtype == "bf16" ? 0 : 0.004
                        case .batchedDecisionsPerSecond: value = 124
                        case .millisecondsPerDecision: value = 18
                        case .nextTokenKL: value = dtype == "bf16" ? 0 : 0.0005
                        default: value = 0.35
                        }
                        let origin =
                            metric == .maximumContextTokens
                            ? StatProvenance(chip: "Model config", date: "2026-10-09", method: "Fixture model config", source: "config.json") : provenance
                        set.set(metric, StatValue(state: metric == .maximumContextTokens ? .declared : .measured, value: value, provenance: origin))
                    }
                }
                row.precisionStats[dtype] = set
            }
            row.recommendedPrecision = kind == .chat ? "int4" : "bf16"
            row.recommendationPending = kind == .sound
            row.recommendationReason = kind == .sound ? "Precision measurements are pending." : "CPU fixture recommendation, never real measurements."
            return row
        }
    }
    private func controller() -> TableController {
        let controller = TableController(service: FixtureService(rows: Self.rows, kinds: Self.kinds), chip: "M5 Max")
        controller.apply(rows: Self.rows, kinds: Self.kinds)
        return controller
    }
    /// The row's settings card as the gear opens it, in the Models menu's table in an unordered window, laid out.
    private func cardView(_ controller: TableController, _ row: ModelRow) async throws -> (NSWindow, NSView) {
        let host = SettingsCardHost.of(controller)
        await host.open(row)
        return (try #require(host.window), host.view)
    }
    private func host(_ controller: TableController, trackingEvents: Bool = false) -> (NSWindow, NSHostingView<ModelTable>) {
        _ = NSApplication.shared
        let host = NSHostingView(rootView: ModelTable(controller: controller))
        let frame = NSRect(x: -9000, y: -9000, width: ModelTable.width(controller), height: ModelTable.height(controller))
        let window: NSWindow =
            trackingEvents
            ? TableTrackingWindow(contentRect: frame, styleMask: .borderless, backing: .buffered, defer: false)
            : NSWindow(contentRect: frame, styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host; window.appearance = Appearance.shared.nsAppearance
        host.layoutSubtreeIfNeeded()
        return (window, host)
    }
    private func views<T: NSView>(_ root: NSView, _: T.Type) -> [T] {
        (root as? T).map { [$0] } ?? root.subviews.flatMap { views($0, T.self) }
    }
    private func hit(_ window: NSWindow, view: NSView, point: NSPoint) throws -> (NSView, NSEvent) {
        let location = view.convert(point, to: nil)
        let root = try #require(window.contentView?.superview)
        let target = try #require(root.hitTest(root.convert(location, from: nil)))
        let event = try #require(
            NSEvent.mouseEvent(
                with: .leftMouseDown, location: location, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                windowNumber: window.windowNumber, context: nil, eventNumber: 1, clickCount: 1, pressure: 1))
        return (target, event)
    }

    @Test func firstFrameLinesFitAndClicksChooseTheDrawnLine() async throws {
        let controller = controller()
        await controller.refresh()
        let (window, host) = host(controller); defer { window.close() }
        let columns = views(host, PrecisionLinesView.self)
        #expect(columns.count == Self.rows.count)
        #expect(abs(host.fittingSize.width - ModelTable.width(controller)) < 1)
        for v in columns {
            // Each fixture row has its int4 on disk: one line.
            #expect(v.lines.map(\.dtype) == ["int4"])
            let pill = v.pillRect(0)
            let (target, event) = try hit(window, view: v, point: NSPoint(x: pill.midX, y: pill.midY))
            #expect(target === v)
            target.mouseDown(with: event)
        }
        // Each row's one line is its chosen int4 already: a click on it writes nothing.
        #expect(controller.previewPrecision.isEmpty)
        let rectangles =
            (columns.map { $0.convert($0.bounds, to: host) }
                + views(host, GearView.self).map { $0.convert($0.bounds, to: host) }
                + views(host, RowActionView.self).map { $0.convert($0.bounds, to: host) })
        for (index, rectangle) in rectangles.enumerated() {
            #expect(rectangle.minX >= 0 && rectangle.maxX <= host.bounds.width)
            for other in rectangles.dropFirst(index + 1) { #expect(!rectangle.intersects(other)) }
        }
    }

    @Test func precisionClickPersistsThroughCloseReopenAndServiceRestartWithoutLoading() async throws {
        let support = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".build/precision-click-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        let row = ModelRow(
            id: "saved", name: "Saved fixture", kind: "chat", precisions: ["bf16", "int8", "int4"], states: ["bf16": .init(state: .loaded), "int4": .init(state: .ready)],
            fast: false, fastExact: false)
        let service = FixtureService(rows: [row], kinds: Self.kinds, supportDirectory: support)
        let controller = TableController(service: service)
        await controller.refresh()
        let (window, host) = host(controller); defer { window.close() }
        let lines = try #require(views(host, PrecisionLinesView.self).first)
        // The 4-bit line is the second (16-bit on top).
        #expect(lines.lines.map(\.dtype) == ["bf16", "int4"])
        let (target, event) = try hit(window, view: lines, point: NSPoint(x: lines.pillRect(1).midX, y: lines.pillRect(1).midY))
        #expect(target === lines)
        target.mouseDown(with: event)
        // Close immediately: the preference write must survive menu teardown.
        controller.discardPreviews(); controller.stop(); window.close()
        try await waitUntil { controller.rows.first?.selectedPrecision == "int4" && controller.previewPrecision.isEmpty }
        let saved = try ArsenalConfiguration.read(support.appendingPathComponent("config.json"))
        #expect(saved.selections[row.id]?.precision == "int4" && saved.selections[row.id]?.explicitSelection == true)
        let after = try #require(await service.rows().first)
        #expect(after.states == row.states && after.residentDType == "bf16" && after.needsReload)
        #expect(await service.calls.isEmpty)
        #expect(await service.approvedGets.isEmpty)
        let reopened = TableController(service: service)
        await reopened.refresh()
        let shown = reopened.presentation(after)
        #expect(shown.selected == "int4" && shown.loadedPrecision == "bf16" && shown.action == .reload)
        #expect(shown.status.text == "4-bit on next load")
        let (nextWindow, nextHost) = self.host(reopened); defer { nextWindow.close() }
        let reopenedLines = try #require(views(nextHost, PrecisionLinesView.self).first)
        #expect(reopenedLines.lines.filter(\.selected).map(\.dtype) == ["int4"] && reopenedLines.lines.filter(\.loaded).map(\.dtype) == ["bf16"])
        let restarted = FixtureService(rows: [row], kinds: Self.kinds, supportDirectory: support)
        #expect(await restarted.rows().first?.selectedPrecision == "int4")
        func clickAction() throws {
            nextHost.layoutSubtreeIfNeeded()
            let action = try #require(views(nextHost, RowActionView.self).first)
            let (target, event) = try hit(nextWindow, view: action, point: NSPoint(x: action.buttonRect.midX, y: action.buttonRect.midY))
            #expect(target === action)
            target.mouseDown(with: event)
        }
        #expect(await service.rows().first?.residentDType == "bf16")
        try clickAction() // Explicit Reload is the first action allowed to replace the resident.
        try await waitUntil { reopened.rows.first?.residentDType == "int4" }
        #expect(await service.calls == ["load saved int4"])
    }

    @Test func unsupportedPrecisionIsDisabledAndAStaleClickIsRefusedWithTheReason() async throws {
        let support = URL(fileURLWithPath: FileManager.default.currentDirectoryPath).appendingPathComponent(".build/precision-refusal-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: support) }
        var row = ModelRow(
            id: "refused", name: "Refused fixture", kind: "chat", precisions: ["bf16", "int4"], states: ["bf16": .init(state: .ready), "int4": .init(state: .ready)])
        row.formats = [.init(id: "int4", bitsPerWeight: 4, source: .derived, runners: [])]
        let service = FixtureService(rows: [row], kinds: Self.kinds, supportDirectory: support)
        let controller = TableController(service: service)
        await controller.refresh()
        let (window, host) = host(controller); defer { window.close() }
        let segment = try #require(views(host, PrecisionLinesView.self).first)
        // The 4-bit line (the second) is the one that cannot be chosen.
        let fourBit = NSPoint(x: segment.pillRect(1).midX, y: segment.pillRect(1).midY)
        #expect(segment.lines.map(\.dtype) == ["bf16", "int4"] && !segment.lines[1].choosable)
        let (target, event) = try hit(window, view: segment, point: fourBit)
        target.mouseDown(with: event)
        #expect(controller.previewPrecision.isEmpty)
        // UI metadata can lag a format change: the service must still refuse the event.
        var stale = row; stale.formats = []
        controller.apply(rows: [stale], kinds: Self.kinds)
        host.layoutSubtreeIfNeeded()
        let (staleTarget, staleEvent) = try hit(window, view: segment, point: fourBit)
        staleTarget.mouseDown(with: staleEvent)
        try await waitUntil { controller.lastError != nil && controller.previewPrecision.isEmpty }
        #expect(controller.lastError == "Refused fixture: No runner in this Arsenal yet.")
        #expect(await service.rows().first?.selectedPrecision == "bf16")
        #expect(!FileManager.default.fileExists(atPath: support.appendingPathComponent("config.json").path))
    }

    @Test func missingFiguresAreTrianglesAndPartialMeasurementsKeepTheirColumn() throws {
        // Nothing measured: every figure is missing (the triangle), and its hover says so; no spanning strip.
        for row in try Self.installedModelFixtures() {
            for column in TableFigures.columns {
                let figure = TableFigures.figure(row, precision: row.selectedPrecision, column: column, host: "M5 Max")
                #expect(figure.missing && figure.help.contains(TableFigures.notMeasured))
            }
        }
        let rejected = try #require(Self.rows.first { $0.kind == "images" })
        // int4 that failed its gate is offered; its figures were never measured.
        let rejectedSpeed = TableFigures.figure(rejected, precision: "int4", column: .speed, host: "M5 Max")
        #expect(rejectedSpeed.missing && rejectedSpeed.help == "int4 \u{00b7} " + TableFigures.notMeasured)
        var partial = try #require(Self.rows.first { $0.kind == "decisions" })
        partial.precisionStats["int4"]?.set(.peakMemoryGB, StatValue())
        #expect(TableFigures.figure(partial, precision: "int4", column: .speed, host: "M5 Max").value == "18 ms")
        #expect(TableFigures.figure(partial, precision: "int4", column: .memory, host: "M5 Max").missing)
        partial.precisionStats["int4"]?.runner = "old"
        #expect(TableFigures.figure(partial, precision: "int4", column: .speed, host: "M5 Max").missing)
        partial.precisionStats["int4"]?.availability = .unavailable("Old runner failed")
        #expect(ModelHelp.unavailable(partial)["4"] == nil)
    }

    @Test func groupArrowsAndAccessibilitySaveRapidMovesInOrder() async throws {
        let controller = controller()
        let (window, host) = host(controller); defer { window.close() }
        let header = try #require(GroupHeaderView.headers(in: host).first)
        let original = controller.groups.map(\.kind.id)
        let (target, event) = try hit(window, view: header, point: NSPoint(x: header.downRect.midX, y: header.downRect.midY))
        #expect(target === header)
        target.mouseDown(with: event)
        #expect(controller.groups[1].kind.id == original[0])
        // Exercise accessibility through the header's declared action, while the first save is still queued.
        let action = try #require(header.accessibilityCustomActions()?.first { $0.name == "Move down" })
        #expect(action.handler?() == true)
        #expect(controller.groups[2].kind.id == original[0])
        try await waitUntil { (await (controller.service as? FixtureService)?.saved.count) == 2 }
        let order = controller.groups.map(\.kind.id)
        let saved = await (controller.service as? FixtureService)?.saved
        #expect(saved?.last == order)
        await controller.refresh()
        #expect(controller.groups.map(\.kind.id) == order)
    }

    @Test func refreshStartedBeforeAMoveCannotRestoreTheOldOrder() async throws {
        let service = TableOrderRaceService()
        let controller = TableController(service: service)
        await controller.refresh()
        let original = controller.groups.map(\.kind.id)
        await service.holdNextKindRead()
        let stale = Task { await controller.refresh() }
        try await waitUntil { await service.holding }
        controller.moveGroup(original[0], by: 1)
        try await waitUntil { await service.saved.count == 1 }
        await controller.refresh() // New order settles first.
        let latest = controller.groups.map(\.kind.id)
        await service.releaseKindRead()
        await stale.value
        #expect(latest != original)
        #expect(controller.groups.map(\.kind.id) == latest)
        #expect(controller.kinds.map(\.id) == latest)
        await controller.refreshRows() // The next download-progress update uses the cached kinds.
        #expect(controller.groups.map(\.kind.id) == latest)
        #expect(await service.saved.last == latest)
    }

    @Test func aBrokenLocalFolderCannotBlockSectionOrderPersistence() async throws {
        // As the service reports a broken folder on the table: its state failed with the folder's problem.
        let local = ModelRow(
            id: "broken-folder", name: "Broken local folder", kind: "local", precisions: [],
            states: ["bf16": .init(state: .failed, error: "This local model folder is missing.", phase: "localFolderError")])
        let service = FixtureService(rows: Self.rows + [local], kinds: Self.kinds)
        let controller = TableController(service: service)
        await controller.refresh()
        let (window, host) = host(controller); defer { window.close() }
        let header = try #require(GroupHeaderView.headers(in: host).first)
        let first = header.kindID
        let localHeader = try #require(GroupHeaderView.headers(in: host).first { $0.kindID == "local" })
        #expect(localHeader.accessibilityCustomActions()?.isEmpty == true)
        #expect(localHeader.toolTipText(at: NSPoint(x: 20, y: localHeader.bounds.midY)).contains("Unreadable local folders"))
        let (target, event) = try hit(window, view: header, point: NSPoint(x: header.downRect.midX, y: header.downRect.midY))
        target.mouseDown(with: event)
        try await waitUntil { await service.saved.count == 1 }
        let saved = try #require(await service.saved.last)
        #expect(Set(saved) == Set(Self.kinds.map(\.id)))
        #expect(!saved.contains("local"))
        #expect(controller.groups[1].kind.id == first)
        await controller.refresh()
        #expect(controller.groups[1].kind.id == first)
        #expect(controller.groups.last?.kind.id == "local")
        #expect(controller.lastError == nil)
    }

    @Test func theTableUsesTheAcceptedFastPathWithoutAnExecutionPathControl() async throws {
        let row = ModelRow(id: "auto", name: "Automatic path fixture", kind: "chat", precisions: ["bf16"], states: ["bf16": .init(state: .ready)], fast: false)
        let service = FixtureService(rows: [row], kinds: Self.kinds)
        let controller = TableController(service: service)
        await controller.refresh()
        #expect(controller.presentation(row).fast)
        let (window, host) = host(controller); defer { window.close() }
        let action = try #require(views(host, RowActionView.self).first)
        let (target, event) = try hit(window, view: action, point: NSPoint(x: action.buttonRect.midX, y: action.buttonRect.midY))
        #expect(target === action)
        target.mouseDown(with: event)
        try await waitUntil { controller.rows.first?.residentDType != nil }
        #expect(controller.rows.first?.fast == true)
        #expect(!ModelHelp.text(controller.rows[0], presentation: controller.presentation(controller.rows[0])).contains("Fast"))
    }

    @Test func figuresUseEachKindsUnitAndNothingIsRecommended() throws {
        let chat = try #require(Self.rows.first { $0.kind == "chat" })
        for column in PrecisionColumn.all { #expect(!ModelHelp.precision(chat, column: column, presentation: RowPresentation(chat)).contains("Recommend")) }
        // int8 has no column: no cell's words name it.
        #expect(PrecisionColumn.all.allSatisfy { !ModelHelp.precision(chat, column: $0, presentation: RowPresentation(chat)).contains("int8") })
        #expect(TableFigures.figure(chat, precision: "int4", column: .speed, host: "M5 Max").value == "42 tok/s")
        let decisions = try #require(Self.rows.first { $0.kind == "decisions" })
        #expect(TableFigures.figure(decisions, precision: "int4", column: .speed, host: "M5 Max").value == "18 ms")
        #expect(TableFigures.figure(decisions, precision: "int4", column: .memory, host: "M5 Max").value == "7.82 GB")
        let voices = try #require(Self.rows.first { $0.kind == "voices" })
        #expect(TableFigures.figure(voices, precision: "int4", column: .speed, host: "M5 Max").value == "40×")
        #expect(TableFigures.figure(voices, precision: "int4", column: .speed, host: "M5 Max").help.contains("first audio 0.35 s"))
        #expect(TableFigures.format(.realTimeFactor, 0) == nil)
        #expect(TableFigures.format(.peakMemoryGB, 15.92) == "14.8 GB" && TableFigures.format(.secondsPerImage, 1.07) == "1.1 s")
    }

    @Test func dragUsesTheReleasePositionThroughTheWindowEventPath() throws {
        let controller = controller()
        let (window, host) = host(controller, trackingEvents: true); defer { window.close() }
        let tracking = try #require(window as? TableTrackingWindow)
        let headers = GroupHeaderView.headers(in: host)
        let first = try #require(headers.first), last = try #require(headers.last)
        let original = first.kindID
        let (target, down) = try hit(window, view: first, point: NSPoint(x: 20, y: first.bounds.midY))
        #expect(target === first)
        func event(_ type: NSEvent.EventType, _ location: NSPoint) throws -> NSEvent {
            try #require(
                NSEvent.mouseEvent(
                    with: type, location: location, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                    windowNumber: window.windowNumber, context: nil, eventNumber: 1, clickCount: 1, pressure: type == .leftMouseUp ? 0 : 1))
        }
        // Supply native events at the window boundary, without entering NSApp's launch loop or posting system input.
        tracking.events = [
            try event(.leftMouseDragged, headers[1].convert(NSPoint(x: 20, y: headers[1].bounds.midY), to: nil)),
            try event(.leftMouseUp, last.convert(NSPoint(x: 20, y: last.extent), to: nil))
        ]
        target.mouseDown(with: down)
        #expect(controller.groups.last?.kind.id == original)
        #expect(controller.drag == nil)
        #expect(tracking.events.isEmpty)
    }

    /// Keep loaded is no column: it is the first switch in every model's settings, reached through the gear by the real
    /// event path (hit-test, then a click), and it writes the service's per-model preference.
    @Test func keepLoadedIsTheFirstSettingOfEveryModelAndReachesResidencyService() async throws {
        let controller = controller()
        let (window, host) = host(controller); defer { window.close() }
        #expect(views(host, MenuSwitch.self).isEmpty)
        // Every model has a gear, with Keep loaded first in its settings.
        #expect(views(host, GearView.self).count == controller.rows.count)
        for row in controller.rows { #expect(controller.settings[row.id]?.first?.id == TableController.keepKey) }
        let row = try #require(controller.rows.first { $0.name.hasPrefix("Long") })
        let index = try #require(controller.groups.flatMap(\.rows).firstIndex { $0.id == row.id })
        let gear = views(host, GearView.self).sorted { $0.convert($0.bounds, to: host).minY < $1.convert($1.bounds, to: host).minY }[index]
        let (gearTarget, gearDown) = try hit(window, view: gear, point: NSPoint(x: gear.bounds.midX, y: gear.bounds.midY))
        #expect(gearTarget === gear)
        gearTarget.mouseDown(with: gearDown)
        #expect(controller.settingsCard.openID == row.id)
        // The card opens in the same table, beside the gear.
        for _ in 0..<8 { host.layoutSubtreeIfNeeded(); await Task.yield() }
        defer { controller.settingsCard.close() }
        let settingsHostWindow = window
        let switches = views(host, MenuSwitch.self).sorted { $0.convert($0.bounds, to: nil).maxY > $1.convert($1.bounds, to: nil).maxY }
        let keep = try #require(switches.first)
        #expect(keep.exposed?.accessibilityLabel() == "Keep loaded" && keep.state == .on)
        #expect(keep.toolTip == "Stays in memory until you unload it.")
        let (target, _) = try hit(settingsHostWindow, view: keep, point: NSPoint(x: keep.bounds.midX, y: keep.bounds.midY))
        #expect(target === keep)
        keep.performClick(nil)
        try await waitUntil { controller.rows.first { $0.id == row.id }?.keepHot == false }
        #expect(controller.settings[row.id]?.first?.value == .bool(false))
        #expect(controller.lastError == nil)
    }

    /// Keep loaded is a residency preference, not a folder setting: Restore defaults resets the load settings and leaves
    /// it, and Keep loaded alone does not offer Restore defaults.
    @Test func restoreDefaultsLeavesKeepLoaded() async throws {
        let row = try #require(Self.rows.first { $0.kind == "chat" })
        #expect(row.keepHot)
        let service = FixtureService(rows: [row], metadata: [row.id: Self.settings], kinds: Self.kinds)
        let controller = TableController(service: service)
        await controller.refresh()
        #expect(!SettingsPane.footer(controller.settings[row.id] ?? [], loaded: false).defaults)
        controller.set(row, "context", .number(131_072))
        try await waitUntil { SettingsPane.footer(controller.settings[row.id] ?? [], loaded: false).defaults }
        controller.restoreDefaults(try #require(controller.rows.first))
        try await waitUntil { (try? await service.settings(id: row.id).first { $0.id == "context" }?.value.number) == 262_144 }
        await controller.refresh()
        #expect(controller.rows.first?.keepHot == true)
        #expect(controller.settings[row.id]?.first { $0.id == TableController.keepKey }?.value == .bool(true))
        #expect(controller.lastError == nil)
    }

    /// Keep loaded stays usable while the model downloads, derives or loads (the service accepts it then, as it did the
    /// table's checkbox): by a click through the hit-test and by accessibility. Only load-time settings lock.
    @Test(arguments: [ModelState.downloading, .deriving, .loading])
    func keepLoadedWorksWhileTheModelIsBusy(_ state: ModelState) async throws {
        var row = try #require(Self.rows.first { $0.kind == "chat" })
        row.states[row.selectedPrecision] = PrecisionState(state: state, progress: state == .downloading ? 0.3 : nil)
        let service = FixtureService(rows: [row], metadata: [row.id: Self.settings], kinds: Self.kinds)
        let controller = TableController(service: service)
        await controller.refresh()
        let first = try #require(controller.rows.first)
        #expect(controller.presentation(first).locked)
        let (window, host) = try await cardView(controller, first)
        defer { SettingsCardHost.close(controller) }
        let switches = views(host, MenuSwitch.self)
        let keep = try #require(switches.first { $0.exposed?.accessibilityLabel() == "Keep loaded" })
        #expect(!switches.contains { $0.exposed?.accessibilityLabel() == "Reason before replying" }, "a request setting is the apps'")
        let context = try #require(views(host, MenuField.self).first { $0.accessibilityLabel() == "Context window" })
        // SwiftUI applies the controls' state on its next update: the load-time field locks; Keep loaded does not.
        try await waitUntil { !context.isEnabled }
        #expect(keep.isEnabled)
        #expect(keep.toolTip == "Stays in memory until you unload it.")
        let (target, _) = try hit(window, view: keep, point: NSPoint(x: keep.bounds.midX, y: keep.bounds.midY))
        #expect(target === keep)
        keep.performClick(nil)
        try await waitUntil { controller.rows.first?.keepHot == false }
        #expect(controller.rows.first?.states[row.selectedPrecision]?.state == state)
        host.layoutSubtreeIfNeeded()
        #expect(keep.isEnabled && keep.cell?.isAccessibilityEnabled() == true)
        // The element accessibility exposes for the switch (its cell, a checkbox with the switch subrole) is enabled; the
        // accessibility press reaches the switch's action (as the settings switch test drives it).
        #expect(keep.exposed?.isAccessibilityEnabled() == true)
        _ = keep.accessibilityPerformPress()
        try await waitUntil { controller.rows.first?.keepHot == true }
        #expect(controller.lastError == nil)
    }

    /// A load choice is segments in the card and locks with its row: once a Get starts, a click on another segment and
    /// any other path (`set`) change nothing.
    @Test(.timeLimit(.minutes(1))) func aLoadChoiceLocksWhenTheModelTurnsBusy() async throws {
        var row = try #require(Self.rows.first { $0.kind == "chat" })
        row.states = ["int4": .init(state: .ready)]
        let style: JSONObject = [
            "setting": [
                ["id": "style", "title": "Style", "type": "choice", "default": "calm", "choices": ["calm", "bright", "warm", "dramatic"], "applies": "load"],
                ["id": "tone", "title": "Tone", "type": "choice", "default": "a", "choices": ["a", "b", "c", "d"], "applies": "request", "maps_to": "request.tone"]
            ]
        ]
        let service = FixtureService(rows: [row] + Self.rows.filter { $0.kind != "chat" }, metadata: [row.id: style], kinds: Self.kinds)
        func value() async -> String? { try? await service.settings(id: row.id).first { $0.id == "style" }?.value.string }
        let controller = TableController(service: service)
        defer { controller.stop() }
        await controller.refresh()
        let ready = try #require(controller.rows.first { $0.id == row.id })
        let host = SettingsCardHost.of(controller)
        defer { host.close() }
        await host.open(ready)
        let segments = try #require(host.views(MenuSegments.self).first { $0.accessibilityLabel() == "Style" })
        #expect(segments.isEnabled && segments.segmentCount == 4 && host.views(MenuSegments.self).count == 1, "Tone is the apps'")
        // A pick through the segment's own mouse-down.
        let point = segments.convert(NSPoint(x: segments.cell(2).midX, y: segments.bounds.midY), to: nil)
        func click(_ control: NSView, _ number: Int) throws {
            control.mouseDown(
                with: try #require(
                    NSEvent.mouseEvent(
                        with: .leftMouseDown, location: point, modifierFlags: [], timestamp: TimeInterval(number), windowNumber: host.window?.windowNumber ?? 0,
                        context: nil, eventNumber: number, clickCount: 1, pressure: 1)))
        }
        try click(segments, 1)
        // Saving the value precedes the controller's follow-up refresh. Drain the whole write before injecting
        // a busy row, otherwise that refresh can publish the fixture's ready row over it under parallel load.
        await controller.waitUntilSettled()
        #expect(await value() == "warm")
        // A Get starts: the load setting locks; Keep loaded stays live.
        var busy = try #require(controller.rows.first { $0.id == row.id })
        busy.states["int4"] = PrecisionState(state: .downloading, progress: 0.1)
        controller.apply(rows: controller.rows.map { $0.id == busy.id ? busy : $0 }, kinds: Self.kinds)
        #expect(!controller.settingEnabled(busy, "style") && controller.settingEnabled(busy, TableController.keepKey))
        controller.set(busy, "style", "dramatic")
        await host.settle()
        let locked = try #require(host.views(MenuSegments.self).first { $0.accessibilityLabel() == "Style" })
        // Observe SwiftUI's actual AppKit update; a fixed number of layout turns is not a completion signal.
        let disabled = AsyncStream<Bool> { continuation in
            let observation = locked.observe(\.isEnabled, options: [.initial, .new]) { _, change in
                guard let enabled = change.newValue else { return }
                continuation.yield(enabled)
                if !enabled { continuation.finish() }
            }
            continuation.onTermination = { _ in observation.invalidate() }
        }
        for await enabled in disabled where !enabled { break }
        #expect(!locked.isEnabled)
        try click(locked, 2)
        await controller.waitUntilSettled()
        #expect(await value() == "warm")
        #expect(controller.rows.first { $0.id == busy.id }?.states["int4"]?.state == .downloading)
        #expect(controller.lastError == nil)
    }

    /// An empty table has no figure headings and is as wide as its Get more models row and the Refresh glyph.
    @Test func anEmptyTableIsTheGetMoreModelsRow() throws {
        let controller = TableController(service: FixtureService(rows: [], kinds: Self.kinds))
        controller.apply(rows: [], kinds: Self.kinds)
        let (window, host) = host(controller); defer { window.close() }
        #expect(ModelTable.width(controller) < ModelTable.width(slots: ModelTable.slots(controller.kinds)) / 3)
        #expect(views(host, TooltipCell.TooltipView.self).allSatisfy { $0.toolTip != ModelTable.paramsHeaderHelp && $0.toolTip != TableStats.headerHelp("speed") })
        let pinned = try #require(views(host, PinnedRowView.self).first)
        #expect(pinned.convert(pinned.bounds, to: host).width == ModelTable.rowWidth(controller))
        let refresh = try #require(views(host, RefreshControlView.self).first)
        let frame = refresh.convert(refresh.bounds, to: host)
        #expect(frame.maxX <= host.bounds.maxX - ModelTable.trailingMargin + 0.5 && frame.minX > pinned.convert(pinned.bounds, to: host).midX)
        // A footer line widens it, never past the grid.
        controller.lastError = String(repeating: "A long catalog problem. ", count: 3)
        #expect(ModelTable.width(controller) > ModelTable.emptyRowWidth && ModelTable.width(controller) <= ModelTable.width(slots: 3))
    }

    /// The table asks the service whether it would refuse Refresh (a Load still verifying files has no Loading row yet)
    /// and keeps the glyph off for exactly that long.
    @Test func refreshFollowsTheServicesRefusalBeforeAnyRowShowsBusy() async throws {
        let stub = StubModelsService(rows: Self.rows)
        let controller = TableController(service: stub)
        controller.start(); defer { controller.stop() }
        try await waitUntil { !controller.rows.isEmpty }
        #expect(controller.refreshAllowed)
        await stub.setRefreshRefusal("Finish Get or Load before refreshing models.")
        try await waitUntil { !controller.refreshAllowed }
        #expect(controller.refreshHelp == RealModelsService.refreshBusyMessage)
        #expect(!controller.rows.contains { $0.states.values.contains { [.downloading, .deriving, .loading].contains($0.state) } })
        await stub.setRefreshRefusal(nil)
        try await waitUntil { controller.refreshAllowed }
        #expect(controller.refreshHelp == RefreshControl.help)
    }

    @Test func typedStatsStayPerPrecisionWithDeltasOnlyOnThisMacsChip() throws {
        let chat = try #require(Self.rows.first { $0.kind == "chat" })
        let int4 = TableFigures.figure(chat, precision: "int4", column: .speed, host: "M5 Max")
        #expect(int4.value == "42 tok/s" && int4.reference == nil)
        #expect(int4.delta == TableFigure.Delta(text: "2.0× faster", tone: .better))
        #expect(int4.help.contains("Measured on M5 Max · 9 Oct 2026"))
        let bf16 = TableFigures.figure(chat, precision: "bf16", column: .speed, host: "M5 Max")
        #expect(bf16.value == "21 tok/s" && bf16.delta == nil)
        // Another Mac: the M5 Max figure stays, named beneath, with no delta.
        let elsewhere = TableFigures.figure(chat, precision: "int4", column: .speed, host: "M3 Pro")
        #expect(elsewhere.value == "42 tok/s" && elsewhere.reference == "M5 Max" && elsewhere.delta == nil)
        #expect(elsewhere.help.contains("Not measured on your Mac; measured on M5 Max."))
        #expect(TableFigures.figure(chat, precision: "int4", column: .speed, host: nil).reference == "M5 Max")
        #expect(TableFigures.figure(chat, precision: "int4", column: .speed, host: "M5 Max", modePreviewChanged: true).missing)
        let sound = try #require(Self.rows.first { $0.kind == "sound" })
        #expect(TableFigures.figure(sound, precision: "int4", column: .speed, host: "M5 Max").missing)
        let images = try #require(Self.rows.first { $0.kind == "images" })
        #expect(TableFigures.figure(images, precision: "int4", column: .memory, host: "M5 Max").missing)
        // int4 that failed its gate is offered with the warning (the int4 policy), not greyed.
        #expect(ModelHelp.unavailable(images)["4"] == nil && images.precisionQualityWarnings["int4"] == PrecisionStatSet.int4QualityWarning)
    }

    @Test func speedDeltasFollowVellasWording() {
        #expect(TableFigures.speedDelta(.millisecondsPerDecision, value: 6, base: 8.1) == .init(text: "35% faster", tone: .better))
        #expect(TableFigures.speedDelta(.realTimeFactor, value: 1 / 39.0, base: 1 / 46.8) == .init(text: "17% slower", tone: .worse))
        #expect(TableFigures.speedDelta(.realTimeFactor, value: 1 / 46.5, base: 1 / 46.8) == .init(text: "same", tone: .same))
        #expect(TableFigures.speedDelta(.tokensPerSecond, value: 96.4, base: 38.2) == .init(text: "2.5× faster", tone: .better))
        #expect(TableFigures.chipName("m5-max") == "M5 Max" && TableFigures.measuredHere(StatProvenance(chip: "m5-max", date: "", method: ""), host: "M5 Max"))
        #expect(!TableFigures.measuredHere(StatProvenance(chip: "m5-max", date: "", method: ""), host: "M5 Pro"))
    }

    @Test func contextFieldAcceptsItsDisplayedTokenUnit() {
        let state = SettingState(declaration: ["id": "context", "type": "number", "unit": "tokens", "min": 1024, "max": 262_144], value: 131_072)
        #expect(state.display(state.value) == "128K")
        #expect(state.number(from: "128K") == 131_072)
        #expect(state.number(from: "512K") == 262_144)
    }

    static let settings: JSONObject = [
        "runner": [["name": "native", "language": "swift", "engine": "chat"]],
        "setting": [
            ["id": "context", "title": "Context window", "type": "number", "default": 262_144, "min": 1024, "max": 262_144, "unit": "tokens", "applies": "load"],
            ["id": "temperature", "title": "Randomness", "type": "slider", "default": 0.6, "min": 0, "max": 2, "step": 0.1, "applies": "request"],
            ["id": "thinking", "title": "Reason before replying", "type": "toggle", "default": false, "applies": "request"]
        ]
    ]

    @Test func settingsControlsCarryLabelsAndCommitThroughTheirNativeActions() async throws {
        let row = try #require(Self.rows.first { $0.kind == "chat" })
        let service = FixtureService(rows: [row], metadata: [row.id: Self.settings], kinds: Self.kinds)
        let controller = TableController(service: service)
        await controller.refresh()
        let (window, host) = try await cardView(controller, row)
        let card = SettingsCardHost.of(controller)
        defer { SettingsCardHost.close(controller) }
        let toggle = try #require(views(host, MenuSwitch.self).first { $0.exposed?.accessibilityLabel() == "Keep loaded" })
        // VoiceOver reads the element accessibility exposes for the control: a named switch.
        let exposed = try #require(NSAccessibility.unignoredDescendant(of: toggle) as? NSCell)
        #expect(exposed === toggle.cell)
        #expect(exposed.accessibilityLabel() == "Keep loaded")
        #expect(exposed.accessibilityRole() == .checkBox && exposed.accessibilitySubrole() == .switch)
        let (target, _) = try hit(window, view: toggle, point: NSPoint(x: toggle.bounds.midX, y: toggle.bounds.midY))
        #expect(target === toggle)
        _ = toggle.accessibilityPerformPress()
        try await waitUntil { controller.rows.first?.keepHot == false }
        let field = try #require(views(host, NSTextField.self).first { $0.accessibilityLabel() == "Context window" })
        field.stringValue = "128K"
        #expect(field.sendAction(field.action, to: field.target))
        try await waitUntil { (try? await service.settings(id: row.id).first { $0.id == "context" }?.value.number) == 131_072 }
        #expect(controller.lastError == nil)
        // One runner is not a choice: no Runner row at all (it shows only as a picker for two or more), no pop-up.
        #expect(views(host, NSPopUpButton.self).isEmpty)
        let runner = try #require(controller.settings[row.id]?.first { $0.id == TableController.runnerKey })
        #expect(SettingsPane.plainValue(runner) && runner.display(runner.value) == "Native (Swift \u{00b7} MLX)")
        #expect(!views(host, TooltipCell.TooltipView.self).contains { $0.toolTip == "Only available runner for this model." })
        // Every control sits inside the card's form column, clear of its edges.
        let form = card.frame.insetBy(dx: SettingsPane.M.padding - 0.5, dy: 0)
        for control in card.views(MenuSwitch.self) + card.views(NSTextField.self) + card.views(NSSlider.self) {
            let r = control.convert(control.bounds, to: card.view.hosting)
            #expect(form.contains(r))
        }
    }

    static func installedModelFixtures() throws -> [ModelRow] {
        let root = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
        let ids = [
            "laya-english", "laya-multilingual", "laya-typed-decisions", "von-1.1", "von-1.2", "d1-3b",
            "kokoro-82m", "pocket-tts", "breeze-tts-2", "qwen3-tts-base", "qwen3-tts-voicedesign",
            "stable-audio-3-medium", "stable-audio-3-small-sfx", "re-use", "qwen3.8-27b", "flux2-klein-4b"
        ]
        return try ids.map { id in
            let model = try ModelManifest.load(shippedOrRetiredFolder(root, id))
            var row = ModelRow(
                id: id, name: model.name, kind: model.kind, params: model.params, licence: model.licence.spdx,
                precisions: model.precisions, states: Dictionary(uniqueKeysWithValues: model.precisions.map { ($0, .init(state: .ready)) }),
                selectedPrecision: model.basePrecision, fast: true, runner: model.runners[0].name, summary: model.summary)
            row.formats = model.formats
            row.precisionStats = PrecisionStatsReader.sets(kind: model.kind, baseDtype: model.basePrecision, runner: row.runner, supported: row.precisions, results: [])
            let context = model.nativeContext ?? model.context
            if let context {
                row.metrics["native_context"] = .number(Double(context))
                for dtype in row.precisions where row.kind == "chat" {
                    row.precisionStats[dtype]?.set(
                        .maximumContextTokens,
                        StatValue(
                            state: .declared, value: Double(context),
                            provenance: .init(chip: "Model config", date: "2026-10-09", method: "Declared supported context", source: "models/" + id + "/model.toml")))
                }
            }
            row.recommendedPrecision = model.basePrecision; row.recommendationPending = true
            row.recommendationReason = "Precision measurements are pending."
            return row
        }
    }

    @Test func installedFixtureCoversAllSixteenModelNamesAndKnownContext() throws {
        let rows = try Self.installedModelFixtures()
        #expect(rows.count == 16)
        #expect(rows.filter { $0.kind == "decisions" }.count == 6)
        let chat = try #require(rows.first { $0.id == "qwen3.8-27b" })
        #expect(TableFigures.figure(chat, precision: chat.selectedPrecision, column: .memory, host: "M5 Max").missing)
        for row in rows {
            #expect(TableFigures.figure(row, precision: row.selectedPrecision, column: .speed, host: "M5 Max").missing)
        }
    }

    @Test(.enabled(if: ProcessInfo.processInfo.environment["ARSENAL_TABLE_RENDERS"] != nil))
    func rendersAtRealWidthBeforeInteraction() async throws {
        let directory = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("docs/screens")
        for light in [false, true] {
            let name = light ? "light" : "dark"
            Appearance.shared.set(light ? .light : .dark)
            let realRows = try Self.installedModelFixtures()
            let real = TableController(service: FixtureService(rows: realRows, kinds: Self.kinds))
            real.apply(rows: realRows, kinds: Self.kinds)
            let (window, host) = host(real); defer { window.close() }
            // No run-loop turn or interaction before this bitmap: the menu's first layout pass.
            let bitmap = try #require(host.bitmapImageRepForCachingDisplay(in: host.bounds))
            host.cacheDisplay(in: host.bounds, to: bitmap)
            try bitmap.representation(using: .png, properties: [:])?.write(to: directory.appendingPathComponent("polish-table-first-frame-\(name).png"))
            await QA.renderTable(real, to: directory.appendingPathComponent("polish-table-real-models-\(name).png"))
            let controller = controller()
            await QA.renderTable(controller, to: directory.appendingPathComponent("polish-table-\(name).png"))
            var partial = Self.rows
            if let index = partial.firstIndex(where: { $0.kind == "chat" }) {
                partial[index].precisionStats["int4"]?.set(.nextTokenAgreementPercent, StatValue())
                partial[index].precisionStats["int4"]?.set(.firstTokenSeconds, StatValue())
                partial[index].precisionStats["int4"]?.set(.batteryPercentPer100000Tokens, StatValue())
                partial[index].selectedPrecision = "int8"
            }
            let pending = TableController(service: FixtureService(rows: partial, kinds: Self.kinds))
            pending.apply(rows: partial, kinds: Self.kinds)
            await QA.renderTable(pending, to: directory.appendingPathComponent("polish-table-pending-\(name).png"))
            partial[partial.firstIndex(where: { $0.kind == "chat" })!].selectedPrecision = "int4"
            pending.apply(rows: partial, kinds: Self.kinds)
            await QA.renderTable(pending, to: directory.appendingPathComponent("polish-table-partial-\(name).png"))
        }
        Appearance.shared.set(.dark)
    }
}

/// The drag handler and hit testing are real; the native tracking events come from this bounded offscreen queue.
private final class TableTrackingWindow: NSWindow {
    var events: [NSEvent] = []
    override func nextEvent(matching mask: NSEvent.EventTypeMask) -> NSEvent? {
        guard let first = events.first, mask.contains(NSEvent.EventTypeMask(rawValue: 1 << first.type.rawValue)) else { return nil }
        return events.removeFirst()
    }
}

/// A precise ordering race, with no sleeps or worker processes.
private actor TableOrderRaceService: ModelsService {
    private var order = KindSpec.defaults.map(\.id)
    private(set) var saved: [[String]] = []
    private var holdNext = false
    private var held: CheckedContinuation<[KindSpec], Never>?
    private var snapshot: [KindSpec] = []
    var holding: Bool { held != nil }
    func holdNextKindRead() { holdNext = true }
    func releaseKindRead() { held?.resume(returning: snapshot); held = nil }
    func rows() async -> [ModelRow] { await PolishTableTests.rows }
    func kinds() async -> [KindSpec] {
        let kinds = await PolishTableTests.kinds
        let ordered = order.compactMap { id in kinds.first { $0.id == id } }
        if holdNext {
            holdNext = false; snapshot = ordered
            return await withCheckedContinuation { held = $0 }
        }
        return ordered
    }
    func persistedKindOrder() async -> [String] { order }
    func setKindOrder(_ value: [String]) { order = value; saved.append(value) }
    func events() -> AsyncStream<ServiceEvent> { AsyncStream { $0.finish() } }
    func get(id: String, precision: String) {}
    func cancelGet(id: String) {}
    func load(id: String, precision: String?, fast: Bool?) {}
    func unload(id: String) {}
    func setKeepHot(id: String, on: Bool) {}
    func delete(id: String) {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) -> JSONObject { [:] }
    func runRaw(id: String, op: String, params: JSONObject) -> Data { Data() }
    func menuActions() -> [LoadedAction] { [] }
    func memory() -> MemorySnapshot { MemorySnapshot() }
    func catalogStatus() -> CatalogStatus { CatalogStatus() }
}

@MainActor @Suite(.menuState, .serialized) final class TableFigureStateTests {
    /// Peak memory shows only measured figures: a derived estimate is a missing figure (a triangle, "not measured"),
    /// with no delta. A result recorded on another core count of the same chip is another configuration.
    @Test func derivedMemoryIsMissingAndCoreCountsFollowTheDataContract() throws {
        let laya = try #require(Fixtures.figureRows().first { $0.id == "laya-multilingual" })
        let memory = TableFigures.figure(laya, precision: "int4", column: .memory, host: "M5 Max")
        #expect(memory.missing && memory.value == nil && memory.reference == nil && memory.delta == nil)
        #expect(memory.help == "int4 · \(TableFigures.notMeasured)")
        #expect(TableFigures.figure(laya, precision: "int4", column: .speed, host: "M5 Max").missing)
        let p = StatProvenance(chip: "m5-max", date: "", method: "", gpuCores: 40)
        #expect(TableFigures.measuredHere(p, host: "M5 Max", gpuCores: 40) && !TableFigures.measuredHere(p, host: "M5 Max", gpuCores: 32))
        #expect(TableFigures.measuredHere(StatProvenance(chip: "M5 Max", date: "", method: ""), host: "m5-max", gpuCores: 32))
        // On another Mac only speed names the chip; memory depends on the weights.
        let elsewhere = TableFigures.figure(laya, precision: "fp16", column: .memory, host: "M3 Pro")
        #expect(elsewhere.value == "1.18 GB" && elsewhere.reference == nil)
        #expect(TableFigures.figure(laya, precision: "fp16", column: .speed, host: "M3 Pro").reference == "M5 Max")
    }

    /// Get more models prints the table's peak memory in its Peak memory column, else the table's triangle (no `~` estimate).
    @Test func getMoreModelsShowsTheTablesPeakMemory() throws {
        let rows = Fixtures.figureRows()
        let laya = try #require(rows.first { $0.id == "laya-multilingual" })
        let item = BrowserItem(CatalogEntry(row: laya, metadata: [:], sizesByDtype: ["fp16": 1, "int8": 1, "int4": 1], added: true), kind: nil)
        #expect(item.ramText == "1.18 GB") // the 16-bit copy the table shows for a saved 8-bit choice
        let audio = try #require(rows.first { $0.id == "stable-audio-3-medium" })
        let unmeasured = BrowserItem(CatalogEntry(row: audio, metadata: [:], sizesByDtype: ["fp16": 1], added: true), kind: nil)
        #expect(unmeasured.ramText.isEmpty && unmeasured.ramHelp(physicalBytes: 128 << 30)?.contains("Not measured") == true)
    }

    // MARK: In-process event path: the hosted table in an unordered window, hit-tested the way AppKit routes a click.

    private func hosted() -> (TableController, NSWindow, NSView) {
        let rows = Fixtures.figureRows()
        let controller = TableController(service: FixtureService(rows: rows, kinds: Fixtures.figureKinds), chip: "M5 Max")
        controller.apply(rows: rows, kinds: Fixtures.figureKinds)
        let host = NSHostingView(rootView: ModelTable(controller: controller))
        let frame = NSRect(x: -9000, y: -9000, width: ModelTable.width(controller), height: ModelTable.height(controller))
        let window = NSWindow(contentRect: frame, styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false
        window.contentView = host; window.appearance = Appearance.shared.nsAppearance
        host.layoutSubtreeIfNeeded()
        return (controller, window, host)
    }
    private func views<T: NSView>(_ root: NSView, _: T.Type) -> [T] {
        (root as? T).map { [$0] } ?? root.subviews.flatMap { views($0, T.self) }
    }
    /// The view AppKit would deliver a click at `point` (in `view`) to, and the event.
    private func hit(_ window: NSWindow, view: NSView, point: NSPoint) throws -> (NSView, NSEvent) {
        let location = view.convert(point, to: nil)
        let root = try #require(window.contentView?.superview)
        let target = try #require(root.hitTest(root.convert(location, from: nil)))
        let event = try #require(
            NSEvent.mouseEvent(
                with: .leftMouseDown, location: location, modifierFlags: [], timestamp: ProcessInfo.processInfo.systemUptime,
                windowNumber: window.windowNumber, context: nil, eventNumber: 1, clickCount: 1, pressure: 1))
        return (target, event)
    }
    /// The row's Precision column: the columns top to bottom follow the groups' rows.
    private func control(_ controller: TableController, _ host: NSView, _ id: String) throws -> PrecisionLinesView {
        let order = controller.groups.flatMap(\.rows).map(\.id)
        let controls = views(host, PrecisionLinesView.self).sorted { $0.convert($0.bounds, to: nil).maxY > $1.convert($1.bounds, to: nil).maxY }
        #expect(controls.count == order.count)
        return controls[try #require(order.firstIndex(of: id))]
    }
    private func centre(_ c: PrecisionLinesView, dtype: String) throws -> NSPoint {
        let pill = c.pillRect(try #require(c.lines.firstIndex { $0.dtype == dtype }))
        return NSPoint(x: pill.midX, y: pill.midY)
    }

    @Test func precisionClicksReachTheDrawnLineAndDisabledTiersRefuse() async throws {
        let (controller, window, host) = hosted()
        defer { window.close() }
        await controller.refresh()
        // The lines grow the rows: the window follows the table's height.
        window.setContentSize(NSSize(width: ModelTable.width(controller), height: ModelTable.height(controller)))
        host.layoutSubtreeIfNeeded()
        // d1 declares 16-bit and int4 only: its two lines are those, and int4 is offered.
        let d1 = try control(controller, host, "d1-3b")
        #expect(d1.lines.map(\.copy.label) == ["16-bit", "4-bit"] && d1.lines[1].choosable)
        // Stable Audio's int4 is offered: a click on its line selects it.
        let audio = try control(controller, host, "stable-audio-3-medium")
        let (target, event) = try hit(window, view: audio, point: try centre(audio, dtype: "int4"))
        #expect(target === audio)
        target.mouseDown(with: event)
        try await waitUntil { controller.presentation(controller.rows.first { $0.id == "stable-audio-3-medium" }!).selected == "int4" }
        // The loaded row's lines draw on the loaded blue; an unloaded one on the menu.
        let laya = try control(controller, host, "laya-multilingual")
        #expect(laya.hot && !audio.hot)
    }

    /// Hovers reach a cell through AppKit's tooltip rects (SwiftUI's hosting view takes the hit test itself): the
    /// cell's tooltip view spans exactly the figure cell, and at its centre it is the innermost tooltip, which wins.
    @Test func figureHoversSitOverTheirCells() throws {
        let (_, window, host) = hosted(); defer { window.close() }
        _ = window
        let tips = views(host, TooltipCell.TooltipView.self).filter { $0.toolTip != nil }
        func reaches(_ text: (String) -> Bool) throws {
            let view = try #require(tips.first { text($0.toolTip ?? "") })
            let rect = view.convert(view.bounds, to: nil)
            #expect(abs(rect.width - ModelTable.W.metric) < 0.5 && abs(rect.height - ModelTable.rowHeight) < 0.5)
            let centre = NSPoint(x: rect.midX, y: rect.midY)
            let covering = tips.filter { $0.convert($0.bounds, to: nil).contains(centre) }
            let innermost = covering.min { $0.bounds.width * $0.bounds.height < $1.bounds.width * $1.bounds.height }
            #expect(innermost === view)
        }
        // A missing figure: the triangle's cell carries why.
        try reaches { $0 == "fp16 · " + TableFigures.notMeasured }
        // A measured 4-bit figure: its precision leads its hover (8-bit has no cell).
        try reaches { $0.hasPrefix("int4 \u{00b7} ") }
    }
}
