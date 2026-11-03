// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import SwiftUI
import Testing

@testable import Arsenal

/// Round 2 of the Models table: Vella's precision control, and int4 offered everywhere with a triangle in
/// the int4 cell whose words show only on hover. Checked on the real hosted table (the figure fixture), first frame,
/// through AppKit's own tooltips and hit testing.
@MainActor @Suite(.menuState, .serialized) final class TableVellaTests {
    @Test func zeroMatrixTierHoverStatesTheActualHalfFormat() {
        var row = ModelRow(id: "rife", name: "RIFE", kind: "restoration", precisions: ["fp16", "int8", "int4"], selectedPrecision: "int4")
        row.precisionDtypes = ["int8": "fp16", "int4": "fp16"]
        #expect(PrecisionColumn.label(row, "4") == "fp16" && PrecisionColumn.label(row, "8") == "fp16")
        row.precisionNotes = ["int4": "Runs as FP16; no eligible Linear/Embedding matrices and no weight compression."]
        let hover = ModelHelp.precision(row, column: "4", presentation: RowPresentation(row))
        #expect(hover.contains("Runs as FP16") && hover.contains("no weight compression"))
    }
    /// Rows whose int4 is offered and not passed (from the service's own `precisionQualityWarnings`): Kokoro's passed,
    /// My Video (a local folder) declares no int4.
    static let flagged: Set<String> = ["laya-multilingual", "d1-3b", "flux2-klein-4b", "stable-audio-3-medium", "re-use", "my-chat-4b"]

    /// int4's triangle sits on the int4 line only, on rows whose int4 is flagged, on disk and can be chosen; its words are
    /// the first line of that line's hover and in no other hover. A precision not on disk has no line (RE-USE).
    @Test func int4WarningSitsOnTheInt4LineOnly() async throws {
        let (controller, window, host) = await hosted(); defer { window.close() }
        var warned: Set<String> = []
        for row in controller.groups.flatMap(\.rows) {
            let v = try column(controller, host, row.id)
            let p = controller.presentation(row)
            for line in v.lines {
                let retired = "\(line.dtype) \u{00b7} \(line.copy.size). \(PrecisionLine.retiredLine)"
                let expected = line.column.map { ModelHelp.precisionTooltip(row, column: $0, presentation: p) } ?? retired
                #expect(v.tooltip(.pill(v.lines.firstIndex(of: line)!)) == expected, "\(row.id) \(line.dtype)")
            }
            if v.lines.contains { $0.warned } { warned.insert(row.id) }
            if row.precisionQualityWarnings["int4"] != nil, let int4 = v.lines.first(where: { $0.column == "4" }), int4.choosable {
                #expect(v.lines.filter(\.warned).map(\.column) == ["4"], "\(row.id)")
                #expect(int4.help.hasPrefix("int4 may lose quality from 4-bit quantization."))
                #expect(v.lines.first { $0.column == "16" }?.help.contains("4-bit") == false)
            } else {
                #expect(!v.lines.contains { $0.warned }, "\(row.id)")
            }
        }
        // RE-USE is flagged but has nothing on disk: no line, so no triangle.
        #expect(warned == Self.flagged.subtracting(["re-use"]))
        // No text in the table: the warning is in no cell's own hover either (figures, names, status lines).
        let hovers = views(host, TooltipCell.TooltipView.self).compactMap(\.toolTip)
        #expect(!hovers.isEmpty && !hovers.contains { $0.contains("4-bit quantization") })
    }

    /// A flagged int4 that cannot run (here a runner that does not support it) shows its reason and no triangle, and
    /// cannot be chosen.
    @Test func unavailableInt4ShowsItsReasonNotTheWarning() async throws {
        var row = ModelRow(
            id: "synthetic", name: "Synthetic", kind: "decisions", params: "1B", precisions: ["bf16", "int8", "int4"], selectedPrecision: "bf16",
            fastExact: false)
        for p in row.precisions { row.states[p] = PrecisionState(state: .ready) }
        row.precisionStats = ["int4": Fixtures.stats(.decisions, "int4", unavailable: "This runner does not support this precision")]
        #expect(row.precisionQualityWarnings["int4"] != nil)
        let (controller, window, host) = await hosted(rows: [row]); defer { window.close() }
        let v = try column(controller, host, "synthetic")
        let int4 = try #require(v.lines.first { $0.column == "4" })
        #expect(!int4.choosable && !int4.warned)
        #expect(int4.help == "Unavailable. This runner does not support this precision.")
    }

    /// Reason lines read the same everywhere: "<State>. <Sentence>." (review r2 #5).
    @Test func reasonLinesShareOneForm() throws {
        for row in Fixtures.figureRows() {
            for (_, reason) in ModelHelp.unavailable(row) {
                #expect(reason.hasPrefix("Not offered. ") || reason.hasPrefix("Unavailable. "), "\(row.id): \(reason)")
                #expect(reason.hasSuffix(".") && !reason.contains(": "), "\(row.id): \(reason)")
            }
        }
        let video = try #require(Fixtures.figureRows().first { $0.id == "my-video-5b" })
        #expect(ModelHelp.unavailable(video)["4"] == "Not offered. This folder declares no 4-bit format.")
    }

    /// The status hover and the precision hovers tell one story (review r2 #4): it names each offered precision's own
    /// check, never "no quality check" for the model while a precision's hover says its check did not pass.
    @Test func statusAgreesWithEachPrecisionsCheck() throws {
        for row in Fixtures.figureRows() where !row.local {
            // The badge's hover (a busy, pending or local row shows its own line instead, about that state).
            let help = RowPresentation.badgeLine(row, chip: "M5 Max").help
            #expect(!help.contains("No quality check yet"), "\(row.id): \(help)")
            switch row.precisionQualityWarnings["int4"] {
            case PrecisionStatSet.int4QualityWarning?: #expect(help.contains("int4 did not pass its quality check."), "\(row.id): \(help)")
            case PrecisionStatSet.int4UncheckedWarning?: #expect(help.contains("int4 has no quality check yet."), "\(row.id): \(help)")
            default: #expect(!row.precisions.contains("int4") || help.contains("int4 passed its quality check."), "\(row.id): \(help)")
            }
        }
        let flux = try #require(Fixtures.figureRows().first { $0.id == "flux2-klein-4b" })
        // int8 is not offered: its quality is not listed; the 16-bit and int4 lines are.
        #expect(RowPresentation(flux).status.help.hasPrefix("bf16 has no quality check yet. int4 did not pass"))
    }

    /// While a row is busy its precision hover says what is busy, in plain words (review r2 #6).
    @Test func lockedHoverSaysWhatIsBusy() throws {
        var row = try #require(Fixtures.figureRows().first { $0.id == "stable-audio-3-medium" })
        row.states["int8"] = PrecisionState(state: .deriving)
        let p = RowPresentation(row)
        #expect(p.locked)
        let tip = ModelHelp.precisionTooltip(row, column: "16", presentation: p)
        #expect(tip.hasSuffix("Locked while int8 is made on this Mac; you can choose again when it finishes."))
        row.states["int8"] = PrecisionState(state: .downloading)
        #expect(ModelHelp.lockLine(row) == "Locked while int8 downloads; you can choose again when it finishes.")
    }

    /// The int4 triangle reads apart from the missing-figure triangle (yellow, beside a figure): orange pixels on the int4
    /// line of an unchosen flagged int4, none where int4 passed (Kokoro), and on the chosen gold line the triangle is ink.
    @Test func int4TriangleIsOrangeOnTheLineAndInkOnTheGold() async throws {
        let (controller, window, host) = await hosted(); defer { window.close() }
        func orange(_ id: String) throws -> Int {
            let v = try column(controller, host, id)
            let i = try #require(v.lines.firstIndex { $0.dtype == "int4" })
            let rep = try #require(v.bitmapImageRepForCachingDisplay(in: v.bounds))
            v.cacheDisplay(in: v.bounds, to: rep)
            let scale = CGFloat(rep.pixelsWide) / v.bounds.width
            let pill = v.pillRect(i)
            var count = 0
            for x in Int(pill.minX * scale)..<min(rep.pixelsWide, Int(pill.maxX * scale)) {
                for y in Int(pill.minY * scale)..<min(rep.pixelsHigh, Int(pill.maxY * scale)) {
                    guard let p = rep.colorAt(x: x, y: y)?.usingColorSpace(.sRGB) else { continue }
                    // Orange (about 240, 150, 70 here); the gold (236, 190, 72) and the yellow are greener.
                    if p.redComponent > 0.85, p.greenComponent > 0.5, p.greenComponent < 0.68, p.blueComponent < 0.45 { count += 1 }
                }
            }
            return count
        }
        #expect(try orange("stable-audio-3-medium") > 6)
        #expect(try orange("kokoro-82m") == 0)
        #expect(try orange("my-chat-4b") == 0)
        #expect(TierControl.warningColor != NSColor.systemYellow)
    }

    /// Loaded: the white edge on the line that runs. Kokoro runs its chosen 16-bit; Laya runs a leftover int8 (no longer
    /// offered), so its edge is on the int8 line and the gold on the 16-bit line it will load next.
    @Test func loadedEdgeMarksTheLineThatRuns() async throws {
        let (controller, window, host) = await hosted(); defer { window.close() }
        let kokoro = try column(controller, host, "kokoro-82m")
        #expect(kokoro.hot && kokoro.lines.filter(\.selected).map(\.dtype) == ["fp16"] && kokoro.lines.filter(\.loaded).map(\.dtype) == ["fp16"])
        let laya = try column(controller, host, "laya-multilingual")
        #expect(laya.hot && laya.lines.filter(\.selected).map(\.dtype) == ["fp16"] && laya.lines.filter(\.loaded).map(\.dtype) == ["int8"])
        let audio = try column(controller, host, "stable-audio-3-medium")
        #expect(!audio.hot && !audio.lines.contains { $0.loaded })
    }

    /// First frame: every row's column at its final size; its lines 16-bit, 8-bit, 4-bit from the top; the bolt (with its
    /// hover) only on rows with custom kernels, its slot kept on every row so the lines line up.
    @Test func firstFrameGeometryAndBolts() async throws {
        let (controller, window, host) = await hosted(); defer { window.close() }
        let columns = views(host, PrecisionLinesView.self)
        #expect(columns.count == controller.rows.count)
        let lefts = Set(columns.map { ($0.convert($0.bounds, to: host).minX * 2).rounded() })
        #expect(lefts.count == 1)
        for v in columns {
            #expect(abs(v.frame.width - PrecisionLines.width) < 0.5)
            #expect(v.frame.height >= PrecisionLines.height(lines: v.lines.count))
            let bits = v.lines.map { PrecisionColumn.bits($0.dtype) ?? 0 }
            #expect(bits == bits.sorted(by: >), "16-bit on top, then 8-bit, then 4-bit")
        }
        let bolts = columns.filter { $0.boltRect != nil }
        #expect(bolts.count == controller.rows.filter(\.hasFastKernels).count && bolts.count == 1)
        #expect(bolts.first.map { $0.tooltip(.bolt) } == TierControl.boltHelp)
    }

    /// The Precision heading is centred over the lines' pills.
    @Test func precisionHeadingSitsOverTheLines() async throws {
        let (_, window, host) = await hosted(); defer { window.close() }
        let heading = try #require(views(host, TooltipCell.TooltipView.self).first { $0.toolTip == TierControl.headerHelp })
        let v = try #require(views(host, PrecisionLinesView.self).first { !$0.lines.isEmpty })
        let a = heading.convert(heading.bounds, to: host), b = v.convert(v.pillRect(0), to: host)
        #expect(abs(a.midX - b.midX) < 0.5, "heading \(a.midX), lines \(b.midX)")
    }

    /// A click on a flagged int4 line, delivered where AppKit's hit testing sends it, chooses int4: the warning informs,
    /// it never blocks.
    @Test func clickOnAFlaggedInt4LineChoosesIt() async throws {
        let (controller, window, host) = await hosted(); defer { window.close() }
        let audio = try column(controller, host, "stable-audio-3-medium")
        let i = try #require(audio.lines.firstIndex { $0.dtype == "int4" })
        let point = NSPoint(x: audio.pillRect(i).midX, y: audio.pillRect(i).midY)
        let (target, event) = try hit(window, view: audio, point: point)
        #expect(target === audio)
        target.mouseDown(with: event)
        try await waitUntil { controller.presentation(controller.rows.first { $0.id == "stable-audio-3-medium" }!).selected == "int4" }
    }

    // MARK: Hosting

    private func waitUntil(seconds: Double = 5, _ condition: @MainActor () async -> Bool) async throws {
        let deadline = Date().addingTimeInterval(seconds)
        while !(await condition()) {
            if Date() > deadline { Issue.record("timed out"); return }
            try await Task.sleep(for: .milliseconds(20))
        }
    }

    private func hosted(rows: [ModelRow] = Fixtures.figureRows(), chip: String = "M5 Max") async -> (TableController, NSWindow, NSView) {
        let controller = TableController(service: FixtureService(rows: rows, kinds: Fixtures.figureKinds), chip: chip)
        controller.apply(rows: rows, kinds: Fixtures.figureKinds)
        // The store's on-disk listing: the Precision lines.
        await controller.refresh()
        let host = NSHostingView(rootView: ModelTable(controller: controller))
        // Offscreen and never ordered in: nothing appears on the screen.
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
    /// The row's Precision column: the columns top to bottom follow the groups' rows.
    private func column(_ controller: TableController, _ host: NSView, _ id: String) throws -> PrecisionLinesView {
        let order = controller.groups.flatMap(\.rows).map(\.id)
        let columns = views(host, PrecisionLinesView.self).sorted { $0.convert($0.bounds, to: nil).maxY > $1.convert($1.bounds, to: nil).maxY }
        #expect(columns.count == order.count)
        return columns[try #require(order.firstIndex(of: id))]
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
}
