// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import SwiftUI
import Testing

@testable import Arsenal

/// The Models table's weights: 16-bit only with int4 beside it, one line per copy on disk with its
/// trash, every on-disk model listed, a single-confirmation Delete of a model's weights, the unlisted-weights row, and
/// one store call per click.
@MainActor @Suite(.menuState, .serialized) final class TableWeightsTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-table-weights-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    /// A fixture row of `id` with the given state per precision and the bytes each copy takes.
    private func row(_ id: String, _ states: [String: ModelState], selected: String? = nil, bytes: [String: Int64] = [:]) -> ModelRow {
        var r = Fixtures.rows(.firstLaunch).first { $0.id == id }!
        for (dtype, state) in states { r.states[dtype] = PrecisionState(state: state) }
        for (dtype, size) in bytes { r.downloadBytes[dtype] = size }
        if let selected { r.selectedPrecision = selected }
        return r
    }
    private func qwen(_ states: [String: ModelState] = ["bf16": .ready, "int4": .ready], selected: String = "bf16") -> ModelRow {
        row("qwen3.8-27b", states, selected: selected, bytes: ["bf16": 54_000_000_000, "int4": 15_000_000_000])
    }
    /// A controller over `rows`, with the store's on-disk listing and orphans from the fixture, read once.
    private func controller(
        _ rows: [ModelRow], unlisted: [OnDiskModel] = [], orphans: Int64 = 0, confirm: @escaping (ModelRow, Int64) -> Bool = { _, _ in false }
    ) async -> (TableController, FixtureService) {
        let service = FixtureService(rows: rows, supportDirectory: support)
        await service.setUnlisted(unlisted)
        await service.setOrphanBytes(orphans)
        let c = TableController(service: service, chip: "M5 Max")
        c.confirmDeleteWeights = confirm
        await c.refresh()
        return (c, service)
    }
    private func current(_ c: TableController, _ id: String) -> ModelRow { c.rows.first { $0.id == id }! }

    // MARK: Item 1: 16-bit only

    @Test func thePrecisionControlHasSixteenAndFourOnly() {
        #expect(TierControl.columns == ["16", "4"])
        #expect(PrecisionColumn.all == ["16", "4"])
        #expect(PrecisionColumn.of("int8") == nil && PrecisionColumn.isEightBit("int8"))
        #expect(PrecisionColumn.of("bf16") == "16" && PrecisionColumn.of("fp16") == "16" && PrecisionColumn.of("int4") == "4")
    }

    @Test func aSavedEightBitPreferenceFallsBackToTheSixteenBitCopy() {
        var r = row("qwen3.8-27b", ["bf16": .ready, "int4": .ready])
        r.precisions = ["bf16", "int8", "int4"]
        r.selectedPrecision = "int8"
        #expect(RowPresentation(r).selected == "bf16")
        #expect(RowPresentation(r).action == .load)
    }

    @Test func getDefaultsToTheSixteenBitCopyForEveryFixtureRow() {
        for r in Fixtures.realisticRows() + Fixtures.rows(.firstLaunch) {
            let selected = RowPresentation(r).selected
            #expect(!PrecisionColumn.isEightBit(selected, formats: r.formats), "\(r.id) defaults to \(selected)")
        }
    }

    /// Every shipped manifest starts on a format that is not 8-bit: nothing defaults to int8.
    @Test func noShippedModelDefaultsToEightBit() throws {
        let models = URL(fileURLWithPath: #filePath).deletingLastPathComponent().deletingLastPathComponent().deletingLastPathComponent()
            .appendingPathComponent("models")
        let folders = try FileManager.default.contentsOfDirectory(at: models, includingPropertiesForKeys: nil)
            .filter { FileManager.default.fileExists(atPath: $0.appendingPathComponent("model.toml").path) }
        var read = 0
        for folder in folders {
            guard let manifest = try? ModelManifest.load(folder, requirePins: false) else { continue }
            read += 1
            #expect(!PrecisionColumn.isEightBit(manifest.defaultPrecision, formats: manifest.formats), "\(manifest.id) defaults to \(manifest.defaultPrecision)")
        }
        #expect(read >= 38, "only \(read) of \(folders.count) manifests read")
    }

    @Test func aRowsDefaultFormatIsItsSixteenBitCopyWhenItHasOne() {
        #expect(RowPresentation(row("flux2-klein-4b", [:])).selected == "bf16")
        #expect(RowPresentation(qwen([:], selected: "bf16")).selected == "bf16")
    }

    // MARK: Item 2: copies on disk above the Precision control

    @Test func copyLinesListOnlyThePrecisionsOnDisk() async {
        let (c, _) = await controller([qwen(["bf16": .ready, "int4": .notInstalled])])
        let copies = c.weights(current(c, "qwen3.8-27b")).copies
        #expect(copies.map(\.label) == ["16-bit"])
        #expect(copies.map(\.size) == ["54.0 GB"])
        #expect(copies.first?.help == "Delete the 16-bit copy from this Mac")
    }

    @Test func copyLinesShowBothPrecisionsWithTheirSizesSixteenFirst() async {
        let (c, _) = await controller([qwen()])
        let weights = c.weights(current(c, "qwen3.8-27b"))
        #expect(weights.copies.map(\.label) == ["16-bit", "4-bit"])
        #expect(weights.copies.map(\.size) == ["54.0 GB", "15.0 GB"])
        #expect(weights.bytes == 69_000_000_000)
    }

    @Test func aLeftoverEightBitCopyStillGetsItsLineAndTrash() async {
        var r = qwen(["bf16": .ready])
        r.precisions = ["bf16", "int8", "int4"]
        r.states["int8"] = PrecisionState(state: .ready)
        r.downloadBytes["int8"] = 27_000_000_000
        let (c, service) = await controller([r], confirm: { _, _ in true })
        let copy = c.weights(current(c, "qwen3.8-27b")).copies.first { $0.label == "8-bit" }
        #expect(copy?.enabled == true)
        c.deleteCopy(current(c, "qwen3.8-27b"), "int8")
        try? await waitUntil { c.removing.isEmpty }
        #expect(await service.deletes == ["precision qwen3.8-27b int8"])
    }

    @Test func aLoadedCopyCannotBeDeletedAndSaysWhy() async {
        let (c, service) = await controller([qwen(["bf16": .ready, "int4": .loaded], selected: "int4")])
        let r = current(c, "qwen3.8-27b")
        let copy = c.weights(r).copies.first { $0.label == "4-bit" }
        #expect(copy?.enabled == false)
        #expect(copy?.help == "Unload to delete")
        c.deleteCopy(r, "int4")
        #expect(c.removing.isEmpty)
        #expect(await service.deletes.isEmpty)
    }

    @Test func aCopyWaitsWhileAnotherPrecisionDownloads() async {
        let (c, _) = await controller([qwen(["bf16": .ready, "int4": .downloading])])
        let copy = c.weights(current(c, "qwen3.8-27b")).copies.first
        #expect(copy?.blocked == "Wait for the download to finish")
        #expect(copy?.enabled == false)
    }

    @Test func aCopyDeleteRunsOnceAndTheLineGoesAfterTheRefresh() async throws {
        let (c, service) = await controller([qwen()], confirm: { _, _ in true })
        let r = current(c, "qwen3.8-27b")
        c.deleteCopy(r, "int4")
        c.deleteCopy(r, "int4")
        try await waitUntil { c.removing.isEmpty && c.weights(current(c, "qwen3.8-27b")).copies.count == 1 }
        #expect(await service.deletes == ["precision qwen3.8-27b int4"])
        #expect(c.weights(current(c, "qwen3.8-27b")).copies.map(\.label) == ["16-bit"])
        #expect(c.rowErrors["qwen3.8-27b"] == nil)
    }

    @Test func aCopyTrashIsOffWhileItsDeleteRuns() async throws {
        let (c, service) = await controller([qwen()], confirm: { _, _ in true })
        await service.setDelay("deletePrecision", .milliseconds(400))
        c.deleteCopy(current(c, "qwen3.8-27b"), "int4")
        let during = c.weights(current(c, "qwen3.8-27b")).copies.first { $0.label == "4-bit" }
        #expect(during?.enabled == false && during?.blocked == "Deleting\u{2026}")
        try await waitUntil(timeout: 5) { c.removing.isEmpty }
        #expect(c.weights(current(c, "qwen3.8-27b")).copies.map(\.label) == ["16-bit"])
    }

    /// The 16-bit copy's trash is off while the 4-bit copy built from it is on disk, and the tooltip names that copy; the 4-bit
    /// copy's own trash stays on.
    @Test func aSixteenBitCopyWithAFourBitCopyBuiltFromItIsOffAndSaysWhy() {
        let disk = OnDiskModel(
            id: "qwen3.8-27b",
            precisions: [
                OnDiskPrecision(dtype: "bf16", bytes: 54_000_000_000, inUse: false, dependents: ["int4"]),
                OnDiskPrecision(dtype: "int4", bytes: 15_000_000_000, inUse: false)
            ], cacheBytes: 0, offered: true)
        let weights = WeightsPresentation.make(qwen(), disk: disk, working: [], removing: [])
        #expect(weights.copies.map(\.label) == ["16-bit", "4-bit"])
        #expect(weights.copies[0].enabled == false)
        #expect(weights.copies[0].help == "Delete the 4-bit copy first: it is built from this one.")
        #expect(weights.copies[1].enabled && weights.copies[1].help == "Delete the 4-bit copy from this Mac")
    }

    // MARK: The Precision column: a line per precision on disk, 16-bit over 8-bit over 4-bit

    @Test func aRowGrowsToFitItsLines() {
        #expect(ModelTable.rowSpan(lines: 0) == ModelTable.rowHeight && ModelTable.rowSpan(lines: 1) == ModelTable.rowHeight)
        #expect(ModelTable.rowSpan(lines: 3) == 3 * PrecisionLines.lineHeight + 2 * ModelTable.linesPadding)
        #expect(ModelTable.rowSpan(lines: 3) > ModelTable.rowHeight && ModelTable.rowSpan(lines: 3) > ModelTable.rowSpan(lines: 2))
        // Never smaller than the menu's secondary text (11 pt), and the table's own value size.
        #expect(PrecisionLines.labelFont.pointSize >= NSFont.smallSystemFontSize && PrecisionLines.sizeFont.pointSize == ModelTable.valueSize)
    }

    /// Laya's int8 and int4 are hardlinks to its fp16 file (made at load), so the store counts 846 MB three times; each
    /// line shows that precision's own size, the figure Arsenal's MCP reports (`CatalogEntry.sizesByDtype`).
    @Test func eachLineShowsItsPrecisionsOwnSizeNotTheSharedFile() {
        var laya = ModelRow(id: "laya-english", name: "Laya English", kind: "decisions", precisions: ["fp16", "int8", "int4"], selectedPrecision: "fp16")
        for p in laya.precisions { laya.states[p] = PrecisionState(state: .ready) }
        let shared: Int64 = 846_226_754
        let disk = OnDiskModel(
            id: laya.id, precisions: ["fp16", "int4", "int8"].map { OnDiskPrecision(dtype: $0, bytes: shared, inUse: false) }, cacheBytes: 0, offered: true)
        let catalog: [String: Int64] = ["fp16": 846_226_754, "int8": 449_557_963, "int4": 238_001_274]
        let weights = WeightsPresentation.make(laya, disk: disk, sizes: catalog, working: [], removing: [])
        #expect(weights.copies.map(\.label) == ["16-bit", "8-bit", "4-bit"])
        #expect(weights.copies.map(\.size) == ["846 MB", "450 MB", "238 MB"])
        // Without a catalog figure (a model the catalog does not list), the store's own count.
        #expect(WeightsPresentation.make(laya, disk: disk, working: [], removing: []).copies.map(\.size) == ["846 MB", "846 MB", "846 MB"])
    }

    /// The fixed order whatever the store lists first, and no line for a precision that is not on disk.
    @Test func linesRunSixteenEightFourAndSkipWhatIsNotOnDisk() {
        var r = qwen(["bf16": .ready, "int4": .ready])
        r.precisions = ["bf16", "int8", "int4"]
        func lines(_ dtypes: [String]) -> [String] {
            let disk = OnDiskModel(id: r.id, precisions: dtypes.map { OnDiskPrecision(dtype: $0, bytes: 1_000_000_000, inUse: false) }, cacheBytes: 0, offered: true)
            return PrecisionLine.make(r, presentation: RowPresentation(r), weights: WeightsPresentation.make(r, disk: disk, working: [], removing: [])).map(\.copy.label)
        }
        #expect(lines(["int4", "int8", "bf16"]) == ["16-bit", "8-bit", "4-bit"])
        #expect(lines(["int4", "bf16"]) == ["16-bit", "4-bit"])
        #expect(lines(["int8"]) == ["8-bit"])
        #expect(lines([]).isEmpty)
    }

    /// The chosen line is gold, the loaded one carries the edge, a leftover int8 keeps its size and trash but cannot be
    /// chosen, and a busy row's lines cannot be chosen.
    @Test func linesCarryTheChoiceTheLoadAndTheirTrash() {
        var r = qwen(["bf16": .ready, "int4": .loaded], selected: "int4")
        r.precisions = ["bf16", "int8", "int4"]
        r.states["int8"] = PrecisionState(state: .ready)
        let disk = OnDiskModel(
            id: r.id,
            precisions: [
                OnDiskPrecision(dtype: "bf16", bytes: 54_000_000_000, inUse: false), OnDiskPrecision(dtype: "int8", bytes: 27_000_000_000, inUse: false),
                OnDiskPrecision(dtype: "int4", bytes: 15_000_000_000, inUse: true)
            ], cacheBytes: 0, offered: true)
        let lines = PrecisionLine.make(r, presentation: RowPresentation(r), weights: WeightsPresentation.make(r, disk: disk, working: [], removing: []))
        #expect(lines.map(\.column) == ["16", nil, "4"])
        #expect(lines.map(\.selected) == [false, false, true] && lines.map(\.loaded) == [false, false, true])
        #expect(lines.map(\.choosable) == [true, false, true])
        #expect(lines[1].help.hasSuffix(PrecisionLine.retiredLine) && lines[1].copy.enabled)
        #expect(lines[2].copy.help == WeightsPresentation.unloadFirst && !lines[2].copy.enabled)
        var busy = r
        busy.states["bf16"] = PrecisionState(state: .downloading)
        let busyWeights = WeightsPresentation.make(busy, disk: disk, working: [], removing: [])
        #expect(!PrecisionLine.make(busy, presentation: RowPresentation(busy), weights: busyWeights).contains { $0.choosable })
    }

    /// On the hosted table: every row's column holds its lines inside the row, the columns of neighbouring rows never
    /// overlap, and a trash click on a line deletes that copy only.
    @Test func linesFitTheirRowsWithoutOverlapAndTheTrashDeletesThatCopy() async throws {
        let rows = Fixtures.figureRows()
        let service = FixtureService(rows: rows, kinds: Fixtures.figureKinds, supportDirectory: support)
        let c = TableController(service: service, chip: "M5 Max")
        // The trash asks first (with the copy's size); this click is confirmed.
        var asked: [String?] = []
        c.confirmDeleteWeights = { _, _ in
            asked.append(DeleteConfirm.copy?.label); return true
        }
        await c.refresh()
        let host = NSHostingView(rootView: ModelTable(controller: c))
        let window = NSWindow(
            contentRect: NSRect(x: -9000, y: -9000, width: ModelTable.width(c), height: ModelTable.height(c)), styleMask: .borderless, backing: .buffered, defer: false)
        window.isReleasedWhenClosed = false; window.contentView = host
        defer { window.close() }
        host.layoutSubtreeIfNeeded()
        func columns() -> [PrecisionLinesView] {
            func all(_ v: NSView) -> [PrecisionLinesView] { ((v as? PrecisionLinesView).map { [$0] } ?? []) + v.subviews.flatMap(all) }
            return all(host).sorted { $0.convert($0.bounds, to: host).minY < $1.convert($1.bounds, to: host).minY }
        }
        let shown = columns()
        #expect(shown.count == c.rows.count)
        #expect(shown.contains { $0.lines.count == 3 }, "rows with 16-, 8- and 4-bit on disk")
        for (index, v) in shown.enumerated() {
            let frame = v.convert(v.bounds, to: host)
            #expect(frame.height >= PrecisionLines.height(lines: v.lines.count), "the row is tall enough for its lines")
            for i in v.lines.indices {
                #expect(v.bounds.contains(v.pillRect(i)) && v.bounds.contains(v.trashRect(i)), "line \(i) inside its row")
                if i > 0 { #expect(v.pillRect(i).minY >= v.pillRect(i - 1).maxY, "line \(i) under line \(i - 1)") }
            }
            for other in shown.dropFirst(index + 1) { #expect(!frame.intersects(other.convert(other.bounds, to: host))) }
        }
        // A click on the chosen line writes nothing; each line's accessibility frame is the line on screen.
        let chosen = try #require(shown.first { v in v.lines.contains { $0.selected && $0.choosable } })
        let index = try #require(chosen.lines.firstIndex { $0.selected })
        #expect(!chosen.press(.pill(index)) && c.previewPrecision.isEmpty)
        let parts = try #require(chosen.accessibilityChildren() as? [PrecisionLinesView.Element])
        #expect(parts.count == 2 * chosen.lines.count)
        let screen = window.convertToScreen(chosen.convert(chosen.pillRect(index), to: nil))
        #expect(parts[2 * index].accessibilityFrame() == screen)
        let audio = try #require(shown.first { v in v.lines.contains { $0.dtype == "int4" && $0.copy.enabled } && !v.hot })
        let i = try #require(audio.lines.firstIndex { $0.dtype == "int4" })
        let point = audio.convert(NSPoint(x: audio.trashRect(i).midX, y: audio.trashRect(i).midY), to: nil)
        let root = try #require(window.contentView?.superview)
        let target = try #require(root.hitTest(root.convert(point, from: nil)))
        #expect(target === audio)
        target.mouseDown(
            with: try #require(
                NSEvent.mouseEvent(
                    with: .leftMouseDown, location: point, modifierFlags: [], timestamp: 0, windowNumber: window.windowNumber, context: nil, eventNumber: 1,
                    clickCount: 1, pressure: 1)))
        try await waitUntil {
            let deletes = await service.deletes
            return c.removing.isEmpty && !deletes.isEmpty
        }
        let deletes = await service.deletes
        #expect(deletes.count == 1 && (deletes.first ?? "").hasSuffix(" int4"))
        #expect(asked == ["4-bit"])
    }

    // MARK: Item 3: every model with weights on disk is a row

    @Test func weightsTheCatalogDoesNotListShowAsRowsInTheirOwnGroup() async {
        let retired = OnDiskModel(id: "retired-model", precisions: [OnDiskPrecision(dtype: "bf16", bytes: 3_100_000_000, inUse: false)], cacheBytes: 0, offered: false)
        let (c, _) = await controller([qwen()], unlisted: [retired])
        #expect(c.rows.contains { $0.id == "retired-model" && $0.kind == Unlisted.kind })
        #expect(c.groups.last?.kind.id == Unlisted.kind && c.groups.last?.rows.map(\.id) == ["retired-model"])
        let unlisted = c.rows.first { $0.id == "retired-model" }!
        #expect(c.weights(unlisted).copies.map(\.size) == ["3.1 GB"])
        #expect(RowPresentation(unlisted).action == .remove)
    }

    @Test func aModelWithCacheOnlyStillShowsItsRow() async {
        let cached = OnDiskModel(id: "cache-only", precisions: [], cacheBytes: 500_000_000, offered: false)
        let (c, _) = await controller([qwen()], unlisted: [cached])
        #expect(c.rows.contains { $0.id == "cache-only" })
        #expect(c.weights(current(c, "cache-only")).hasWeights)
    }

    @Test func deletingUnlistedWeightsRemovesTheRowWithoutAnyRemoveCall() async throws {
        let retired = OnDiskModel(id: "retired-model", precisions: [OnDiskPrecision(dtype: "bf16", bytes: 3_100_000_000, inUse: false)], cacheBytes: 0, offered: false)
        let (c, service) = await controller([qwen()], unlisted: [retired], confirm: { _, _ in true })
        c.deleteWeights(current(c, "retired-model"))
        try await waitUntil { c.removing.isEmpty && !c.rows.contains { $0.id == "retired-model" } }
        #expect(await service.deletes == ["all retired-model"])
        #expect(c.lastError == nil)
    }

    // MARK: Item 4: a model's Delete removes all its weights, after one confirmation

    @Test func deleteAsksOnceWithTheSizeAndRemovesEveryWeightAndTheRow() async throws {
        final class Asked: @unchecked Sendable { var sizes: [Int64] = [] }
        let asked = Asked()
        let (c, service) = await controller(
            [qwen()],
            confirm: { _, bytes in
                asked.sizes.append(bytes); return true
            })
        c.deleteWeights(current(c, "qwen3.8-27b"))
        try await waitUntil { c.removing.isEmpty && !c.rows.contains { $0.id == "qwen3.8-27b" } }
        #expect(asked.sizes == [69_000_000_000])
        #expect(await service.deletes == ["all qwen3.8-27b"])
        #expect(!c.rows.contains { $0.id == "qwen3.8-27b" })
    }

    @Test func aDeclinedDeleteChangesNothing() async throws {
        let (c, service) = await controller([qwen()], confirm: { _, _ in false })
        c.deleteWeights(current(c, "qwen3.8-27b"))
        try await Task.sleep(for: .milliseconds(50))
        #expect(await service.deletes.isEmpty)
        #expect(c.rows.contains { $0.id == "qwen3.8-27b" })
    }

    @Test func twoClicksOnDeleteAskOnceAndDeleteOnce() async throws {
        final class Count: @unchecked Sendable { var asks = 0 }
        let asks = Count()
        let (c, service) = await controller(
            [qwen()],
            confirm: { _, _ in
                asks.asks += 1; return true
            })
        c.deleteWeights(current(c, "qwen3.8-27b"))
        c.deleteWeights(current(c, "qwen3.8-27b"))
        try await waitUntil { c.removing.isEmpty && !c.rows.contains { $0.id == "qwen3.8-27b" } }
        #expect(asks.asks == 1)
        #expect(await service.deletes == ["all qwen3.8-27b"])
    }

    @Test func aLoadedModelsDeleteIsRefusedBeforeAnyQuestion() async throws {
        final class Count: @unchecked Sendable { var asks = 0 }
        let asks = Count()
        let (c, service) = await controller(
            [qwen(["bf16": .ready, "int4": .loaded], selected: "int4")],
            confirm: { _, _ in
                asks.asks += 1; return true
            })
        let r = current(c, "qwen3.8-27b")
        #expect(c.weights(r).deleteBlocked == "Unload to delete")
        c.deleteWeights(r)
        try await Task.sleep(for: .milliseconds(50))
        #expect(asks.asks == 0)
        #expect(await service.deletes.isEmpty)
        #expect(c.lastError == "\(r.name): Unload to delete")
    }

    @Test func theDeleteSheetNamesTheSizeWithOneDeleteButton() {
        let alert = DeleteConfirm.alert(qwen(), bytes: 69_000_000_000)
        #expect(alert.messageText == "Delete \(qwen().name)?")
        #expect(alert.informativeText.contains("Frees up to 69.0 GB"))
        #expect(alert.buttons.map(\.title) == ["Cancel", "Delete"])
    }

    @Test func aRemoveActionOfACatalogRemovedRowDeletesItsWeights() async throws {
        var r = qwen()
        r.states = ["bf16": PrecisionState(state: .ready, phase: "catalogRemoved"), "int4": PrecisionState(state: .ready, phase: "catalogRemoved")]
        let presentation = RowPresentation(r)
        #expect(presentation.action == .remove)
        #expect(presentation.actionTitle == "Delete")
        let (c, service) = await controller([r], confirm: { _, _ in true })
        c.perform(current(c, "qwen3.8-27b"))
        try await waitUntil { c.removing.isEmpty && !c.rows.contains { $0.id == "qwen3.8-27b" } }
        #expect(await service.deletes == ["all qwen3.8-27b"])
    }

    // MARK: Item 5: unlisted weights

    @Test func theUnlistedWeightsRowShowsOnlyWhileThereAreBytes() async throws {
        let (c, service) = await controller([qwen()], orphans: 0)
        #expect(!ModelTable.orphanShown(c))
        let before = ModelTable.height(c)
        await service.setOrphanBytes(3_100_000_000)
        await c.refresh()
        #expect(ModelTable.orphanShown(c) && c.orphanBytes == 3_100_000_000)
        #expect(ModelTable.height(c) == before + ModelTable.rowHeight + ModelTable.rowGap)
    }

    @Test func deleteUnlistedWeightsIsOneClickAndNeverAutomatic() async throws {
        let (c, service) = await controller([qwen()], orphans: 3_100_000_000)
        #expect(await service.deletes.isEmpty)
        c.deleteOrphans()
        c.deleteOrphans()
        try await waitUntil { c.removing.isEmpty && c.orphanBytes == 0 }
        #expect(await service.deletes == ["orphans"])
        #expect(!ModelTable.orphanShown(c))
    }

    // MARK: Item 6: one store call per click, controls off while in flight

    @Test func twoClicksOnGetStartOneDownload() async throws {
        // A failed Get keeps the row on the table without weights; its Try Again is a Get.
        let (c, service) = await controller([row("flux2-klein-4b", ["bf16": .failed], selected: "bf16", bytes: ["bf16": 15_980_140_843])])
        let r = current(c, "flux2-klein-4b")
        c.perform(r)
        c.perform(r)
        #expect(c.working == ["flux2-klein-4b"])
        try await waitUntil { c.working.isEmpty }
        #expect(await service.approvedGets.count == 1)
    }
}
