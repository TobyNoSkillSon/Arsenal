// SPDX-License-Identifier: MIT
import AppKit
import Foundation
import ArsenalCore
import SwiftUI
import Testing

@testable import Arsenal

/// The UX pass over the Models table, its settings card, Activity and Get more models: behaviour and wording only.
/// Alerts are answered by the menu's injected presenter (`ModelsMenu.presentAlert`); none is shown.
@MainActor @Suite(.menuState, .serialized) final class TableUXFixTests {
    private let support = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-table-ux-\(UUID().uuidString)")
    deinit { try? FileManager.default.removeItem(at: support) }

    private func qwen() -> ModelRow {
        var r = Fixtures.rows(.firstLaunch).first { $0.id == "qwen3.8-27b" }!
        r.states = ["bf16": PrecisionState(state: .ready), "int4": PrecisionState(state: .ready)]
        r.downloadBytes = ["bf16": 54_000_000_000, "int4": 15_000_000_000]
        r.selectedPrecision = "bf16"
        return r
    }
    private func controller(_ rows: [ModelRow]) async -> (TableController, FixtureService) {
        let service = FixtureService(rows: rows, supportDirectory: support)
        let c = TableController(service: service, chip: "M5 Max")
        await c.refresh()
        return (c, service)
    }
    private func current(_ c: TableController, _ id: String) -> ModelRow { c.rows.first { $0.id == id }! }

    // MARK: 1. A precision line's trash asks first, with the size

    @Test func theCopyQuestionIsWordedLikeTheRowDelete() {
        let kokoro = ModelRow(id: "kokoro-82m", name: "Kokoro 82M", kind: "voices", precisions: ["fp16"], selectedPrecision: "fp16")
        let alert = DeleteConfirm.copyAlert(kokoro, copy: .init(label: "16-bit", downloadable: true), bytes: 164_000_000)
        #expect(alert.messageText == "Delete the 16-bit copy of Kokoro 82M?")
        #expect(alert.informativeText == "164 MB is freed. It can be downloaded again.")
        #expect(alert.buttons.map(\.title) == ["Cancel", "Delete"])
        #expect(alert.buttons[1].hasDestructiveAction)
        // Outside a copy question the same entry point keeps the model's whole Delete.
        #expect(DeleteConfirm.alert(kokoro, bytes: 341_000_000).messageText == "Delete Kokoro 82M?")
        #expect(
            DeleteConfirm.$copy.withValue(.init(label: "4-bit", downloadable: false)) { DeleteConfirm.alert(kokoro, bytes: 0) }.informativeText
                == "It is no longer offered, so it cannot be downloaded again.")
    }

    @Test func aCopyTrashAsksThroughTheMenuAndCancelKeepsTheCopy() async throws {
        let (c, service) = await controller([qwen()])
        let menu = ModelsMenu(controller: c)
        var asked: [NSAlert] = []
        var answer = NSApplication.ModalResponse.alertFirstButtonReturn
        menu.presentAlert = { alert in
            asked.append(alert); return answer
        }
        let line = try #require(c.weights(current(c, "qwen3.8-27b")).copies.first { $0.dtype == "int4" })

        c.deleteCopy(current(c, "qwen3.8-27b"), "int4")
        try await Task.sleep(for: .milliseconds(50))
        #expect(asked.count == 1)
        #expect(asked.first?.messageText == "Delete the 4-bit copy of \(qwen().name)?")
        #expect(asked.first?.informativeText == "\(line.size) is freed. It can be downloaded again.")
        #expect(await service.deletes.isEmpty)
        #expect(c.removing.isEmpty)

        answer = .alertSecondButtonReturn
        c.deleteCopy(current(c, "qwen3.8-27b"), "int4")
        try await waitUntil { c.removing.isEmpty && c.weights(current(c, "qwen3.8-27b")).copies.count == 1 }
        #expect(asked.count == 2)
        #expect(await service.deletes == ["precision qwen3.8-27b int4"])
        // The model's whole Delete still asks its own question.
        #expect(DeleteConfirm.copy == nil)
    }

    @Test func aRefusedCopyDeleteNeverAsks() async {
        var r = qwen()
        r.states["int4"] = PrecisionState(state: .loaded)
        r.selectedPrecision = "int4"
        let (c, service) = await controller([r])
        var asked = 0
        c.confirmDeleteWeights = { _, _ in
            asked += 1; return true
        }
        c.deleteCopy(current(c, "qwen3.8-27b"), "int4")
        #expect(asked == 0)
        #expect(await service.deletes.isEmpty)
        #expect(c.lastError == "\(r.name): Unload to delete")
    }

    // MARK: 2. Raw file errors are worded and attached to their model

    static let raw = "The file \u{201C}diarize_session.input.json\u{201D} couldn\u{2019}t be opened because there is no such file."

    @Test func theLiveRawFileErrorIsWordedForItsModelWithTheRawTextOnHover() async throws {
        let service = FixtureService(rows: [qwen()], supportDirectory: support)
        let c = TableController(service: service, chip: "M5 Max")
        c.start()
        defer { c.stop() }
        try await waitUntil { !c.rows.isEmpty }
        await service.emit(.error(id: "qwen3.8-27b", message: Self.raw))
        try await waitUntil { c.lastError != nil }
        #expect(c.lastError == "\(qwen().name): a file it needs is missing")
        let footer = try #require(ModelTable.footer(c))
        #expect(footer.text == "\(qwen().name): a file it needs is missing")
        #expect(footer.help == Self.raw)
        #expect(!footer.text.contains("couldn"))

        // Without a model it still never shows raw.
        await service.emit(.error(id: nil, message: Self.raw))
        try await waitUntil { c.lastError == "A file it needs is missing" }
        #expect(ModelTable.footer(c)?.help == Self.raw)
    }

    @Test func aRawMessageFromAnActionGoesToTheFooterAndTheSettingsLineInPlainWords() async {
        let (c, _) = await controller([qwen()])
        let r = current(c, "qwen3.8-27b")
        c.fail(r, Self.raw)
        #expect(c.lastError == "\(r.name): a file it needs is missing")
        #expect(c.rowErrors[r.id] == "A file it needs is missing")
        #expect(ModelTable.footer(c)?.help == Self.raw)
        // Arsenal's own words pass through unchanged, hover included.
        c.fail(r, "Unload to delete")
        #expect(c.lastError == "\(r.name): Unload to delete")
        #expect(ModelTable.footer(c)?.help == "\(r.name): Unload to delete")
    }

    @Test func aLineSetElsewhereWithARawMessageIsRewordedInTheFooter() async {
        let (c, _) = await controller([qwen()])
        c.lastError = "Local models folder: " + Self.raw
        let footer = ModelTable.footer(c)
        #expect(footer?.text == "Local models folder: a file it needs is missing")
        #expect(footer?.help == "Local models folder: " + Self.raw)
        // Get more models words its refusals the same way.
        let line = BrowserController.errorLine("Kokoro 82M: " + Self.raw)
        #expect(line.text == "Kokoro 82M: a file it needs is missing" && line.help == "Kokoro 82M: " + Self.raw)
        #expect(BrowserController.errorLine("Kokoro 82M: still loading; wait, then Remove.").help == nil)
    }

    @Test func wordingCoversTheCommonFileFailures() {
        #expect(ErrorWording.plain(Self.raw) == "a file it needs is missing")
        #expect(
            ErrorWording.plain("The file \u{201C}x\u{201D} couldn\u{2019}t be opened because you don\u{2019}t have permission to view it.")
                == "Arsenal is not allowed to open a file it needs")
        #expect(ErrorWording.plain("The file \u{201C}x\u{201D} couldn\u{2019}t be saved in the folder \u{201C}y\u{201D}.") == "a file could not be saved")
        #expect(ErrorWording.plain("The data couldn\u{2019}t be read because it isn\u{2019}t in the correct format.") == "a file it needs could not be read")
        #expect(ErrorWording.plain("The operation couldn\u{2019}t be completed. (NSCocoaErrorDomain error 4.)") == "something went wrong on this Mac")
        #expect(ErrorWording.plain("Unload to delete") == nil)
        #expect(ErrorWording.plain("Not enough memory to load Qwen.") == nil)
    }

    // MARK: 3. A failed download: full reason on hover, Try Again

    @Test func aFailedGetReadsTryAgainAndKeepsTheWholeReasonOnHover() async throws {
        let reason = "Download stopped: Hugging Face timed out after 30 s while fetching model-00003-of-00004.safetensors (2.1 GB left)."
        var r = qwen()
        r.states = ["bf16": PrecisionState(state: .failed, error: reason)]
        let p = RowPresentation(r)
        #expect(p.action == .get && p.failed)
        #expect(p.actionTitle == "Try Again \u{00b7} 54.0 GB")
        // The size shows only when it fits the button's slot; the column never widens.
        let font = NSFont.systemFont(ofSize: ModelTable.valueSize)
        #expect(["Try Again \u{00b7} 54.0 GB", "Try Again"].contains(RowAction.shownTitle(p.actionTitle, font: font)))
        #expect(RowAction.buttonWidth(RowAction.shownTitle(p.actionTitle, font: font), font: font) <= RowAction.buttonWidth)
        // The line says it plainly under its lead (round 1 cut the raw reason beside Try Again); the hover keeps it whole.
        #expect(p.line.tone == .error && p.line.text == "Download stopped: the connection stalled")
        #expect(p.line.help == reason)
        #expect(ModelHelp.action(r, presentation: p).hasPrefix("Tries again. "))
        // A raw reason is worded on the line; the hover keeps it whole.
        r.states = ["bf16": PrecisionState(state: .failed, error: Self.raw)]
        let raw = RowPresentation(r)
        #expect(raw.line.text == "A file it needs is missing" && raw.line.help == Self.raw)

        // Try Again runs Get again.
        let (c, service) = await controller([r])
        c.confirmGet = { _, _ in true }
        c.perform(current(c, "qwen3.8-27b"))
        try await waitUntil { await !service.approvedGets.isEmpty }
        #expect(await service.approvedGets.first?.dtype == "bf16")
    }

    @Test func aGetThatNeverFailedKeepsItsSize() {
        var r = qwen()
        r.states = [:]
        let p = RowPresentation(r)
        #expect(!p.failed && p.actionTitle.hasPrefix("Get"))
    }

    // MARK: 4. Remove = off the list; Delete = off the disk

    @Test func removeAndDeleteMeanOneThingEach() {
        var weighed = qwen()
        weighed.states = ["bf16": PrecisionState(state: .ready, phase: "catalogRemoved")]
        #expect(RowPresentation(weighed).action == .remove)
        #expect(RowPresentation(weighed).actionTitle == "Delete")
        var listed = qwen()
        listed.states = ["bf16": PrecisionState(state: .notInstalled, phase: "catalogRemoved")]
        #expect(RowPresentation(listed).actionTitle == "Remove")
        #expect(ModelHelp.action(listed, presentation: RowPresentation(listed)) == "Removes it from the table.")
        #expect(RowAction.deleteHelp == "Remove from the table")
        var withdrawn = qwen()
        withdrawn.withdrawnTiers = ["int8": 27_000_000_000]
        #expect(!ModelHelp.withdrawnHelp(withdrawn).contains("Remove"))
        #expect(ModelHelp.withdrawnHelp(withdrawn).contains("Deletes the old copy from this Mac."))
        #expect(!ModelHelp.removedLine.contains("Removed"))
    }

    // MARK: 5. Keep loaded, never Keep Hot

    @Test func theSettingsSayKeepLoaded() {
        let keep = TableController.keepSetting(qwen())
        #expect(keep.title == "Keep loaded")
        let help: String = keep.declaration["help"]?.string ?? ""
        #expect(!help.lowercased().contains("hot"))
    }

    // MARK: 6. Activity: macOS free memory under its own name, set apart from the menu's Room for models; memory now

    @Test func activityNamesSystemFreeApartFromTheMenusRoomForModels() throws {
        let machine = ActivityFeed.Machine(memoryUsedMB: 60_000, memoryTotalMB: 128_000, arsenalMB: 300)
        let parts = ActivityText.memoryParts(machine: machine, models: [])
        let free = try #require(parts.first { $0.kind == .free })
        // Activity keeps its own measurement: total minus used.
        #expect(free.name == "Free" && free.mb == 68_000)
        let entry = try #require(ActivityText.legend(parts, models: [], totalMB: machine.memoryTotalMB).first { $0.kind == .free })
        // The legend reads "Free 66.4" (round 1 truncated "System free…" with no number); the hover names it in full.
        #expect(entry.name == "Free")
        #expect(entry.help?.hasPrefix("System free: memory macOS has free now.") == true)
        // The hover says how it differs from the menu's Room for models and names Arsenal's limit as the menu does.
        let limit = MemoryDisplay.text(decimalGB: MacMemoryGuard.safeInUseMB(totalMB: 128_000) / 1000)
        #expect(entry.help?.contains("Room for models") == true && entry.help?.contains(limit + " limit") == true)
        // The menu's Memory names its own figure the same way.
        #expect(MenuContent.memoryLines(MemorySnapshot(totalMB: 128_000, availableMB: 60_000), rows: [])[0].hasPrefix("Room for models: "))
        #expect(ActivityText.placeholderParts(.dark).contains { $0.name == "Free" })
        // A card's figure is memory now; the table's peak is at most.
        #expect(ActivityText.nowHelp(1_380).hasPrefix("1.3 GB in memory now."))
    }

    @Test func thePeakHoverSaysAtMost() {
        #expect(TableFigures.spelled(.peakMemoryGB, 1.29).hasSuffix(" at most"))
    }

    // MARK: 7. Background-only models have no Show in menu switch

    @Test func backgroundOnlyModelsHaveNoShowInMenuSwitch() async {
        var decider = ModelRow(id: "laya", name: "Laya", kind: "decisions", precisions: ["fp16"], selectedPrecision: "fp16")
        decider.tools = ["operation"]
        #expect(TableController.toolsSetting(decider) == nil)
        #expect(TableController.toolsSetting(decider, declared: ["decide"]) == nil)
        var embedder = ModelRow(id: "embed", name: "Embed", kind: "embeddings", precisions: ["fp16"], selectedPrecision: "fp16")
        embedder.tools = ["operation"]
        #expect(TableController.toolsSetting(embedder) == nil)
        var voice = ModelRow(id: "voice", name: "Voice", kind: "voices", precisions: ["fp16"], selectedPrecision: "fp16")
        voice.tools = ["voice"]
        #expect(TableController.toolsSetting(voice) != nil)
        // A decision model that also has a tool of its own keeps the switch.
        decider.tools = ["chat"]
        #expect(TableController.toolsSetting(decider) != nil)

        // Through the controller, as the card reads its rows.
        var laya = ModelRow(id: "laya", name: "Laya", kind: "decisions", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        laya.tools = ["operation"]
        voice.states = ["fp16": .init(state: .ready)] // On the table: its weights are on this Mac.
        let (c, _) = await controller([laya, voice])
        #expect(c.settings["laya"]?.contains { $0.id == TableController.toolsKey } == false)
        #expect(c.settings["laya"]?.contains { $0.id == TableController.keepKey } == true)
        #expect(c.settings["voice"]?.contains { $0.id == TableController.toolsKey } == true)
    }
}
