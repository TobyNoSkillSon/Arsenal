// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// v2 table: the pinned row, the red pending row and its Reload, settings, Remove and the local-code consent.
@MainActor @Suite(.menuState, .serialized) final class V2TableTests {
    static let settings: JSONObject = [
        "setting": [
            [
                "id": "context_limit", "title": "Context", "type": "choice", "default": 32768, "choices": [8192, 32768, 131072, 262144], "applies": "load",
                "maps_to": "config.context_limit", "help": "Longest conversation the model keeps."
            ],
            [
                "id": "temperature", "title": "Temperature", "type": "slider", "default": 0.6, "min": 0, "max": 2, "step": 0.1, "applies": "request",
                "maps_to": "request.temperature"
            ],
            ["id": "thinking", "title": "Thinking", "type": "toggle", "default": false, "applies": "request", "maps_to": "request.thinking"]
        ]
    ]
    private func row(
        _ states: [String: PrecisionState] = ["bf16": .init(state: .ready), "int8": .init(state: .ready)], fast: Bool = true, precisions: [String] = ["bf16", "int8"]
    ) -> ModelRow {
        ModelRow(id: "m", name: "M", kind: "chat", params: "27B", precisions: precisions, states: states, selectedPrecision: "bf16", fast: fast, fastExact: true)
    }
    private func live(_ rows: [ModelRow], metadata: [String: JSONObject] = [:]) async -> (TableController, FixtureService) {
        let service = FixtureService(rows: rows, metadata: metadata, kinds: Fixtures.fallbackKinds)
        let c = TableController(service: service, chip: "M5 Max")
        await c.refresh()
        return (c, service)
    }

    // MARK: Pinned row and layout

    @Test func dayOneIsTheHeaderOnePinnedRowAndTheFooter() {
        #expect(ModelTable.height([]) == ModelTable.chromeHeight() + ModelTable.rowHeight + ModelTable.rowGap)
        // With groups the pinned row sits below a group rule, after every group.
        #expect(ModelTable.height([2]) == ModelTable.chromeHeight() + ModelTable.groupsHeight([2]) + ModelTable.groupRuleHeight + ModelTable.rowHeight + ModelTable.rowGap)
        // Width does not jump when the first group appears (Speed and Peak memory for every kind) and includes the gear column.
        #expect(ModelTable.slots([]) == 2)
        #expect(
            ModelTable.W.row(slots: 3) - ModelTable.W.gear - ModelTable.W.spacing
                == ModelTable.W.fixed.reduce(0, +) + 3 * ModelTable.W.metric + ModelTable.W.action + CGFloat(ModelTable.W.fixed.count + 3) * ModelTable.W.spacing + 2
                * ModelTable.W.rowPadding)
    }

    @Test func thePinnedRowOpensTheBrowser() {
        let c = TableController(service: FixtureService(rows: []), chip: nil)
        var opened = 0
        c.openBrowser = { opened += 1 }
        c.openBrowser()
        #expect(opened == 1)
        #expect(PinnedRow.title == "Get more models")
        // The row's words say it: no tooltip.
        #expect(PinnedRow.help == nil)
    }

    // MARK: Red row

    @Test func notLoadedIsNeverRed() {
        var r = row(); r.needsReload = true
        let p = RowPresentation(r, previewPrecision: "int8", previewFast: false)
        #expect(!p.pending && p.action == .load)
    }

    @Test func aDtypeAFlipOrALoadSettingOnALoadedRowIsPendingAndRevertingClearsIt() {
        // The other cell is int4 (int8 has no cell).
        let loaded = row(["bf16": .init(state: .loaded), "int4": .init(state: .ready)], precisions: ["bf16", "int4"])
        #expect(!RowPresentation(loaded).pending && RowPresentation(loaded).action == .unload)
        let dtype = RowPresentation(loaded, previewPrecision: "int4")
        #expect(dtype.pending && dtype.action == .reload && dtype.actionTitle == "Reload")
        #expect(RowPresentation(loaded, previewFast: false).pending)
        // An 8-bit choice is no choice here: the loaded row stays as it is.
        #expect(!RowPresentation(loaded, previewPrecision: "int8").pending)
        // Back to what is loaded: not pending (computed against the loaded configuration).
        #expect(!RowPresentation(loaded, previewPrecision: "bf16", previewFast: true).pending)
        var setting = loaded; setting.needsReload = true
        #expect(RowPresentation(setting).pending && RowPresentation(setting).action == .reload)
    }

    @Test func reloadAppliesSettingsInPlaceOrLoadsTheNewDtype() async throws {
        let (c, service) = await live([row(["bf16": .init(state: .ready), "int4": .init(state: .ready)], precisions: ["bf16", "int4"])], metadata: ["m": Self.settings])
        c.perform(c.rows[0])
        try await waitUntil { c.presentation(c.rows[0]).hot }
        // A load-time setting changed: the row turns red with Reload; Reload = applySettings (same dtype and mode).
        c.set(c.rows[0], "context_limit", 131072)
        try await waitUntil { c.rows.first?.needsReload == true }
        #expect(c.presentation(c.rows[0]).pending)
        #expect(c.pendingChanges(c.rows[0]) == ["Context 128K"])
        #expect(
            ModelHelp.action(c.rows[0], presentation: c.presentation(c.rows[0]), changes: c.pendingChanges(c.rows[0])).hasPrefix(
                "Reload applies: Context 128K."))
        c.perform(c.rows[0])
        try await waitUntil { c.rows.first?.needsReload == false }
        #expect(await service.calls.contains("apply m"))
        // A dtype change: Reload loads the other dtype (its cell is int4; int8 has none).
        c.select(c.rows[0], column: "4")
        #expect(c.pendingChanges(c.rows[0]) == ["int4"])
        c.perform(c.rows[0])
        // (The stub keeps the old dtype loaded beside the new one; core's service swaps them.)
        try await waitUntil { await service.calls.last == "load m int4" }
    }

    @Test func aRequestSettingNeverReddensTheRow() async throws {
        let (c, _) = await live([row(["bf16": .init(state: .ready)])], metadata: ["m": Self.settings])
        c.perform(c.rows[0])
        try await waitUntil { c.presentation(c.rows[0]).hot }
        c.set(c.rows[0], "temperature", 1.2)
        try await waitUntil { c.settings["m"]?.first { $0.id == "temperature" }?.value == 1.2 }
        #expect(!c.presentation(c.rows[0]).pending)
    }

    @Test func notLoadedTheLoadTooltipListsChangedLoadSettings() async throws {
        let (c, _) = await live([row()], metadata: ["m": Self.settings])
        c.set(c.rows[0], "context_limit", 8192)
        try await waitUntil { c.loadChoices(c.rows[0]) == ["Context 8K"] }
        let help = ModelHelp.action(c.rows[0], presentation: c.presentation(c.rows[0]), loadChoices: c.loadChoices(c.rows[0]))
        #expect(help == "Loads 16-bit \u{00b7} Context 8K")
        #expect(!c.presentation(c.rows[0]).pending)
    }

    // MARK: Settings card

    @Test func settingsOpenOneCardAtATimeAndDefaultsRestoreLoadSettingsOnly() async throws {
        let (c, _) = await live([row()], metadata: ["m": Self.settings])
        defer { c.settingsCard.close() }
        // A chat model declares the chat tool: its settings end with "Show in menu".
        #expect(c.settings["m"]?.map(\.id) == [TableController.keepKey, TableController.runnerKey, "context_limit", "temperature", "thinking", TableController.toolsKey])
        let height = ModelTable.height(c)
        c.openSettings(c.rows[0])
        // The gear opens the card over the table; the table keeps its size.
        #expect(c.settingsCard.openID == "m" && c.settingsCard.state.id == "m" && ModelTable.height(c) == height)
        // Again: the gear closes it.
        c.openSettings(c.rows[0])
        #expect(c.settingsCard.openID == nil && c.settingsCard.state.id == nil)
        c.openSettings(c.rows[0])
        #expect(c.settingsCard.openID == "m")
        // A request setting away from its default is the apps' business: nothing for Restore defaults.
        c.set(c.rows[0], "thinking", true)
        try await waitUntil { c.settings["m"]?.first { $0.id == "thinking" }?.value == .bool(true) }
        #expect(!SettingsPane.footer(c.settings["m"] ?? [], loaded: false).defaults && c.restorable(c.rows[0]).isEmpty)
        c.set(c.rows[0], "context_limit", 8192)
        try await waitUntil { SettingsPane.footer(c.settings["m"] ?? [], loaded: false).defaults }
        c.restoreDefaults(c.rows[0])
        try await waitUntil { c.settings["m"]?.first { $0.id == "context_limit" }.map { $0.value == $0.defaultValue } == true }
        #expect(c.settings["m"]?.first { $0.id == "thinking" }?.value == .bool(true))
        c.settingsCard.close("m")
        #expect(c.settingsCard.openID == nil)
    }

    @Test func theCardShowsLoadSettingsOnlyAndItsHeightFollowsTheContent() {
        let states = (Self.settings["setting"]?.array ?? []).compactMap(\.object).map { SettingState(declaration: $0, value: $0["default"]!) }
        #expect(PaneLayout.items(states, panel: nil).map(\.id) == ["context_limit"])
        let metrics = SettingsPane.M.self
        let plain = SettingsPane.height(states, loaded: false)
        #expect(plain == metrics.top + metrics.bottom + metrics.rowHeight)
        // A pending load setting on a loaded model adds Reload at the foot.
        let pending = states.map { s in
            var s = s; s.pending = s.id == "context_limit"; return s
        }
        #expect(SettingsPane.height(pending, loaded: true) == plain + metrics.footerHeight)
        #expect(SettingsPane.height(pending, loaded: false) == plain)
    }

    @Test func controlsAndValuesFollowTheDeclaration() {
        func s(_ d: JSONObject) -> SettingState { SettingState(declaration: d, value: d["default"] ?? .null) }
        #expect(s(["type": "toggle", "default": true]).control == .toggle)
        #expect(s(["type": "choice", "choices": ["a", "b", "c"]]).control == .segments)
        #expect(s(["type": "choice", "choices": ["denoise", "restore", "upscale"]]).control == .list) // labels > 6 characters
        #expect(s(["id": "context_limit", "type": "choice", "choices": [8192, 32768, 131072, 262144]]).control == .list)
        #expect(s(["type": "voice"]).control == .voice && s(["type": "slider"]).control == .slider && s(["type": "number"]).control == .number)
        let context = s(["id": "context_limit", "type": "choice"])
        #expect(context.display(32768) == "32K")
        let slider = s(["type": "slider", "min": 0.25, "max": 4, "step": 0.05])
        #expect(slider.number(from: "1.234") == 1.25)
        #expect(slider.number(from: "9") == 4)
        #expect(slider.number(from: "abc") == nil)
        #expect(slider.display(1.25) == "1.25")
        #expect(s(["type": "text", "applies": "load", "help": "Path."]).tooltip == "Path. Applies after Reload.")
        #expect(s(["type": "text", "applies": "request"]).tooltip == "Applies to the next request.")
    }

    @Test func voicesComeFromTheVoicesOp() {
        let list = VoiceChoice.list(["voices": [["id": "af_heart", "name": "Heart", "language": "en"], ["voice_id": "v2"]]])
        #expect(list == [VoiceChoice(id: "af_heart", name: "Heart", language: "en"), VoiceChoice(id: "v2", name: "v2", language: "")])
        #expect(list[0].title == "Heart \u{00b7} English")
        // A preset id used as its own name reads as a name; American and British English by name.
        let kokoro = VoiceChoice.list(["voices": [["id": "bf_emma", "name": "bf_emma", "language": "en-GB"]]])
        #expect(kokoro == [VoiceChoice(id: "bf_emma", name: "Emma", language: "en-GB")] && kokoro[0].languageName == "British English")
    }

    // MARK: Remove

    @Test func removeWithoutWeightsIsImmediateWithWeightsAsks() async throws {
        // A failed Get: on the table with no weights.
        let none = ModelRow(
            id: "a", name: "A", kind: "voices", precisions: ["fp16"], states: ["fp16": .init(state: .failed, error: "Get failed.")], selectedPrecision: "fp16")
        let have = ModelRow(id: "b", name: "B", kind: "voices", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16", sizeBytes: 341_000_000)
        let (c, service) = await live([none, have])
        var asked: [(id: String, bytes: Int64)] = []
        c.confirmDeleteWeights = { row, bytes in
            asked.append((row.id, bytes)); return true
        }
        c.remove(c.rows[0])
        try await waitUntil { c.rows.map(\.id) == ["b"] }
        #expect(asked.isEmpty)
        c.remove(c.rows[0])
        try await waitUntil { c.rows.isEmpty }
        // One question, with the size that goes; Delete removes every weight, then takes the row off the table.
        #expect(asked.map(\.id) == ["b"] && asked.first?.bytes == 341_000_000)
        #expect(await service.deletes == ["all b"])
        #expect(await service.calls == ["remove a keep", "remove b keep"])
        let alert = DeleteConfirm.alert(have, bytes: 341_000_000)
        #expect(alert.messageText == "Delete B?")
        #expect(alert.informativeText.hasPrefix("Frees up to 341 MB."))
        #expect(alert.buttons.map(\.title) == ["Cancel", "Delete"])
        #expect(RowAction.deleteHelp == "Remove from the table")
    }

    @Test func cancellingRemoveKeepsTheRow() async throws {
        let have = ModelRow(id: "b", name: "B", kind: "voices", precisions: ["fp16"], states: ["fp16": .init(state: .ready)], selectedPrecision: "fp16")
        let (c, service) = await live([have])
        c.confirmDeleteWeights = { _, _ in false }
        c.remove(c.rows[0])
        try await Task.sleep(for: .milliseconds(50))
        let calls = await service.calls
        #expect(c.rows.count == 1 && calls.isEmpty)
    }

    // MARK: Local code consent

    /// A model whose Get failed: on the table with no weights, its action Gets again.
    private func failedGet(_ row: ModelRow) -> ModelRow {
        var row = row
        row.states = [row.selectedPrecision: .init(state: .failed, error: "Get failed.")]
        return row
    }

    /// A loaded local model whose folder changed: the service marks it unapproved (new digest) and pending (Reload).
    private func changedWhileLoaded() -> ModelRow {
        var row = StubModelsService.localFixture
        row.states = ["bf16": .init(state: .loaded)]; row.needsReload = true
        return row
    }

    @Test func reloadOfAChangedLocalFolderAsksAndRefusalRunsNothing() async throws {
        let (c, service) = await live([changedWhileLoaded()])
        #expect(c.presentation(c.rows[0]).action == .reload)
        var asked = 0
        c.confirmCode = { _ in
            asked += 1; return false
        }
        c.perform(c.rows[0])
        try await Task.sleep(for: .milliseconds(50))
        #expect(asked == 1)
        #expect(await service.calls.isEmpty)
        await c.refresh()
        #expect(c.rows[0].needsCodeConsent && c.rows[0].state == .loaded)
    }

    @Test func reloadOfAChangedLocalFolderApprovesOnlyTheShownDigestOnYes() async throws {
        let (c, service) = await live([changedWhileLoaded()])
        var shown: String?
        c.confirmCode = { row in
            shown = row.contentDigest; return true
        }
        c.perform(c.rows[0])
        try await waitUntil { await service.calls.count >= 2 }
        #expect(shown == String(repeating: "a", count: 64))
        #expect(await service.calls.prefix(2) == ["approve stub-local aaaa app", "apply stub-local"])
        try await waitUntil { c.rows.first?.needsCodeConsent == false }
    }

    /// The Local badge says the folder is unsigned and who approved which contents when (O-9).
    @Test func theLocalBadgeShowsWhoApprovedWhichContents() async throws {
        // On the table: its weights are on this Mac (Load asks for the approval).
        var local = StubModelsService.localFixture
        local.states = ["bf16": .init(state: .ready)]
        let (c, _) = await live([local])
        var status = c.presentation(c.rows[0]).status
        #expect(status.text == "Local")
        #expect(status.help.contains("not reviewed or signed by Arsenal") && status.help.contains("not approved yet"))
        c.confirmCode = { _ in true }
        c.perform(c.rows[0])
        try await waitUntil { c.rows.first?.approval != nil }
        status = c.presentation(c.rows[0]).status
        #expect(status.help.contains("Approved by app \u{00b7} ") && status.help.contains("contents aaaaaaaa"))
        #expect(!status.help.contains("changed since"))
        // Changed after approval: the old approval stays visible, with why it asks again.
        var changed = c.rows[0]; changed.needsCodeConsent = true
        #expect(ModelHelp.local(changed).contains("the folder has changed since"))
        #expect(BrowserItem(CatalogEntry(row: c.rows[0]), kind: nil).statusHelp == ModelHelp.local(c.rows[0]))
    }

    @Test func aSettingsWarningReachesTheRowTooltip() async {
        var r = row(); r.settingsWarning = "Context 131072 is no longer offered; using the default 32768."
        let (c, _) = await live([r])
        #expect(ModelHelp.text(c.rows[0], presentation: c.presentation(c.rows[0])).hasSuffix("\nContext 131072 is no longer offered; using the default 32768."))
    }

    @Test func unloadNeverAsksOrApproves() async throws {
        var row = StubModelsService.localFixture
        row.states = ["bf16": .init(state: .loaded)]
        let (c, service) = await live([row])
        #expect(c.presentation(c.rows[0]).action == .unload)
        var asked = 0
        c.confirmCode = { _ in
            asked += 1; return true
        }
        c.perform(c.rows[0])
        try await waitUntil { c.rows.first?.state == .ready }
        #expect(asked == 0)
        #expect(!(await service.calls.contains { $0.hasPrefix("approve") }))
        #expect(c.rows[0].needsCodeConsent)
    }

    @Test func consentWithoutADigestIsNeverApproved() async throws {
        var row = failedGet(StubModelsService.localFixture)
        row.contentDigest = nil
        let (c, service) = await live([row])
        var asked = 0
        c.confirmCode = { _ in
            asked += 1; return true
        }
        c.perform(c.rows[0])
        try await Task.sleep(for: .milliseconds(50))
        #expect(asked == 0)
        #expect(await service.calls.isEmpty)
        #expect(c.lastError != nil)
    }

    @Test func localCodeAsksOnceThenApprovesTheShownDigestAndGets() async throws {
        let local = failedGet(StubModelsService.localFixture)
        let (c, service) = await live([local])
        var asked = 0
        c.confirmCode = { _ in
            asked += 1; return false
        }
        c.perform(c.rows[0])
        try await waitUntil { asked == 1 }
        let none = await service.calls
        #expect(asked == 1 && none.isEmpty)
        c.confirmCode = { _ in
            asked += 1; return true
        }
        c.perform(c.rows[0])
        try await waitUntil { c.rows.first?.needsCodeConsent == false }
        #expect(await service.calls.first == "approve stub-local aaaa app")
        // Approved: the next action does not ask again.
        try await waitUntil { c.presentation(c.rows[0]).busyText == nil }
        let before = asked
        c.perform(c.rows[0])
        try await Task.sleep(for: .milliseconds(50))
        #expect(asked == before)
        let alert = ModelsMenu.codeAlert(local)
        #expect(alert.messageText == "Run Local runner fixture\u{2019}s own code?")
        #expect(alert.informativeText.contains("its Rust runner was not reviewed or signed by Arsenal"))
        #expect(alert.buttons.map(\.title) == ["Cancel", "Run"])
        // A Swift runner is the bundled worker: only the folder's Metal kernels are unreviewed.
        var kernelsOnly = local; kernelsOnly.runnerLanguage = "swift"
        let kernelAlert = ModelsMenu.codeAlert(kernelsOnly).informativeText
        #expect(kernelAlert.contains("Swift executable or Metal kernels") && kernelAlert.contains("your permissions"))
    }
}
