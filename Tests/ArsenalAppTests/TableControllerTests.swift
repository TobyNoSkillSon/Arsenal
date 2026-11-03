// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

@MainActor @Suite(.menuState, .serialized) final class TableControllerTests {
    private func controller(_ rows: [ModelRow] = Fixtures.rows(.loaded)) -> TableController {
        let c = TableController(service: FixtureService(rows: rows), chip: "M5 Max")
        c.apply(rows: rows, kinds: Fixtures.fallbackKinds)
        return c
    }

    /// The first-launch models with these on the table as failed Gets: no weights yet, their action Gets again.
    static func failedGets(_ ids: [String]) -> [ModelRow] {
        Fixtures.rows(.firstLaunch).map { row in
            var row = row
            if ids.contains(row.id) { row.states[row.selectedPrecision] = PrecisionState(state: .failed, error: "Get failed.") }
            return row
        }
    }

    @Test func groupsFollowTheKindsOrderAndHideEmptyKinds() {
        #expect(controller().groups.map(\.kind.id) == ["voices", "images", "chat"])
        let onlyChat = controller(Fixtures.rows(.loaded).filter { $0.kind == "chat" })
        #expect(onlyChat.groups.map(\.kind.id) == ["chat"])
    }

    @Test func rowsKeepCatalogOrderWhateverIsSelected() {
        let c = controller()
        let before = c.groups.flatMap { $0.rows.map(\.id) }
        let qwen = c.rows.first { $0.id == "qwen3.8-27b" }!
        c.select(qwen, column: "16")
        #expect(c.presentation(qwen).selected == "bf16")
        #expect(c.groups.flatMap { $0.rows.map(\.id) } == before)
    }

    /// The move shows at once and is saved through the service with every kind once (core persists it as
    /// config.json `kindOrder`); the next `kinds()` comes back in that order.
    @Test func movingAGroupSavesTheOrderThroughTheService() async throws {
        let service = FixtureService(rows: Fixtures.rows(.loaded), kinds: Fixtures.fallbackKinds)
        let c = TableController(service: service, chip: "M5 Max")
        await c.refresh()
        #expect(c.groups.map(\.kind.id) == ["voices", "images", "chat"])
        c.moveGroup("chat", toSlot: 0)
        #expect(c.groups.map(\.kind.id) == ["chat", "voices", "images"])
        try await waitUntil { await service.saved.count == 1 }
        // Every kind once, the hidden one (decisions: no rows) after the visible ones.
        #expect(await service.saved.last == ["chat", "voices", "images", "decisions"])
        c.moveGroup("voices", by: 1)
        c.moveGroup("chat", toSlot: 3)
        try await waitUntil { await service.saved.count == 3 }
        #expect(await service.saved.last == ["images", "voices", "chat", "decisions"])
        try await waitUntil { c.groups.map(\.kind.id) == ["images", "voices", "chat"] }
        // A new controller (next launch) reads the saved order.
        let next = TableController(service: service, chip: "M5 Max")
        await next.refresh()
        #expect(next.groups.map(\.kind.id) == ["images", "voices", "chat"])
    }

    @Test func movesAtTheEdgesDoNothing() {
        let c = controller()
        c.moveGroup("voices", by: -1)
        c.moveGroup("chat", by: 1)
        #expect(c.groups.map(\.kind.id) == ["voices", "images", "chat"])
    }

    @Test func closingTheMenuKeepsPrecisionPreferences() async throws {
        let c = controller()
        // qwen runs at int4 (loaded): its 16-bit choice is a pending Reload.
        let qwen = c.rows.first { $0.id == "qwen3.8-27b" }!
        c.select(qwen, column: "16")
        #expect(c.presentation(qwen).action == .reload)
        c.discardPreviews()
        try await waitUntil { c.rows.first { $0.id == qwen.id }?.selectedPrecision == "bf16" && c.previewPrecision.isEmpty }
        #expect(c.presentation(c.rows.first { $0.id == qwen.id }!).action == .reload)
    }

    @Test func aBusyRowTakesNoChoices() {
        let c = controller(Fixtures.rows(.deriving))
        let kokoro = c.rows.first { $0.id == "kokoro-82m" }!
        c.select(kokoro, column: "16")
        #expect(c.previewPrecision[kokoro.id] == nil && c.presentation(kokoro).locked)
    }

    /// Get → Load → Unload through the controller on the stub service: the row's action follows the service.
    @Test func theActionCycleGoesThroughTheService() async throws {
        let rows = Self.failedGets(["kokoro-82m"])
        let service = FixtureService(rows: rows)
        let c = TableController(service: service)
        await c.refresh()
        func kokoro() -> ModelRow { c.rows.first { $0.id == "kokoro-82m" }! }
        #expect(c.presentation(kokoro()).action == .get)
        c.perform(kokoro())
        try await waitUntil { c.presentation(kokoro()).action == .load }
        c.perform(kokoro())
        try await waitUntil { c.presentation(kokoro()).action == .unload }
        #expect(c.presentation(kokoro()).hot)
        c.perform(kokoro())
        try await waitUntil { c.presentation(kokoro()).action == .load }
        #expect(c.lastError == nil)
    }

    @Test func aLargeGetAsksFirstAndASmallOneDoesNot() async throws {
        let rows = Self.failedGets(["kokoro-82m", "flux2-klein-4b"])
        let c = TableController(service: FixtureService(rows: rows))
        await c.refresh()
        var asked: [(String, Int64?)] = []
        c.confirmGet = { row, plan in
            asked.append((row.id, plan.totalDownloadBytes)); return false
        }
        c.perform(c.rows.first { $0.id == "flux2-klein-4b" }!)
        try await waitUntil { !asked.isEmpty }
        #expect(asked.map(\.0) == ["flux2-klein-4b"])
        #expect(asked.first?.1 == 15_980_140_843)
        c.perform(c.rows.first { $0.id == "kokoro-82m" }!)
        try await waitUntil { c.presentation(c.rows.first { $0.id == "kokoro-82m" }!).action == .load }
        #expect(asked.count == 1)
        #expect(c.rows.first { $0.id == "flux2-klein-4b" }!.state == .failed)
    }

    /// v2.2 item 6: a small Get that also needs shared assets asks once, listing them with sizes; a
    /// bundled part (0 bytes) is not listed. (A part of unknown size is an error, not a question: V22ReviewFixTests.)
    @Test func aGetWithDependenciesAsksWithTheListAndCancelDownloadsNothing() async throws {
        let service = FixtureService(rows: Self.failedGets(["kokoro-82m"]))
        await service.setPlan(
            GetPlan(
                id: "kokoro-82m", dtype: "fp16", runner: "mlx-audio",
                items: [
                    GetPlanItem(id: "weights", name: "Kokoro 82M weights", kind: "weights", downloadBytes: 342_000_000),
                    GetPlanItem(id: "voices", name: "Voice assets", kind: "asset", downloadBytes: 48_000_000),
                    GetPlanItem(id: "tokenizer", name: "Tokenizer assets", kind: "asset", downloadBytes: 12_000_000),
                    GetPlanItem(id: "native", name: "Arsenal native engine", kind: "engine", downloadBytes: 0)
                ]))
        let c = TableController(service: service)
        await c.refresh()
        var plans: [GetPlan] = []
        c.confirmGet = { _, plan in
            plans.append(plan); return false
        }
        let kokoro = c.rows.first { $0.id == "kokoro-82m" }!
        c.perform(kokoro)
        try await waitUntil { !plans.isEmpty }
        let alert = ModelsMenu.getAlert(kokoro, plan: plans[0])
        #expect(alert.messageText == "Get Kokoro 82M \u{00b7} 402 MB?")
        let expected = [
            "To run it, Arsenal downloads and sets up:", "\u{2022} Kokoro 82M weights \u{00b7} 342 MB", "\u{2022} Voice assets \u{00b7} 48 MB",
            "\u{2022} Tokenizer assets \u{00b7} 12 MB"
        ]
        #expect(alert.informativeText == expected.joined(separator: "\n"))
        #expect(alert.buttons.map(\.title) == ["Get", "Cancel"])
        #expect(ModelsMenu.plainName("Native engine and anyio, certifi, click, mflux, torch") == "Native engine and 5 libraries")
        #expect(ModelsMenu.plainName("Bundled engine") == "Bundled engine")
        try await Task.sleep(for: .milliseconds(100))
        #expect(c.rows.first { $0.id == "kokoro-82m" }!.state == .failed)
        #expect(!TableController.asks(GetPlan(id: "k", dtype: "fp16", runner: "native", items: [plans[0].items[0], plans[0].items[3]])))
    }

    @Test func aServiceErrorReachesTheFooter() async throws {
        let c = TableController(service: FixtureService(rows: Self.failedGets(["kokoro-82m"])))
        await c.refresh()
        var kokoro = c.rows.first { $0.id == "kokoro-82m" }!
        // A row the service has as ready, but the stub has not installed: its load refuses.
        kokoro.states["fp16"] = PrecisionState(state: .ready)
        c.perform(kokoro)
        try await waitUntil { c.lastError != nil }
        #expect(c.lastError!.hasPrefix("Kokoro 82M: "))
    }

    /// The footer shows errors about table models and general ones, never about a model removed from the table.
    @Test func errorsAboutModelsOutsideTheTableStayOut() async throws {
        let rows = Self.failedGets(["kokoro-82m"])
        let service = FixtureService(rows: rows.filter { $0.id != "flux2-klein-4b" }, catalog: rows.filter { $0.id == "flux2-klein-4b" })
        let c = TableController(service: service)
        c.start()
        defer { c.stop() }
        try await waitUntil { !c.rows.isEmpty }
        #expect(!c.rows.contains { $0.id == "flux2-klein-4b" })
        await service.emit(.error(id: "flux2-klein-4b", message: "removed"))
        await service.emit(.error(id: "kokoro-82m", message: "kokoro failed"))
        try await waitUntil { c.lastError != nil }
        // Attached to the model it belongs to, by name.
        #expect(c.lastError == "Kokoro 82M: kokoro failed")
        await service.emit(.error(id: nil, message: "general"))
        try await waitUntil { c.lastError == "general" }
    }

    @Test func dropSlotRule() {
        let spans: [(top: CGFloat, extent: CGFloat)] = [(0, 100), (100, 60), (160, 200)]
        #expect(GroupHeaderView.slot(pointerY: 10, spans: spans) == 0)
        #expect(GroupHeaderView.slot(pointerY: 60, spans: spans) == 1)
        #expect(GroupHeaderView.slot(pointerY: 140, spans: spans) == 2)
        #expect(GroupHeaderView.slot(pointerY: 300, spans: spans) == 3)
    }

    @Test func hostChipName() {
        #expect(HostChip.display("Apple M5 Max") == "M5 Max")
        #expect(HostChip.display("") == nil)
    }
}

@MainActor func waitUntil(timeout: TimeInterval = 3, _ condition: @MainActor () async -> Bool) async throws {
    let deadline = Date().addingTimeInterval(timeout)
    while !(await condition()) {
        if Date() > deadline { Issue.record("condition not met within \(timeout) s"); return }
        try await Task.sleep(for: .milliseconds(10))
    }
}
