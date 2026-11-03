// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// Gated snapshots and saves exercise the controller; nothing runs a worker or posts input.
private actor OrderOwnershipService: ModelsService {
    var currentKinds = Fixtures.fallbackKinds
    var currentRows = Fixtures.rows(.loaded)
    private var order = Fixtures.fallbackKinds.map(\.id)
    private var nextRead = 0
    private var heldReads: [Int: CheckedContinuation<Void, Never>] = [:]
    private var gatedReads: Set<Int> = []
    private var gatedWrite = false
    private var write: CheckedContinuation<Void, Never>?
    private var failWrite = false
    var ignoreWrite = false
    private var gateRowRead = false
    private var rowRead: CheckedContinuation<Void, Never>?
    func gateRows() { gateRowRead = true }
    var readingRows: Bool { rowRead != nil }
    func releaseRows() { rowRead?.resume(); rowRead = nil }
    private(set) var saved: [[String]] = []
    var saving: Bool { write != nil }
    func gateRead() -> Int { let id = nextRead; gatedReads.insert(id); return id }
    func reading(_ id: Int) -> Bool { heldReads[id] != nil }
    func releaseRead(_ id: Int) { heldReads.removeValue(forKey: id)?.resume() }
    func gateWrite(fail: Bool = false) { gatedWrite = true; failWrite = fail }
    func releaseWrite() { write?.resume(); write = nil }
    func ignoreNextWrite() { ignoreWrite = true }
    func replaceKinds(_ kinds: [KindSpec]) { currentKinds = kinds }
    func replaceRows(_ rows: [ModelRow]) { currentRows = rows }
    func rows() async -> [ModelRow] {
        let snapshot = currentRows
        if gateRowRead { gateRowRead = false; await withCheckedContinuation { rowRead = $0 } }
        return snapshot
    }
    func kinds() async -> [KindSpec] {
        let id = nextRead; nextRead += 1
        let snapshot = currentKinds.sorted { (order.firstIndex(of: $0.id) ?? Int.max) < (order.firstIndex(of: $1.id) ?? Int.max) }
        if gatedReads.remove(id) != nil { await withCheckedContinuation { heldReads[id] = $0 } }
        return snapshot
    }
    func persistedKindOrder() async -> [String] { order }
    func setKindOrder(_ value: [String]) async throws {
        if gatedWrite { gatedWrite = false; await withCheckedContinuation { write = $0 } }
        if failWrite { failWrite = false; throw WorkerFailure(code: "fixture", message: "Fixture disk is read-only.") }
        if ignoreWrite { ignoreWrite = false; return }
        order = value; saved.append(value)
    }
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

@MainActor @Suite(.menuState, .serialized) struct TableOrderOwnershipTests {
    @Test func successfulReturnWithDifferentReadBackDoesNotConfirmOrder() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh()
        let original = table.groupOrder
        await service.ignoreNextWrite(); table.moveGroup("chat", toSlot: 0)
        try await waitUntil { table.lastError != nil }
        #expect(table.groupOrder == original)
        #expect(table.lastError?.contains("stored group order differs") == true)
        // A later failed save must still revert to the original, never the falsely acknowledged move.
        table.lastError = nil
        await service.gateWrite(fail: true); table.moveGroup("voices", by: 1)
        try await waitUntil { await service.saving }
        await service.releaseWrite(); try await waitUntil { table.lastError != nil }
        #expect(table.groupOrder == original)
    }
    enum Interleaving: CaseIterable {
        case beforeMove, duringSaveFinishesBeforeSave, duringSaveFinishesAfterSave, afterSaveFinishesAfterSecondMove
    }
    @Test(arguments: Interleaving.allCases)
    func refreshNeverWritesTheUserOrder(interleaving: Interleaving) async throws {
        let service = OrderOwnershipService()
        // Keep one controller/service pair; its first read is the only ordering seed.
        let table = TableController(service: service)
        await table.refresh()
        let original = table.groupOrder
        await service.gateWrite()
        let stale: Task<Void, Never>
        let read: Int
        if interleaving == .beforeMove {
            read = await service.gateRead(); stale = Task { await table.refresh() }
            try await waitUntil { await service.reading(read) }
            table.moveGroup("chat", toSlot: 0)
        } else {
            table.moveGroup("chat", toSlot: 0)
            try await waitUntil { await service.saving }
            if interleaving == .afterSaveFinishesAfterSecondMove {
                await service.releaseWrite(); try await waitUntil { await service.saved.count == 1 }
            }
            read = await service.gateRead(); stale = Task { await table.refresh() }
            try await waitUntil { await service.reading(read) }
        }
        if interleaving != .afterSaveFinishesAfterSecondMove { try await waitUntil { await service.saving } }
        if interleaving == .afterSaveFinishesAfterSecondMove { table.moveGroup("voices", by: 1) }
        let chosen = table.groupOrder
        #expect(chosen != original)
        for _ in 0..<3 { await table.refreshRows(); #expect(table.groupOrder == chosen && table.kinds.map(\.id) == chosen) }
        if interleaving == .duringSaveFinishesBeforeSave {
            await service.releaseRead(read); await stale.value
            #expect(table.groupOrder == chosen && table.kinds.map(\.id) == chosen)
        }
        await service.releaseWrite()
        let count = interleaving == .afterSaveFinishesAfterSecondMove ? 2 : 1
        try await waitUntil { await service.saved.count == count }
        await table.refresh(); await table.refreshRows()
        if interleaving != .duringSaveFinishesBeforeSave { await service.releaseRead(read); await stale.value }
        await table.refreshRows()
        #expect(table.groupOrder == chosen && table.kinds.map(\.id) == chosen)
        #expect(await service.saved.last == chosen)
    }

    @Test func overlappingInitialReadsSeedOnlyOnce() async throws {
        let service = OrderOwnershipService()
        let controller = TableController(service: service)
        let read = await service.gateRead(), stale = Task { await controller.refresh() }
        try await waitUntil { await service.reading(read) }
        await controller.refresh()
        controller.moveGroup("chat", toSlot: 0)
        let chosen = controller.groupOrder
        try await waitUntil { await service.saved.count == 1 }
        await service.releaseRead(read); await stale.value; await controller.refreshRows()
        #expect(controller.groupOrder == chosen && controller.kinds.map(\.id) == chosen)
    }

    @Test func refreshReconcilesMembershipAndMetadataWithoutReordering() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh(); table.moveGroup("chat", toSlot: 0)
        try await waitUntil { await service.saved.count == 1 }
        let read = await service.gateRead(), refresh = Task { await table.refresh() }
        try await waitUntil { await service.reading(read) }
        let retained = table.kinds.filter { $0.id != "images" }.reversed().map { kind in
            KindSpec(id: kind.id, title: kind.title + " updated", order: kind.order, columns: kind.columns)
        }
        await service.replaceKinds([KindSpec(id: "new", title: "New", order: -1, columns: [])] + retained)
        await service.releaseRead(read); await refresh.value
        let expected = table.groupOrder.filter { $0 != "images" } + ["new"]
        await table.refresh()
        #expect(table.groupOrder == expected && table.kinds.map(\.id) == expected)
        #expect(table.kinds.first?.title.hasSuffix(" updated") == true)
        #expect(await service.saved.count == 1) // Membership reads do not persist anything.
    }

    @Test func oldRefreshCannotRemoveANewKindAfterItIsMovedAndSaved() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        let all = Fixtures.fallbackKinds
        let initial = all.filter { ["images", "voices"].contains($0.id) }
        await service.replaceKinds(initial)
        await service.replaceRows(Fixtures.rows(.loaded).filter { ["images", "voices"].contains($0.kind) })
        await table.refresh()
        let read = await service.gateRead(), old = Task { await table.refresh() }
        try await waitUntil { await service.reading(read) }
        let latestKinds = initial + all.filter { $0.id == "chat" }
        var latestRows = Fixtures.rows(.loaded).filter { ["images", "voices", "chat"].contains($0.kind) }
        for index in latestRows.indices { latestRows[index].name += " latest"; latestRows[index].toolsShown = false }
        await service.replaceKinds(latestKinds); await service.replaceRows(latestRows)
        await table.refresh()
        table.moveGroup("chat", toSlot: 0)
        let chosen = ["chat"] + initial.map(\.id)
        try await waitUntil { await service.saved.last == chosen }
        let latestSettings = table.settings, latestRunners = table.runners
        let chat = try #require(latestRows.first { $0.kind == "chat" })
        defer { table.settingsCard.close() }
        table.openSettings(chat)
        await service.releaseRead(read); await old.value
        #expect(table.groupOrder == chosen && table.kinds.map(\.id) == chosen)
        #expect(table.rows == latestRows && table.settings == latestSettings && table.runners == latestRunners)
        #expect(table.settingsCard.openID == chat.id)
        await table.refreshRows(); await table.refresh()
        #expect(table.groupOrder == chosen && table.kinds.map(\.id) == chosen)
        #expect(await service.saved.last == chosen)
    }

    @Test(arguments: [false, true]) func absentKindKeepsItsPositionUntilTheUserSaves(pruneBySaving: Bool) async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh(); table.moveGroup("chat", toSlot: 0)
        try await waitUntil { await service.saved.count == 1 }
        let remembered = table.groupOrder, all = table.kinds
        await service.replaceKinds(all.filter { $0.id != "chat" }); await table.refresh()
        #expect(table.groupOrder == remembered.filter { $0 != "chat" })
        if pruneBySaving {
            table.moveGroup("voices", by: 1)
            try await waitUntil { await service.saved.count == 2 }
            #expect(await service.saved.last?.contains("chat") == false)
        }
        let remaining = table.groupOrder
        await service.replaceKinds(all); await table.refresh(); await table.refreshRows()
        let expected = pruneBySaving ? remaining + ["chat"] : remembered
        #expect(table.groupOrder == expected && table.kinds.map(\.id) == expected)
    }

    @Test func absentKindKeepsItsConfirmedPositionAfterARejectedSave() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh(); table.moveGroup("chat", toSlot: 0)
        try await waitUntil { await service.saved.count == 1 }
        let remembered = table.groupOrder, all = table.kinds
        await service.replaceKinds(all.filter { $0.id != "chat" }); await table.refresh()
        await service.gateWrite(fail: true); table.moveGroup("voices", by: 1)
        try await waitUntil { await service.saving }
        await service.releaseWrite()
        try await waitUntil { table.lastError != nil }
        #expect(table.groupOrder == remembered.filter { $0 != "chat" })
        await service.replaceKinds(all); await table.refresh()
        #expect(table.groupOrder == remembered && table.kinds.map(\.id) == remembered)
        #expect(await service.saved.last == remembered)
    }

    @Test func progressRefreshDropsAnOlderFullResultEntirely() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh()
        let read = await service.gateRead(), old = Task { await table.refresh() }
        try await waitUntil { await service.reading(read) }
        var latest = table.rows
        for index in latest.indices { latest[index].name += " latest" }
        await service.replaceRows(latest); await table.refreshRows()
        let currentKinds = table.kinds, currentSettings = table.settings
        await service.releaseRead(read); await old.value
        #expect(table.rows == latest && table.kinds == currentKinds && table.settings == currentSettings)
    }

    @Test func newerFullRefreshDropsAnOlderProgressResult() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh(); await service.gateRows()
        let old = Task { await table.refreshRows() }
        try await waitUntil { await service.readingRows }
        var latest = table.rows
        for index in latest.indices { latest[index].name += " latest" }
        await service.replaceRows(latest); await table.refresh()
        await service.releaseRows(); await old.value
        #expect(table.rows == latest)
    }

    @Test func initialProgressRefreshReadsThePersistedOrderEvenWithNoRows() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        let persisted = ["chat", "images", "voices", "decisions"]
        try await service.setKindOrder(persisted); await service.replaceRows([])
        await table.refreshRows()
        #expect(table.groupOrder == persisted && table.kinds.map(\.id) == persisted)
    }

    @Test func unknownKindsAppendInStableOrderDespiteSnapshotOrder() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh()
        let remembered = table.groupOrder, initial = table.kinds
        let new = [KindSpec(id: "a", title: "A", order: -1, columns: []), KindSpec(id: "b", title: "B", order: -1, columns: [])]
        await service.replaceKinds(initial + new.reversed()); await table.refresh()
        #expect(table.groupOrder == remembered + ["a", "b"])
        await service.replaceKinds(new + initial.reversed()); await table.refresh()
        #expect(table.groupOrder == remembered + ["a", "b"])
        #expect(await service.saved.isEmpty)
    }

    @Test(arguments: [false, true]) func failedSaveRevertsToLastConfirmedOrder(afterSuccessfulMove: Bool) async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh()
        if afterSuccessfulMove {
            table.moveGroup("chat", toSlot: 0)
            try await waitUntil { await service.saved.count == 1 }
            await table.refresh()
        }
        let confirmed = table.groupOrder
        await service.gateWrite(fail: true)
        table.moveGroup("voices", by: 1)
        try await waitUntil { await service.saving }
        #expect(table.groupOrder != confirmed)
        await table.refresh(); await table.refreshRows()
        await service.releaseWrite()
        try await waitUntil { table.lastError != nil }
        #expect(table.groupOrder == confirmed && table.kinds.map(\.id) == confirmed)
        #expect(table.lastError?.contains("Fixture disk is read-only.") == true)
        await table.refresh(); await table.refreshRows()
        #expect(table.groupOrder == confirmed)
    }

    @Test func failedOlderSaveDoesNotEraseANewerQueuedMove() async throws {
        let service = OrderOwnershipService(), table = TableController(service: service)
        await table.refresh(); await service.gateWrite(fail: true)
        table.moveGroup("chat", toSlot: 0)
        try await waitUntil { await service.saving }
        table.moveGroup("voices", by: 1)
        let chosen = table.groupOrder
        await table.refreshRows(); await service.releaseWrite()
        try await waitUntil { await service.saved.count == 1 }
        #expect(table.groupOrder == chosen && table.kinds.map(\.id) == chosen)
        #expect(await service.saved.last == chosen)
        #expect(table.lastError?.contains("Fixture disk is read-only.") == true)
    }
}
