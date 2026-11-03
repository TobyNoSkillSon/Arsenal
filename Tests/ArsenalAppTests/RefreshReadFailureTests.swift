// SPDX-License-Identifier: MIT
import Foundation
import Testing
import ArsenalCore
@testable import Arsenal

/// Persistent failures are injected behind the real TableController/AppDelegate subscription, including every
/// menu read. Faulty getters deliberately publish every event kind as read feedback before returning. Independent
/// state changes use ordinary events and must survive the same refresh scope.
@MainActor @Suite(.menuState, .serialized) struct RefreshReadFailureTests {
    @Test(arguments: RefreshReadService.Read.allCases)
    func persistentReadFailureSettlesAfterExactlyOneRefresh(read: RefreshReadService.Read) async throws {
        let root = FileManager.default.temporaryDirectory.appendingPathComponent("arsenal-refresh-failure-" + UUID().uuidString)
        defer { try? FileManager.default.removeItem(at: root) }
        let service = RefreshReadService(failure: read)
        let app = AppDelegate(service: service, paths: ArsenalPaths(support: root))
        defer { app.table.stop(); app.tools.closeAll() }
        var completed = 0
        app.table.start {
            await app.refreshMenuState(); completed += 1
        }
        try await waitUntil { completed == 1 && app.table.lastError == "Fixture: " + read.reason }
        let counts = await service.counts
        #expect(counts[read] == (read == .settings ? 2 : 1))
        #expect(app.table.lastError == "Fixture: " + read.reason)
        if read == .voices { #expect(app.tools.errors["fixture"] == "Voices unavailable: " + read.reason) }
        if read == .admission || read == .operations {
            #expect(app.tools.admissions["fixture"]?[read == .operations ? .operation : .image]?.refusal?.message == read.reason)
        }
        try await Task.sleep(for: .milliseconds(220))
        #expect(completed == 1)
        #expect(await service.counts == counts)
        // A real state change after settling retries once; the persistent failure cannot perpetuate that retry.
        await service.stateChanged()
        try await waitUntil { completed == 2 }
        try await Task.sleep(for: .milliseconds(220))
        #expect(completed == 2)
        #expect(await service.counts[read] == (read == .settings ? 4 : 2))
        #expect(app.table.lastError == "Fixture: " + read.reason)
    }

    @Test func progressReadFailureSettlesWithoutPromotingItselfForever() async throws {
        let service = RefreshReadService(failure: .rows)
        let table = TableController(service: service)
        defer { table.stop() }
        var completed = 0
        table.start { completed += 1 }
        try await waitUntil { completed == 1 && !table.isRefreshing }
        await service.progressChanged()
        // One rows read fails and asks for a full refresh; that full read must not request another.
        try await waitUntil { completed == 2 }
        try await Task.sleep(for: .milliseconds(220))
        #expect(completed == 2)
        #expect(await service.counts[.rows] == 3)
        #expect(table.lastError == "Fixture: " + RefreshReadService.Read.rows.reason)
    }

    @Test func rowsReadCannotRepeatOnItsOwnProgressEvent() async throws {
        let service = RefreshReadService(failure: .rows, progressOnly: true)
        let table = TableController(service: service)
        defer { table.stop() }
        var completed = 0
        table.start { completed += 1 }
        try await waitUntil { completed == 1 && !table.isRefreshing }
        await service.progressChanged()
        try await waitUntil { await service.counts[.rows] == 2 }
        try await Task.sleep(for: .milliseconds(220))
        #expect(completed == 1 && table.rows.first?.verification == RefreshReadService.Read.rows.reason)
        #expect(await service.counts[.rows] == 2)
    }

    @Test func lateFailureEventShowsItsReasonWithoutRetryingAnyRead() async throws {
        let service = RefreshReadService()
        let table = TableController(service: service)
        defer { table.stop() }
        var completed = 0
        table.start { completed += 1 }
        try await waitUntil { completed == 1 && !table.isRefreshing }
        let counts = await service.counts
        await service.reportFailure()
        try await waitUntil { table.lastError == "Fixture: Delayed read failure." }
        try await Task.sleep(for: .milliseconds(220))
        #expect(completed == 1 && !table.isRefreshing)
        #expect(await service.counts == counts)
    }

    private func waitUntil(_ condition: @MainActor () async -> Bool) async throws {
        let deadline = ContinuousClock.now.advanced(by: .seconds(3))
        while !(await condition()) {
            guard ContinuousClock.now < deadline else { throw ToolError("Refresh did not settle") }
            try await Task.sleep(for: .milliseconds(5))
        }
    }
}

actor RefreshReadService: ModelsService {
    enum Read: String, Sendable, CaseIterable {
        case rows, kinds, order, catalogStatus, refreshRefusal, catalog, settings, panels
        case actions, sections, memory, residency, memorySettings, admission, operations, voices
        var reason: String { "Persistent " + rawValue + " failure." }
    }
    let failure: Read?
    let progressOnly: Bool
    private var continuation: AsyncStream<ServiceEvent>.Continuation?
    private(set) var counts: [Read: Int] = [:]
    private var inventory = "empty", voice = "af_heart", saved = false
    var voiceDelay: Duration = .zero
    private var rowsError: String?
    private var verifiedCatalog = false
    func setVerifiedCatalog() { verifiedCatalog = true }
    func setRowsError(_ message: String) { rowsError = message }
    private var currentRows: [ModelRow]?
    private var rowRevision = 0
    private var pauseRows = false
    private var rowsGate: CheckedContinuation<Void, Never>?
    private var pauseWaiters: [CheckedContinuation<Void, Never>] = []
    private var rowWaiters: [(Int, CheckedContinuation<Void, Never>)] = []
    var rowsPaused: Bool { rowsGate != nil }
    func waitForPausedRows() async {
        guard rowsGate == nil else { return }
        await withCheckedContinuation { pauseWaiters.append($0) }
    }
    func waitForRowsRead(_ count: Int) async {
        guard counts[.rows, default: 0] < count else { return }
        await withCheckedContinuation { rowWaiters.append((count, $0)) }
    }
    func holdNextRowsRead() { pauseRows = true }
    func releaseRowsRead() { rowsGate?.resume(); rowsGate = nil }
    @discardableResult func changeRows(_ rows: [ModelRow]) -> CatalogStatus {
        currentRows = rows
        rowRevision += 1
        let status = CatalogStatus(source: "fixture-change-\(rowRevision)", verified: true)
        // Independent notifications can arrive together; only one trailing full refresh should be retained.
        continuation?.yield(.rowsChanged)
        continuation?.yield(.memory(MemorySnapshot()))
        continuation?.yield(.catalog(status))
        return status
    }
    init(failure: Read? = nil, progressOnly: Bool = false) { self.failure = failure; self.progressOnly = progressOnly }
    private func read(_ read: Read) async -> Bool {
        counts[read, default: 0] += 1
        guard failure == read else { return false }
        if progressOnly {
            continuation?.yield(.readFeedback(.progress(id: "fixture", precision: "bf16", progress: 0.5))); return true
        }
        continuation?.yield(.readFeedback(.error(id: "fixture", message: read.reason)))
        continuation?.yield(.readFeedback(.rowsChanged))
        continuation?.yield(.readFeedback(.memory(MemorySnapshot())))
        continuation?.yield(.readFeedback(.catalog(CatalogStatus(error: read.reason))))
        continuation?.yield(.readFeedback(.progress(id: "fixture", precision: "bf16", progress: 0.5)))
        // Return immediately: buffered failure events must still be covered when the last read completes.
        return true
    }
    func rows() async -> [ModelRow] {
        let failed = await read(.rows)
        let ready = rowWaiters.filter { $0.0 <= counts[.rows, default: 0] }
        rowWaiters.removeAll { $0.0 <= counts[.rows, default: 0] }
        for (_, waiter) in ready { waiter.resume() }
        if let rowsError { continuation?.yield(.readFeedback(.error(id: "fixture", message: rowsError))) }
        var row = ModelRow(id: "fixture", name: "Fixture", kind: "voices", states: ["bf16": .init(state: .loaded)])
        row.tools = ["voice", "image", "operation"]
        if failed { row.verification = Read.rows.reason }
        let snapshot = currentRows ?? [row]
        if pauseRows {
            pauseRows = false
            await withCheckedContinuation {
                rowsGate = $0
                let waiters = pauseWaiters; pauseWaiters = []
                for waiter in waiters { waiter.resume() }
            }
        }
        return snapshot
    }
    func kinds() async -> [KindSpec] { _ = await read(.kinds); return KindSpec.defaults }
    func persistedKindOrder() async -> [String] { _ = await read(.order); return ["voices"] }
    func catalogStatus() async -> CatalogStatus { CatalogStatus(verified: verifiedCatalog, error: await read(.catalogStatus) ? Read.catalogStatus.reason : nil) }
    func refreshRefusal() async -> String? { await read(.refreshRefusal) ? Read.refreshRefusal.reason : nil }
    func catalog(query: CatalogQuery) async -> [CatalogEntry] { _ = await read(.catalog); return [] }
    func settingsPanel(id: String) async -> SettingsPanel? { _ = await read(.panels); return nil }
    func settings(id: String) async throws -> [SettingState] {
        if await read(.settings) { throw WorkerFailure(code: "fixture", message: Read.settings.reason) }
        return [SettingState(declaration: ["id": "voice", "type": "voice", "saved_voices_sha256": .string(inventory)], value: .string(voice))]
    }
    func voiceInventory(id: String) async throws -> JSONObject {
        if await read(.voices) { throw WorkerFailure(code: "fixture", message: Read.voices.reason) }
        try await Task.sleep(for: voiceDelay)
        return ["voices": .array([["id": "af_heart", "name": "Heart"]] + (saved ? [["id": "voice-saved", "name": "Narrator"]] : []))]
    }
    func setInventory(_ inventory: String, voice: String, saved: Bool) { self.inventory = inventory; self.voice = voice; self.saved = saved }
    func setVoiceDelay(_ delay: Duration) { voiceDelay = delay }
    func runRefusal(id: String, op: String) async -> WorkerFailure? { await read(.admission) ? WorkerFailure(code: "fixture", message: Read.admission.reason) : nil }
    func primaryOperationAdmission(id: String, preferred: String?) async -> RunAdmission {
        RunAdmission(operation: "speak", refusal: await read(.operations) ? WorkerFailure(code: "fixture", message: Read.operations.reason) : nil)
    }
    func menuActions() async -> [LoadedAction] { _ = await read(.actions); return [] }
    func menuSections() async -> [LoadedMenuSection] { _ = await read(.sections); return [] }
    func memory() async -> MemorySnapshot { _ = await read(.memory); return MemorySnapshot() }
    func residencySettings() async -> ResidencySettings { _ = await read(.residency); return ResidencySettings() }
    func memorySettings() async -> MemorySettings { _ = await read(.memorySettings); return MemorySettings() }
    func events() -> AsyncStream<ServiceEvent> { let (stream, c) = AsyncStream<ServiceEvent>.makeStream(); continuation = c; return stream }
    func reportFailure() { continuation?.yield(.error(id: "fixture", message: "Delayed read failure.")) }
    func stateChanged() { continuation?.yield(.rowsChanged) }
    func progressChanged() { continuation?.yield(.progress(id: "fixture", precision: "bf16", progress: 0.5)) }
    func get(id: String, precision: String) {}
    func cancelGet(id: String) {}
    func load(id: String, precision: String?, fast: Bool?) {}
    func unload(id: String) {}
    func setKeepHot(id: String, on: Bool) {}
    func delete(id: String) {}
    func setRunner(id: String, runner: String?) {}
    func run(id: String, op: String, params: JSONObject, sink: RunSink?) throws -> JSONObject { throw WorkerFailure(code: "fixture", message: "Refresh must never call run.") }
    func runRaw(id: String, op: String, params: JSONObject) -> Data { Data() }
    func refresh() {}
    func activity() -> ActivitySnapshot { ActivitySnapshot() }
    func activityStream(machine: Bool) -> AsyncStream<ActivitySnapshot> { AsyncStream { $0.finish() } }
}
