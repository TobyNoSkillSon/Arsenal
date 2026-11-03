// SPDX-License-Identifier: MIT
import Foundation
import Testing
@testable import Arsenal
import ArsenalCore

/// Gates scheduling only. Every order read/write uses RealModelsService and its real config.json.
/// Rows are weight-free presentation fixtures so the production controller can exercise user moves.
@MainActor private final class OrderModelService: ModelsService {
    let real: RealModelsService
    let file: URL
    var inventory: [KindSpec] = []
    private var nextFull: Int?
    private var nextRows: Int?
    private var reads: [Int: CheckedContinuation<Void, Never>] = [:]
    private var write: CheckedContinuation<Bool, Never>?
    private(set) var writing: [String]?
    private(set) var completedWrites = 0
    private(set) var readBacks = 0
    private(set) var rowReads = 0
    init(real: RealModelsService, file: URL) { self.real = real; self.file = file }
    func arm(_ id: Int, full: Bool) { if full { nextFull = id } else { nextRows = id } }
    func reading(_ id: Int) -> Bool { reads[id] != nil }
    func release(_ id: Int) { reads.removeValue(forKey: id)?.resume() }
    func finishWrite(fail: Bool) { write?.resume(returning: fail); write = nil }
    func rows() async -> [ModelRow] {
        let snapshot = inventory.sorted { $0.id < $1.id }.map { ModelRow(id: $0.id, name: $0.title, kind: $0.id) }
        rowReads += 1
        if let id = nextRows { nextRows = nil; await withCheckedContinuation { reads[id] = $0 } }
        return snapshot
    }
    func kinds() async -> [KindSpec] {
        let snapshot = await real.kinds()
        if let id = nextFull { nextFull = nil; await withCheckedContinuation { reads[id] = $0 } }
        return snapshot
    }
    func persistedKindOrder() async -> [String] { readBacks += 1; return await real.persistedKindOrder() }
    func setKindOrder(_ order: [String]) async throws {
        writing = order
        let fail = await withCheckedContinuation { write = $0 }
        let bytes = try Data(contentsOf: file)
        // An actual rejected atomic config write, rather than a stubbed service error.
        if fail { try Data("broken json".utf8).write(to: file) }
        defer {
            if fail { try? bytes.write(to: file, options: .atomic) }
            writing = nil; completedWrites += 1
        }
        try await real.setKindOrder(order)
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

/// Independent list model: no production sorter, merger or controller helper is used.
private struct OrderReference {
    static let defaults = ["c", "a", "b", "d", "e"]
    var present: [String] = []
    var displayedPresent: [String] = []
    var user: [String] = []
    var confirmed: [String] = []
    var persisted: [String] = []
    var writes: [[String]] = []
    var sequence = 0
    var applied = 0
    var display: [String] {
        user.filter(displayedPresent.contains) + Self.defaults.filter { displayedPresent.contains($0) && !user.contains($0) }
    }
    mutating func move(from: Int, slot: Int) -> String? {
        var requested = display
        let id = requested.remove(at: from)
        requested.insert(id, at: slot > from ? slot - 1 : slot)
        guard requested != display else { return nil }
        user = requested; writes.append(requested); return id
    }
    mutating func finish(fail: Bool) -> Bool {
        let requested = writes.removeFirst()
        let valid = requested.allSatisfy { present.contains($0) || persisted.contains($0) }
        if !fail && valid { persisted = requested; confirmed = requested; return true }
        if writes.isEmpty { user = confirmed }
        return false
    }
}

private struct OrderRandom {
    var state: UInt64
    mutating func choose(_ count: Int) -> Int {
        state = state &* 6_364_136_223_846_793_005 &+ 1_442_695_040_888_963_407
        return Int((state >> 32) % UInt64(count))
    }
}

@MainActor private final class OrderModelRun {
    struct Read {
        let id: Int
        let full: Bool
        let snapshot: [String]
        let task: Task<Void, Never>
    }
    let fixture = OrderRestartFixture()
    var service: OrderModelService!
    var table: TableController!
    var model = OrderReference()
    var pending: [Read] = []
    var counts: [String: Int] = [:]
    func count(_ event: String) { counts[event, default: 0] += 1 }
    func open() async throws {
        let real = try await fixture.open()
        service = OrderModelService(real: real, file: fixture.configFile)
        service.inventory = await real.kinds()
        table = TableController(service: service)
        await table.refresh()
        model.user = model.persisted; model.confirmed = model.persisted
        model.displayedPresent = model.present; model.sequence = 1; model.applied = 1
    }
    func inventory(_ ids: [String]) async throws {
        try fixture.inventory(ids)
        model.present = ids
        if let service {
            try await service.real.refresh()
            service.inventory = await service.real.kinds()
        }
    }
    func startRead(full: Bool) async throws {
        model.sequence += 1
        let id = model.sequence, snapshot = model.present, rowsBefore = service.rowReads
        service.arm(id, full: full)
        let controller = table!
        let task = Task { if full { await controller.refresh() } else { await controller.refreshRows() } }
        try await eventually { self.service.reading(id) && self.service.rowReads > rowsBefore }
        pending.append(Read(id: id, full: full, snapshot: snapshot, task: task))
        count(full ? "fullStarted" : "rowStarted")
    }
    func completeRead(_ index: Int) async {
        let read = pending.remove(at: index)
        if read.id > model.applied {
            if read.full {
                model.displayedPresent = read.snapshot; model.applied = read.id
            } else if Set(read.snapshot) != Set(model.displayedPresent) {
                // The production row path escalates changed membership to a fresh full read.
                model.sequence += 1; model.applied = model.sequence; model.displayedPresent = model.present
                count("rowEscalated")
            } else {
                model.applied = read.id
            }
        } else {
            count(read.full ? "staleFull" : "staleRow")
        }
        service.release(read.id); await read.task.value
        count(read.full ? "fullCompleted" : "rowCompleted")
    }
    func move(from: Int, slot: Int) async throws {
        let queued = !model.writes.isEmpty
        guard let id = model.move(from: from, slot: slot) else { count("moveNoOp"); return }
        if queued { count("queuedMove") }
        table.moveGroup(id, toSlot: slot)
        try await eventually { self.service.writing != nil }
        #expect(service.writing == model.writes.first)
        count("move")
    }
    func finishWrite(fail: Bool) async throws {
        if !pending.isEmpty { count("saveDuringRead") }
        let completed = service.completedWrites, readBacks = service.readBacks
        table.lastError = nil
        let succeeded = model.finish(fail: fail)
        service.finishWrite(fail: fail)
        try await eventually {
            self.service.completedWrites > completed && (succeeded ? self.service.readBacks > readBacks : self.table.lastError != nil)
                && (self.model.writes.isEmpty || self.service.writing == self.model.writes.first)
        }
        // Let the read-back's main-actor continuation finish before the next scheduled event.
        for _ in 0..<8 { await Task.yield() }
        count(succeeded ? "saveSuccess" : "saveFailure")
    }
    func restart() async throws {
        if !model.writes.isEmpty { count("restartWithSave") }
        if !pending.isEmpty { count("restartWithRead") }
        // Resolve writes at the shutdown boundary; held reads may finish after newer reads before reopening.
        while !model.writes.isEmpty { try await finishWrite(fail: true) }
        while !pending.isEmpty { await completeRead(pending.count - 1) }
        await service.real.shutdown(); try await open(); count("restart")
    }
    func check(seed: UInt64, step: Int) async throws {
        let context = "seed=\(seed), step=\(step), events=\(counts)"
        #expect(table.groupOrder == model.display, Comment(rawValue: context))
        #expect(table.kinds.map(\.id) == model.display, Comment(rawValue: context))
        #expect(try fixture.saved() == model.persisted, Comment(rawValue: context))
        #expect(await service.real.persistedKindOrder() == model.persisted, Comment(rawValue: context))
    }
    /// Cooperative event waits have a deadline; no timer polling is needed in these CPU-only schedules.
    private func eventually(_ condition: () -> Bool) async throws {
        let deadline = ContinuousClock.now + .seconds(3)
        while !condition() {
            guard ContinuousClock.now < deadline else { throw WorkerFailure(code: "test_timeout", message: "Order model gate did not settle.") }
            await Task.yield()
        }
    }
}

@MainActor @Suite(.menuState, .serialized) struct TableOrderModelTests {
    @Test func newlyDiscoveredKindMovedFirstIsStoredVerbatimAndSurvivesRestart() async throws {
        let run = OrderModelRun()
        try await run.inventory(["a", "b"]); try await run.open()
        try await run.service.real.setKindOrder(["b", "a"])
        await run.service.real.shutdown(); run.model.persisted = ["b", "a"]; try await run.open()
        try await run.inventory(["a", "b", "c"])
        try await run.startRead(full: true); await run.completeRead(0)
        #expect(run.table.groupOrder == ["b", "a", "c"])
        try await run.move(from: 2, slot: 0)
        try await run.finishWrite(fail: false)
        #expect(try run.fixture.saved() == ["c", "b", "a"])
        try await run.restart(); try await run.check(seed: 0, step: 0)
        #expect(run.table.groupOrder == ["c", "b", "a"])
        await run.service.real.shutdown()
    }

    @Test(arguments: [UInt64(7), 0xC0FFEE, 0x5EED, 0xDEADBEEF])
    func randomizedProductionConfigSchedules(seed: UInt64) async throws {
        let run = OrderModelRun()
        var random = OrderRandom(state: seed)
        try await run.inventory(["a", "b"]); try await run.open()
        for step in 0..<1_000 {
            switch random.choose(10) {
            case 0, 1:
                if !run.model.display.isEmpty {
                    let from = random.choose(run.model.display.count), slot = random.choose(run.model.display.count + 1)
                    try await run.move(from: from, slot: slot)
                }
            case 2, 3:
                if !run.model.writes.isEmpty { try await run.finishWrite(fail: random.choose(3) == 0) }
            case 4:
                let missing = OrderReference.defaults.filter { !run.model.present.contains($0) }
                if !missing.isEmpty { try await run.inventory(run.model.present + [missing[random.choose(missing.count)]]); run.count("appeared") }
            case 5:
                if !run.model.present.isEmpty {
                    var ids = run.model.present; ids.remove(at: random.choose(ids.count))
                    try await run.inventory(ids); run.count("disappeared")
                }
            case 6, 7:
                if run.pending.count < 6 { try await run.startRead(full: random.choose(2) == 0) }
            case 8 where !run.pending.isEmpty:
                await run.completeRead(random.choose(run.pending.count))
            case 9:
                try await run.restart()
            default: break
            }
            try await run.check(seed: seed, step: step)
        }
        while !run.model.writes.isEmpty { try await run.finishWrite(fail: false) }
        while !run.pending.isEmpty { await run.completeRead(run.pending.count - 1) }
        try await run.restart(); try await run.check(seed: seed, step: 1_000)
        for event in [
            "move", "saveSuccess", "saveFailure", "appeared", "disappeared", "staleFull", "staleRow", "queuedMove", "saveDuringRead", "restartWithSave", "restartWithRead"
        ] {
            #expect(run.counts[event, default: 0] > 0)
        }
        print("Order model seed=\(seed), randomSteps=1000, events=\(run.counts.sorted { $0.key < $1.key })")
        await run.service.real.shutdown()
    }
}
